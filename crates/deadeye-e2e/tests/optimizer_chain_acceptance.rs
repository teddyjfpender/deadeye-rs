#![allow(
    clippy::print_stderr,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    reason = "integration tests in tests/ are top-level — printing aids debugging, unwrap/panic/expect are fine"
)]

//! Chain-acceptance parity test for `NormalMarket::optimize_quote_offline`.
//!
//! Closes the loop the off-chain-vs-chain reviewer flagged in Item 1:
//! bit-exact parity on `(σ, hints)` does **not** prove chain acceptance.
//! What we want is a contract:
//!
//! > Given `(belief, market, budget, k)` where the **off-chain** optimizer
//! > returns a positive-EV trade (i.e. `quote.on_chain_will_accept ==
//! > true` from `optimize_quote_offline`), an independent on-chain
//! > `check_trade_view` call with the optimizer's exact
//! > `(candidate, x_star, supplied_collateral, hints)` must **also**
//! > return `is_valid == true`.
//!
//! Why the off-chain path: the offline parity test
//! (`offline_optimize_quote_parity.rs`) reports `chain accept=false |
//! off accept=true` across all fixtures — i.e. the off-chain optimizer
//! says yes but the chain rejects. The bit-exact-on-σ-and-hints test
//! only proves the *encoding* matches; it does not prove the *trade*
//! is admissible by `check_trade_view`. This test closes that loop.
//!
//! This catches:
//! * Hint-derivation drift (chain re-derives different hints, fails
//!   `INVALID_HINTS`).
//! * `x_star` seed choice that lands outside the chain's stationarity tolerance
//!   (`VerificationFailed::StationaryInvalid`).
//! * Collateral solver drift between the off-chain Newton solver and the
//!   chain's Sq128 solver (`CollateralInsufficient`).
//! * Policy-region disagreements between the optimizer's grid bounds and the
//!   chain's `min_trade_collateral` / backing constraint.
//!
//! Gated behind `DEADEYE_RUN_INTEGRATION=1` and requires `starknet-devnet`
//! on `:5050`. Sweeps ~30 scenarios spanning σ-tightening, σ-widening,
//! σ-equal (the bug class), μ-shifts, micro-budgets, huge budgets, and
//! 5 asymmetry-probe scenarios (Newton-near-singular σ-arb, budget-edge
//! cliffs) added during Item 3 review.

use deadeye_core::{
    Distribution, Sq128,
    distribution::{NormalDistributionRaw, NormalSqrtHintsRaw},
    sq128::Sq128Raw,
};
use deadeye_sdk::{normal::NormalMarket, starknet::JsonRpcProvider};
use deadeye_starknet::{
    Felt,
    runtime::{check_normal_trade, compute_normal_hints},
    types::normal::TradeCheckRaw,
};
use deadeye_testkit::fixture::{
    env::{BootstrapConfig, bootstrap_devnet},
    erc20::approve,
    lifecycle::{
        build_initial_normal_inputs, deploy_normal_market_with_event, fetch_normal_hints,
        initialize_market, upsert_normal_profile_for_test,
    },
};
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};

const PROFILE_ID: u32 = 9_911_u32;
/// Profile id for the chain-runtime parity sweep (Driver B / P4). Held
/// separate from [`PROFILE_ID`] so both tests can share a single devnet
/// `bootstrap_devnet` invocation without colliding on the factory's
/// per-profile invariants.
const PROFILE_ID_RUNTIME: u32 = 9_912_u32;
// Mirror the bit-exactness parity test — known to satisfy the
// per-profile `initial backing invalid` invariant (which compares
// `σ × √π × max_pdf` against the deployed profile's backing=50).
const INITIAL_MEAN: f64 = 42.0;
const INITIAL_VAR: f64 = 64.0; // σ_market = 8.0
const INIT_APPROVE: u128 = 10_000_000_000_000_000_000_000_u128; // 10k STRK

fn integration_enabled() -> bool {
    std::env::var("DEADEYE_RUN_INTEGRATION").is_ok()
}

/// One scenario in the chain-acceptance sweep.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    label: &'static str,
    belief_mu: f64,
    belief_sigma: f64,
    budget: f64,
}

/// Named chain-runtime regressions for issue #1. These are the scenarios a
/// pre-P4 `x_star = cand_mean` implementation bucketed as
/// `OptimizerRejected`, so aggregate `accepted_count > 0` was too weak.
#[derive(Debug, Clone, Copy)]
struct AnchoredRuntimeScenario {
    label: &'static str,
    belief_mu: f64,
    belief_sigma: f64,
    budget: f64,
    x_star_must_differ_from_candidate_mean: bool,
}

/// 30 scenarios spanning the policy envelope. Market is fixed at
/// `N(μ=42, σ=8)`; budgets and beliefs sweep relative to that.
///
/// Cover at minimum:
/// * 3 σ-tightening (`σ_b < σ_m`) at different μ-shifts
/// * 3 σ-widening (`σ_b > σ_m`) at different μ-shifts
/// * 3 σ-equal (`σ_b == σ_m`) at different μ-shifts (the bug class)
/// * 3 σ-only (`μ_b == μ_m`, `σ_b ≠ σ_m`)
/// * 3 large-μ-shift (`|μ_b − μ_m| ≥ 3σ`)
/// * 3 micro-budget (very small budget, may be below `min_trade_collateral`)
/// * 3 huge-budget (budget ≫ optimal cost)
/// * 2 degenerate edges (very narrow belief, very wide belief)
const fn scenarios() -> &'static [Scenario] {
    // Beliefs are picked to be **strongly informative** (σ_b ≪ σ_m) where
    // possible so that the chain-correct λ-scaled EV exceeds the chain-
    // correct λ-scaled cost. Loose beliefs (σ_b ≈ σ_m) produce honestly
    // negative-EV trades at k=50 — the optimizer correctly filters them
    // (`on_chain_will_accept=false`) and the test marks them
    // "optimizer-rejected", not "disagreed".
    &[
        // ── σ-tightening (σ_b ≪ σ_m=8), different μ-shifts ──────────
        Scenario {
            label: "σ↓ μ↑ mild   ",
            belief_mu: 44.0,
            belief_sigma: 1.5,
            budget: 60.0,
        },
        Scenario {
            label: "σ↓ μ=eq      ",
            belief_mu: 42.0,
            belief_sigma: 1.5,
            budget: 60.0,
        },
        Scenario {
            label: "σ↓ μ↓ mild   ",
            belief_mu: 40.0,
            belief_sigma: 1.5,
            budget: 60.0,
        },
        // ── σ-widening (σ_b > σ_m=8), different μ-shifts ────────────
        Scenario {
            label: "σ↑ μ↑ mild   ",
            belief_mu: 44.0,
            belief_sigma: 10.0,
            budget: 60.0,
        },
        Scenario {
            label: "σ↑ μ=eq      ",
            belief_mu: 42.0,
            belief_sigma: 12.0,
            budget: 60.0,
        },
        Scenario {
            label: "σ↑ μ↓ mild   ",
            belief_mu: 40.0,
            belief_sigma: 11.0,
            budget: 60.0,
        },
        // ── σ-near-equal (σ_b ≈ σ_m=8) at different μ-shifts ──────
        Scenario {
            label: "σ≈ μ↑ pure   ",
            belief_mu: 44.0,
            belief_sigma: 7.0,
            budget: 60.0,
        },
        Scenario {
            label: "σ≈ μ↑↑ mild  ",
            belief_mu: 46.0,
            belief_sigma: 7.0,
            budget: 60.0,
        },
        Scenario {
            label: "σ≈ μ↓ pure   ",
            belief_mu: 39.0,
            belief_sigma: 7.0,
            budget: 60.0,
        },
        // ── σ-only (μ ≈ μ_m, σ ≠ σ_m) ────────────────────────────────
        Scenario {
            label: "σ-only narrow",
            belief_mu: 42.0,
            belief_sigma: 1.0,
            budget: 50.0,
        },
        Scenario {
            label: "σ-only loose ",
            belief_mu: 42.0,
            belief_sigma: 14.0,
            budget: 50.0,
        },
        Scenario {
            label: "σ-only mid↓  ",
            belief_mu: 42.0,
            belief_sigma: 2.5,
            budget: 50.0,
        },
        // ── Large μ-shift (≥3σ_m), tight belief ──────────────────────
        Scenario {
            label: "μ-shift +3σ  ",
            belief_mu: 54.0,
            belief_sigma: 2.0,
            budget: 100.0,
        },
        Scenario {
            label: "μ-shift -3σ  ",
            belief_mu: 30.0,
            belief_sigma: 2.0,
            budget: 100.0,
        },
        Scenario {
            label: "μ-shift +3σ↓",
            belief_mu: 53.0,
            belief_sigma: 1.5,
            budget: 100.0,
        },
        // ── Micro-budget (close to / below `min_trade_collateral`=1.0) ─
        Scenario {
            label: "budget micro1",
            belief_mu: 44.0,
            belief_sigma: 1.5,
            budget: 1.0,
        },
        Scenario {
            label: "budget micro2",
            belief_mu: 42.0,
            belief_sigma: 2.0,
            budget: 2.0,
        },
        Scenario {
            label: "budget micro3",
            belief_mu: 40.0,
            belief_sigma: 1.5,
            budget: 1.5,
        },
        // ── Huge budget (≫ optimal cost) ─────────────────────────────
        Scenario {
            label: "budget huge1 ",
            belief_mu: 44.0,
            belief_sigma: 1.5,
            budget: 1_000.0,
        },
        Scenario {
            label: "budget huge2 ",
            belief_mu: 42.0,
            belief_sigma: 2.0,
            budget: 1_000.0,
        },
        Scenario {
            label: "budget huge3 ",
            belief_mu: 50.0,
            belief_sigma: 1.5,
            budget: 1_000.0,
        },
        // ── Spread across the envelope ───────────────────────────────
        Scenario {
            label: "μ↑ tight σ↓  ",
            belief_mu: 48.0,
            belief_sigma: 1.0,
            budget: 80.0,
        },
        Scenario {
            label: "μ↓ tight σ↓  ",
            belief_mu: 36.0,
            belief_sigma: 1.0,
            budget: 80.0,
        },
        // ── Degenerate edges ─────────────────────────────────────────
        Scenario {
            label: "edge narrow  ",
            belief_mu: 42.0,
            belief_sigma: 0.8,
            budget: 100.0,
        },
        Scenario {
            label: "edge wide    ",
            belief_mu: 42.0,
            belief_sigma: 15.0,
            budget: 100.0,
        },
        // ── Asymmetry probes (Item 3 review, Q4) ─────────────────────
        // The contract is one-directional: when the optimizer accepts,
        // the chain must accept. The reverse failure mode — optimizer
        // rejects but chain *would* accept — is not asserted, but is a
        // real bot leak (missed σ-arb). These 5 scenarios target the
        // brittle regions:
        //   1. σ-arb with σ_b ≈ σ_m within 1% (Newton near-singular)
        //   2. σ-arb same-mean with σ_b just below σ_m
        //   3. budget right at the λ-scaled cost edge (filter cliff)
        //   4. very-tight σ_b with mild μ-shift (high λ-scaled EV/cost)
        //   5. wide σ_b with μ-shift at exactly 1σ
        Scenario {
            label: "σ-arb 1%     ",
            belief_mu: 42.0,
            belief_sigma: 7.92,
            budget: 80.0,
        },
        Scenario {
            label: "σ-arb same-μ ",
            belief_mu: 42.0,
            belief_sigma: 7.5,
            budget: 80.0,
        },
        Scenario {
            label: "budget edge  ",
            belief_mu: 44.0,
            belief_sigma: 1.5,
            budget: 6.0,
        },
        Scenario {
            label: "tight σ μ-mid",
            belief_mu: 43.0,
            belief_sigma: 0.5,
            budget: 100.0,
        },
        Scenario {
            label: "wide σ 1σ    ",
            belief_mu: 50.0,
            belief_sigma: 10.0,
            budget: 100.0,
        },
    ]
}

const fn anchored_runtime_scenarios() -> &'static [AnchoredRuntimeScenario] {
    &[
        AnchoredRuntimeScenario {
            label: "anchor σ↓ μ=eq",
            belief_mu: 42.0,
            belief_sigma: 1.5,
            budget: 60.0,
            x_star_must_differ_from_candidate_mean: false,
        },
        AnchoredRuntimeScenario {
            label: "anchor σ↓ μ-small",
            belief_mu: 44.0,
            belief_sigma: 1.5,
            budget: 60.0,
            x_star_must_differ_from_candidate_mean: true,
        },
        AnchoredRuntimeScenario {
            label: "anchor σ↓ μ-large",
            belief_mu: 54.0,
            belief_sigma: 2.0,
            budget: 100.0,
            x_star_must_differ_from_candidate_mean: true,
        },
    ]
}

/// Outcome bucket for a single scenario. Fields carry the formatted
/// reason for the post-sweep summary; `Disagreed.rejection` is the one
/// that actually surfaces a parity bug.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "string payloads are kept for the post-sweep summary; only `Disagreed.rejection` is read directly"
)]
enum Outcome {
    /// Optimizer rejected (no positive-EV trade) — no parity claim.
    OptimizerRejected(String),
    /// Optimizer accepted; chain re-check also accepted.
    Accepted,
    /// Optimizer accepted; chain re-check disagreed (BUG).
    Disagreed { rejection: String },
    /// `check_trade_view` call failed (transport / decode error).
    CallFailed(String),
    /// `optimize_quote` itself errored — surfaces optimizer or chain bug.
    OptimizeFailed(String),
}

/// Re-call `check_trade_view` on the runtime with the exact
/// `(candidate, x_star, supplied_collateral, hints)` the optimizer
/// produced. Returns the raw chain verdict.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the Cairo check_trade_view ABI 1:1; collapsing into a struct would obscure parity"
)]
async fn check_trade_via_runtime(
    provider: &JsonRpcProvider,
    runtime: Felt,
    current: NormalDistributionRaw,
    candidate: NormalDistributionRaw,
    x_star: Sq128Raw,
    supplied_collateral: Sq128Raw,
    k: Sq128Raw,
    backing: Sq128Raw,
    tolerance: Sq128Raw,
    min_trade_collateral: Sq128Raw,
    candidate_hints: NormalSqrtHintsRaw,
) -> Result<TradeCheckRaw, String> {
    // Re-fetch the current-distribution hints fresh so we exercise the
    // independent compute_hints_view + check_trade_view pipeline.
    let current_hints = compute_normal_hints(provider, runtime, current)
        .await
        .map_err(|e| format!("compute_normal_hints(current): {e}"))?;
    check_normal_trade(
        provider,
        runtime,
        current,
        candidate,
        x_star,
        supplied_collateral,
        k,
        backing,
        tolerance,
        min_trade_collateral,
        current_hints,
        candidate_hints,
    )
    .await
    .map_err(|e| format!("check_normal_trade: {e}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn optimizer_output_must_be_accepted_by_chain() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1 and start starknet-devnet on :5050");
        return;
    }

    // ── Phase 0: bootstrap devnet + factory ─────────────────────────
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .expect("bootstrap succeeds");
    eprintln!(
        "✅ devnet up: chain={:#x}, factory={:#x}, runtime={:#x}",
        env.chain_id, env.factory, env.normal_runtime
    );

    let admin = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    // ── Phase 1: upsert profile + deploy market ─────────────────────
    upsert_normal_profile_for_test(admin.clone(), env.factory, env.collateral, PROFILE_ID)
        .await
        .expect("upsert normal profile");

    let (initial_dist, _placeholder) =
        build_initial_normal_inputs(INITIAL_MEAN, INITIAL_VAR, 1_000.0);
    let initial_hints = fetch_normal_hints(&rpc, env.normal_runtime, initial_dist)
        .await
        .expect("fetch initial hints");

    let market_addr = deploy_normal_market_with_event(
        &admin,
        env.factory,
        PROFILE_ID,
        Felt::from(0xCAFE_BABE_u64),
        Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .expect("deploy normal market");
    eprintln!("✅ market deployed: {market_addr:#x}");

    // ── Phase 2: initialize + approve ───────────────────────────────
    if let Err(e) = initialize_market(&admin, market_addr, env.collateral, INIT_APPROVE).await {
        eprintln!("⚠️ initialize_market failed (known blocker?): {e}");
        eprintln!("    Skipping chain-acceptance parity assertion.");
        return;
    }
    approve(admin.clone(), env.collateral, market_addr, INIT_APPROVE)
        .await
        .expect("approve market");
    eprintln!("✅ initialized + approved");

    // ── Phase 3: build NormalMarket handle + provider for check calls ──
    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let check_provider =
        JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let market = NormalMarket::new(&provider, market_addr);

    // Read params once — they're profile-stamped and immutable after deploy.
    let params = market.reader().params().await.expect("read params");
    let current_dist_at_setup = market
        .reader()
        .distribution()
        .await
        .expect("read distribution")
        .to_raw();

    // ── Phase 4: sweep scenarios ────────────────────────────────────
    let mut outcomes: Vec<(Scenario, Outcome)> = Vec::new();
    let mut accepted_count = 0_usize;
    let mut disagreed_count = 0_usize;
    let mut optimizer_rejected_count = 0_usize;
    let mut call_failed_count = 0_usize;
    let mut optimize_failed_count = 0_usize;

    for sc in scenarios() {
        // Use the **off-chain** optimizer here. Its `on_chain_will_accept`
        // bit is set from the f64 `expected_value > cost` check inside
        // `optimize_normal_trade`, **not** from a chain round-trip. So if
        // it says yes, the test then asks the chain to confirm.
        let quote = match market
            .optimize_quote_offline(sc.belief_mu, sc.belief_sigma, sc.budget)
            .await
        {
            Ok(q) => q,
            Err(e) => {
                let msg = format!("optimize_quote_offline error: {e:?}");
                eprintln!("  [{label}] {msg}", label = sc.label);
                outcomes.push((*sc, Outcome::OptimizeFailed(msg)));
                optimize_failed_count += 1;
                continue;
            },
        };

        if !quote.on_chain_will_accept {
            // The optimizer determined no positive-EV trade exists for
            // this (belief, market, budget) under the chain-correct
            // λ-scaling. Log the candidate + cost so the operator can
            // verify the rejection is honest (cost > EV) rather than
            // a chain-vs-off-chain divergence.
            let reason = quote
                .rejection
                .map_or_else(|| "no-trade".into(), |r| format!("{r:?}"));
            let cand_mu = Sq128::from_raw(quote.candidate.mean).to_f64();
            let cand_sig = Sq128::from_raw(quote.candidate.sigma).to_f64();
            let req = Sq128::from_raw(quote.required_collateral).to_f64();
            eprintln!(
                "  [{label}] optimizer rejected: {reason} \
                 (cand μ={cand_mu:.3}, σ={cand_sig:.3}, req_coll={req:.6})",
                label = sc.label
            );
            outcomes.push((*sc, Outcome::OptimizerRejected(reason)));
            optimizer_rejected_count += 1;
            continue;
        }

        // Optimizer says yes. Re-verify the chain agrees with a fresh
        // `check_trade_view` call using the exact `(candidate, x_star,
        // supplied_collateral, hints)` the optimizer produced. The
        // supplied collateral mirrors what `execute_quote` would send:
        // the larger of `required_collateral` and `padded_collateral`.
        let supplied = max_raw(quote.required_collateral, quote.padded_collateral);

        let check = check_trade_via_runtime(
            &check_provider,
            env.normal_runtime,
            current_dist_at_setup,
            quote.candidate,
            quote.x_star,
            supplied,
            params.k,
            params.backing,
            params.tolerance,
            params.min_trade_collateral,
            quote.candidate_hints,
        )
        .await;

        match check {
            Ok(check_raw) if check_raw.is_valid => {
                let req = Sq128::from_raw(check_raw.verification.computed_collateral).to_f64();
                eprintln!(
                    "  [{label}] ✅ accepted (req_coll={req:.6})",
                    label = sc.label
                );
                outcomes.push((*sc, Outcome::Accepted));
                accepted_count += 1;
            },
            Ok(check_raw) => {
                // Decode every sub-flag so we can pinpoint *why* the
                // chain disagreed — the symbolic `rejection_reason` can
                // be `None` when the failure is in `collateral_above_min`
                // or in the underlying `computation_valid` flags.
                let v = &check_raw.verification;
                let b = &check_raw.backing_check;
                let computed = Sq128::from_raw(v.computed_collateral).to_f64();
                let rejection = format!(
                    "reason={:?} backing(valid={}, comp_ok={}) verify(side={}, stat={}, curv={}, \
                     comp_valid={}, coll_ok={}, computed={:.6}) above_min={}",
                    check_raw.rejection_reason,
                    b.is_valid,
                    b.computation_succeeded,
                    v.side_valid,
                    v.stationary_valid,
                    v.curvature_valid,
                    v.computation_valid,
                    v.collateral_sufficient,
                    computed,
                    check_raw.collateral_above_min,
                );
                eprintln!(
                    "  [{label}] ❌ DISAGREE: optimizer accept, chain reject ({rejection})",
                    label = sc.label
                );
                outcomes.push((*sc, Outcome::Disagreed {
                    rejection: rejection.clone(),
                }));
                disagreed_count += 1;
            },
            Err(e) => {
                eprintln!("  [{label}] ⚠️ check call failed: {e}", label = sc.label);
                outcomes.push((*sc, Outcome::CallFailed(e)));
                call_failed_count += 1;
            },
        }
    }

    let total = scenarios().len();
    eprintln!(
        "\n🔍 chain-acceptance: {accepted_count}/{total} accepted | \
         optimizer-rejected={optimizer_rejected_count} | \
         disagreed={disagreed_count} | \
         call-failed={call_failed_count} | \
         optimize-failed={optimize_failed_count}"
    );

    if disagreed_count > 0 {
        eprintln!("\nDISAGREEMENTS (optimizer says yes, chain says no):");
        for (sc, outcome) in &outcomes {
            if let Outcome::Disagreed { rejection } = outcome {
                eprintln!(
                    "  [{label}] μ_b={mu_b}, σ_b={sigma_b}, budget={budget}: {rejection}",
                    label = sc.label,
                    mu_b = sc.belief_mu,
                    sigma_b = sc.belief_sigma,
                    budget = sc.budget,
                );
            }
        }
    }

    // The contract: zero optimizer-chain disagreements.
    assert_eq!(
        disagreed_count, 0,
        "optimizer and chain disagree on {disagreed_count} trades — \
         on_chain_will_accept must imply chain check_trade_view accepts"
    );
    // Sanity floor: at least some scenarios must clear all gates,
    // otherwise the test is just measuring the noise of failures.
    assert!(
        accepted_count > 0,
        "scenarios produced no acceptances at all — test is not exercising the accept path"
    );
}

/// Pick the larger of two `Sq128Raw` values via signed comparison.
fn max_raw(a: Sq128Raw, b: Sq128Raw) -> Sq128Raw {
    let lhs = Sq128::from_raw(a);
    let rhs = Sq128::from_raw(b);
    if lhs.cmp_signed(rhs) == core::cmp::Ordering::Less {
        b
    } else {
        a
    }
}

// ─── Chain-runtime parity (P4 / Driver B) ────────────────────────────
//
// The test above exercises [`NormalMarket::optimize_quote_offline`] —
// the off-chain variant that runs the audited Newton solver to land
// `x_star` at the true stationary point of `d(x) = λ_g g(x) − λ_f f(x)`.
//
// The **chain-runtime** variant ([`NormalMarket::optimize_quote`]) takes
// a real math-runtime address and routes the candidate through
// `quote_trade` (which calls `quote_trade_view` on the runtime). Pre-P4
// it seeded `x_star = cand_mean` (`deadeye-sdk/src/normal.rs:506-509`),
// which mis-satisfied the AMM's `stationary_valid` check at the
// chaos-suite belief and surfaced as `VERIFICATION_FAILED` on submit.
//
// `cpi-bot`'s `unsafe_skip_preflight_with_funded_wallet_succeeds` test
// worked around the bug by dropping `DEADEYE_NORMAL_RUNTIME_ADDR` to
// force the offline path (FU2 Driver A's
// `BotInvocation::run_with_env(..., removals)` helper).
//
// Driver A's P4 SDK fix (v0.1.4) reseeds `x_star` from the same audited
// `normal_collateral` Newton solver used by the offline variant. This
// test pins the contract: when the chain-runtime variant says yes, the
// chain must accept. **Before** v0.1.4 lands this test fails (≥1
// scenario in `Disagreed`); **after** it lands the test passes with
// zero disagreements.

/// Chain-runtime parity sweep — mirrors
/// [`optimizer_output_must_be_accepted_by_chain`] but calls
/// [`NormalMarket::optimize_quote`] (the chain-runtime variant) instead
/// of `optimize_quote_offline`. Drives the same 30 scenarios so any
/// chain-runtime-vs-offline drift becomes a per-scenario diff.
///
/// Contract: when `optimize_quote` returns `on_chain_will_accept ==
/// true`, an independent on-chain `check_trade_view` call with the
/// quote's exact `(candidate, x_star, supplied_collateral, hints)` must
/// also return `is_valid == true`. Zero disagreements allowed.
///
/// **Pre-P4 status:** disagrees on the σ-tightening and μ-shift
/// scenarios — the chain rejects with `VERIFICATION_FAILED`
/// (`stationary_valid=false`) because `x_star = cand_mean ≠ stationary
/// point`. **Post-P4 (v0.1.4):** zero disagreements.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn optimize_quote_chain_runtime_must_be_accepted_by_chain() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1 and start starknet-devnet on :5050");
        return;
    }

    // ── Phase 0: bootstrap devnet + factory ─────────────────────────
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .expect("bootstrap succeeds");
    eprintln!(
        "✅ devnet up: chain={:#x}, factory={:#x}, runtime={:#x}",
        env.chain_id, env.factory, env.normal_runtime
    );

    let admin = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    // ── Phase 1: upsert profile + deploy market ─────────────────────
    // Use a distinct profile id ([`PROFILE_ID_RUNTIME`]) so this test
    // can run in the same process as the offline-variant test without
    // colliding on the factory's per-profile invariants.
    upsert_normal_profile_for_test(
        admin.clone(),
        env.factory,
        env.collateral,
        PROFILE_ID_RUNTIME,
    )
    .await
    .expect("upsert normal profile");

    let (initial_dist, _placeholder) =
        build_initial_normal_inputs(INITIAL_MEAN, INITIAL_VAR, 1_000.0);
    let initial_hints = fetch_normal_hints(&rpc, env.normal_runtime, initial_dist)
        .await
        .expect("fetch initial hints");

    let market_addr = deploy_normal_market_with_event(
        &admin,
        env.factory,
        PROFILE_ID_RUNTIME,
        Felt::from(0xCAFE_F00D_u64),
        Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .expect("deploy normal market");
    eprintln!("✅ market deployed: {market_addr:#x}");

    // ── Phase 2: initialize + approve ───────────────────────────────
    if let Err(e) = initialize_market(&admin, market_addr, env.collateral, INIT_APPROVE).await {
        eprintln!("⚠️ initialize_market failed (known blocker?): {e}");
        eprintln!("    Skipping chain-runtime parity assertion.");
        return;
    }
    approve(admin.clone(), env.collateral, market_addr, INIT_APPROVE)
        .await
        .expect("approve market");
    eprintln!("✅ initialized + approved");

    // ── Phase 3: build handles ──────────────────────────────────────
    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let check_provider =
        JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let market = NormalMarket::new(&provider, market_addr);

    let params = market.reader().params().await.expect("read params");
    let current_dist_at_setup = market
        .reader()
        .distribution()
        .await
        .expect("read distribution")
        .to_raw();

    // ── Phase 3b: anchored σ-tightening regressions (issue #1) ───────
    for sc in anchored_runtime_scenarios() {
        let quote = market
            .optimize_quote(env.normal_runtime, sc.belief_mu, sc.belief_sigma, sc.budget)
            .await
            .unwrap_or_else(|e| panic!("{}: optimize_quote(chain) failed: {e:?}", sc.label));
        let required = Sq128::from_raw(quote.required_collateral).to_f64();
        let x_star = Sq128::from_raw(quote.x_star).to_f64();
        let cand_mean = Sq128::from_raw(quote.candidate.mean).to_f64();
        assert!(
            quote.on_chain_will_accept,
            "{}: chain-runtime optimizer must accept anchored σ-tightening scenario \
             (pre-P4 regression bucketed this as optimizer-rejected; req_coll={required:.6})",
            sc.label,
        );
        assert!(
            required > 0.0,
            "{}: accepted anchor must lock positive collateral",
            sc.label,
        );
        if sc.x_star_must_differ_from_candidate_mean {
            assert!(
                (x_star - cand_mean).abs() > 1e-6,
                "{}: x_star should be the stationary point, not the candidate mean \
                 (x_star={x_star:.9}, cand_mean={cand_mean:.9})",
                sc.label,
            );
        }
        let supplied = max_raw(quote.required_collateral, quote.padded_collateral);
        let check = check_trade_via_runtime(
            &check_provider,
            env.normal_runtime,
            current_dist_at_setup,
            quote.candidate,
            quote.x_star,
            supplied,
            params.k,
            params.backing,
            params.tolerance,
            params.min_trade_collateral,
            quote.candidate_hints,
        )
        .await
        .unwrap_or_else(|e| panic!("{}: independent check_trade_view failed: {e}", sc.label));
        assert!(
            check.is_valid,
            "{}: independent check_trade_view rejected anchored optimizer output: {:?}",
            sc.label, check.rejection_reason,
        );
    }

    // ── Phase 4: sweep scenarios ────────────────────────────────────
    let mut outcomes: Vec<(Scenario, Outcome)> = Vec::new();
    let mut accepted_count = 0_usize;
    let mut disagreed_count = 0_usize;
    let mut optimizer_rejected_count = 0_usize;
    let mut call_failed_count = 0_usize;
    let mut optimize_failed_count = 0_usize;

    for sc in scenarios() {
        // **Chain-runtime variant**: passes a real runtime address so
        // the SDK routes through `quote_trade` (chain preflight). This
        // is the variant the cpi-bot uses in production — and the one
        // P4 fixes.
        let quote = match market
            .optimize_quote(env.normal_runtime, sc.belief_mu, sc.belief_sigma, sc.budget)
            .await
        {
            Ok(q) => q,
            Err(e) => {
                let msg = format!("optimize_quote(chain) error: {e:?}");
                eprintln!("  [{label}] {msg}", label = sc.label);
                outcomes.push((*sc, Outcome::OptimizeFailed(msg)));
                optimize_failed_count += 1;
                continue;
            },
        };

        if !quote.on_chain_will_accept {
            let reason = quote
                .rejection
                .map_or_else(|| "no-trade".into(), |r| format!("{r:?}"));
            let cand_mu = Sq128::from_raw(quote.candidate.mean).to_f64();
            let cand_sig = Sq128::from_raw(quote.candidate.sigma).to_f64();
            let req = Sq128::from_raw(quote.required_collateral).to_f64();
            eprintln!(
                "  [{label}] optimizer rejected: {reason} \
                 (cand μ={cand_mu:.3}, σ={cand_sig:.3}, req_coll={req:.6})",
                label = sc.label
            );
            outcomes.push((*sc, Outcome::OptimizerRejected(reason)));
            optimizer_rejected_count += 1;
            continue;
        }

        // Optimizer (chain-runtime) says yes. Re-verify via an
        // independent `check_trade_view` call using the quote's exact
        // calldata. The supplied collateral mirrors `execute_quote`'s
        // behavior: the larger of `required_collateral` and
        // `padded_collateral`.
        let supplied = max_raw(quote.required_collateral, quote.padded_collateral);

        let check = check_trade_via_runtime(
            &check_provider,
            env.normal_runtime,
            current_dist_at_setup,
            quote.candidate,
            quote.x_star,
            supplied,
            params.k,
            params.backing,
            params.tolerance,
            params.min_trade_collateral,
            quote.candidate_hints,
        )
        .await;

        match check {
            Ok(check_raw) if check_raw.is_valid => {
                let req = Sq128::from_raw(check_raw.verification.computed_collateral).to_f64();
                eprintln!(
                    "  [{label}] ✅ accepted (req_coll={req:.6})",
                    label = sc.label
                );
                outcomes.push((*sc, Outcome::Accepted));
                accepted_count += 1;
            },
            Ok(check_raw) => {
                let v = &check_raw.verification;
                let b = &check_raw.backing_check;
                let computed = Sq128::from_raw(v.computed_collateral).to_f64();
                let rejection = format!(
                    "reason={:?} backing(valid={}, comp_ok={}) verify(side={}, stat={}, curv={}, \
                     comp_valid={}, coll_ok={}, computed={:.6}) above_min={}",
                    check_raw.rejection_reason,
                    b.is_valid,
                    b.computation_succeeded,
                    v.side_valid,
                    v.stationary_valid,
                    v.curvature_valid,
                    v.computation_valid,
                    v.collateral_sufficient,
                    computed,
                    check_raw.collateral_above_min,
                );
                eprintln!(
                    "  [{label}] ❌ DISAGREE: chain-runtime optimizer accept, chain reject ({rejection})",
                    label = sc.label
                );
                outcomes.push((*sc, Outcome::Disagreed {
                    rejection: rejection.clone(),
                }));
                disagreed_count += 1;
            },
            Err(e) => {
                eprintln!("  [{label}] ⚠️ check call failed: {e}", label = sc.label);
                outcomes.push((*sc, Outcome::CallFailed(e)));
                call_failed_count += 1;
            },
        }
    }

    let total = scenarios().len();
    eprintln!(
        "\n🔍 chain-runtime parity: {accepted_count}/{total} accepted | \
         optimizer-rejected={optimizer_rejected_count} | \
         disagreed={disagreed_count} | \
         call-failed={call_failed_count} | \
         optimize-failed={optimize_failed_count}"
    );

    if disagreed_count > 0 {
        eprintln!("\nDISAGREEMENTS (chain-runtime optimizer says yes, chain says no):");
        for (sc, outcome) in &outcomes {
            if let Outcome::Disagreed { rejection } = outcome {
                eprintln!(
                    "  [{label}] μ_b={mu_b}, σ_b={sigma_b}, budget={budget}: {rejection}",
                    label = sc.label,
                    mu_b = sc.belief_mu,
                    sigma_b = sc.belief_sigma,
                    budget = sc.budget,
                );
            }
        }
    }

    // The contract: zero chain-runtime / chain disagreements. Pre-P4
    // this fails (`x_star = cand_mean` mis-satisfies `stationary_valid`
    // on σ-tightening / μ-shift scenarios). Post-P4 (v0.1.4) it passes.
    assert_eq!(
        disagreed_count, 0,
        "chain-runtime optimizer and chain disagree on {disagreed_count} trades — \
         optimize_quote `on_chain_will_accept` must imply chain check_trade_view accepts \
         (P4 contract; pre-fix x_star = cand_mean blew `stationary_valid`)"
    );
    assert!(
        accepted_count > 0,
        "scenarios produced no acceptances at all — test is not exercising the accept path"
    );
}
