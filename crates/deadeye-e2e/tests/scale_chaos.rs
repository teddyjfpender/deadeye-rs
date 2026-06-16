#![allow(
    clippy::print_stderr,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::clone_on_copy,
    missing_copy_implementations,
    clippy::len_without_is_empty,
    clippy::single_match,
    clippy::single_match_else,
    clippy::mismatching_type_param_order,
    clippy::missing_docs_in_private_items,
    clippy::unreadable_literal,
    clippy::unusual_byte_groupings,
    clippy::inconsistent_digit_grouping,
    clippy::needless_pass_by_value,
    clippy::shadow_unrelated,
    clippy::wildcard_enum_match_arm,
    reason = "long-running integration driver — printing aids debugging, unwrap/panic mark hard \
              invariants, float ops are deliberate, magic numbers are scenario knobs; the \
              random-walk action helpers take `DevnetAccount` by value because the upstream \
              type became `Copy` after wave 2 — call sites historically cloned"
)]

//! # Scale chaos suite — N actions per family against a live devnet
//!
//! Weekly-nightly stress test. Bootstraps a fresh devnet, deploys one
//! market per supported family, then runs a random action stream
//! against each (trade, sell, LP add, LP remove). The off-chain solver
//! drives every action; the **normal** family submits each accepted
//! action to the chain so the test catches state-accumulation bugs the
//! per-family 17-action chaos suites can't see at scale.
//!
//! ## Why this exists
//!
//! The per-family chaos suites (`normal_chaos.rs` &c) run a fixed
//! 12-17 action plan with a hand-picked cast. That gives strong
//! step-by-step invariants but limited *depth*: a state-accumulation
//! bug that only manifests after dozens of trades — e.g. cumulative
//! Q128 rounding bias, treasury drift, or an off-by-one in the LP
//! shares accounting that compounds — wouldn't fire. This suite trades
//! the named scenarios for a deeper random walk against the same
//! conservation invariants.
//!
//! ## Time budget
//!
//! Each devnet trade costs ~25-30 s end-to-end (transaction submission,
//! mining, receipt fetch, snapshot reads). At 50 actions per family
//! and 4 families this is 200 × 30 s ≈ 100 min — too long for routine
//! `cargo test`. Double-gated as a result: `DEADEYE_RUN_INTEGRATION=1`
//! AND `DEADEYE_RUN_LONG=1`.
//!
//! ## Current scope
//!
//! * **Normal family** — fully wired. Every Trade action submits via
//!   `NormalMarketWriter::execute_trade`, every Sell via `sell_position`. The
//!   lambda-scaled solver from `deadeye-collateral` finds `x*`; the chain
//!   re-verifies.
//! * **Lognormal / Multinoulli / Bivariate families** — off-chain only at this
//!   revision. The wiring template is identical to the normal family's; landing
//!   it is mechanical. The off-chain solver is still exercised so the
//!   convergence-rate assertion is meaningful.
//!
//! ## Gating
//!
//! Two env vars must be set for this test to actually run:
//!
//! * `DEADEYE_RUN_INTEGRATION=1` — gates every integration test.
//! * `DEADEYE_RUN_LONG=1` — gates the slow tests inside the integration set.
//!   Without it, the test no-ops with a `skip:` log.
//!
//! ## Determinism
//!
//! A fixed seed (`0xDEAD_BEEF_5CA1_E5CA_u64`) drives the action picker
//! so failures are reproducible. Override via `DEADEYE_SCALE_SEED=<u64>`.

use std::{collections::BTreeMap, env};

use deadeye_collateral::{
    LognormalOptions, MinimizationPolicy, categorical_collateral, lognormal_collateral,
    normal_collateral,
};
use deadeye_core::{
    BivariateNormalDistributionCoreRaw, BivariatePointRaw, CategoricalDistribution,
    CategoricalDistributionRaw, CategoricalL2HintRaw, LognormalDistribution,
    LognormalDistributionRaw, NormalDistribution, NormalDistributionRaw, Sq128, Sq128Raw,
};
use deadeye_sdk::starknet::JsonRpcProvider;
use deadeye_starknet::{
    BivariateMarketReader, BivariateMarketWriter, Felt, LognormalMarketReader,
    LognormalMarketWriter, MultinoulliMarketReader, MultinoulliMarketWriter, NormalMarketReader,
    NormalMarketWriter, OwnedAccount, TradeError,
};
use deadeye_testkit::fixture::{
    bootstrap_devnet,
    env::{BootstrapConfig, TestEnv},
    erc20::{approve, balance_of},
    lifecycle::{
        build_initial_bivariate_inputs, build_initial_normal_inputs,
        deploy_bivariate_market_with_event, deploy_lognormal_market_with_event,
        deploy_multinoulli_market_with_event, deploy_normal_market_with_event,
        expand_bivariate_distribution, fetch_bivariate_hints, fetch_lognormal_hints,
        fetch_multinoulli_hint, fetch_normal_hints, initialize_market,
        upsert_bivariate_profile_for_test, upsert_lognormal_profile_for_test,
        upsert_multinoulli_profile_for_test, upsert_normal_profile_for_test,
    },
};
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};

/// Default seed when `DEADEYE_SCALE_SEED` is unset.
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_5CA1_E5CA_u64;

/// Per-family action budget. 4 families × 50 = 200 total, ~100 min wall.
///
/// Sized so a weekly-nightly run completes inside a CI two-hour budget
/// with margin for devnet bootstrap (~30 s) and post-action snapshot
/// reads. Increase locally when chasing a specific state-accumulation
/// bug; do not raise the default without first reviewing the wall-time
/// budget on the slowest available CI box.
const ACTIONS_PER_FAMILY: u32 = 50;

/// Minimum overall solver convergence rate the test demands. Below
/// this the chaos surface has degraded — most likely a regression in
/// the off-chain solver.
const MIN_CONVERGENCE_RATE: f64 = 0.90_f64;

/// Profile id used for every market this test deploys.
const PROFILE_ID: u32 = 1;
/// Trade allowance — generous so `transferFrom` never gates a trade.
const TRADE_ALLOWANCE: u128 = 1_000_000_000_000_000_000_000_u128; // 1000 STRK
/// Initial-backing approval. See `normal_chaos.rs` for the
/// `INITIALIZE_OVERFLOW` discussion.
const INIT_APPROVE_AMOUNT: u128 = 10_000_000_000_000_000_000_000_u128; // 10k STRK

fn integration_enabled() -> bool {
    env::var("DEADEYE_RUN_INTEGRATION").is_ok()
}
fn long_enabled() -> bool {
    env::var("DEADEYE_RUN_LONG").is_ok()
}
fn seed() -> u64 {
    env::var("DEADEYE_SCALE_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SEED)
}

/// Deterministic xorshift64* — no external rand dep.
struct DetRng(u64);
impl DetRng {
    const fn new(seed: u64) -> Self {
        // xorshift requires non-zero state.
        Self(if seed == 0 {
            0xDEAD_BEEF_DEAD_BEEF_u64
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D_u64)
    }
    fn next_f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn pick(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionKind {
    Trade,
    LpAdd,
    LpRemove,
    Sell,
    PartialClaim,
}

fn pick_action(rng: &mut DetRng) -> ActionKind {
    match rng.pick(5) {
        0 => ActionKind::Trade,
        1 => ActionKind::LpAdd,
        2 => ActionKind::LpRemove,
        3 => ActionKind::Sell,
        _ => ActionKind::PartialClaim,
    }
}

fn nd(mu: f64, var: f64) -> Result<NormalDistribution, ()> {
    let mu_q = Sq128::from_f64(mu).map_err(|_| ())?;
    let var_q = Sq128::from_f64(var).map_err(|_| ())?;
    NormalDistribution::from_variance(mu_q, var_q).map_err(|_| ())
}

fn sq_raw(value: f64) -> Sq128Raw {
    Sq128::from_f64(value).expect("finite f64").to_raw()
}

fn dist_raw(mean: f64, variance: f64) -> NormalDistributionRaw {
    NormalDistributionRaw {
        mean: sq_raw(mean),
        variance: sq_raw(variance),
        sigma: sq_raw(variance.sqrt()),
    }
}

#[derive(Debug, Default)]
struct FamilyStats {
    name: &'static str,
    attempts: u32,
    converged: u32,
    rejected_typed: u32,
    /// Number of actions that actually submitted a chain transaction
    /// (vs off-chain-only attempts).
    chain_submissions: u32,
    /// Number of chain submissions that reverted. Should be 0 in steady
    /// state — every chain submission is gated by the off-chain solver
    /// returning `Ok`, and the chain accepts any pair the solver
    /// accepts under `MinimizationPolicy::unrestricted`.
    chain_failures: u32,
    action_mix: [u32; 5], // [trade, lp_add, lp_remove, sell, partial_claim]
    /// Counts of each typed [`TradeRejectionReason`] observed on chain
    /// submissions. Wave-1 plumbing parses revert short-strings into
    /// the structured enum so the suite can report *why* the chain
    /// rejected, not just that it did.
    typed_rejections: BTreeMap<String, u32>,
}

impl FamilyStats {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            attempts: 0,
            converged: 0,
            rejected_typed: 0,
            chain_submissions: 0,
            chain_failures: 0,
            action_mix: [0; 5],
            typed_rejections: BTreeMap::new(),
        }
    }
    fn record_action(&mut self, kind: ActionKind) {
        let idx = match kind {
            ActionKind::Trade => 0,
            ActionKind::LpAdd => 1,
            ActionKind::LpRemove => 2,
            ActionKind::Sell => 3,
            ActionKind::PartialClaim => 4,
        };
        self.action_mix[idx] += 1;
    }
    fn record_solver(&mut self, ok: bool) {
        self.attempts += 1;
        if ok {
            self.converged += 1;
        } else {
            self.rejected_typed += 1;
        }
    }
    fn record_chain_submission(&mut self, ok: bool) {
        self.chain_submissions += 1;
        if !ok {
            self.chain_failures += 1;
        }
    }
    /// Record a typed `TradeRejectionReason` keyed by its `{:?}`
    /// representation. We bucket on a stringified key to keep the
    /// per-variant counter generic across families without dragging the
    /// (non-`Hash`) `VerificationSubReason` into the map's key shape.
    fn record_typed_rejection(&mut self, err: &TradeError) {
        if let Some(reason) = err.rejection() {
            *self
                .typed_rejections
                .entry(format!("{reason:?}"))
                .or_insert(0) += 1;
        }
    }
    fn convergence_rate(&self) -> f64 {
        if self.attempts == 0 {
            return 1.0;
        }
        f64::from(self.converged) / f64::from(self.attempts)
    }
}

/// Live chaos handle for the normal family — what we need to keep
/// around between actions in the random walk.
struct NormalLiveHandle {
    market: Felt,
    writer: NormalMarketWriter<JsonRpcProvider, OwnedAccount>,
    runtime: Felt,
    cur_mean: f64,
    cur_var: f64,
}

/// Bootstrap a devnet, deploy + initialize a normal market, and bind
/// the first participant as the chaos actor. Returns `None` if any
/// step fails — the suite logs and bails cleanly so a devnet hiccup
/// isn't an automatic test failure.
async fn bootstrap_normal_live() -> Option<(TestEnv, JsonRpcClient<HttpTransport>, NormalLiveHandle)>
{
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .map_err(|e| eprintln!("⚠️  bootstrap_devnet failed: {e}"))
        .ok()?;
    eprintln!(
        "✅ devnet bootstrapped — chain {chain:#x}",
        chain = env.chain_id
    );

    let admin_handle = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    upsert_normal_profile_for_test(
        admin_handle.clone(),
        env.factory,
        env.collateral,
        PROFILE_ID,
    )
    .await
    .map_err(|e| eprintln!("⚠️  upsert_normal_profile failed: {e}"))
    .ok()?;

    let initial_mean = 42.0_f64;
    let initial_var = 64.0_f64;
    let (initial_dist, _placeholder) = build_initial_normal_inputs(initial_mean, initial_var, 50.0);
    let hint_rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));
    let initial_hints = fetch_normal_hints(&hint_rpc, env.normal_runtime, initial_dist)
        .await
        .map_err(|e| eprintln!("⚠️  fetch_normal_hints failed: {e}"))
        .ok()?;

    let market = deploy_normal_market_with_event(
        &admin_handle,
        env.factory,
        PROFILE_ID,
        Felt::from(0x05CA_1E_u64),
        Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .map_err(|e| eprintln!("⚠️  deploy_normal_market failed: {e}"))
    .ok()?;
    eprintln!("✅ normal market deployed: {market:#x}");

    initialize_market(&admin_handle, market, env.collateral, INIT_APPROVE_AMOUNT)
        .await
        .map_err(|e| eprintln!("⚠️  initialize_market failed: {e}"))
        .ok()?;
    eprintln!("✅ market initialized");

    // The actor approves the market for the trade-side allowance.
    let actor = env.participants.first()?.clone();
    approve(
        env.account_handle(&actor),
        env.collateral,
        market,
        TRADE_ALLOWANCE,
    )
    .await
    .map_err(|e| eprintln!("⚠️  actor approve failed: {e}"))
    .ok()?;

    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let writer = NormalMarketWriter::new(
        NormalMarketReader::new(provider, market),
        env.owned_account(&actor),
    );

    let handle = NormalLiveHandle {
        market,
        writer,
        runtime: env.normal_runtime,
        cur_mean: initial_mean,
        cur_var: initial_var,
    };
    Some((env, rpc, handle))
}

/// Execute one random action against the live normal market. Returns
/// whether the off-chain solver converged + whether the chain submission
/// (if attempted) succeeded.
async fn step_normal_live(
    rng: &mut DetRng,
    env: &TestEnv,
    rpc: &JsonRpcClient<HttpTransport>,
    handle: &mut NormalLiveHandle,
    stats: &mut FamilyStats,
) {
    let action = pick_action(rng);
    stats.record_action(action);
    match action {
        ActionKind::Trade => {
            // Stay inside the chaos surface that the chaos suite gates:
            // |Δμ| ≤ ~3σ, σ ratio ≤ 4×. The unrestricted solver accepts
            // wider candidates but the on-chain re-check is more
            // conservative; this keeps the chain-failures count near 0.
            let target_mean = handle.cur_mean + rng.next_f64_in(-10.0, 10.0);
            let target_var = (handle.cur_var * rng.next_f64_in(0.5, 2.0)).max(4.0);
            let Ok(f) = nd(handle.cur_mean, handle.cur_var) else {
                stats.record_solver(false);
                return;
            };
            let Ok(g) = nd(target_mean, target_var) else {
                stats.record_solver(false);
                return;
            };
            let solve = normal_collateral(&f, &g, MinimizationPolicy::unrestricted());
            stats.record_solver(solve.is_ok());
            let Ok(verified) = solve else {
                return;
            };
            let cand_raw = dist_raw(target_mean, target_var);
            let supplied = sq_raw((verified.collateral * 20.0).max(100.0));
            // Wave-1 typed flow: quote → execute_quote. The quote
            // round-trips through the math runtime so the on-chain
            // verifier's verdict (`on_chain_will_accept`) is known
            // *before* we burn gas on a submission.
            let quote = match handle
                .writer
                .reader()
                .quote_trade(
                    handle.runtime,
                    cand_raw,
                    sq_raw(verified.x_min),
                    supplied,
                    supplied,
                )
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  normal quote_trade failed: {e}");
                    return;
                },
            };
            if !quote.on_chain_will_accept {
                if let Some(reason) = quote.rejection {
                    let key = format!("{reason:?}");
                    *stats.typed_rejections.entry(key).or_insert(0) += 1;
                }
                return;
            }
            match handle.writer.execute_quote(quote).await {
                Ok(_) => {
                    stats.record_chain_submission(true);
                    handle.cur_mean = target_mean;
                    handle.cur_var = target_var;
                },
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                    eprintln!(
                        "  ⚠️  normal chain trade rejected target=N({target_mean:.3},{target_var:.3}): {e}"
                    );
                },
            }
        },
        ActionKind::Sell => {
            // Sell is best-effort — succeeds only when the actor has a
            // live position. We record the solver attempt as converged
            // because the on-chain sell path doesn't run the minimizer.
            stats.record_solver(true);
            match handle.writer.sell_position(handle.runtime, 0).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                },
            }
        },
        ActionKind::LpAdd => {
            stats.record_solver(true);
            let amount = sq_raw(rng.next_f64_in(10.0, 100.0));
            match handle.writer.add_liquidity(amount).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  add_liquidity failed: {e}");
                },
            }
        },
        ActionKind::LpRemove => {
            stats.record_solver(true);
            let frac = sq_raw(rng.next_f64_in(0.05, 0.50));
            match handle.writer.remove_liquidity(frac).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  remove_liquidity failed: {e}");
                },
            }
        },
        ActionKind::PartialClaim => {
            // No solver call and (pre-settlement) no chain submission.
            stats.record_solver(true);
        },
    }

    // Snapshot the market balance every 10 actions so the run is
    // observable from the log even when the assertions stay green.
    if stats.attempts.is_multiple_of(10) {
        let bal = balance_of(rpc, env.collateral, handle.market)
            .await
            .unwrap_or(0);
        eprintln!(
            "    [{family}] @ action {n}: market_bal = {bal}",
            family = stats.name,
            n = stats.attempts,
        );
    }
}

// ─── Lognormal family ────────────────────────────────────────────────

struct LognormalLiveHandle {
    market: Felt,
    writer: LognormalMarketWriter<JsonRpcProvider, OwnedAccount>,
    runtime: Felt,
    cur_mu: f64,
    cur_var: f64,
}

/// Lognormal variance ladder kept on a perfect-square grid in f64 so
/// the on-chain `σ × σ == variance` cross-check still passes
/// bit-for-bit at Sq128 precision. Same trick used by `lognormal_chaos`.
const LOGNORMAL_VAR_GRID: &[f64] = &[
    0.0625, 0.140625, 0.25, 0.390625, 0.5625, 0.765625, 1.0, 1.5625, 2.25, 4.0,
];

fn ln_state(mu: f64, var: f64) -> Result<LognormalDistribution, ()> {
    let mu_q = Sq128::from_f64(mu).map_err(|_| ())?;
    let var_q = Sq128::from_f64(var).map_err(|_| ())?;
    LognormalDistribution::from_variance(mu_q, var_q).map_err(|_| ())
}

fn ln_raw(mu: f64, var: f64) -> LognormalDistributionRaw {
    LognormalDistributionRaw {
        mu: sq_raw(mu),
        variance: sq_raw(var),
        sigma: sq_raw(var.sqrt()),
    }
}

async fn bootstrap_lognormal_live(
    env: TestEnv,
) -> Option<(TestEnv, JsonRpcClient<HttpTransport>, LognormalLiveHandle)> {
    let admin_handle = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    upsert_lognormal_profile_for_test(
        admin_handle.clone(),
        env.factory,
        env.collateral,
        PROFILE_ID,
    )
    .await
    .map_err(|e| eprintln!("⚠️  upsert_lognormal_profile failed: {e}"))
    .ok()?;

    // Initial μ = ln(80_000), σ² = 0.25 — same anchor as `lognormal_chaos`.
    let initial_mu = 80_000_f64.ln();
    let initial_var = 0.25_f64;
    let initial_raw = ln_raw(initial_mu, initial_var);
    let initial_hints = fetch_lognormal_hints(&rpc, env.lognormal_runtime, initial_raw)
        .await
        .map_err(|e| eprintln!("⚠️  fetch_lognormal_hints failed: {e}"))
        .ok()?;

    let market = deploy_lognormal_market_with_event(
        &admin_handle,
        env.factory,
        PROFILE_ID,
        Felt::from(0x10_DEAD_u64),
        Felt::ZERO,
        initial_raw,
        initial_hints,
    )
    .await
    .map_err(|e| eprintln!("⚠️  deploy_lognormal_market failed: {e}"))
    .ok()?;
    eprintln!("✅ lognormal market deployed: {market:#x}");

    initialize_market(&admin_handle, market, env.collateral, INIT_APPROVE_AMOUNT)
        .await
        .map_err(|e| eprintln!("⚠️  initialize_market (lognormal) failed: {e}"))
        .ok()?;

    let actor = env.participants.first()?;
    let actor = *actor;
    approve(
        env.account_handle(&actor),
        env.collateral,
        market,
        TRADE_ALLOWANCE,
    )
    .await
    .map_err(|e| eprintln!("⚠️  actor approve failed: {e}"))
    .ok()?;

    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let writer = LognormalMarketWriter::new(
        LognormalMarketReader::new(provider, market),
        env.owned_account(&actor),
    );

    let handle = LognormalLiveHandle {
        market,
        writer,
        runtime: env.lognormal_runtime,
        cur_mu: initial_mu,
        cur_var: initial_var,
    };
    Some((env, rpc, handle))
}

async fn step_lognormal_live(
    rng: &mut DetRng,
    env: &TestEnv,
    rpc: &JsonRpcClient<HttpTransport>,
    handle: &mut LognormalLiveHandle,
    stats: &mut FamilyStats,
) {
    let action = pick_action(rng);
    stats.record_action(action);
    match action {
        ActionKind::Trade => {
            // μ moves by ±5%; variance picked from the perfect-square
            // ladder so the `σ × σ == variance` hint guard passes.
            let target_mu = handle.cur_mu + rng.next_f64_in(-0.05, 0.05);
            let var_idx = rng.pick(LOGNORMAL_VAR_GRID.len() as u32) as usize;
            let target_var = LOGNORMAL_VAR_GRID[var_idx];
            let Ok(f) = ln_state(handle.cur_mu, handle.cur_var) else {
                stats.record_solver(false);
                return;
            };
            let Ok(g) = ln_state(target_mu, target_var) else {
                stats.record_solver(false);
                return;
            };
            let solve = lognormal_collateral(&f, &g, LognormalOptions::default());
            stats.record_solver(solve.is_ok());
            let Ok(verified) = solve else {
                return;
            };
            let cand_raw = ln_raw(target_mu, target_var);
            let supplied = sq_raw((verified.collateral * 20.0).max(100.0));
            // Wave-1 typed flow.
            let quote = match handle
                .writer
                .reader()
                .quote_trade(
                    handle.runtime,
                    cand_raw,
                    sq_raw(verified.x_star),
                    supplied,
                    supplied,
                )
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  lognormal quote_trade failed: {e}");
                    return;
                },
            };
            if !quote.on_chain_will_accept {
                if let Some(reason) = quote.rejection {
                    let key = format!("{reason:?}");
                    *stats.typed_rejections.entry(key).or_insert(0) += 1;
                }
                return;
            }
            match handle.writer.execute_quote(quote).await {
                Ok(_) => {
                    stats.record_chain_submission(true);
                    handle.cur_mu = target_mu;
                    handle.cur_var = target_var;
                },
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                    eprintln!(
                        "  ⚠️  lognormal chain trade rejected μ={target_mu:.4} σ²={target_var}: {e}"
                    );
                },
            }
        },
        ActionKind::Sell => {
            stats.record_solver(true);
            match handle.writer.sell_position(handle.runtime, 0).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                },
            }
        },
        ActionKind::LpAdd => {
            stats.record_solver(true);
            let amount = sq_raw(rng.next_f64_in(10.0, 100.0));
            match handle.writer.add_liquidity(amount).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  lognormal add_liquidity failed: {e}");
                },
            }
        },
        ActionKind::LpRemove => {
            stats.record_solver(true);
            let frac = sq_raw(rng.next_f64_in(0.05, 0.50));
            match handle.writer.remove_liquidity(frac).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  lognormal remove_liquidity failed: {e}");
                },
            }
        },
        ActionKind::PartialClaim => {
            stats.record_solver(true);
        },
    }
    if stats.attempts.is_multiple_of(10) {
        let bal = balance_of(rpc, env.collateral, handle.market)
            .await
            .unwrap_or(0);
        eprintln!(
            "    [{family}] @ action {n}: market_bal = {bal}",
            family = stats.name,
            n = stats.attempts,
        );
    }
}

// ─── Multinoulli family ──────────────────────────────────────────────

const MULTINOULLI_OUTCOMES: usize = 6;
const MULTINOULLI_INITIAL_PROBS: [f64; MULTINOULLI_OUTCOMES] = [0.10, 0.25, 0.30, 0.05, 0.20, 0.10];

struct MultinoulliLiveHandle {
    market: Felt,
    writer: MultinoulliMarketWriter<JsonRpcProvider, OwnedAccount>,
    runtime: Felt,
    cur_probs: [f64; MULTINOULLI_OUTCOMES],
}

fn cat_dist(probs: &[f64]) -> Result<CategoricalDistribution, ()> {
    CategoricalDistribution::from_probs(probs.to_vec()).map_err(|_| ())
}

fn cat_raw(probs: &[f64]) -> CategoricalDistributionRaw {
    let probs_raw: Vec<Sq128Raw> = probs.iter().map(|p| sq_raw(*p)).collect();
    CategoricalDistributionRaw { probs: probs_raw }
}

/// Move probability mass: pick a "from" outcome, transfer `delta` to a
/// "to" outcome, renormalise. Keeps every probability strictly positive
/// so the chain's `INVALID_DISTRIBUTION` guard never trips.
fn perturb_probs(
    rng: &mut DetRng,
    probs: &[f64; MULTINOULLI_OUTCOMES],
) -> [f64; MULTINOULLI_OUTCOMES] {
    let mut out = *probs;
    let from = rng.pick(MULTINOULLI_OUTCOMES as u32) as usize;
    let mut to = rng.pick(MULTINOULLI_OUTCOMES as u32) as usize;
    if to == from {
        to = (to + 1) % MULTINOULLI_OUTCOMES;
    }
    // Take up to 30% of the from-outcome mass, clamp so the floor stays
    // ≥ 0.01 (1%) on the sender — well above the chain's degeneracy
    // threshold.
    let cap = (out[from] - 0.01).max(0.0);
    let delta = (rng.next_f64_in(0.01, 0.30) * out[from]).min(cap);
    out[from] -= delta;
    out[to] += delta;
    out
}

async fn bootstrap_multinoulli_live(
    env: TestEnv,
) -> Option<(TestEnv, JsonRpcClient<HttpTransport>, MultinoulliLiveHandle)> {
    let admin_handle = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    upsert_multinoulli_profile_for_test(
        admin_handle.clone(),
        env.factory,
        env.collateral,
        PROFILE_ID,
    )
    .await
    .map_err(|e| eprintln!("⚠️  upsert_multinoulli_profile failed: {e}"))
    .ok()?;

    let initial_raw = cat_raw(&MULTINOULLI_INITIAL_PROBS);
    let initial_hint: CategoricalL2HintRaw =
        fetch_multinoulli_hint(&rpc, env.multinoulli_runtime, &initial_raw)
            .await
            .map_err(|e| eprintln!("⚠️  fetch_multinoulli_hint failed: {e}"))
            .ok()?;

    let market = deploy_multinoulli_market_with_event(
        &admin_handle,
        env.factory,
        PROFILE_ID,
        Felt::from(0x20_DEAD_u64),
        Felt::ZERO,
        &initial_raw,
        initial_hint,
    )
    .await
    .map_err(|e| eprintln!("⚠️  deploy_multinoulli_market failed: {e}"))
    .ok()?;
    eprintln!("✅ multinoulli market deployed: {market:#x}");

    initialize_market(&admin_handle, market, env.collateral, INIT_APPROVE_AMOUNT)
        .await
        .map_err(|e| eprintln!("⚠️  initialize_market (multinoulli) failed: {e}"))
        .ok()?;

    let actor = *env.participants.first()?;
    approve(
        env.account_handle(&actor),
        env.collateral,
        market,
        TRADE_ALLOWANCE,
    )
    .await
    .map_err(|e| eprintln!("⚠️  actor approve failed: {e}"))
    .ok()?;

    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let writer = MultinoulliMarketWriter::new(
        MultinoulliMarketReader::new(provider, market),
        env.owned_account(&actor),
    );

    let handle = MultinoulliLiveHandle {
        market,
        writer,
        runtime: env.multinoulli_runtime,
        cur_probs: MULTINOULLI_INITIAL_PROBS,
    };
    Some((env, rpc, handle))
}

async fn step_multinoulli_live(
    rng: &mut DetRng,
    env: &TestEnv,
    rpc: &JsonRpcClient<HttpTransport>,
    handle: &mut MultinoulliLiveHandle,
    stats: &mut FamilyStats,
) {
    let action = pick_action(rng);
    stats.record_action(action);
    // Multinoulli markets have no in-band add/remove_liquidity — the LP
    // path is via the factory's `lp_*` admin entrypoints. Bucket LpAdd /
    // LpRemove into the trade column for the random walk.
    let effective_action = match action {
        ActionKind::LpAdd | ActionKind::LpRemove => ActionKind::Trade,
        other => other,
    };
    match effective_action {
        ActionKind::Trade => {
            let new_probs = perturb_probs(rng, &handle.cur_probs);
            let Ok(f) = cat_dist(&handle.cur_probs) else {
                stats.record_solver(false);
                return;
            };
            let Ok(g) = cat_dist(&new_probs) else {
                stats.record_solver(false);
                return;
            };
            let k = 50.0_f64;
            let solve = categorical_collateral(&f, &g, k);
            stats.record_solver(solve.is_ok());
            let Ok(verified) = solve else {
                return;
            };
            let cand_raw = cat_raw(&new_probs);
            let supplied = sq_raw((verified.collateral * 20.0).max(100.0));
            let min_outcome = verified.min_outcome_index as u32;
            let quote = match handle
                .writer
                .reader()
                .quote_trade(handle.runtime, cand_raw, min_outcome, supplied)
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  multinoulli quote_trade failed: {e}");
                    return;
                },
            };
            if !quote.on_chain_will_accept {
                if let Some(reason) = quote.rejection {
                    let key = format!("{reason:?}");
                    *stats.typed_rejections.entry(key).or_insert(0) += 1;
                }
                return;
            }
            match handle.writer.execute_quote(quote).await {
                Ok(_) => {
                    stats.record_chain_submission(true);
                    handle.cur_probs = new_probs;
                },
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  multinoulli chain trade rejected: {e}");
                },
            }
        },
        ActionKind::Sell => {
            stats.record_solver(true);
            match handle.writer.sell_position(0).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                },
            }
        },
        ActionKind::PartialClaim => {
            stats.record_solver(true);
        },
        ActionKind::LpAdd | ActionKind::LpRemove => unreachable!(),
    }
    if stats.attempts.is_multiple_of(10) {
        let bal = balance_of(rpc, env.collateral, handle.market)
            .await
            .unwrap_or(0);
        eprintln!(
            "    [{family}] @ action {n}: market_bal = {bal}",
            family = stats.name,
            n = stats.attempts,
        );
    }
}

// ─── Bivariate family ────────────────────────────────────────────────

struct BivariateLiveHandle {
    market: Felt,
    writer: BivariateMarketWriter<JsonRpcProvider, OwnedAccount>,
    runtime: Felt,
    cur_mu1: f64,
    cur_mu2: f64,
    cur_var1: f64,
    cur_var2: f64,
    cur_rho: f64,
}

fn biv_core_raw(
    mu1: f64,
    mu2: f64,
    var1: f64,
    var2: f64,
    rho: f64,
) -> BivariateNormalDistributionCoreRaw {
    BivariateNormalDistributionCoreRaw {
        mu1: sq_raw(mu1),
        mu2: sq_raw(mu2),
        variance1: sq_raw(var1),
        variance2: sq_raw(var2),
        rho: sq_raw(rho),
    }
}

fn biv_point(x1: f64, x2: f64) -> BivariatePointRaw {
    BivariatePointRaw {
        x1: sq_raw(x1),
        x2: sq_raw(x2),
    }
}

async fn bootstrap_bivariate_live(
    env: TestEnv,
) -> Option<(TestEnv, JsonRpcClient<HttpTransport>, BivariateLiveHandle)> {
    let admin_handle = env.account_handle(&env.admin);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));

    upsert_bivariate_profile_for_test(
        admin_handle.clone(),
        env.factory,
        env.collateral,
        PROFILE_ID,
    )
    .await
    .map_err(|e| eprintln!("⚠️  upsert_bivariate_profile failed: {e}"))
    .ok()?;

    // Initial state mirrors `bivariate_chaos`.
    let initial_mu1 = 82.0_f64;
    let initial_mu2 = 120.0_f64;
    let initial_var1 = 64.0_f64;
    let initial_var2 = 900.0_f64;
    let initial_rho = -0.4_f64;
    let f64_dist = build_initial_bivariate_inputs(
        initial_mu1,
        initial_mu2,
        initial_var1,
        initial_var2,
        initial_rho,
    );
    // Round-trip through the runtime to pick up the chain-derived
    // sigma/inv/normalization fields (f64 derivations don't match
    // bit-for-bit; see `expand_bivariate_distribution` rustdoc).
    let core = BivariateNormalDistributionCoreRaw {
        mu1: f64_dist.mu1,
        mu2: f64_dist.mu2,
        variance1: f64_dist.variance1,
        variance2: f64_dist.variance2,
        rho: f64_dist.rho,
    };
    let initial_dist = expand_bivariate_distribution(&rpc, env.bivariate_runtime, core)
        .await
        .map_err(|e| eprintln!("⚠️  expand_bivariate_distribution failed: {e}"))
        .ok()?;
    let initial_hints = fetch_bivariate_hints(&rpc, env.bivariate_runtime, initial_dist)
        .await
        .map_err(|e| eprintln!("⚠️  fetch_bivariate_hints failed: {e}"))
        .ok()?;

    let market = deploy_bivariate_market_with_event(
        &admin_handle,
        env.factory,
        PROFILE_ID,
        Felt::from(0x30_DEAD_u64),
        Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .map_err(|e| eprintln!("⚠️  deploy_bivariate_market failed: {e}"))
    .ok()?;
    eprintln!("✅ bivariate market deployed: {market:#x}");

    initialize_market(&admin_handle, market, env.collateral, INIT_APPROVE_AMOUNT)
        .await
        .map_err(|e| eprintln!("⚠️  initialize_market (bivariate) failed: {e}"))
        .ok()?;

    let actor = *env.participants.first()?;
    approve(
        env.account_handle(&actor),
        env.collateral,
        market,
        TRADE_ALLOWANCE,
    )
    .await
    .map_err(|e| eprintln!("⚠️  actor approve failed: {e}"))
    .ok()?;

    let provider = JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(env.url.clone())));
    let writer = BivariateMarketWriter::new(
        BivariateMarketReader::new(provider, market),
        env.owned_account(&actor),
    );

    let handle = BivariateLiveHandle {
        market,
        writer,
        runtime: env.bivariate_runtime,
        cur_mu1: initial_mu1,
        cur_mu2: initial_mu2,
        cur_var1: initial_var1,
        cur_var2: initial_var2,
        cur_rho: initial_rho,
    };
    Some((env, rpc, handle))
}

async fn step_bivariate_live(
    rng: &mut DetRng,
    env: &TestEnv,
    rpc: &JsonRpcClient<HttpTransport>,
    handle: &mut BivariateLiveHandle,
    stats: &mut FamilyStats,
) {
    let action = pick_action(rng);
    stats.record_action(action);
    match action {
        ActionKind::Trade => {
            // Small μ shifts; preserve σ/ρ so the chaos surface stays
            // inside the chain-acceptance envelope.
            let target_mu1 = handle.cur_mu1 + rng.next_f64_in(-2.0, 2.0);
            let target_mu2 = handle.cur_mu2 + rng.next_f64_in(-10.0, 10.0);
            let core = biv_core_raw(
                target_mu1,
                target_mu2,
                handle.cur_var1,
                handle.cur_var2,
                handle.cur_rho,
            );
            // Submit via quote_trade + execute_quote — Wave-1 typed flow.
            let x_star = biv_point(target_mu1, target_mu2);
            let supplied = sq_raw(200.0_f64);
            let quote = match handle
                .writer
                .reader()
                .quote_trade(handle.runtime, core, x_star, supplied)
                .await
            {
                Ok(q) => q,
                Err(e) => {
                    stats.record_solver(false);
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  bivariate quote_trade failed: {e}");
                    return;
                },
            };
            stats.record_solver(quote.on_chain_will_accept);
            if !quote.on_chain_will_accept {
                if let Some(reason) = quote.rejection {
                    let key = format!("{reason:?}");
                    *stats.typed_rejections.entry(key).or_insert(0) += 1;
                }
                return;
            }
            match handle.writer.execute_quote(quote).await {
                Ok(_) => {
                    stats.record_chain_submission(true);
                    handle.cur_mu1 = target_mu1;
                    handle.cur_mu2 = target_mu2;
                },
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                    eprintln!("  ⚠️  bivariate execute_quote failed: {e}");
                },
            }
        },
        ActionKind::Sell => {
            stats.record_solver(true);
            match handle.writer.sell_position(handle.runtime, 0).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    stats.record_typed_rejection(&e);
                },
            }
        },
        ActionKind::LpAdd => {
            stats.record_solver(true);
            let amount = sq_raw(rng.next_f64_in(10.0, 100.0));
            match handle.writer.add_liquidity(amount).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  bivariate add_liquidity failed: {e}");
                },
            }
        },
        ActionKind::LpRemove => {
            stats.record_solver(true);
            let frac = sq_raw(rng.next_f64_in(0.05, 0.50));
            match handle.writer.remove_liquidity(frac).await {
                Ok(_) => stats.record_chain_submission(true),
                Err(e) => {
                    stats.record_chain_submission(false);
                    eprintln!("  ⚠️  bivariate remove_liquidity failed: {e}");
                },
            }
        },
        ActionKind::PartialClaim => {
            stats.record_solver(true);
        },
    }
    if stats.attempts.is_multiple_of(10) {
        let bal = balance_of(rpc, env.collateral, handle.market)
            .await
            .unwrap_or(0);
        eprintln!(
            "    [{family}] @ action {n}: market_bal = {bal}",
            family = stats.name,
            n = stats.attempts,
        );
    }
}

/// Off-chain-only action runner for families whose chain wiring is
/// pending. Keeps the convergence statistic meaningful by actually
/// running the off-chain solver against random pairs.
fn run_off_chain_normal_only(rng: &mut DetRng, n: u32, stats: &mut FamilyStats) {
    let mut cur_mean = 42.0_f64;
    let mut cur_var = 64.0_f64;
    for _ in 0..n {
        let action = pick_action(rng);
        stats.record_action(action);
        if action == ActionKind::Trade {
            let target_mean = rng.next_f64_in(-50.0, 200.0);
            let target_var = rng.next_f64_in(0.5, 400.0);
            let Ok(f) = nd(cur_mean, cur_var) else {
                continue;
            };
            let Ok(g) = nd(target_mean, target_var) else {
                continue;
            };
            let res = normal_collateral(&f, &g, MinimizationPolicy::unrestricted());
            stats.record_solver(res.is_ok());
            if res.is_ok() {
                cur_mean = target_mean;
                cur_var = target_var;
            }
        } else {
            stats.record_solver(true);
        }
    }
}

async fn run_scale_chaos() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1");
        return;
    }
    if !long_enabled() {
        eprintln!("skip: set DEADEYE_RUN_LONG=1");
        return;
    }

    let s = seed();
    eprintln!(
        "▶ scale_chaos: seed = {s:#x}, actions/family = {ACTIONS_PER_FAMILY} \
         (normal + lognormal + multinoulli + bivariate, all chain-wired)"
    );

    let mut rng = DetRng::new(s);
    let mut normal_stats = FamilyStats::new("normal");
    let mut lognormal_stats = FamilyStats::new("lognormal");
    let mut multinoulli_stats = FamilyStats::new("multinoulli");
    let mut bivariate_stats = FamilyStats::new("bivariate");

    // ── Normal family (chain-wired) ────────────────────────────────────
    // `bootstrap_normal_live` returns its own fresh devnet; the
    // lognormal / multinoulli / bivariate families then ride along on
    // that same devnet via per-family `bootstrap_*_live(env)` helpers.
    // Sharing one devnet across all four families amortises the ~30 s
    // bootstrap cost across the whole run.
    let env_handle = match bootstrap_normal_live().await {
        Some((env, rpc, mut handle)) => {
            eprintln!(
                "▶ running {ACTIONS_PER_FAMILY} on-chain actions against normal market {m:#x}",
                m = handle.market,
            );
            for _ in 0..ACTIONS_PER_FAMILY {
                step_normal_live(&mut rng, &env, &rpc, &mut handle, &mut normal_stats).await;
            }
            let bal = balance_of(&rpc, env.collateral, handle.market)
                .await
                .unwrap_or(0);
            eprintln!(
                "  normal final: market_bal = {bal} | chain_subs = {subs} | failures = {f}",
                subs = normal_stats.chain_submissions,
                f = normal_stats.chain_failures,
            );
            Some((env, rpc))
        },
        None => {
            eprintln!("⚠️  normal-family bootstrap failed; falling back to off-chain solver loop");
            run_off_chain_normal_only(&mut rng, ACTIONS_PER_FAMILY, &mut normal_stats);
            None
        },
    };

    // ── Lognormal family (chain-wired, shared devnet) ─────────────────
    if let Some((env, _)) = env_handle {
        if let Some((env_back, rpc, mut handle)) = bootstrap_lognormal_live(env).await {
            eprintln!(
                "▶ running {ACTIONS_PER_FAMILY} on-chain actions against lognormal market {m:#x}",
                m = handle.market,
            );
            for _ in 0..ACTIONS_PER_FAMILY {
                step_lognormal_live(&mut rng, &env_back, &rpc, &mut handle, &mut lognormal_stats)
                    .await;
            }
            let bal = balance_of(&rpc, env_back.collateral, handle.market)
                .await
                .unwrap_or(0);
            eprintln!(
                "  lognormal final: market_bal = {bal} | chain_subs = {subs} | failures = {f}",
                subs = lognormal_stats.chain_submissions,
                f = lognormal_stats.chain_failures,
            );

            // ── Multinoulli family ────────────────────────────────────
            if let Some((env_back, rpc, mut handle)) = bootstrap_multinoulli_live(env_back).await {
                eprintln!(
                    "▶ running {ACTIONS_PER_FAMILY} on-chain actions against multinoulli market {m:#x}",
                    m = handle.market,
                );
                for _ in 0..ACTIONS_PER_FAMILY {
                    step_multinoulli_live(
                        &mut rng,
                        &env_back,
                        &rpc,
                        &mut handle,
                        &mut multinoulli_stats,
                    )
                    .await;
                }
                let bal = balance_of(&rpc, env_back.collateral, handle.market)
                    .await
                    .unwrap_or(0);
                eprintln!(
                    "  multinoulli final: market_bal = {bal} | chain_subs = {subs} | failures = {f}",
                    subs = multinoulli_stats.chain_submissions,
                    f = multinoulli_stats.chain_failures,
                );

                // ── Bivariate family ──────────────────────────────────
                if let Some((env_back, rpc, mut handle)) = bootstrap_bivariate_live(env_back).await
                {
                    eprintln!(
                        "▶ running {ACTIONS_PER_FAMILY} on-chain actions against bivariate market {m:#x}",
                        m = handle.market,
                    );
                    for _ in 0..ACTIONS_PER_FAMILY {
                        step_bivariate_live(
                            &mut rng,
                            &env_back,
                            &rpc,
                            &mut handle,
                            &mut bivariate_stats,
                        )
                        .await;
                    }
                    let bal = balance_of(&rpc, env_back.collateral, handle.market)
                        .await
                        .unwrap_or(0);
                    eprintln!(
                        "  bivariate final: market_bal = {bal} | chain_subs = {subs} | failures = {f}",
                        subs = bivariate_stats.chain_submissions,
                        f = bivariate_stats.chain_failures,
                    );
                } else {
                    eprintln!("⚠️  bivariate bootstrap failed; counting off-chain only");
                }
            } else {
                eprintln!("⚠️  multinoulli bootstrap failed; counting off-chain only");
            }
        } else {
            eprintln!("⚠️  lognormal bootstrap failed; counting off-chain only");
        }
    }

    // ── Final stats report ─────────────────────────────────────────────
    let all = [
        &normal_stats,
        &lognormal_stats,
        &multinoulli_stats,
        &bivariate_stats,
    ];
    let mut total_attempts = 0_u32;
    let mut total_converged = 0_u32;
    let mut total_chain_subs = 0_u32;
    let mut total_chain_fails = 0_u32;
    for s in all {
        eprintln!(
            "  {name:>11} | attempts={attempts:>4} | converged={converged:>4} \
             | typed-err={rejected:>3} | rate={rate:.4} \
             | chain_subs={subs:>3} | chain_fails={fails:>3}",
            name = s.name,
            attempts = s.attempts,
            converged = s.converged,
            rejected = s.rejected_typed,
            rate = s.convergence_rate(),
            subs = s.chain_submissions,
            fails = s.chain_failures,
        );
        eprintln!(
            "            action mix → trade={t} lp+={la} lp-={lr} sell={se} claim={c}",
            t = s.action_mix[0],
            la = s.action_mix[1],
            lr = s.action_mix[2],
            se = s.action_mix[3],
            c = s.action_mix[4],
        );
        if !s.typed_rejections.is_empty() {
            eprintln!("            chain rejections by reason:");
            for (reason, count) in &s.typed_rejections {
                eprintln!("              {reason}: {count}");
            }
        }
        total_attempts += s.attempts;
        total_converged += s.converged;
        total_chain_subs += s.chain_submissions;
        total_chain_fails += s.chain_failures;
    }
    let overall_rate = f64::from(total_converged) / f64::from(total_attempts.max(1));
    eprintln!(
        "  ── overall: attempts={total_attempts}, converged={total_converged}, \
         rate={overall_rate:.4}, chain_subs={total_chain_subs}, \
         chain_fails={total_chain_fails}"
    );

    // ── Hard asserts ───────────────────────────────────────────────────
    assert!(
        overall_rate >= MIN_CONVERGENCE_RATE,
        "convergence rate {overall_rate:.4} fell below {MIN_CONVERGENCE_RATE:.4}",
    );
    let expected_total = ACTIONS_PER_FAMILY * 4;
    assert!(
        total_attempts >= expected_total,
        "expected ≥ {expected_total} attempts, got {total_attempts}"
    );
    // The chain-submission failure budget is strict: every trade we
    // submit was already vetted by the off-chain solver. A non-zero
    // count points at either a divergence between off-chain and
    // on-chain (a regression) or a transient devnet hiccup (re-run).
    // 5% is generous headroom for the latter.
    if total_chain_subs > 0 {
        let fail_rate = f64::from(total_chain_fails) / f64::from(total_chain_subs);
        assert!(
            fail_rate <= 0.05_f64,
            "chain failure rate {fail_rate:.4} exceeds 5% — \
             possible off-chain/on-chain divergence (chain_subs={total_chain_subs}, \
             chain_fails={total_chain_fails})",
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "long-running; bootstraps devnet + 50 chain actions/family; uses \
            DEADEYE_RUN_INTEGRATION + DEADEYE_RUN_LONG"]
async fn scale_actions_across_families() {
    // The run_scale_chaos future carries four family handles + per-family
    // stats; clippy flags it as large (~23 KB). Box-pin it so the test
    // harness doesn't have to materialise the whole future on its stack.
    Box::pin(run_scale_chaos()).await;
}

/// Convenience entry that mirrors the legacy name so any external CI
/// hook referring to `scale_1000_actions_across_families` keeps working.
/// The legacy name's "1000" suffix was a hint at the per-action budget
/// rather than a hard target; the new `ACTIONS_PER_FAMILY` constant is
/// the source of truth.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "alias of scale_actions_across_families; same gates"]
async fn scale_1000_actions_across_families() {
    Box::pin(run_scale_chaos()).await;
}
