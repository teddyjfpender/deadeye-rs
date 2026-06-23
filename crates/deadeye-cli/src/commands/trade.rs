//! `deadeye trade …` — preview-first trading flow (Driver B).
//!
//! `quote` is read-only; it preflights via `quote_trade` and prints a
//! verdict plus a copy-pasteable execute hint. `execute` re-runs the
//! quote (chain state may have moved), confirms, and submits via the
//! family writer. `loop` is the EV-gated arbitrage loop: each tick
//! re-reads the market, re-loads the belief, runs the optimizer, and
//! submits at most one trade — only when every EV / risk / budget gate
//! passes, and only when `--execute` is set (observe-only otherwise).
//! `journal` opens / replays the on-disk journal.
//!
//! The market family is auto-detected per command (issue #38) — normal
//! and lognormal AMMs are wire-identical, so detection is semantic
//! (indexer / class hash / factory), never a reader probe; pass
//! `--family` to force it.

use anyhow::{Context as _, Result};
use deadeye_core::Sq128;
use deadeye_sdk::{
    DeadeyeClient,
    bulk::Family,
    journal::{EntryKind, JournalEntry, TradeJournal},
};
use deadeye_starknet::{
    Account, Felt, LognormalMarketReader, LognormalMarketWriter, NormalMarketReader,
    NormalMarketWriter, TradeRejectionReason,
};
use serde_json::json;

use crate::{
    cli::{TradeCmd, TradeExecuteArgs, TradeJournalArgs, TradeQuoteArgs},
    commands::{
        call_bundle::{
            CalldataInput, CalldataResult, WriteMode, build_write_account, calldata_result, labels,
            print_calldata_json, simulate_calls, submit_external_calls,
        },
        render_helpers::{
            QuoteResult, SubmissionResult, pretty_rejection, submission_from_receipt,
            submission_from_trade_error,
        },
        runtime_resolver::{
            build_provider, family_label, parse_felt, resolve_family, resolve_runtime,
            resolve_runtime_opt,
        },
    },
    context::{AppContext, CliProvider},
    output::OutputMode,
};

/// Multiplier applied to the offline-computed required collateral when sizing
/// the amount the trade actually supplies. Collateral is a *returned* margin
/// lock (not a cost), so over-supplying is free; a margin is required because
/// the on-chain Q128.128 `collateral_sufficient` check rejects a supply that
/// equals the f64 estimate on any rounding gap. 5% comfortably covers the
/// fixed-point delta while staying close to the webapp's buffered collateral.
pub(crate) const COLLATERAL_BUFFER: f64 = 1.05;

pub(crate) async fn run(action: TradeCmd, ctx: &AppContext, confirm: bool) -> Result<()> {
    match action {
        TradeCmd::Quote(args) => quote(ctx, args).await,
        TradeCmd::Execute(args) => execute(ctx, args, confirm).await,
        TradeCmd::Loop(args) => super::trade_loop::run(ctx, args, confirm).await,
        TradeCmd::Journal(args) => journal_cmd(ctx, args),
    }
}

// ─── shared candidate solving (quote / execute / loop) ─────────────────

/// EV-max candidate solved from a state snapshot plus the sizing policy
/// (issue #33): fractional-Kelly stake cap, then the `--max-cvar` budget
/// walk-down. Pure f64 — zero RPC; re-optimising at smaller budgets reuses
/// the snapshot.
pub(crate) struct NormalSolve {
    /// The optimizer's chain-bit-exact quote at the final effective budget.
    pub(crate) quote: deadeye_starknet::NormalTradeQuote,
    /// Optimizer expected value (XP) under the belief.
    pub(crate) expected_value: f64,
    /// Which constraint bound the stake (`budget`, `kelly-…`, `cvar-cap`).
    pub(crate) sizing_basis: String,
}

/// Outcome of [`solve_normal_candidate`]: the CVaR-unreachable case is typed
/// so `trade loop` can journal it as a skip while `quote`/`execute` keep
/// treating it as a hard error.
#[expect(
    clippy::large_enum_variant,
    reason = "short-lived, one per solve — boxing the quote buys nothing"
)]
pub(crate) enum NormalSolveOutcome {
    /// A candidate was found at the final effective budget.
    Solved(NormalSolve),
    /// Even the smallest viable stake violates the `--max-cvar` cap.
    CvarUnreachable {
        /// 5% CVaR (XP, negative = loss) of the smallest stake tried.
        cvar: f64,
        /// The configured cap that could not be met.
        max_cvar: f64,
    },
}

impl NormalSolveOutcome {
    /// Unwrap as a solve, converting CVaR-unreachable into the historical
    /// `trade quote` / `trade execute` error message.
    pub(crate) fn into_solve(self) -> Result<NormalSolve> {
        match self {
            Self::Solved(s) => Ok(s),
            Self::CvarUnreachable { cvar, max_cvar } => anyhow::bail!(
                "--max-cvar {max_cvar} XP is unreachable: the smallest viable \
                 stake still has 5% CVaR {cvar:.4} XP — raise --max-cvar or \
                 shrink the belief distance"
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct NormalMicroPolicy {
    min_ev: f64,
    min_collateral: Option<f64>,
    bankroll: Option<f64>,
    kelly_mult: Option<f64>,
    max_cvar: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NormalMicroStats {
    evaluated: usize,
    chain_rejected: usize,
    over_budget: usize,
    below_min_collateral: usize,
    below_min_ev: usize,
    cvar_rejected: usize,
    kelly_rejected: usize,
    best_seen_ev: Option<f64>,
    best_seen_collateral: Option<f64>,
}

impl NormalMicroStats {
    fn record_seen(&mut self, ev: f64, collateral: f64) {
        if self.best_seen_ev.is_none_or(|best| ev > best) {
            self.best_seen_ev = Some(ev);
            self.best_seen_collateral = Some(collateral);
        }
    }
}

fn normal_no_trade_reason(
    quote: &deadeye_starknet::NormalTradeQuote,
    expected_value: Option<f64>,
    budget: Option<f64>,
) -> Option<String> {
    let required = Sq128::from_raw(quote.required_collateral).to_f64();
    if quote.on_chain_will_accept && required > 0.0 {
        return None;
    }
    if let Some(rejection) = quote.rejection {
        return Some(format!(
            "candidate rejected by chain preflight: {rejection:?}"
        ));
    }
    if required <= 0.0 {
        return Some(
            "no trade: candidate equals the current market or requires zero collateral".to_owned(),
        );
    }
    if let Some(ev) = expected_value
        && ev <= 0.0
    {
        return Some(format!(
            "no trade: candidate EV {ev:+.6} XP is not positive under the supplied belief"
        ));
    }
    if let Some(budget) = budget
        && required > budget
    {
        return Some(format!(
            "no trade: required collateral {required:.6} XP exceeds budget {budget:.6} XP"
        ));
    }
    Some("no trade: candidate failed the offline acceptance gates".to_owned())
}

fn micro_no_trade_reason(
    stats: NormalMicroStats,
    budget: f64,
    min_ev: f64,
    min_collateral: f64,
    max_cvar: Option<f64>,
    kelly_active: bool,
) -> String {
    let best = match (stats.best_seen_ev, stats.best_seen_collateral) {
        (Some(ev), Some(collateral)) => {
            format!(
                "; best chain-accepted candidate seen had EV {ev:+.6} XP at {collateral:.6} XP collateral"
            )
        },
        _ => String::new(),
    };
    let mut gates = format!(
        "micro search found no candidate clearing chain acceptance, budget {budget:.6} XP, \
         min EV {min_ev:.6} XP, and min collateral {min_collateral:.6} XP"
    );
    if let Some(max_cvar) = max_cvar {
        gates.push_str(&format!(", with --max-cvar {max_cvar:.6} XP"));
    }
    if kelly_active {
        gates.push_str(", with Kelly sizing");
    }
    format!(
        "{gates}{best} (evaluated {}, chain-rejected {}, over-budget {}, below-min-collateral {}, \
         below-min-ev {}, cvar-rejected {}, kelly-rejected {})",
        stats.evaluated,
        stats.chain_rejected,
        stats.over_budget,
        stats.below_min_collateral,
        stats.below_min_ev,
        stats.cvar_rejected,
        stats.kelly_rejected,
    )
}

/// Fine local search for small positive-EV normal trades (issue #42).
///
/// The default optimizer keeps its historical coarse EV-max behavior. This
/// opt-in path deliberately searches the rectangle between the live market and
/// the supplied belief so tiny mean/σ disagreements can surface as
/// chain-acceptable candidates.
fn solve_normal_micro_candidate(
    snapshot: &deadeye_sdk::normal::NormalMarketStateSnapshot,
    belief_mean: f64,
    belief_sigma: f64,
    budget: f64,
    policy: &NormalMicroPolicy,
) -> Result<Result<NormalSolve, String>> {
    anyhow::ensure!(budget > 0.0, "--budget must be > 0");
    anyhow::ensure!(belief_sigma > 0.0, "--belief-sigma must be > 0");
    anyhow::ensure!(
        policy.min_ev >= 0.0,
        "--min-ev must be >= 0 (use 0 to allow any positive-EV micro trade)"
    );
    if let Some(min_collateral) = policy.min_collateral {
        anyhow::ensure!(min_collateral >= 0.0, "--min-collateral must be >= 0");
    }
    if let Some(max_cvar) = policy.max_cvar {
        anyhow::ensure!(max_cvar > 0.0, "--max-cvar must be > 0");
    }

    const STEPS: u32 = 100;
    let min_collateral = policy
        .min_collateral
        .unwrap_or(snapshot.min_trade_collateral)
        .max(snapshot.min_trade_collateral);
    let mut best: Option<NormalSolve> = None;
    let mut best_required = f64::INFINITY;
    let mut stats = NormalMicroStats::default();

    for mean_i in 0..=STEPS {
        let mean_frac = f64::from(mean_i) / f64::from(STEPS);
        let cand_mean = snapshot
            .mean
            .mul_add(1.0 - mean_frac, belief_mean * mean_frac);
        for sigma_i in 0..=STEPS {
            let sigma_frac = f64::from(sigma_i) / f64::from(STEPS);
            let cand_sigma = snapshot
                .sigma
                .mul_add(1.0 - sigma_frac, belief_sigma * sigma_frac);
            if cand_sigma <= 0.0 {
                continue;
            }
            if (cand_mean - snapshot.mean).abs() < 1e-12_f64
                && (cand_sigma - snapshot.sigma).abs() < 1e-12_f64
            {
                continue;
            }
            stats.evaluated += 1;
            let quote = deadeye_sdk::normal::quote_candidate_from_state(
                snapshot,
                cand_mean,
                cand_sigma * cand_sigma,
            )
            .map_err(|e| anyhow::anyhow!("quote_candidate_from_state: {e}"))?;
            if !quote.on_chain_will_accept {
                stats.chain_rejected += 1;
                continue;
            }
            let required = Sq128::from_raw(quote.required_collateral).to_f64();
            let ev = super::risk::expected_pnl(
                snapshot.mean,
                snapshot.sigma,
                cand_mean,
                cand_sigma,
                snapshot.effective_k,
                belief_mean,
                belief_sigma,
            );
            stats.record_seen(ev, required);
            if required > budget {
                stats.over_budget += 1;
                continue;
            }
            if required + 1e-12_f64 < min_collateral {
                stats.below_min_collateral += 1;
                continue;
            }
            if ev <= 0.0 || ev + 1e-12_f64 < policy.min_ev {
                stats.below_min_ev += 1;
                continue;
            }
            if let Some(max_cvar) = policy.max_cvar {
                let cvar = super::risk::cvar_under_belief(
                    snapshot.mean,
                    snapshot.sigma,
                    cand_mean,
                    cand_sigma,
                    snapshot.effective_k,
                    belief_mean,
                    belief_sigma,
                    0.05,
                );
                if cvar.is_finite() && cvar < -max_cvar {
                    stats.cvar_rejected += 1;
                    continue;
                }
            }
            let basis = if let Some(kelly_mult) = policy.kelly_mult {
                let Some(advice) = super::risk::sizing_advice(
                    ev,
                    required,
                    policy.bankroll.unwrap_or(0.0),
                    kelly_mult,
                ) else {
                    stats.kelly_rejected += 1;
                    continue;
                };
                if required > advice.recommended_stake_xp + 1e-12_f64 {
                    stats.kelly_rejected += 1;
                    continue;
                }
                format!("micro-kelly-{kelly_mult:.2}")
            } else {
                "micro-min-ev".to_owned()
            };
            if best.as_ref().is_none_or(|current| {
                ev > current.expected_value
                    || ((ev - current.expected_value).abs() < 1e-12_f64 && required < best_required)
            }) {
                best_required = required;
                best = Some(NormalSolve {
                    quote,
                    expected_value: ev,
                    sizing_basis: basis,
                });
            }
        }
    }

    Ok(best.ok_or_else(|| {
        micro_no_trade_reason(
            stats,
            budget,
            policy.min_ev,
            min_collateral,
            policy.max_cvar,
            policy.kelly_mult.is_some(),
        )
    }))
}

/// Shared by `trade quote`, `trade quote --from-state`, `trade execute`
/// and `trade loop` so every path sizes stakes identically.
pub(crate) fn solve_normal_candidate(
    snapshot: &deadeye_sdk::normal::NormalMarketStateSnapshot,
    belief_mean: f64,
    belief_sigma: f64,
    budget: f64,
    bankroll: Option<f64>,
    kelly_mult: Option<f64>,
    max_cvar: Option<f64>,
) -> Result<NormalSolveOutcome> {
    let quote_at = |b: f64| {
        deadeye_sdk::normal::optimize_quote_from_state(snapshot, belief_mean, belief_sigma, b)
            .context("optimize_quote_offline")
    };
    let (mut q, mut ev) = quote_at(budget)?;
    let mut basis = "budget".to_owned();
    let mut eff = budget;
    let required = Sq128::from_raw(q.required_collateral).to_f64();
    if let Some((cap, kelly_basis)) =
        super::risk::kelly_stake_cap(bankroll, kelly_mult, ev, required)?
        && cap < required
    {
        eff = cap;
        basis = kelly_basis;
        (q, ev) = quote_at(eff)?;
    }
    if let Some(max_cvar) = max_cvar {
        anyhow::ensure!(max_cvar > 0.0, "--max-cvar must be > 0");
        for _ in 0..40 {
            let cm = Sq128::from_raw(q.candidate.mean).to_f64();
            let cs = Sq128::from_raw(q.candidate.sigma).to_f64();
            let cvar = super::risk::cvar_under_belief(
                snapshot.mean,
                snapshot.sigma,
                cm,
                cs,
                snapshot.effective_k,
                belief_mean,
                belief_sigma,
                0.05,
            );
            if !cvar.is_finite() || cvar >= -max_cvar {
                break;
            }
            eff *= 0.75;
            basis = "cvar-cap".to_owned();
            let (q2, ev2) = quote_at(eff)?;
            if Sq128::from_raw(q2.required_collateral).to_f64() <= 0.0 {
                return Ok(NormalSolveOutcome::CvarUnreachable { cvar, max_cvar });
            }
            (q, ev) = (q2, ev2);
        }
    }
    Ok(NormalSolveOutcome::Solved(NormalSolve {
        quote: q,
        expected_value: ev,
        sizing_basis: basis,
    }))
}

/// Lognormal twin of [`solve_normal_candidate`]: EV-max grid search plus the
/// fractional-Kelly stake cap. Belief is in LOG space. (Each optimize call
/// reads 3 views — the lognormal path has no cached snapshot type yet.)
/// Returns `(result, sizing_basis, kelly_cap)` — the cap is `Some` when the
/// fractional-Kelly rule bound the stake.
pub(crate) async fn solve_lognormal_candidate(
    handle: &deadeye_sdk::lognormal::LognormalMarket<'_, CliProvider>,
    belief_mu: f64,
    belief_sigma: f64,
    budget: f64,
    bankroll: Option<f64>,
    kelly_mult: Option<f64>,
) -> Result<(
    deadeye_sdk::lognormal::LognormalOptimizationResult,
    String,
    Option<f64>,
)> {
    let mut sizing_basis = "budget".to_owned();
    let mut kelly_cap = None;
    let mut result = handle
        .optimize_quote_offline_ev(belief_mu, belief_sigma, budget)
        .await
        .map_err(|e| anyhow::anyhow!("optimize_quote_offline_ev (lognormal): {e}"))?;
    if let Some((cap, kelly_basis)) = super::risk::kelly_stake_cap(
        bankroll,
        kelly_mult,
        result.expected_value,
        result.collateral_required,
    )? && cap < result.collateral_required
    {
        sizing_basis = kelly_basis;
        kelly_cap = Some(cap);
        result = handle
            .optimize_quote_offline_ev(belief_mu, belief_sigma, cap)
            .await
            .map_err(|e| anyhow::anyhow!("optimize_quote_offline_ev (lognormal): {e}"))?;
    }
    Ok((result, sizing_basis, kelly_cap))
}

// ─── quote ────────────────────────────────────────────────────────────

pub(crate) async fn quote(ctx: &AppContext, args: TradeQuoteArgs) -> Result<()> {
    // Fetch-once path (issue #14): a saved snapshot makes the quote PURE —
    // zero RPC, so exploring N candidates costs one read total.
    if let Some(path) = &args.from_state {
        let result = quote_normal_from_state(ctx, path, &args)?;
        return ctx.renderer.print(&result);
    }
    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider(ctx)?;
    let client = DeadeyeClient::new(provider);
    let family = resolve_family(ctx, &client, market, args.family).await?;

    let result = match family {
        Family::Normal => quote_normal(&client, market, family, &args).await?,
        Family::Lognormal => quote_lognormal(&client, market, family, &args).await?,
        Family::Multinoulli | Family::Bivariate => {
            anyhow::bail!(
                "trade quote: only normal + lognormal families are wired in Driver B's first cut; \
                 multinoulli / bivariate forthcoming"
            );
        },
    };
    ctx.renderer.print(&result)
}

/// Risk/sizing/lint block shared by the live and `--from-state` quote paths
/// (issues #15 + #24). Pure f64 display math — never touches the verified
/// collateral path.
struct RiskExtras {
    downside_at_market_mean: Option<f64>,
    cvar_5pct: Option<f64>,
    stress_ev: Option<f64>,
    sizing: Option<super::risk::SizingAdvice>,
    warnings: Vec<String>,
}

#[expect(clippy::too_many_arguments, reason = "plain display-math inputs")]
fn compute_risk_extras(
    args: &TradeQuoteArgs,
    market_mean: f64,
    market_sigma: f64,
    effective_k: f64,
    cand_mean: f64,
    cand_sigma: f64,
    expected_value: Option<f64>,
    required_collateral: f64,
    sigma_floor: Option<f64>,
    belief: Option<(f64, f64)>,
    budget: Option<f64>,
) -> RiskExtras {
    use super::risk;
    let downside = Some(risk::pnl_at(
        market_mean,
        market_sigma,
        cand_mean,
        cand_sigma,
        effective_k,
        market_mean,
    ));
    let (cvar, stress) = belief.map_or((None, None), |(bm, bs)| {
        let cvar = risk::cvar_under_belief(
            market_mean,
            market_sigma,
            cand_mean,
            cand_sigma,
            effective_k,
            bm,
            bs,
            0.05,
        );
        let stress = risk::expected_pnl(
            market_mean,
            market_sigma,
            cand_mean,
            cand_sigma,
            effective_k,
            bm,
            bs * 1.5,
        );
        (cvar.is_finite().then_some(cvar), Some(stress))
    });
    let kelly_multiplier = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(risk::preset_fraction));
    if let Some(preset) = args.risk.as_deref()
        && risk::preset_fraction(preset).is_none()
    {
        tracing::warn!(target: "deadeye::risk", preset, "unknown --risk preset; expected conservative|balanced|aggressive");
    }
    let ev_for_sizing = expected_value.or_else(|| {
        belief.map(|(bm, bs)| {
            risk::expected_pnl(
                market_mean,
                market_sigma,
                cand_mean,
                cand_sigma,
                effective_k,
                bm,
                bs,
            )
        })
    });
    let sizing = match (args.bankroll, kelly_multiplier, ev_for_sizing) {
        (Some(bankroll), mult, Some(ev)) => {
            risk::sizing_advice(ev, required_collateral, bankroll, mult.unwrap_or(0.5))
        },
        _ => None,
    };
    let warnings = risk::lint_quote(
        belief,
        market_mean,
        market_sigma,
        cand_mean,
        cand_sigma,
        sigma_floor,
        budget,
        sizing.as_ref(),
    );
    RiskExtras {
        downside_at_market_mean: downside,
        cvar_5pct: cvar,
        stress_ev: stress,
        sizing,
        warnings,
    }
}

/// Pure quote from a saved snapshot (issue #14): zero RPC. Mirrors the
/// offline branches of `quote_normal`, sourcing state from the JSON that
/// `deadeye markets snapshot` produced instead of three live view calls.
fn quote_normal_from_state(
    ctx: &AppContext,
    path: &std::path::Path,
    args: &TradeQuoteArgs,
) -> Result<QuoteResult> {
    use deadeye_sdk::normal::{NormalMarketStateSnapshot, quote_candidate_from_state};
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading state snapshot {}", path.display()))?;
    let snapshot: NormalMarketStateSnapshot = serde_json::from_str(&raw)
        .context("parsing state snapshot (expected `deadeye markets snapshot` JSON)")?;
    // Family gate (issue #38). Snapshots that pre-date the `family` field
    // CANNOT be trusted to be normal: the pre-fix `markets snapshot` happily
    // snapshotted lognormal markets through the wire-identical normal reader,
    // silently emitting log-space values — and production is lognormal-heavy.
    // Refuse unless the operator explicitly asserts `--family normal`.
    let untyped: serde_json::Value = serde_json::from_str(&raw)
        .context("parsing state snapshot (expected `deadeye markets snapshot` JSON)")?;
    if untyped.get("family").is_none() {
        if args.family == Some(crate::cli::FamilyArg::Normal) {
            ctx.renderer.warning(
                "state snapshot pre-dates family stamping (issue #38) — proceeding on \
                 your explicit --family normal assertion; re-take it with \
                 `deadeye markets snapshot` to silence this",
            );
        } else {
            anyhow::bail!(
                "state snapshot {} pre-dates family stamping (issue #38) and cannot be \
                 trusted to be normal-family — pre-fix snapshots of LOGNORMAL markets \
                 look identical and quote garbage. Re-take it with `deadeye markets \
                 snapshot {}`, or pass `--family normal` to assert it really is a \
                 normal market",
                path.display(),
                snapshot.market,
            );
        }
    }
    if snapshot.family != Family::Normal {
        anyhow::bail!(
            "state snapshot {} is for a {} market; --from-state currently supports the \
             normal family only — quote it live instead: `deadeye trade quote {} --family {}`",
            path.display(),
            family_label(snapshot.family),
            snapshot.market,
            family_label(snapshot.family),
        );
    }
    if let Some(flag) = args.family {
        let flagged = super::runtime_resolver::family_from_arg(flag);
        anyhow::ensure!(
            flagged == snapshot.family,
            "--family {} contradicts the state snapshot, which was taken from a {} market",
            family_label(flagged),
            family_label(snapshot.family),
        );
    }
    let market_mean = snapshot.mean;
    let market_sigma = snapshot.sigma;
    let market_hex = snapshot.market.clone();
    anyhow::ensure!(
        args.min_ev.is_some() || args.min_collateral.is_none(),
        "--min-collateral only applies with --min-ev micro-edge search"
    );

    let kelly_mult = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
    let has_direct_candidate = args.mean.is_some() || args.variance.is_some();
    anyhow::ensure!(
        !has_direct_candidate || (args.min_ev.is_none() && args.min_collateral.is_none()),
        "--min-ev/--min-collateral select optimizer micro-edge search; omit them with explicit --mean/--variance"
    );
    let (quote, belief, budget, expected_value, sizing_basis, no_trade_reason) =
        if has_direct_candidate {
            let mean = args
                .mean
                .context("`--mean` is required when quoting an explicit candidate")?;
            let variance = args
                .variance
                .context("`--variance` is required when quoting an explicit candidate")?;
            let q = quote_candidate_from_state(&snapshot, mean, variance)
                .map_err(|e| anyhow::anyhow!("quote_candidate_from_state: {e}"))?;
            let belief = args
                .belief
                .map(|belief_mean| (belief_mean, args.belief_sigma.unwrap_or(market_sigma)));
            let cand_mean = Sq128::from_raw(q.candidate.mean).to_f64();
            let cand_sigma = Sq128::from_raw(q.candidate.sigma).to_f64();
            let expected_value = belief.map(|(belief_mean, belief_sigma)| {
                super::risk::expected_pnl(
                    market_mean,
                    market_sigma,
                    cand_mean,
                    cand_sigma,
                    snapshot.effective_k,
                    belief_mean,
                    belief_sigma,
                )
            });
            let no_trade_reason = normal_no_trade_reason(&q, expected_value, args.budget);
            (
                q,
                belief,
                args.budget,
                expected_value,
                None,
                no_trade_reason,
            )
        } else if let (Some(belief_mean), Some(budget)) = (args.belief, args.budget) {
            let belief_sigma = args.belief_sigma.unwrap_or(market_sigma);
            let solve = if let Some(min_ev) = args.min_ev {
                anyhow::ensure!(
                    args.runtime.is_none(),
                    "--min-ev micro-edge search is offline-only; drop --runtime"
                );
                let policy = NormalMicroPolicy {
                    min_ev,
                    min_collateral: args.min_collateral,
                    bankroll: args.bankroll,
                    kelly_mult,
                    max_cvar: args.max_cvar,
                };
                match solve_normal_micro_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    &policy,
                )? {
                    Ok(solve) => solve,
                    Err(reason) => {
                        let q = deadeye_sdk::normal::quote_candidate_from_state(
                            &snapshot,
                            market_mean,
                            market_sigma * market_sigma,
                        )
                        .map_err(|e| anyhow::anyhow!("quote_candidate_from_state: {e}"))?;
                        return Ok(QuoteResult {
                            family: "normal",
                            market: market_hex,
                            candidate_mean: Some(market_mean),
                            candidate_variance: Some(market_sigma * market_sigma),
                            candidate_sigma: Some(market_sigma),
                            candidate_mu1: None,
                            candidate_mu2: None,
                            candidate_rho: None,
                            x_star: Some(Sq128::from_raw(q.x_star).to_f64()),
                            required_collateral: Some(0.0),
                            padded_collateral: Some(0.0),
                            sigma_floor: None,
                            market_mean: Some(market_mean),
                            market_sigma: Some(market_sigma),
                            belief_mean: Some(belief_mean),
                            belief_sigma: Some(belief_sigma),
                            expected_value: Some(0.0),
                            budget: Some(budget),
                            on_chain_will_accept: false,
                            rejection: None,
                            no_trade_reason: Some(reason),
                            downside_at_market_mean: None,
                            cvar_5pct: None,
                            stress_ev: None,
                            sizing: None,
                            sizing_basis: Some("micro-min-ev".to_owned()),
                            warnings: Vec::new(),
                            execute_hint: format!(
                                "deadeye trade quote {} --family normal --belief {:.6} --belief-sigma {:.6} --budget {:.6} --min-ev {:.6}",
                                snapshot.market, belief_mean, belief_sigma, budget, min_ev
                            ),
                        });
                    },
                }
            } else {
                // Sizing policy (issue #33) — pure re-quotes, zero RPC.
                solve_normal_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    args.bankroll,
                    kelly_mult,
                    args.max_cvar,
                )?
                .into_solve()?
            };
            let no_trade_reason =
                normal_no_trade_reason(&solve.quote, Some(solve.expected_value), Some(budget));
            (
                solve.quote,
                Some((belief_mean, belief_sigma)),
                Some(budget),
                Some(solve.expected_value),
                Some(solve.sizing_basis),
                no_trade_reason,
            )
        } else {
            anyhow::bail!("`--mean`/`--variance` or `--belief`/`--budget` is required")
        };

    let cand_mean = Sq128::from_raw(quote.candidate.mean).to_f64();
    let cand_sigma = Sq128::from_raw(quote.candidate.sigma).to_f64();
    let cand_variance = Sq128::from_raw(quote.candidate.variance).to_f64();
    let req_collat = Sq128::from_raw(quote.required_collateral).to_f64();
    let extras = compute_risk_extras(
        args,
        market_mean,
        market_sigma,
        snapshot.effective_k,
        cand_mean,
        cand_sigma,
        expected_value,
        req_collat,
        None,
        belief,
        budget,
    );
    let execute_hint = format!(
        "deadeye trade execute {} --family normal --mean {:.6} --variance {:.6} --max-collateral {:.6}",
        market_hex,
        cand_mean,
        cand_variance,
        req_collat * 1.10
    );
    Ok(QuoteResult {
        family: "normal",
        market: market_hex,
        candidate_mean: Some(cand_mean),
        candidate_variance: Some(cand_variance),
        candidate_sigma: Some(cand_sigma),
        candidate_mu1: None,
        candidate_mu2: None,
        candidate_rho: None,
        x_star: Some(Sq128::from_raw(quote.x_star).to_f64()),
        required_collateral: Some(req_collat),
        padded_collateral: Some(Sq128::from_raw(quote.padded_collateral).to_f64()),
        // The snapshot has no live backing read; floor gating is offline-only
        // here and the execute path still chain-verifies.
        sigma_floor: None,
        market_mean: Some(market_mean),
        market_sigma: Some(market_sigma),
        belief_mean: belief.map(|(m, _)| m),
        belief_sigma: belief.map(|(_, s)| s),
        expected_value,
        budget,
        on_chain_will_accept: quote.on_chain_will_accept,
        rejection: quote.rejection.as_ref().map(pretty_rejection),
        no_trade_reason,
        downside_at_market_mean: extras.downside_at_market_mean,
        cvar_5pct: extras.cvar_5pct,
        stress_ev: extras.stress_ev,
        sizing: extras.sizing,
        sizing_basis,
        warnings: extras.warnings,
        execute_hint,
    })
}

async fn quote_normal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    family: Family,
    args: &TradeQuoteArgs,
) -> Result<QuoteResult> {
    let market_handle = client.normal_market(market);
    // Offline by default: a runtime address is an *optional* chain-faithful
    // override, never required for a read-only quote (issue #4).
    let runtime = resolve_runtime_opt(args.runtime.as_deref(), family)?;

    // ONE state fetch (issue #14): distribution + params + lp_info in a
    // single snapshot; σ-floor and effective-k derive locally from it.
    let snapshot = market_handle
        .state_snapshot()
        .await
        .context("reading market state snapshot")?;
    let current = snapshot
        .distribution()
        .context("reconstructing market distribution")?;
    let market_mean = snapshot.mean;
    let market_sigma = snapshot.sigma;
    let effective_k = snapshot.effective_k;
    let sigma_floor = Some(deadeye_sdk::normal::normal_sigma_floor(
        effective_k,
        snapshot.pool_backing_xp,
    ));
    anyhow::ensure!(
        args.min_ev.is_some() || args.min_collateral.is_none(),
        "--min-collateral only applies with --min-ev micro-edge search"
    );

    let kelly_mult = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
    let has_direct_candidate = args.mean.is_some() || args.variance.is_some();
    anyhow::ensure!(
        !has_direct_candidate || (args.min_ev.is_none() && args.min_collateral.is_none()),
        "--min-ev/--min-collateral select optimizer micro-edge search; omit them with explicit --mean/--variance"
    );
    let (quote, belief, budget, expected_value, sizing_basis, no_trade_reason) =
        if has_direct_candidate {
            let mean = args
                .mean
                .context("`--mean` is required when quoting an explicit candidate")?;
            let variance = args
                .variance
                .context("`--variance` is required when quoting an explicit candidate")?;
            let q = if let Some(rt) = runtime {
                // Optional chain-faithful path for a fixed candidate.
                let cand_dist = deadeye_core::NormalDistribution::from_variance(
                    Sq128::from_f64(mean)?,
                    Sq128::from_f64(variance)?,
                )?;
                // Encode the raw FROM the dist so (σ, σ²) stays Sq128-exact —
                // an f64-sqrt σ quantized independently fails the runtime's
                // consistency check with an opaque Option::None (issue #36).
                let candidate = deadeye_core::Distribution::to_raw(&cand_dist);
                let x_star = match deadeye_sdk::collateral::normal_collateral(
                    &current,
                    &cand_dist,
                    deadeye_sdk::collateral::MinimizationPolicy::standard(),
                ) {
                    Ok(s) => Sq128::from_f64(s.x_min)?.to_raw(),
                    Err(_) => candidate.mean,
                };
                let supplied = Sq128::from_f64(args.pad.max(0.0))?.to_raw();
                market_handle
                    .reader()
                    .quote_trade(rt, candidate, x_star, supplied, supplied)
                    .await
                    .map_err(|e| anyhow::anyhow!("quote_trade: {e}"))?
            } else {
                // Default: fully client-side quote (no runtime, no tx, no gas).
                deadeye_sdk::normal::quote_candidate_from_state(&snapshot, mean, variance)
                    .context("quote_candidate_from_state")?
            };
            let belief = args
                .belief
                .map(|belief_mean| (belief_mean, args.belief_sigma.unwrap_or(market_sigma)));
            let cand_mean = Sq128::from_raw(q.candidate.mean).to_f64();
            let cand_sigma = Sq128::from_raw(q.candidate.sigma).to_f64();
            let expected_value = belief.map(|(belief_mean, belief_sigma)| {
                super::risk::expected_pnl(
                    market_mean,
                    market_sigma,
                    cand_mean,
                    cand_sigma,
                    effective_k,
                    belief_mean,
                    belief_sigma,
                )
            });
            let no_trade_reason = normal_no_trade_reason(&q, expected_value, args.budget);
            (
                q,
                belief,
                args.budget,
                expected_value,
                None,
                no_trade_reason,
            )
        } else if let (Some(belief_mean), Some(budget)) = (args.belief, args.budget) {
            let belief_sigma = args.belief_sigma.unwrap_or(market_sigma);
            let (q, ev, basis) = if let Some(min_ev) = args.min_ev {
                anyhow::ensure!(
                    runtime.is_none(),
                    "--min-ev micro-edge search is offline-only; drop --runtime"
                );
                let policy = NormalMicroPolicy {
                    min_ev,
                    min_collateral: args.min_collateral,
                    bankroll: args.bankroll,
                    kelly_mult,
                    max_cvar: args.max_cvar,
                };
                match solve_normal_micro_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    &policy,
                )? {
                    Ok(solve) => (
                        solve.quote,
                        Some(solve.expected_value),
                        Some(solve.sizing_basis),
                    ),
                    Err(reason) => {
                        let q = deadeye_sdk::normal::quote_candidate_from_state(
                            &snapshot,
                            market_mean,
                            market_sigma * market_sigma,
                        )
                        .context("quote_candidate_from_state")?;
                        return Ok(QuoteResult {
                            family: family_label(family),
                            market: format!("{market:#x}"),
                            candidate_mean: Some(market_mean),
                            candidate_variance: Some(market_sigma * market_sigma),
                            candidate_sigma: Some(market_sigma),
                            candidate_mu1: None,
                            candidate_mu2: None,
                            candidate_rho: None,
                            x_star: Some(Sq128::from_raw(q.x_star).to_f64()),
                            required_collateral: Some(0.0),
                            padded_collateral: Some(0.0),
                            sigma_floor,
                            market_mean: Some(market_mean),
                            market_sigma: Some(market_sigma),
                            belief_mean: Some(belief_mean),
                            belief_sigma: Some(belief_sigma),
                            expected_value: Some(0.0),
                            budget: Some(budget),
                            on_chain_will_accept: false,
                            rejection: None,
                            no_trade_reason: Some(reason),
                            downside_at_market_mean: None,
                            cvar_5pct: None,
                            stress_ev: None,
                            sizing: None,
                            sizing_basis: Some("micro-min-ev".to_owned()),
                            warnings: Vec::new(),
                            execute_hint: format!(
                                "deadeye trade quote {market:#x} --family normal --belief {belief_mean:.6} --belief-sigma {belief_sigma:.6} --budget {budget:.6} --min-ev {min_ev:.6}"
                            ),
                        });
                    },
                }
            } else if let Some(rt) = runtime {
                anyhow::ensure!(
                    kelly_mult.is_none() && args.max_cvar.is_none(),
                    "sizing caps (--kelly/--risk/--max-cvar) run on the offline quote path — \
                     drop --runtime"
                );
                // Chain-runtime path doesn't surface the optimizer EV.
                let q = market_handle
                    .optimize_quote(rt, belief_mean, belief_sigma, budget)
                    .await
                    .context("optimize_quote (chain runtime)")?;
                (q, None, None)
            } else {
                // Offline path returns the optimizer's expected value (XP).
                // Reuses the snapshot — no params/lp re-read (issue #14).
                // Sizing policy (issue #33): the belief is never touched; the
                // stake is capped by re-optimising at a smaller budget, and
                // `sizing_basis` names whichever constraint bound.
                let solve = solve_normal_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    args.bankroll,
                    kelly_mult,
                    args.max_cvar,
                )?
                .into_solve()?;
                (
                    solve.quote,
                    Some(solve.expected_value),
                    Some(solve.sizing_basis),
                )
            };
            let no_trade_reason = normal_no_trade_reason(&q, ev, Some(budget));
            (
                q,
                Some((belief_mean, belief_sigma)),
                Some(budget),
                ev,
                basis,
                no_trade_reason,
            )
        } else {
            anyhow::bail!("`--mean`/`--variance` or `--belief`/`--budget` is required")
        };

    let cand_mean = Sq128::from_raw(quote.candidate.mean).to_f64();
    let cand_sigma = Sq128::from_raw(quote.candidate.sigma).to_f64();
    let cand_variance = Sq128::from_raw(quote.candidate.variance).to_f64();
    let req_collat = Sq128::from_raw(quote.required_collateral).to_f64();

    // σ-floor gate at the CLI level too — covers the optimizer/belief path,
    // whose grid can otherwise propose a σ below the backing floor.
    let sub_floor = sigma_floor.is_some_and(|sf| cand_sigma + 1e-12 < sf);
    let accept = quote.on_chain_will_accept && !sub_floor;
    let rejection = if accept {
        None
    } else if sub_floor {
        Some(pretty_rejection(&TradeRejectionReason::SigmaTooLow))
    } else {
        quote.rejection.as_ref().map(pretty_rejection)
    };

    let execute_hint = format!(
        "deadeye trade execute {:#x} --family normal --mean {:.6} --variance {:.6} --max-collateral {:.6}",
        market,
        cand_mean,
        cand_variance,
        req_collat * 1.10
    );

    let extras = compute_risk_extras(
        args,
        market_mean,
        market_sigma,
        effective_k,
        cand_mean,
        cand_sigma,
        expected_value,
        req_collat,
        sigma_floor,
        belief,
        budget,
    );

    Ok(QuoteResult {
        family: family_label(family),
        market: format!("{market:#x}"),
        candidate_mean: Some(cand_mean),
        candidate_variance: Some(cand_variance),
        candidate_sigma: Some(cand_sigma),
        candidate_mu1: None,
        candidate_mu2: None,
        candidate_rho: None,
        x_star: Some(Sq128::from_raw(quote.x_star).to_f64()),
        required_collateral: Some(req_collat),
        padded_collateral: Some(Sq128::from_raw(quote.padded_collateral).to_f64()),
        sigma_floor,
        market_mean: Some(market_mean),
        market_sigma: Some(market_sigma),
        belief_mean: belief.map(|(m, _)| m),
        belief_sigma: belief.map(|(_, s)| s),
        expected_value,
        budget,
        on_chain_will_accept: accept,
        rejection,
        no_trade_reason: no_trade_reason.or_else(|| {
            sub_floor
                .then(|| "no trade: candidate sigma is below the market backing floor".to_owned())
        }),
        downside_at_market_mean: extras.downside_at_market_mean,
        cvar_5pct: extras.cvar_5pct,
        stress_ev: extras.stress_ev,
        sizing: extras.sizing,
        sizing_basis,
        warnings: extras.warnings,
        execute_hint,
    })
}

/// Lognormal optimizer quote (log-space belief + budget) — fully offline.
async fn quote_lognormal_optimized(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    belief_mu: f64,
    budget: f64,
    args: &TradeQuoteArgs,
) -> Result<QuoteResult> {
    anyhow::ensure!(
        args.min_ev.is_none() && args.min_collateral.is_none(),
        "--min-ev/--min-collateral are normal-family micro-edge search flags"
    );
    let handle = client.lognormal_market(market);
    let current = handle
        .distribution()
        .await
        .context("reading lognormal market distribution")?;
    let market_mu = current.mu().to_f64();
    let market_sigma = deadeye_core::Distribution::sigma(&current).to_f64();
    let belief_sigma = args.belief_sigma.unwrap_or(market_sigma);

    anyhow::ensure!(
        args.max_cvar.is_none(),
        "--max-cvar is normal-family only for now (lognormal risk extras are not yet wired)"
    );
    let kelly_mult = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
    let mut sizing_basis = "budget".to_owned();
    let mut result = handle
        .optimize_quote_offline_ev(belief_mu, belief_sigma, budget)
        .await
        .map_err(|e| anyhow::anyhow!("optimize_quote_offline_ev (lognormal): {e}"))?;
    // Fractional-Kelly stake cap (issue #33): re-optimise at the capped
    // budget; the belief itself never changes.
    if let Some((cap, kelly_basis)) = super::risk::kelly_stake_cap(
        args.bankroll,
        kelly_mult,
        result.expected_value,
        result.collateral_required,
    )? && cap < result.collateral_required
    {
        sizing_basis = kelly_basis;
        result = handle
            .optimize_quote_offline_ev(belief_mu, belief_sigma, cap)
            .await
            .map_err(|e| anyhow::anyhow!("optimize_quote_offline_ev (lognormal): {e}"))?;
    }

    if result.collateral_required <= 0.0 {
        anyhow::bail!(
            "lognormal optimizer found no positive-EV trade inside the policy region under \
             budget {budget} XP — either the market already prices your belief, or the \
             per-trade movement caps bind (σ ratio ≤ 4×, |Δμ| ≤ 4σ per trade); for a \
             far-away belief, ladder multiple trades"
        );
    }

    let mut warnings = Vec::new();
    if result.belief_utilization < 0.999 && !result.is_budget_sufficient {
        warnings.push(format!(
            "single trade expresses {:.0}% of your belief shift (per-trade caps: σ ratio ≤4×, \
             |Δμ| ≤ 4σ_market) — execute, then re-quote from the new market state to ladder \
             the remainder",
            result.belief_utilization * 100.0,
        ));
    }

    warnings.push(format!(
        "collateral {:.4} XP is the off-chain optimizer's estimate — the chain-certified \
         requirement at execute time can differ (the on-chain λ calibration is not fully \
         mirrored off-chain yet); `trade execute` \
         probes the chain first and only proceeds while the certified cost stays under your \
         --max-collateral ceiling",
        result.collateral_required,
    ));
    // Hint in `--belief` form: execute re-runs this same optimizer against
    // live state and submits its candidate — no lossy hand-off through
    // --mean/--variance, and the chain probe certifies x* before submission
    // (issues #30 + #32). --max-collateral carries the BUDGET (the documented
    // \"max collateral the trader will risk\"), not estimate×1.1 — the chain
    // truth can exceed the f64 estimate, and the post-probe ceiling check is
    // the real guard.
    let execute_hint = format!(
        "deadeye trade execute {market:#x} --family lognormal --belief {belief_mu:.6} \
         --belief-sigma {belief_sigma:.6} --budget {budget:.6} --max-collateral {budget:.6}"
    );
    Ok(QuoteResult {
        family: "lognormal",
        market: format!("{market:#x}"),
        candidate_mean: Some(result.optimized_mu),
        candidate_variance: Some(result.optimized_variance),
        candidate_sigma: Some(result.optimized_sigma),
        candidate_mu1: None,
        candidate_mu2: None,
        candidate_rho: None,
        x_star: Some(result.x_star),
        required_collateral: Some(result.collateral_required),
        padded_collateral: Some(result.collateral_required * (1.0 + args.pad.max(0.0))),
        sigma_floor: None,
        market_mean: Some(market_mu),
        market_sigma: Some(market_sigma),
        belief_mean: Some(belief_mu),
        belief_sigma: Some(belief_sigma),
        expected_value: Some(result.expected_value),
        budget: Some(budget),
        // The optimizer only emits policy-region candidates; execute still
        // chain-probes + verifies before submitting.
        on_chain_will_accept: true,
        rejection: None,
        no_trade_reason: None,
        downside_at_market_mean: None,
        cvar_5pct: None,
        stress_ev: None,
        sizing: super::risk::sizing_advice(
            result.expected_value,
            result.collateral_required,
            args.bankroll.unwrap_or(0.0),
            kelly_mult.unwrap_or(0.5),
        ),
        sizing_basis: Some(sizing_basis),
        warnings,
        execute_hint,
    })
}

async fn quote_lognormal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    family: Family,
    args: &TradeQuoteArgs,
) -> Result<QuoteResult> {
    anyhow::ensure!(
        args.min_ev.is_none() && args.min_collateral.is_none(),
        "--min-ev/--min-collateral are normal-family micro-edge search flags"
    );
    // Optimizer path (issue: lognormal optimizer): --belief/--budget runs the
    // EV-max grid search fully client-side — no runtime, no tx. Belief is in
    // LOG space, matching the on-chain (μ, σ).
    if let (Some(belief_mu), Some(budget)) = (args.belief, args.budget) {
        return quote_lognormal_optimized(client, market, belief_mu, budget, args).await;
    }
    let runtime = resolve_runtime(args.runtime.as_deref(), family)?;
    let provider = client.provider();
    let reader = LognormalMarketReader::new(provider, market);
    let mean = args
        .mean
        .context("--mean is required for lognormal quote")?;
    let variance = args
        .variance
        .context("--variance is required for lognormal quote")?;
    let supplied = Sq128::from_f64(args.pad.max(0.0))?.to_raw();
    // x* seed: the f64 minimiser's root. NEVER the candidate's μ — feeding
    // μ_cand as x* fails the verifier's side/stationarity checks for every
    // candidate, which mis-reported all moves as SideInvalid (issues #30/#31).
    let current = reader
        .distribution()
        .await
        .map_err(|e| anyhow::anyhow!("reading market distribution: {e}"))?;
    let cand_dist = deadeye_core::LognormalDistribution::from_variance(
        Sq128::from_f64(mean)?,
        Sq128::from_f64(variance)?,
    )?;
    // Raw FROM the dist: (σ, σ²) must be Sq128-exact or the runtime's
    // compute_hints_view rejects the candidate as Option::None (issue #36).
    let candidate = deadeye_core::Distribution::to_raw(&cand_dist);
    let x_star = match deadeye_sdk::collateral::lognormal_collateral(
        &current,
        &cand_dist,
        deadeye_sdk::collateral::LognormalOptions::default(),
    ) {
        Ok(s) => Sq128::from_f64(s.x_star)?.to_raw(),
        Err(_) => candidate.mu,
    };
    let quote = reader
        .quote_trade(runtime, candidate, x_star, supplied, supplied)
        .await
        .map_err(|e| anyhow::anyhow!("quote_trade: {e}"))?;

    // The runtime check verifies the *supplied* x* at fixed-point precision;
    // a perfect f64 root can still sit just outside its acceptance window
    // (the chain-probe drift, issue #13). Execute probes + certifies x*
    // automatically, so a Side/Stationary rejection here is conservative.
    let mut warnings = Vec::new();
    if matches!(
        quote.rejection,
        Some(deadeye_starknet::TradeRejectionReason::VerificationFailed {
            sub_reason: Some(
                deadeye_starknet::VerificationSubReason::SideInvalid
                    | deadeye_starknet::VerificationSubReason::StationaryInvalid
            ),
        })
    ) {
        warnings.push(
            "the runtime rejected the f64 x* at chain fixed-point precision — `trade execute`              chain-probes and certifies x* automatically, so this candidate may still execute;              treat this preflight as conservative"
                .into(),
        );
    }

    let cand_mu = Sq128::from_raw(quote.candidate.mu).to_f64();
    let cand_sigma = Sq128::from_raw(quote.candidate.sigma).to_f64();
    let req_collat = Sq128::from_raw(quote.required_collateral).to_f64();
    let execute_hint = format!(
        "deadeye trade execute {:#x} --family lognormal --mean {:.6} --variance {:.6} --max-collateral {:.6}",
        market,
        cand_mu,
        cand_sigma * cand_sigma,
        req_collat * 1.10
    );
    let rejection = if quote.on_chain_will_accept {
        None
    } else {
        quote.rejection.as_ref().map(pretty_rejection)
    };

    // Risk extras are normal-family math for now (issue #15) — lognormal
    // quotes render without them.
    Ok(QuoteResult {
        downside_at_market_mean: None,
        cvar_5pct: None,
        stress_ev: None,
        sizing: None,
        sizing_basis: None,
        warnings,
        family: family_label(family),
        market: format!("{market:#x}"),
        candidate_mean: Some(cand_mu),
        candidate_variance: Some(Sq128::from_raw(quote.candidate.variance).to_f64()),
        candidate_sigma: Some(cand_sigma),
        candidate_mu1: None,
        candidate_mu2: None,
        candidate_rho: None,
        x_star: Some(Sq128::from_raw(quote.x_star).to_f64()),
        required_collateral: Some(req_collat),
        padded_collateral: Some(Sq128::from_raw(quote.padded_collateral).to_f64()),
        sigma_floor: None,
        market_mean: None,
        market_sigma: None,
        belief_mean: None,
        belief_sigma: None,
        expected_value: None,
        budget: None,
        on_chain_will_accept: quote.on_chain_will_accept,
        rejection,
        no_trade_reason: (!quote.on_chain_will_accept && quote.rejection.is_none())
            .then(|| "no trade: lognormal candidate failed runtime preflight".to_owned()),
        execute_hint,
    })
}

// ─── execute ───────────────────────────────────────────────────────────

pub(crate) async fn execute(ctx: &AppContext, args: TradeExecuteArgs, confirm: bool) -> Result<()> {
    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider(ctx)?;
    let client = DeadeyeClient::new(provider);
    let family = resolve_family(ctx, &client, market, args.family).await?;
    let label = family_label(family);

    match family {
        Family::Normal => execute_normal(ctx, &client, market, args, confirm, label).await,
        Family::Lognormal => execute_lognormal(ctx, &client, market, args, confirm, label).await,
        Family::Multinoulli | Family::Bivariate => {
            anyhow::bail!(
                "trade execute: only normal + lognormal are wired in Driver B's first cut"
            );
        },
    }
}

async fn execute_normal(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    args: TradeExecuteArgs,
    confirm: bool,
    label: &'static str,
) -> Result<()> {
    // Offline preflight by default (no runtime / no gas); `--runtime` opts
    // into the chain-faithful path. The offline quote also enforces the
    // σ-floor, so a sub-σ-min candidate is rejected before submission.
    let runtime = resolve_runtime_opt(args.runtime.as_deref(), Family::Normal)?;
    let market_handle = client.normal_market(market);
    anyhow::ensure!(
        args.min_ev.is_some() || args.min_collateral.is_none(),
        "--min-collateral only applies with --min-ev micro-edge search"
    );
    let has_direct_candidate = args.mean.is_some() || args.variance.is_some();
    anyhow::ensure!(
        !has_direct_candidate || (args.min_ev.is_none() && args.min_collateral.is_none()),
        "--min-ev/--min-collateral select optimizer micro-edge search; omit them with explicit --mean/--variance"
    );

    // Candidate: explicit --mean/--variance, or the same EV-max optimizer
    // `trade quote --belief/--budget` runs (issue #32) — the executed
    // candidate is exactly the optimizer's certified one, re-derived against
    // live state.
    let (mean, variance) = match (args.mean, args.variance, args.belief, args.budget) {
        (Some(m), Some(v), ..) => (m, v),
        (None, None, Some(belief_mean), Some(budget)) => {
            let snapshot = market_handle
                .state_snapshot()
                .await
                .context("reading market state snapshot")?;
            let belief_sigma = args.belief_sigma.unwrap_or(snapshot.sigma);
            // Sizing policy (issue #33) — same semantics as `trade quote`:
            // the belief never changes; the stake is capped by re-optimising
            // at a smaller budget.
            let kelly_mult = args
                .kelly
                .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
            let solve = if let Some(min_ev) = args.min_ev {
                anyhow::ensure!(
                    runtime.is_none(),
                    "--min-ev micro-edge execution is offline-only; drop --runtime"
                );
                let policy = NormalMicroPolicy {
                    min_ev,
                    min_collateral: args.min_collateral,
                    bankroll: args.bankroll,
                    kelly_mult,
                    max_cvar: args.max_cvar,
                };
                match solve_normal_micro_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    &policy,
                )? {
                    Ok(solve) => solve,
                    Err(reason) => anyhow::bail!("{reason}"),
                }
            } else {
                solve_normal_candidate(
                    &snapshot,
                    belief_mean,
                    belief_sigma,
                    budget,
                    args.bankroll,
                    kelly_mult,
                    args.max_cvar,
                )?
                .into_solve()?
            };
            let q = solve.quote;
            let ev = solve.expected_value;
            let basis = &solve.sizing_basis;
            let required = Sq128::from_raw(q.required_collateral).to_f64();
            if !q.on_chain_will_accept || required <= 0.0 {
                anyhow::bail!(
                    "normal optimizer found no acceptable positive-EV trade under budget {budget} XP — \
                     the market may already price your belief; re-quote with \
                     `trade quote --belief/--budget` for diagnostics"
                );
            }
            let m = Sq128::from_raw(q.candidate.mean).to_f64();
            let v = Sq128::from_raw(q.candidate.variance).to_f64();
            if ctx.renderer.mode() != OutputMode::Json {
                eprintln!(
                    "optimizer: EV-max candidate μ={m:.6}, σ²={v:.6} \
                     (EV {ev:+.4} XP, collateral ~{required:.4} XP, sizing_basis {basis})"
                );
            }
            (m, v)
        },
        _ => anyhow::bail!(
            "normal execute needs either --mean/--variance or --belief/--budget \
             [--belief-sigma] (runs the same EV-max optimizer as `trade quote`)"
        ),
    };

    let opts = NormalSubmitOptions {
        max_collateral: args.max_collateral,
        mode: WriteMode::from_flags(args.dry_run, args.emit_calldata),
        runtime,
        x_star_override: args.x_star,
        journal: args.journal.clone(),
        interactive: true,
        confirm,
        label,
    };
    match submit_normal_quote(ctx, client, market, mean, variance, &opts).await? {
        SubmitOutcome::CeilingExceeded { message } => anyhow::bail!(message),
        SubmitOutcome::PreflightRejected(result)
        | SubmitOutcome::DryRun(result)
        | SubmitOutcome::Submitted { result, .. } => ctx.renderer.print(&result),
        SubmitOutcome::EmitCalldata(result) => print_calldata_json(&result),
    }
}

/// Knobs for [`submit_normal_quote`] / [`submit_lognormal_quote`].
#[derive(Clone)]
pub(crate) struct NormalSubmitOptions {
    /// Hard ceiling (XP) on the gross collateral the trade may supply.
    pub(crate) max_collateral: f64,
    /// Whether this pipeline submits or stops after validation.
    pub(crate) mode: WriteMode,
    /// Optional math-runtime address (chain-faithful preflight for normal;
    /// diagnostic probe fallback for lognormal).
    pub(crate) runtime: Option<Felt>,
    /// Diagnostic x* override (hidden flag; bypasses the chain probe).
    pub(crate) x_star_override: Option<f64>,
    /// Canonical trade-journal path to append on successful submission.
    pub(crate) journal: Option<std::path::PathBuf>,
    /// Allow the interactive confirm prompt (TTY + non-JSON). The trade
    /// loop passes `false`: its own `--execute` flag is the standing
    /// consent and a long-running loop must never block on stdin.
    pub(crate) interactive: bool,
    /// Global `--confirm` (skips the prompt when `interactive`).
    pub(crate) confirm: bool,
    /// Family label for the prompt text.
    pub(crate) label: &'static str,
}

/// Typed result of a submit pipeline, so callers (the `trade execute`
/// renderer and the `trade loop` journal) can react without string-matching
/// `anyhow` prose.
pub(crate) enum SubmitOutcome {
    /// Preflight rejected the candidate — nothing was sent.
    PreflightRejected(SubmissionResult),
    /// Dry run: the multicall was simulated gas-free and nothing was sent.
    DryRun(SubmissionResult),
    /// Calldata emission: the multicall was built for an external signer.
    EmitCalldata(CalldataResult),
    /// The (chain-certified) collateral requirement exceeds `max_collateral`.
    /// Nothing was sent; `trade execute` renders this as a hard error, the
    /// loop as a skip.
    CeilingExceeded {
        /// Human-readable explanation, preserved from the historical bail.
        message: String,
    },
    /// A transaction was submitted; check `result.accepted` / `tx_hash`.
    Submitted {
        /// Render-ready submission verdict.
        result: SubmissionResult,
        /// Gross collateral supplied (XP) — what budget accounting must count.
        supplied_gross_xp: f64,
        /// Chain-certified net requirement (XP).
        required_net_xp: f64,
    },
}

/// Submit pipeline for a fixed normal candidate: offline (or runtime)
/// preflight → collateral buffer sizing → chain-probe x* certification →
/// fresh-wallet grant bundling → optional dry-run → submission + journal.
/// Extracted from `trade execute` so `trade loop` runs the **identical**
/// path (issue #40).
pub(crate) async fn submit_normal_quote(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    mean: f64,
    variance: f64,
    opts: &NormalSubmitOptions,
) -> Result<SubmitOutcome> {
    let market_handle = client.normal_market(market);
    let runtime = opts.runtime;

    let mut quote = if let Some(rt) = runtime {
        let cand_dist = deadeye_core::NormalDistribution::from_variance(
            Sq128::from_f64(mean)?,
            Sq128::from_f64(variance)?,
        )?;
        // Sq128-exact (σ, σ²) — see issue #36.
        let candidate = deadeye_core::Distribution::to_raw(&cand_dist);
        let current = market_handle.distribution().await?;
        let solver = deadeye_sdk::collateral::normal_collateral(
            &current,
            &cand_dist,
            deadeye_sdk::collateral::MinimizationPolicy::standard(),
        )
        .map_err(|e| anyhow::anyhow!("off-chain collateral solver: {e}"))?;
        let x_star = Sq128::from_f64(solver.x_min)?.to_raw();
        let supplied = Sq128::from_f64(opts.max_collateral)?.to_raw();
        market_handle
            .reader()
            .quote_trade(rt, candidate, x_star, supplied, supplied)
            .await
            .map_err(|e| anyhow::anyhow!("preflight quote_trade: {e}"))?
    } else {
        market_handle
            .quote_candidate_offline(mean, variance)
            .await
            .context("quote_candidate_offline preflight")?
    };

    // Size the *supplied* collateral the trade locks. The offline quote's
    // `padded_collateral` defaults to the bare f64-computed required amount
    // with **no margin** — which the on-chain Q128.128 `collateral_sufficient`
    // check rejects (`VERIFICATION_FAILED`) on the slightest rounding gap.
    // Supply a buffered amount instead (collateral is a *returned* margin lock,
    // not a cost), capped by the trader's `--max-collateral` ceiling. This
    // mirrors `trade quote`'s `execute_hint` and the webapp's buffered trade
    // collateral. Skipped for the `--runtime` path, which already supplies
    // `--max-collateral` and was validated by `check_trade_view`.
    if runtime.is_none() && quote.on_chain_will_accept {
        let required = Sq128::from_raw(quote.required_collateral).to_f64();
        let target = required * COLLATERAL_BUFFER;
        let supplied = if opts.max_collateral >= target {
            target
        } else if opts.max_collateral >= required {
            opts.max_collateral
        } else {
            return Ok(SubmitOutcome::CeilingExceeded {
                message: format!(
                    "--max-collateral {:.4} is below the required collateral {:.4}; \
                     raise it to at least ~{:.4} (required × {COLLATERAL_BUFFER}) so the \
                     on-chain collateral check clears",
                    opts.max_collateral, required, target,
                ),
            });
        };
        quote.padded_collateral = Sq128::from_f64(supplied)?.to_raw();
    }

    // Diagnostic override of x* (collateral point) to probe the on-chain
    // verifier's stationary check.
    if let Some(xs) = opts.x_star_override {
        quote.x_star = Sq128::from_f64(xs)?.to_raw();
    }

    if !quote.on_chain_will_accept {
        let rejection = quote.rejection.as_ref().map(pretty_rejection);
        return Ok(SubmitOutcome::PreflightRejected(SubmissionResult {
            action: "trade",
            market: format!("{market:#x}"),
            tx_hash: None,
            call_count: None,
            accepted: false,
            rejection,
            note: Some("preflight rejected — fix the cause and re-quote before retrying".into()),
        }));
    }

    let account = build_write_account(ctx, opts.mode)?;
    let writer_provider = build_provider(ctx)?;
    let writer =
        NormalMarketWriter::new(NormalMarketReader::new(&writer_provider, market), account);

    // Chain-probe `x*` refinement (issue #13 root cause). The AMM verifies
    // stationarity of the λ-scaled PDF difference in its own fixed-point
    // arithmetic, whose acceptance window (≈1e-7 wide in x) sits slightly off
    // the f64 root the off-chain solver finds — so a mathematically-perfect
    // x* still reverts with VERIFICATION_FAILED. Probe `check_trade_view`
    // (gas-free, simulated against the market's own runtime class) around the
    // f64 root and adopt the x* + collateral the chain itself certifies.
    if opts.x_star_override.is_none() {
        match deadeye_starknet::chain_probe::refine_normal_quote(
            writer.account(),
            writer.reader(),
            &quote,
        )
        .await
        {
            Ok(Some(outcome)) => {
                let chain_required = Sq128::from_raw(outcome.computed_collateral).to_f64();
                // `execute_trade` deducts deposit fees from the supplied
                // amount and verifies the NET against the requirement —
                // gross up by the measured net rate, plus a thin margin.
                // The NET supply must also clear the AMM's minimum-trade
                // floor, or a small trade reverts LOW_COLLATERAL.
                let min_trade = match writer.reader().params().await {
                    Ok(p) => Sq128::from_raw(p.min_trade_collateral).to_f64(),
                    Err(_) => 0.0,
                };
                let net_needed = chain_required.max(min_trade);
                let gross_needed = net_needed / outcome.net_rate;
                let buffered = gross_needed * 1.002;
                if buffered > opts.max_collateral {
                    return Ok(SubmitOutcome::CeilingExceeded {
                        message: format!(
                            "chain-verified collateral is {net_needed:.4} XP net \
                             (≈{buffered:.4} XP gross incl. deposit fees and the {min_trade:.4} \
                             XP minimum-trade floor), which exceeds --max-collateral {:.4}; \
                             raise the ceiling",
                            opts.max_collateral,
                        ),
                    });
                }
                quote.x_star = outcome.x_star;
                quote.required_collateral = outcome.computed_collateral;
                quote.padded_collateral = Sq128::from_f64(buffered)?.to_raw();
                if ctx.renderer.mode() != OutputMode::Json {
                    eprintln!(
                        "chain probe: certified x* (offset {:+.3e}, {} round(s)); \
                         collateral {chain_required:.4} XP net → supplying {buffered:.4} \
                         XP gross (fees {:.2}%)",
                        outcome.offset,
                        outcome.rounds,
                        (1.0 - outcome.net_rate) * 100.0,
                    );
                }
            },
            Ok(None) => {
                ctx.renderer.warning(
                    "chain probe could not certify an x* near the off-chain solution; \
                     submitting unrefined (the pre-submit simulation still blocks a \
                     reverting trade before any gas is spent)",
                );
            },
            Err(e) => {
                ctx.renderer.warning(&format!(
                    "chain probe unavailable ({e}); submitting unrefined (the pre-submit \
                     simulation still blocks a reverting trade before any gas is spent)"
                ));
            },
        }
    }

    // Fresh-wallet bootstrap: if the wallet's XP balance can't cover the
    // gross supply and its one-shot initial grant is unclaimed, bundle
    // `claim_initial_grant()` into the same atomic multicall so a brand-new
    // agent wallet can claim + approve + trade in a single transaction.
    let leading = match writer.reader().config().await {
        Ok(config) => {
            bootstrap_grant_calls(
                &writer_provider,
                config.collateral_token,
                config.token_decimals,
                deadeye_starknet::Account::address(writer.account()),
                Sq128::from_raw(quote.padded_collateral).to_f64(),
                ctx,
            )
            .await
        },
        Err(_) => Vec::new(),
    };

    if opts.interactive
        && opts.mode == WriteMode::Submit
        && !opts.confirm
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && ctx.renderer.mode() != OutputMode::Json
    {
        let label = opts.label;
        eprintln!("About to submit {label}-market trade:");
        eprintln!("  market:    {market:#x}");
        eprintln!(
            "  candidate: μ={:.4}, σ²={:.4}",
            Sq128::from_raw(quote.candidate.mean).to_f64(),
            Sq128::from_raw(quote.candidate.variance).to_f64()
        );
        eprintln!(
            "  required collateral: ~{:.4} XP",
            Sq128::from_raw(quote.required_collateral).to_f64()
        );
        eprintln!(
            "  supplied:  {:.4} XP",
            Sq128::from_raw(quote.padded_collateral).to_f64()
        );
        super::confirm_or_bail("Continue?")?;
    }

    // No-submit modes: build the full multicall once, simulate it gas-free,
    // then either render the verdict (`--dry-run`) or emit the calls for an
    // external signer (`--emit-calldata`).
    if opts.mode.is_no_submit() || ctx.config.uses_external_signer() {
        let labels = trade_call_labels(leading.len());
        let mut calls = leading;
        calls.extend(
            writer
                .build_trade_calls(&quote)
                .await
                .map_err(|e| anyhow::anyhow!("build trade calls: {e}"))?,
        );
        let simulation = simulate_calls("trade(dry-run)", market, writer.account(), &calls).await;
        if opts.mode == WriteMode::EmitCalldata {
            let account_address = deadeye_starknet::Account::address(writer.account());
            return Ok(SubmitOutcome::EmitCalldata(calldata_result(
                CalldataInput {
                    action: "trade(emit-calldata)",
                    family: Some(opts.label),
                    target: market,
                    account: account_address,
                    calls: &calls,
                    labels: &labels,
                    simulation: &simulation,
                    computed_collateral: Some(Sq128::from_raw(quote.required_collateral).to_f64()),
                },
            )));
        }
        if ctx.config.uses_external_signer() {
            if !simulation.accepted {
                return Ok(SubmitOutcome::PreflightRejected(simulation));
            }
            let supplied_gross_xp = Sq128::from_raw(quote.padded_collateral).to_f64();
            let required_net_xp = Sq128::from_raw(quote.required_collateral).to_f64();
            let submission = submit_external_calls(
                ctx,
                "trade",
                market,
                deadeye_starknet::Account::address(writer.account()),
                &calls,
                &labels,
            )
            .await?;
            return Ok(SubmitOutcome::Submitted {
                result: submission,
                supplied_gross_xp,
                required_net_xp,
            });
        }
        return Ok(SubmitOutcome::DryRun(simulation));
    }

    let supplied_gross_xp = Sq128::from_raw(quote.padded_collateral).to_f64();
    let required_net_xp = Sq128::from_raw(quote.required_collateral).to_f64();
    let result = match writer.execute_quote_bundled(quote, leading).await {
        Ok(receipt) => {
            if let Some(path) = &opts.journal {
                let _ = append_normal_journal(path, market, &writer, &quote, receipt);
            }
            submission_from_receipt("trade", format!("{market:#x}"), receipt)
        },
        Err(e) => submission_from_trade_error("trade", format!("{market:#x}"), &e),
    };
    Ok(SubmitOutcome::Submitted {
        result,
        supplied_gross_xp,
        required_net_xp,
    })
}

/// Decide whether the trade multicall needs a leading `claim_initial_grant()`
/// to bootstrap a fresh wallet: returns `[claim]` when the trader's XP
/// balance cannot cover `gross_supply` AND the one-shot grant is unclaimed,
/// `[]` otherwise (including on read failures — the pre-submit simulation
/// remains the safety net).
async fn bootstrap_grant_calls<P>(
    provider: &P,
    collateral_token: Felt,
    token_decimals: u8,
    trader: Felt,
    gross_supply: f64,
    ctx: &AppContext,
) -> Vec<deadeye_starknet::Call>
where
    P: deadeye_starknet::Provider + Sync,
{
    let token = deadeye_starknet::CollateralTokenReader::new(provider, collateral_token);
    let (Ok(balance), Ok(claimed)) = (
        token.balance_of(trader).await,
        token.has_claimed_initial_grant(trader).await,
    ) else {
        return Vec::new();
    };
    #[expect(clippy::cast_precision_loss, reason = "balance compare is approximate")]
    let balance_xp = balance.low() as f64 / 10f64.powi(i32::from(token_decimals));
    if balance.high() > 0 || balance_xp >= gross_supply || claimed {
        return Vec::new();
    }
    if ctx.renderer.mode() != OutputMode::Json {
        eprintln!(
            "fresh wallet: balance {balance_xp:.4} XP < supply {gross_supply:.4} XP and the \
             initial grant is unclaimed — bundling claim_initial_grant() into the multicall"
        );
    }
    vec![deadeye_starknet::build_claim_initial_grant_call(
        collateral_token,
    )]
}

fn trade_call_labels(leading_count: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(leading_count + 2);
    out.extend((0..leading_count).map(|_| "claim_initial_grant".to_owned()));
    out.extend(labels(["approve", "execute_trade"]));
    out
}

async fn execute_lognormal(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    args: TradeExecuteArgs,
    confirm: bool,
    label: &'static str,
) -> Result<()> {
    anyhow::ensure!(
        args.min_ev.is_none() && args.min_collateral.is_none(),
        "--min-ev/--min-collateral are normal-family micro-edge search flags"
    );
    // A math-runtime address is OPTIONAL and diagnostic-only. The submit path
    // always drafts the quote off-chain and has the chain itself certify
    // x* + hints via the probe. (Issues #30/#31: the old runtime preflight fed
    // `check_trade_view` the candidate's μ as x*, which fails the verifier's
    // side/stationarity checks for EVERY candidate — so all lognormal trades
    // were rejected as SideInvalid/StationaryInvalid before the probe that
    // would have certified them ever ran.)
    let runtime = resolve_runtime_opt(args.runtime.as_deref(), Family::Lognormal)?;

    // Candidate: explicit log-space --mean/--variance, or the same EV-max
    // optimizer `trade quote --belief/--budget` runs (issue #32) — so the
    // executed candidate is exactly the optimizer's certified one, re-derived
    // against live state. Belief is in LOG space, matching the on-chain (μ, σ).
    let (mean, variance) = match (args.mean, args.variance, args.belief, args.budget) {
        (Some(m), Some(v), ..) => (m, v),
        (None, None, Some(belief_mu), Some(budget)) => {
            anyhow::ensure!(
                args.max_cvar.is_none(),
                "--max-cvar is normal-family only for now (lognormal risk extras are not \
                 yet wired)"
            );
            let reader = LognormalMarketReader::new(client.provider(), market);
            let current = reader
                .distribution()
                .await
                .map_err(|e| anyhow::anyhow!("reading market distribution: {e}"))?;
            let market_sigma = deadeye_core::Distribution::sigma(&current).to_f64();
            let belief_sigma = args.belief_sigma.unwrap_or(market_sigma);
            let handle = client.lognormal_market(market);
            let kelly_mult = args
                .kelly
                .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
            let (result, basis, kelly_cap) = solve_lognormal_candidate(
                &handle,
                belief_mu,
                belief_sigma,
                budget,
                args.bankroll,
                kelly_mult,
            )
            .await?;
            if let Some(cap) = kelly_cap
                && ctx.renderer.mode() != OutputMode::Json
            {
                eprintln!("sizing: {basis} caps the stake at {cap:.4} XP");
            }
            if result.collateral_required <= 0.0 {
                anyhow::bail!(
                    "lognormal optimizer found no positive-EV trade inside the policy region \
                     under budget {budget} XP — either the market already prices your belief, \
                     or the per-trade movement caps bind (σ ratio ≤ 4×, |Δμ| ≤ 4σ per trade); \
                     for a far-away belief, ladder multiple trades"
                );
            }
            if ctx.renderer.mode() != OutputMode::Json {
                eprintln!(
                    "optimizer: EV-max candidate μ_log={:.6}, σ²_log={:.6} (EV {:+.4} XP, \
                     collateral ~{:.4} XP)",
                    result.optimized_mu,
                    result.optimized_variance,
                    result.expected_value,
                    result.collateral_required,
                );
            }
            (result.optimized_mu, result.optimized_variance)
        },
        _ => anyhow::bail!(
            "lognormal execute needs either --mean/--variance (log-space) or \
             --belief/--budget [--belief-sigma] (log-space; runs the same EV-max \
             optimizer as `trade quote`)"
        ),
    };

    let opts = NormalSubmitOptions {
        max_collateral: args.max_collateral,
        mode: WriteMode::from_flags(args.dry_run, args.emit_calldata),
        runtime,
        x_star_override: None,
        journal: args.journal.clone(),
        interactive: true,
        confirm,
        label,
    };
    match submit_lognormal_quote(ctx, client, market, mean, variance, &opts).await? {
        SubmitOutcome::CeilingExceeded { message } => anyhow::bail!(message),
        SubmitOutcome::PreflightRejected(result)
        | SubmitOutcome::DryRun(result)
        | SubmitOutcome::Submitted { result, .. } => ctx.renderer.print(&result),
        SubmitOutcome::EmitCalldata(result) => print_calldata_json(&result),
    }
}

/// Submit pipeline for a fixed lognormal candidate (log-space mean/variance):
/// off-chain draft → **mandatory** chain-probe x* + hints certification (with
/// a runtime-verifier diagnostic fallback) → fresh-wallet grant bundling →
/// optional dry-run → submission + journal. Extracted from `trade execute`
/// so `trade loop` runs the identical path (issue #40).
pub(crate) async fn submit_lognormal_quote(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    mean: f64,
    variance: f64,
    opts: &NormalSubmitOptions,
) -> Result<SubmitOutcome> {
    let runtime = opts.runtime;
    let reader = LognormalMarketReader::new(client.provider(), market);
    let current = reader
        .distribution()
        .await
        .map_err(|e| anyhow::anyhow!("reading market distribution: {e}"))?;

    let supplied = Sq128::from_f64(opts.max_collateral)?.to_raw();

    // Off-chain draft: solve x* with the f64 lognormal minimiser; the
    // hints + chain-exact x*/collateral come from the probe below.
    let cand_dist = deadeye_core::LognormalDistribution::from_variance(
        Sq128::from_f64(mean)?,
        Sq128::from_f64(variance)?,
    )?;
    // Encode the raw FROM the dist: the runtime's compute_hints_view verifies
    // (σ, σ²) consistency at Sq128 precision and rejects an f64-sqrt σ that
    // was quantized independently — the issue #36 Brazil/Belgium blocker.
    let candidate = deadeye_core::Distribution::to_raw(&cand_dist);
    let solved = deadeye_sdk::collateral::lognormal_collateral(
        &current,
        &cand_dist,
        deadeye_sdk::collateral::LognormalOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("off-chain lognormal solver: {e}"))?;
    let mut quote = deadeye_starknet::LognormalTradeQuote {
        candidate,
        // Placeholder — replaced by the probe's chain-computed hints.
        candidate_hints: deadeye_starknet::types::lognormal::LognormalSqrtHintsRaw {
            l2_norm_denom: Sq128::ZERO.to_raw(),
            backing_denom: Sq128::ZERO.to_raw(),
        },
        x_star: Sq128::from_f64(solved.x_star)?.to_raw(),
        required_collateral: Sq128::from_f64(solved.collateral)?.to_raw(),
        padded_collateral: supplied,
        on_chain_will_accept: true,
        rejection: None,
    };

    let account = build_write_account(ctx, opts.mode)?;
    let writer_provider = build_provider(ctx)?;
    let writer = LognormalMarketWriter::new(
        LognormalMarketReader::new(&writer_provider, market),
        account,
    );

    // Chain-probe x* certification (issue #13 root cause — fixed-point
    // stationarity drift). MANDATORY: it also supplies the chain-computed
    // candidate hints, without which `execute_trade` rejects the calldata.
    let probe_outcome = deadeye_starknet::chain_probe::refine_lognormal_quote(
        writer.account(),
        writer.reader(),
        &quote,
    )
    .await;
    match probe_outcome {
        Ok(Some(outcome)) => {
            let chain_required = Sq128::from_raw(outcome.computed_collateral).to_f64();
            // The AMM also enforces a minimum-trade floor on the NET supply —
            // for a small trade the probe-certified requirement can sit below
            // it, and supplying only the requirement reverts LOW_COLLATERAL.
            let min_trade = match reader.params().await {
                Ok(p) => Sq128::from_raw(p.min_trade_collateral).to_f64(),
                Err(_) => 0.0,
            };
            let net_needed = chain_required.max(min_trade);
            let gross_needed = net_needed / outcome.net_rate;
            let buffered = gross_needed * 1.002;
            if buffered > opts.max_collateral {
                return Ok(SubmitOutcome::CeilingExceeded {
                    message: format!(
                        "chain-verified collateral is {net_needed:.4} XP net (≈{buffered:.4} XP \
                         gross incl. deposit fees and the {min_trade:.4} XP minimum-trade floor), \
                         which exceeds --max-collateral {:.4}; raise the ceiling",
                        opts.max_collateral,
                    ),
                });
            }
            quote.x_star = outcome.x_star;
            quote.candidate_hints = outcome.candidate_hints;
            quote.required_collateral = outcome.computed_collateral;
            quote.padded_collateral = Sq128::from_f64(buffered)?.to_raw();
            if ctx.renderer.mode() != OutputMode::Json {
                eprintln!(
                    "chain probe: certified x* (offset {:+.3e}, {} round(s)); collateral \
                     {chain_required:.4} XP net → supplying {buffered:.4} XP gross (fees {:.2}%)",
                    outcome.offset,
                    outcome.rounds,
                    (1.0 - outcome.net_rate) * 100.0,
                );
            }
        },
        probe_miss @ (Ok(None) | Err(_)) => {
            // Diagnostic fallback: with a runtime configured, ask its
            // `check_trade_view` (seeded with the SOLVED x*, never μ_cand)
            // for a typed verdict. If it accepts, adopt its chain-computed
            // hints and proceed; otherwise surface the typed rejection.
            if let Some(rt) = runtime {
                let q = reader
                    .quote_trade(rt, candidate, quote.x_star, supplied, supplied)
                    .await
                    .map_err(|e| anyhow::anyhow!("diagnostic quote_trade: {e}"))?;
                if q.on_chain_will_accept {
                    quote.candidate_hints = q.candidate_hints;
                    quote.required_collateral = q.required_collateral;
                    // Size the supply like the probe-success path instead of
                    // leaving the FULL --max-collateral ceiling in place:
                    // buffered requirement (≥ the min-trade floor), clamped
                    // to the ceiling. Collateral is a returned margin lock,
                    // but budget accounting must reflect what is locked.
                    let required = Sq128::from_raw(q.required_collateral).to_f64();
                    let min_trade = match reader.params().await {
                        Ok(p) => Sq128::from_raw(p.min_trade_collateral).to_f64(),
                        Err(_) => 0.0,
                    };
                    let buffered = (required.max(min_trade) * COLLATERAL_BUFFER)
                        .min(opts.max_collateral)
                        .max(required);
                    quote.padded_collateral = Sq128::from_f64(buffered)?.to_raw();
                    ctx.renderer.warning(
                        "chain probe could not certify an x*, but the runtime verifier accepts \
                         the f64 root — submitting the runtime-validated quote (the pre-submit \
                         simulation still blocks a reverting trade before any gas is spent)",
                    );
                } else {
                    let rejection = q.rejection.as_ref().map(pretty_rejection);
                    return Ok(SubmitOutcome::PreflightRejected(SubmissionResult {
                        action: "trade",
                        market: format!("{market:#x}"),
                        tx_hash: None,
                        call_count: None,
                        accepted: false,
                        rejection,
                        note: Some(
                            "chain probe could not certify an x* and the runtime verifier \
                             rejects the candidate — adjust it (e.g. a smaller move) and retry"
                                .into(),
                        ),
                    }));
                }
            } else {
                match probe_miss {
                    Ok(None) => anyhow::bail!(
                        "the chain probe could not certify an x* for this candidate (and the \
                         offline path cannot construct chain-exact hints without it) — adjust \
                         the candidate (e.g. a smaller move) and retry"
                    ),
                    Err(e) => anyhow::bail!(
                        "chain probe unavailable ({e}) — the offline lognormal path needs it \
                         to construct chain-exact hints; retry, or pass --runtime with a \
                         deployed math-runtime address"
                    ),
                    Ok(Some(_)) => unreachable!("handled above"),
                }
            }
        },
    }

    // Fresh-wallet bootstrap (see execute_normal).
    let leading = match writer.reader().config().await {
        Ok(config) => {
            bootstrap_grant_calls(
                &writer_provider,
                config.collateral_token,
                config.token_decimals,
                deadeye_starknet::Account::address(writer.account()),
                Sq128::from_raw(quote.padded_collateral).to_f64(),
                ctx,
            )
            .await
        },
        Err(_) => Vec::new(),
    };

    if opts.interactive
        && opts.mode == WriteMode::Submit
        && !opts.confirm
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && ctx.renderer.mode() != OutputMode::Json
    {
        let label = opts.label;
        eprintln!("About to submit {label}-market trade:");
        eprintln!("  market:    {market:#x}");
        eprintln!(
            "  candidate: μ_log={:.4}, σ²_log={:.4}",
            Sq128::from_raw(quote.candidate.mu).to_f64(),
            Sq128::from_raw(quote.candidate.variance).to_f64()
        );
        eprintln!(
            "  required collateral: ~{:.4} XP",
            Sq128::from_raw(quote.required_collateral).to_f64()
        );
        eprintln!(
            "  supplied:  {:.4} XP",
            Sq128::from_raw(quote.padded_collateral).to_f64()
        );
        super::confirm_or_bail("Continue?")?;
    }

    // No-submit modes: build the full multicall once, simulate it gas-free,
    // then either render the verdict (`--dry-run`) or emit the calls for an
    // external signer (`--emit-calldata`).
    if opts.mode.is_no_submit() || ctx.config.uses_external_signer() {
        let labels = trade_call_labels(leading.len());
        let mut calls = leading;
        calls.extend(
            writer
                .build_trade_calls(&quote)
                .await
                .map_err(|e| anyhow::anyhow!("build trade calls: {e}"))?,
        );
        let simulation = simulate_calls("trade(dry-run)", market, writer.account(), &calls).await;
        if opts.mode == WriteMode::EmitCalldata {
            let account_address = deadeye_starknet::Account::address(writer.account());
            return Ok(SubmitOutcome::EmitCalldata(calldata_result(
                CalldataInput {
                    action: "trade(emit-calldata)",
                    family: Some(opts.label),
                    target: market,
                    account: account_address,
                    calls: &calls,
                    labels: &labels,
                    simulation: &simulation,
                    computed_collateral: Some(Sq128::from_raw(quote.required_collateral).to_f64()),
                },
            )));
        }
        if ctx.config.uses_external_signer() {
            if !simulation.accepted {
                return Ok(SubmitOutcome::PreflightRejected(simulation));
            }
            let supplied_gross_xp = Sq128::from_raw(quote.padded_collateral).to_f64();
            let required_net_xp = Sq128::from_raw(quote.required_collateral).to_f64();
            let submission = submit_external_calls(
                ctx,
                "trade",
                market,
                deadeye_starknet::Account::address(writer.account()),
                &calls,
                &labels,
            )
            .await?;
            return Ok(SubmitOutcome::Submitted {
                result: submission,
                supplied_gross_xp,
                required_net_xp,
            });
        }
        return Ok(SubmitOutcome::DryRun(simulation));
    }

    let supplied_gross_xp = Sq128::from_raw(quote.padded_collateral).to_f64();
    let required_net_xp = Sq128::from_raw(quote.required_collateral).to_f64();
    let result = match writer.execute_quote_bundled(quote, leading).await {
        Ok(receipt) => {
            // Journal the canonical Trade entry (was dropped pre-#40).
            if let Some(path) = &opts.journal {
                let _ = append_lognormal_journal(path, market, &writer, &quote, receipt);
            }
            submission_from_receipt("trade", format!("{market:#x}"), receipt)
        },
        Err(e) => submission_from_trade_error("trade", format!("{market:#x}"), &e),
    };
    Ok(SubmitOutcome::Submitted {
        result,
        supplied_gross_xp,
        required_net_xp,
    })
}

// ─── journal ──────────────────────────────────────────────────────────

fn journal_cmd(ctx: &AppContext, args: TradeJournalArgs) -> Result<()> {
    let path = match args.path {
        Some(p) => p,
        None => default_journal_path()?,
    };
    if !path.exists() {
        ctx.renderer
            .warning(&format!("journal {} does not exist", path.display()));
        return Ok(());
    }
    let entries: Vec<JournalEntry> = TradeJournal::replay(&path)
        .with_context(|| format!("opening journal {}", path.display()))?
        .filter_map(Result::ok)
        .collect();
    let tail_start = entries.len().saturating_sub(args.tail);
    let slice = &entries[tail_start..];
    match ctx.renderer.mode() {
        OutputMode::Json => {
            let json = serde_json::to_string_pretty(slice)?;
            println!("{json}");
        },
        OutputMode::Pretty | OutputMode::Plain => {
            ctx.renderer.header(&format!(
                "Journal {} — {} entries",
                path.display(),
                slice.len()
            ));
            for entry in slice {
                println!(
                    "{:?} family={:?} market={:#x} tx={}",
                    entry.kind,
                    entry.family,
                    entry.market,
                    entry
                        .tx_hash
                        .map(|h| format!("{h:#x}"))
                        .unwrap_or_else(|| "(none)".into()),
                );
            }
        },
    }
    Ok(())
}

pub(crate) fn default_journal_path() -> Result<std::path::PathBuf> {
    let mut dir =
        dirs::data_dir().context("could not locate user data dir; pass --path explicitly")?;
    dir.push("deadeye");
    std::fs::create_dir_all(&dir).ok();
    dir.push("journal.jsonl");
    Ok(dir)
}

fn append_lognormal_journal<P, A>(
    path: &std::path::Path,
    market: Felt,
    writer: &LognormalMarketWriter<P, A>,
    quote: &deadeye_starknet::LognormalTradeQuote,
    receipt: deadeye_starknet::ExecutionReceipt,
) -> Result<()>
where
    P: deadeye_starknet::Provider,
    A: Account,
{
    let mut journal =
        TradeJournal::open(path).with_context(|| format!("opening journal {}", path.display()))?;
    let entry = JournalEntry::new(
        Family::Lognormal,
        market,
        Account::address(writer.account()),
        EntryKind::Trade,
        json!({
            "candidate_mu": Sq128::from_raw(quote.candidate.mu).to_f64(),
            "candidate_variance": Sq128::from_raw(quote.candidate.variance).to_f64(),
            "x_star": Sq128::from_raw(quote.x_star).to_f64(),
            "required_collateral": Sq128::from_raw(quote.required_collateral).to_f64(),
            "padded_collateral": Sq128::from_raw(quote.padded_collateral).to_f64(),
        }),
    )
    .with_tx_hash(receipt.transaction_hash)
    .with_receipt(json!({
        "transaction_hash": format!("{:#x}", receipt.transaction_hash),
        "call_count": receipt.call_count,
    }));
    journal
        .append(&entry)
        .with_context(|| format!("appending to journal {}", path.display()))
}

fn append_normal_journal<P, A>(
    path: &std::path::Path,
    market: Felt,
    writer: &NormalMarketWriter<P, A>,
    quote: &deadeye_starknet::NormalTradeQuote,
    receipt: deadeye_starknet::ExecutionReceipt,
) -> Result<()>
where
    P: deadeye_starknet::Provider,
    A: Account,
{
    let mut journal =
        TradeJournal::open(path).with_context(|| format!("opening journal {}", path.display()))?;
    let entry = JournalEntry::new(
        Family::Normal,
        market,
        Account::address(writer.account()),
        EntryKind::Trade,
        json!({
            "candidate_mean": Sq128::from_raw(quote.candidate.mean).to_f64(),
            "candidate_variance": Sq128::from_raw(quote.candidate.variance).to_f64(),
            "x_star": Sq128::from_raw(quote.x_star).to_f64(),
            "required_collateral": Sq128::from_raw(quote.required_collateral).to_f64(),
            "padded_collateral": Sq128::from_raw(quote.padded_collateral).to_f64(),
        }),
    )
    .with_tx_hash(receipt.transaction_hash)
    .with_receipt(json!({
        "transaction_hash": format!("{:#x}", receipt.transaction_hash),
        "call_count": receipt.call_count,
    }));
    journal
        .append(&entry)
        .with_context(|| format!("appending to journal {}", path.display()))
}
