//! Shared `Render` impls for command-result types.
//!
//! Every Driver-B command outputs a small data record (e.g.
//! [`QuoteResult`], [`SubmissionResult`], [`WatchUpdate`]) implementing
//! [`crate::output::Render`] for pretty / plain rendering and
//! [`serde::Serialize`] for `--output json`.

use std::io::{self, Write};

use deadeye_sdk::SettlementPoint;
use deadeye_starknet::{TradeRejectionReason, VerificationSubReason};
use serde::Serialize;

use crate::output::{Render, Renderer};

/// Human-readable explanation of a [`TradeRejectionReason`].
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RejectionExplanation {
    /// The variant name (`"VerificationFailed"`, `"LowCollateral"`, …).
    pub(crate) variant: String,
    /// Optional sub-variant (e.g. `"CurvatureInvalid"`).
    pub(crate) sub_variant: Option<String>,
    /// One-line summary.
    pub(crate) summary: &'static str,
    /// Suggested action.
    pub(crate) suggested_fix: &'static str,
}

/// The single source-of-truth mapping
/// `TradeRejectionReason` → human-friendly explanation.
pub(crate) fn pretty_rejection(reason: &TradeRejectionReason) -> RejectionExplanation {
    use TradeRejectionReason as R;
    let (variant, sub_variant, summary, suggested_fix): (
        String,
        Option<String>,
        &'static str,
        &'static str,
    ) = match *reason {
        R::InvalidDistribution => (
            "InvalidDistribution".to_owned(),
            None,
            "The candidate distribution failed basic invariants (e.g. σ ≤ 0, σ² ≠ σ·σ).",
            "Re-check the candidate's `mean`, `variance`, and `sigma` fields — they must satisfy σ² = σ·σ and σ > 0.",
        ),
        R::InvalidHints => (
            "InvalidHints".to_owned(),
            None,
            "The sqrt hints didn't round-trip against the math runtime.",
            "Let `quote_trade` fetch fresh hints from the runtime; do not hand-roll them in f64.",
        ),
        R::BackingFail => (
            "BackingFail".to_owned(),
            None,
            "Pool backing check failed — the AMM cannot absorb this trade given current LP shares.",
            "Wait for more LP, reduce the candidate's |Δμ|, or split the trade into smaller steps.",
        ),
        R::SigmaTooLow => (
            "SigmaTooLow".to_owned(),
            None,
            "Candidate σ is below the market's per-trade σ floor.",
            "Widen `--variance` (σ² = variance) so σ ≥ market.min_sigma.",
        ),
        R::LowCollateral => (
            "LowCollateral".to_owned(),
            None,
            "Supplied collateral is below the AMM's minimum-trade-collateral floor.",
            "Increase `--max-collateral` (try at least `min_trade_collateral` × 1.1).",
        ),
        R::VerificationFailed { sub_reason } => {
            let (sub, sum, fix): (Option<&'static str>, &'static str, &'static str) =
                match sub_reason {
                    Some(VerificationSubReason::SideInvalid) => (
                        Some("SideInvalid"),
                        "Verifier rejected the supplied x* (collateral point): wrong side of the payoff crossing at chain precision.",
                        "Not a move-size problem — drop any hand-supplied x* and re-run `trade execute` (it chain-probes and certifies x* automatically). Per-trade caps are σ ratio ≤ 4× and |Δμ| ≤ 4σ_market.",
                    ),
                    Some(VerificationSubReason::StationaryInvalid) => (
                        Some("StationaryInvalid"),
                        "d'(x*) is outside the chain's tolerance — the supplied x* missed the verifier's own fixed-point root.",
                        "Re-run `trade execute` (it chain-probes x* to the verifier's root); if you supplied --x-star by hand, drop it.",
                    ),
                    Some(VerificationSubReason::CurvatureInvalid) => (
                        Some("CurvatureInvalid"),
                        "d''(x*) ≤ 0 at chain precision — the off-chain x* is at a saddle, not a minimum.",
                        "Increase supplied collateral (raise `--max-collateral`) or widen the candidate's σ.",
                    ),
                    Some(VerificationSubReason::CollateralInsufficient) => (
                        Some("CollateralInsufficient"),
                        "Chain-recomputed required collateral exceeded the supplied amount.",
                        "Raise `--max-collateral` — the off-chain solver underestimated by < 5%; +10% pad usually works.",
                    ),
                    Some(VerificationSubReason::MinimumInvalid) => (
                        Some("MinimumInvalid"),
                        "Verifier's minimum-finding routine couldn't converge.",
                        "Re-quote — chain state may have drifted; if it persists, reduce |Δμ|.",
                    ),
                    None => (
                        None,
                        "Generic verifier failure — chain rejected the trade without a refined sub-reason.",
                        "Re-quote to refresh chain state; if it persists, widen σ and re-pad collateral.",
                    ),
                    Some(_) => (
                        None,
                        "Unknown verifier sub-reason — newer SDK variant.",
                        "Update the CLI to match the SDK's VerificationSubReason variants.",
                    ),
                };
            (
                "VerificationFailed".to_owned(),
                sub.map(str::to_owned),
                sum,
                fix,
            )
        },
        R::StaleState { field } => (
            "StaleState".to_owned(),
            Some(field.to_owned()),
            "An `expected_*` guard mismatched live chain state (chain moved between quote + execute).",
            "Re-run `deadeye trade quote` and submit again — this is the canonical MM retry path.",
        ),
        R::MarketSettled => (
            "MarketSettled".to_owned(),
            None,
            "Market has been settled — trading is permanently closed.",
            "There is nothing to do. If you have a position, run `deadeye claim`.",
        ),
        R::MarketPaused => (
            "MarketPaused".to_owned(),
            None,
            "Market is paused — admin froze trading.",
            "Wait for admin to call `unpause_market`. Reads + claims remain available.",
        ),
        R::NoPosition => (
            "NoPosition".to_owned(),
            None,
            "Trader has no position on this market.",
            "Open one via `deadeye trade execute` first, then sell or claim.",
        ),
        R::AlreadyClaimed => (
            "AlreadyClaimed".to_owned(),
            None,
            "Position has already been claimed; no payout remains.",
            "Inspect `deadeye position show` to verify the payout was credited.",
        ),
        R::RequiresAdditionalCollateral => (
            "RequiresAdditionalCollateral".to_owned(),
            None,
            "Trade requires more collateral than was supplied.",
            "Raise `--max-collateral`.",
        ),
        R::NoCollateralOut => (
            "NoCollateralOut".to_owned(),
            None,
            "Sell would return zero collateral — the position is exhausted.",
            "Skip the sell; the position has no remaining payout.",
        ),
        R::ConversionFailed => (
            "ConversionFailed".to_owned(),
            None,
            "Token / internal-unit conversion overflowed.",
            "Reduce the trade size — you're hitting Q128.128 bounds.",
        ),
        R::OnlyOwner => (
            "OnlyOwner".to_owned(),
            None,
            "Entrypoint is restricted to the factory owner.",
            "Run this command from the factory owner's address (see `deadeye account show`).",
        ),
        R::NotAuthorized => (
            "NotAuthorized".to_owned(),
            None,
            "Entrypoint requires admin / owner privilege.",
            "Confirm the active profile's address is on the admin list.",
        ),
        R::MinOutNotMet => (
            "MinOutNotMet".to_owned(),
            None,
            "Slippage floor breached — `min_out` set above actual collateral-out.",
            "Lower `--min-out` or accept the worse fill.",
        ),
        R::InvalidMinOutcome => (
            "InvalidMinOutcome".to_owned(),
            None,
            "Multinoulli `min_outcome_index` did not match the candidate's argmin.",
            "Let the SDK derive `min_outcome_index` — do not hand-pick it.",
        ),
        R::Reentrant => (
            "Reentrant".to_owned(),
            None,
            "Re-entrant call detected — the AMM is currently mid-flight.",
            "Retry once; if it persists, the AMM is wedged and an admin must investigate.",
        ),
        R::MarketNotInitialized => (
            "MarketNotInitialized".to_owned(),
            None,
            "Market has not been initialised yet — admin must seed initial LP.",
            "Admin should run the initialize flow before traders can interact.",
        ),
        R::AlreadySettled => (
            "AlreadySettled".to_owned(),
            None,
            "Settle entrypoint called twice.",
            "Skip — the market is already settled.",
        ),
        R::AlreadyPaused => (
            "AlreadyPaused".to_owned(),
            None,
            "Pause called on an already-paused market.",
            "Skip — market is already paused.",
        ),
        R::NotPaused => (
            "NotPaused".to_owned(),
            None,
            "Unpause called on a market that wasn't paused.",
            "Skip — market is already live.",
        ),
        R::MarketNotSettled => (
            "MarketNotSettled".to_owned(),
            None,
            "Claim called before the market was settled.",
            "Wait for admin to call `settle_*` before claiming.",
        ),
        R::NoClaim => (
            "NoClaim".to_owned(),
            None,
            "Claim path produced an empty result (no entitled payout).",
            "Inspect `deadeye position show` — your position contributed zero at settlement.",
        ),
        R::TraderClaimsPending => (
            "TraderClaimsPending".to_owned(),
            None,
            "LP withdraw is blocked because trader positions remain unclaimed.",
            "Wait for traders to claim, then retry the LP withdraw.",
        ),
        R::OnlyFactory => (
            "OnlyFactory".to_owned(),
            None,
            "Entrypoint is restricted to the factory.",
            "Use `deadeye admin collect-fees` (which routes through the factory).",
        ),
        R::InvalidMatrixMode => (
            "InvalidMatrixMode".to_owned(),
            None,
            "Multinoulli matrix-mode validation failed.",
            "File a bug — the SDK should never submit an invalid matrix mode.",
        ),
        R::InvalidSettlementMode => (
            "InvalidSettlementMode".to_owned(),
            None,
            "Multinoulli LP claim path saw an unexpected settlement-mode discriminant.",
            "File a bug — the chain is in an unexpected state.",
        ),
        R::MissingSnapshotRef => (
            "MissingSnapshotRef".to_owned(),
            None,
            "Multinoulli snapshot bookkeeping is inconsistent.",
            "File a bug — the chain is in an unexpected state.",
        ),
        R::Other { raw } => (
            "Other".to_owned(),
            Some(raw.to_owned()),
            "Unmapped revert reason — the SDK saw a known short string but doesn't have a dedicated variant.",
            "Capture the raw string in your bug report; the SDK team will add a typed variant.",
        ),
        _ => (
            "Unknown".to_owned(),
            None,
            "Unrecognised rejection variant — likely a newer SDK version with additional variants.",
            "Update the CLI to match the SDK's TradeRejectionReason variants.",
        ),
    };
    RejectionExplanation {
        variant,
        sub_variant,
        summary,
        suggested_fix,
    }
}

/// Renderable form of a preflighted trade quote.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct QuoteResult {
    pub(crate) family: &'static str,
    pub(crate) market: String,
    pub(crate) candidate_mean: Option<f64>,
    pub(crate) candidate_variance: Option<f64>,
    pub(crate) candidate_sigma: Option<f64>,
    pub(crate) candidate_mu1: Option<f64>,
    pub(crate) candidate_mu2: Option<f64>,
    pub(crate) candidate_rho: Option<f64>,
    pub(crate) x_star: Option<f64>,
    pub(crate) required_collateral: Option<f64>,
    pub(crate) padded_collateral: Option<f64>,
    /// Backing-derived σ-floor: candidate σ below this is rejected
    /// SIGMA_TOO_LOW.
    pub(crate) sigma_floor: Option<f64>,
    /// Current on-chain market curve.
    pub(crate) market_mean: Option<f64>,
    pub(crate) market_sigma: Option<f64>,
    /// Trader belief (optimizer path only).
    pub(crate) belief_mean: Option<f64>,
    pub(crate) belief_sigma: Option<f64>,
    /// Expected value of the candidate under the belief (XP).
    pub(crate) expected_value: Option<f64>,
    /// Budget cap the optimizer respected (XP).
    pub(crate) budget: Option<f64>,
    pub(crate) on_chain_will_accept: bool,
    pub(crate) rejection: Option<RejectionExplanation>,
    /// Human-readable reason for a deliberate no-trade quote when there is no
    /// typed on-chain rejection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) no_trade_reason: Option<String>,
    /// P&L if the market settles exactly at today's market mean (XP) —
    /// the plain-terms cost of being wrong (issue #24).
    pub(crate) downside_at_market_mean: Option<f64>,
    /// Expected shortfall over the worst 5% of belief-weighted outcomes (XP).
    pub(crate) cvar_5pct: Option<f64>,
    /// EV re-computed with belief σ widened 1.5× — "EV if you're 1σ
    /// overconfident" (issue #24).
    pub(crate) stress_ev: Option<f64>,
    /// Kelly/bankroll sizing recommendation (issue #15).
    pub(crate) sizing: Option<crate::commands::risk::SizingAdvice>,
    /// Which sizing constraint bound the stake: `budget`, `kelly-<frac>`,
    /// or `cvar-cap` (issue #33). None when no sizing policy was applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sizing_basis: Option<String>,
    /// Pre-trade lint findings — warnings, never blocks (issue #24).
    pub(crate) warnings: Vec<String>,
    pub(crate) execute_hint: String,
}

/// Closed-form normal quantiles [P5, P25, P50, P75, P95].
pub(crate) fn normal_quantiles(mean: f64, sigma: f64) -> [f64; 5] {
    const Z: [f64; 5] = [-1.644_854, -0.674_490, 0.0, 0.674_490, 1.644_854];
    Z.map(|z| sigma.mul_add(z, mean))
}

fn fmt_quantiles(q: [f64; 5]) -> String {
    format!(
        "P5 {:.4} | P25 {:.4} | P50 {:.4} | P75 {:.4} | P95 {:.4}",
        q[0], q[1], q[2], q[3], q[4]
    )
}

impl Render for QuoteResult {
    fn render_pretty(&self, r: &Renderer) {
        if self.on_chain_will_accept {
            r.success("Preflight: chain will accept");
        } else {
            r.error("Preflight: chain WILL REJECT this trade");
        }
        r.kv("family", self.family);
        r.kv("market", &self.market);
        if let (Some(mm), Some(ms)) = (self.market_mean, self.market_sigma) {
            r.kv("market_curve", &format!("μ={mm:.6}, σ={ms:.6}"));
            r.kv("market_quantiles", &fmt_quantiles(normal_quantiles(mm, ms)));
        }
        if let (Some(bm), Some(bs)) = (self.belief_mean, self.belief_sigma) {
            r.kv("belief", &format!("μ={bm:.6}, σ={bs:.6}"));
        }
        if let (Some(m), Some(v)) = (self.candidate_mean, self.candidate_variance) {
            r.kv("candidate", &format!("μ={m:.6}, σ²={v:.6}"));
            if let Some(cs) = self.candidate_sigma {
                r.kv(
                    "candidate_quantiles",
                    &fmt_quantiles(normal_quantiles(m, cs)),
                );
            }
            // Market impact of this stake (issue #20): what the quoted
            // collateral buys in curve movement.
            if let (Some(mm), Some(ms), Some(cs), Some(rc)) = (
                self.market_mean,
                self.market_sigma,
                self.candidate_sigma,
                self.required_collateral,
            ) {
                r.kv(
                    "market_impact",
                    &format!(
                        "{rc:.2} XP moves μ by {:+.4} and σ by {:+.4}",
                        m - mm,
                        cs - ms,
                    ),
                );
            }
        }
        if let (Some(m1), Some(m2)) = (self.candidate_mu1, self.candidate_mu2) {
            r.kv("candidate", &format!("μ₁={m1:.6}, μ₂={m2:.6}"));
        }
        if let Some(rho) = self.candidate_rho {
            r.kv("rho", &format!("{rho:.6}"));
        }
        if let Some(xs) = self.x_star {
            r.kv("x_star", &format!("{xs:.6}"));
        }
        if let Some(rc) = self.required_collateral {
            r.kv("required_collateral", &format!("{rc:.6} STRK"));
        }
        if let Some(pc) = self.padded_collateral {
            r.kv("padded_collateral", &format!("{pc:.6} STRK"));
        }
        if let Some(ev) = self.expected_value {
            r.kv("expected_value", &format!("{ev:.6} XP"));
        }
        if let Some(sf) = self.sigma_floor {
            r.kv(
                "sigma_floor",
                &format!("{sf:.6}  (candidate σ must be ≥ this, else SIGMA_TOO_LOW)"),
            );
        }
        if let Some(d) = self.downside_at_market_mean {
            r.kv(
                "if_settles_at_market_mean",
                &format!("{}{d:.4} XP", if d >= 0.0 { "+" } else { "" }),
            );
        }
        if let Some(c) = self.cvar_5pct {
            r.kv(
                "cvar_5pct",
                &format!("{c:.4} XP  (avg of worst 5% outcomes under your belief)"),
            );
        }
        if let Some(sev) = self.stress_ev {
            r.kv(
                "stress_ev",
                &format!("{sev:.4} XP  (EV if your σ is 1.5× too tight)"),
            );
        }
        if let Some(basis) = &self.sizing_basis {
            r.kv("sizing_basis", basis);
        }
        if let Some(sz) = &self.sizing {
            r.kv(
                "sizing",
                &format!(
                    "recommend {:.2} XP  (edge {:.3}/XP, full-Kelly {:.1}%, applied {:.0}% Kelly)",
                    sz.recommended_stake_xp,
                    sz.edge_per_xp,
                    sz.full_kelly_fraction * 100.0,
                    sz.kelly_multiplier * 100.0,
                ),
            );
        }
        for warning in &self.warnings {
            r.warning(warning);
        }
        if let Some(rej) = &self.rejection {
            r.kv(
                "rejected",
                &format!(
                    "{} ({})",
                    rej.variant,
                    rej.sub_variant.as_deref().unwrap_or("-")
                ),
            );
            r.kv("what_this_means", rej.summary);
            r.kv("suggested_fix", rej.suggested_fix);
        }
        if let Some(reason) = &self.no_trade_reason {
            r.kv("no_trade_reason", reason);
        }
        if self.on_chain_will_accept {
            println!();
            println!("  {}", r.dim("to execute, run:"));
            println!("  {}", r.highlight(&self.execute_hint));
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "family: {}", self.family)?;
        writeln!(w, "market: {}", self.market)?;
        if let Some(m) = self.candidate_mean {
            writeln!(w, "candidate_mean: {m}")?;
        }
        if let Some(v) = self.candidate_variance {
            writeln!(w, "candidate_variance: {v}")?;
        }
        if let Some(s) = self.candidate_sigma {
            writeln!(w, "candidate_sigma: {s}")?;
        }
        if let Some(m) = self.candidate_mu1 {
            writeln!(w, "candidate_mu1: {m}")?;
        }
        if let Some(m) = self.candidate_mu2 {
            writeln!(w, "candidate_mu2: {m}")?;
        }
        if let Some(rho) = self.candidate_rho {
            writeln!(w, "candidate_rho: {rho}")?;
        }
        if let Some(xs) = self.x_star {
            writeln!(w, "x_star: {xs}")?;
        }
        if let Some(rc) = self.required_collateral {
            writeln!(w, "required_collateral: {rc}")?;
        }
        if let Some(pc) = self.padded_collateral {
            writeln!(w, "padded_collateral: {pc}")?;
        }
        if let Some(sf) = self.sigma_floor {
            writeln!(w, "sigma_floor: {sf}")?;
        }
        if let Some(ev) = self.expected_value {
            writeln!(w, "expected_value: {ev}")?;
        }
        if let Some(mm) = self.market_mean {
            writeln!(w, "market_mean: {mm}")?;
        }
        if let Some(ms) = self.market_sigma {
            writeln!(w, "market_sigma: {ms}")?;
        }
        if let Some(bm) = self.belief_mean {
            writeln!(w, "belief_mean: {bm}")?;
        }
        if let Some(bs) = self.belief_sigma {
            writeln!(w, "belief_sigma: {bs}")?;
        }
        if let Some(d) = self.downside_at_market_mean {
            writeln!(w, "downside_at_market_mean: {d}")?;
        }
        if let Some(c) = self.cvar_5pct {
            writeln!(w, "cvar_5pct: {c}")?;
        }
        if let Some(sev) = self.stress_ev {
            writeln!(w, "stress_ev: {sev}")?;
        }
        if let Some(sz) = &self.sizing {
            writeln!(w, "recommended_stake_xp: {}", sz.recommended_stake_xp)?;
            writeln!(w, "edge_per_xp: {}", sz.edge_per_xp)?;
            writeln!(w, "kelly_multiplier: {}", sz.kelly_multiplier)?;
        }
        for warning in &self.warnings {
            writeln!(w, "warning: {warning}")?;
        }
        if let Some(b) = self.budget {
            writeln!(w, "budget: {b}")?;
        }
        writeln!(w, "on_chain_will_accept: {}", self.on_chain_will_accept)?;
        if let Some(rej) = &self.rejection {
            writeln!(w, "rejection_variant: {}", rej.variant)?;
            if let Some(sub) = &rej.sub_variant {
                writeln!(w, "rejection_sub_variant: {sub}")?;
            }
            writeln!(w, "rejection_summary: {}", rej.summary)?;
            writeln!(w, "rejection_suggested_fix: {}", rej.suggested_fix)?;
        }
        if let Some(reason) = &self.no_trade_reason {
            writeln!(w, "no_trade_reason: {reason}")?;
        }
        writeln!(w, "execute_hint: {}", self.execute_hint)
    }
}

/// Renderable form of a write-path submission result.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct SubmissionResult {
    pub(crate) action: &'static str,
    pub(crate) market: String,
    pub(crate) tx_hash: Option<String>,
    pub(crate) call_count: Option<usize>,
    pub(crate) accepted: bool,
    pub(crate) rejection: Option<RejectionExplanation>,
    pub(crate) note: Option<String>,
}

impl Render for SubmissionResult {
    fn render_pretty(&self, r: &Renderer) {
        if self.accepted {
            r.success(&format!("{} submitted", self.action));
        } else {
            r.error(&format!("{} REJECTED", self.action));
        }
        r.kv("market", &self.market);
        if let Some(h) = &self.tx_hash {
            r.kv("tx_hash", h);
        }
        if let Some(c) = self.call_count {
            r.kv("call_count", &c.to_string());
        }
        if let Some(rej) = &self.rejection {
            r.kv("rejection", &rej.variant);
            if let Some(sub) = &rej.sub_variant {
                r.kv("sub_variant", sub);
            }
            r.kv("what_this_means", rej.summary);
            r.kv("suggested_fix", rej.suggested_fix);
        }
        if let Some(n) = &self.note {
            r.kv("note", n);
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "action: {}", self.action)?;
        writeln!(w, "market: {}", self.market)?;
        if let Some(h) = &self.tx_hash {
            writeln!(w, "tx_hash: {h}")?;
        }
        if let Some(c) = self.call_count {
            writeln!(w, "call_count: {c}")?;
        }
        writeln!(w, "accepted: {}", self.accepted)?;
        if let Some(rej) = &self.rejection {
            writeln!(w, "rejection_variant: {}", rej.variant)?;
            if let Some(sub) = &rej.sub_variant {
                writeln!(w, "rejection_sub_variant: {sub}")?;
            }
            writeln!(w, "rejection_summary: {}", rej.summary)?;
            writeln!(w, "rejection_suggested_fix: {}", rej.suggested_fix)?;
        }
        if let Some(n) = &self.note {
            writeln!(w, "note: {n}")?;
        }
        Ok(())
    }
}

/// One emitted block-update line from `deadeye watch`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WatchUpdate {
    pub(crate) family: &'static str,
    pub(crate) market: String,
    pub(crate) block_number: u64,
    pub(crate) timestamp_unix_ms: u128,
    pub(crate) mean: Option<f64>,
    pub(crate) sigma: Option<f64>,
    pub(crate) variance: Option<f64>,
    pub(crate) lp_total_backing: Option<f64>,
    pub(crate) lp_total_shares: Option<f64>,
    pub(crate) quote_accepts: Option<bool>,
    pub(crate) quote_required_collateral: Option<f64>,
    pub(crate) quote_x_star: Option<f64>,
}

impl Render for WatchUpdate {
    fn render_pretty(&self, r: &Renderer) {
        println!(
            "{} {} block {}  μ={:.6}  σ={:.6}",
            r.highlight(self.family),
            r.dim(&self.market),
            self.block_number,
            self.mean.unwrap_or(f64::NAN),
            self.sigma.unwrap_or(f64::NAN),
        );
        if let Some(b) = self.lp_total_backing {
            r.kv("LP backing", &format!("{b:.6}"));
        }
        if let Some(s) = self.lp_total_shares {
            r.kv("LP shares", &format!("{s:.6}"));
        }
        if let Some(acc) = self.quote_accepts {
            r.kv("quote_accepts", &format!("{acc}"));
            if let Some(rc) = self.quote_required_collateral {
                r.kv("quote_required_collateral", &format!("{rc:.6} STRK"));
            }
            if let Some(xs) = self.quote_x_star {
                r.kv("quote_x_star", &format!("{xs:.6}"));
            }
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "family: {}", self.family)?;
        writeln!(w, "market: {}", self.market)?;
        writeln!(w, "block_number: {}", self.block_number)?;
        writeln!(w, "timestamp_unix_ms: {}", self.timestamp_unix_ms)?;
        if let Some(m) = self.mean {
            writeln!(w, "mean: {m}")?;
        }
        if let Some(s) = self.sigma {
            writeln!(w, "sigma: {s}")?;
        }
        if let Some(v) = self.variance {
            writeln!(w, "variance: {v}")?;
        }
        if let Some(b) = self.lp_total_backing {
            writeln!(w, "lp_total_backing: {b}")?;
        }
        if let Some(s) = self.lp_total_shares {
            writeln!(w, "lp_total_shares: {s}")?;
        }
        if let Some(acc) = self.quote_accepts {
            writeln!(w, "quote_accepts: {acc}")?;
        }
        if let Some(rc) = self.quote_required_collateral {
            writeln!(w, "quote_required_collateral: {rc}")?;
        }
        if let Some(xs) = self.quote_x_star {
            writeln!(w, "quote_x_star: {xs}")?;
        }
        Ok(())
    }
}

/// Render a `TradeError` → human-readable explanation, returning a
/// fully-populated [`SubmissionResult`] caller can pass to `Renderer::print`.
pub(crate) fn submission_from_trade_error(
    action: &'static str,
    market: String,
    err: &deadeye_starknet::TradeError,
) -> SubmissionResult {
    let rejection = err.rejection().map(|r| pretty_rejection(&r));
    SubmissionResult {
        action,
        market,
        tx_hash: None,
        call_count: None,
        accepted: false,
        rejection,
        note: Some(format!("{err}")),
    }
}

/// Render the `SubmissionResult` from a successful receipt.
pub(crate) fn submission_from_receipt(
    action: &'static str,
    market: String,
    receipt: deadeye_starknet::ExecutionReceipt,
) -> SubmissionResult {
    SubmissionResult {
        action,
        market,
        tx_hash: Some(format!("{:#x}", receipt.transaction_hash)),
        call_count: Some(receipt.call_count),
        accepted: true,
        rejection: None,
        note: None,
    }
}

// ─── Multi-leg position views ────────────────────────────────────────────

/// One leg row for `position show`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct LegRow {
    pub(crate) lot_id: u64,
    pub(crate) settled: bool,
    pub(crate) cancelled: bool,
    /// Entry move "μ_from±σ_from → μ_to±σ_to" from indexed trade events
    /// (best-effort; None when the indexer is unavailable).
    pub(crate) entry: Option<String>,
}

/// `position show` for the trade-lot model: the trader's legs + summary.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PositionLegsView {
    pub(crate) market: String,
    pub(crate) trader: String,
    pub(crate) family: &'static str,
    pub(crate) exists: bool,
    pub(crate) claimed: bool,
    pub(crate) tracks_settlement_claim: bool,
    pub(crate) total_collateral: f64,
    pub(crate) leg_count: usize,
    pub(crate) active_legs: usize,
    pub(crate) legs: Vec<LegRow>,
    /// Market lifecycle (issue #21): settled + value when known.
    pub(crate) market_settled: Option<bool>,
    pub(crate) settlement_value: Option<f64>,
    /// Mark-to-market (issue #20): position valued at `evaluated_at`
    /// (settlement value when settled, else the live market mean).
    pub(crate) evaluated_at: Option<f64>,
    pub(crate) unrealized_pnl: Option<f64>,
    pub(crate) gross_return: Option<f64>,
    /// Outcome interval(s) where the position nets positive (issue #20),
    /// linear-interpolated from an on-chain valuation grid.
    pub(crate) profit_intervals: Vec<(f64, f64)>,
}

impl Render for PositionLegsView {
    fn render_pretty(&self, r: &Renderer) {
        r.header(&format!("Position — {} market", self.family));
        r.kv("market", &self.market);
        r.kv("trader", &self.trader);
        if !self.exists {
            r.kv("position", "none (no legs)");
            return;
        }
        r.kv(
            "total_collateral",
            &format!("{:.6} XP", self.total_collateral),
        );
        r.kv(
            "legs",
            &format!(
                "{} ({} active / claimable)",
                self.leg_count, self.active_legs
            ),
        );
        r.kv("claimed", &self.claimed.to_string());
        r.kv(
            "pending_settlement_claim",
            &self.tracks_settlement_claim.to_string(),
        );
        if let Some(settled) = self.market_settled {
            let label = match (settled, self.settlement_value) {
                (true, Some(v)) => format!("settled at {v:.4}"),
                (true, None) => "settled".to_owned(),
                (false, _) => "open".to_owned(),
            };
            r.kv("market_status", &label);
        }
        if let (Some(at), Some(pnl), Some(gross)) =
            (self.evaluated_at, self.unrealized_pnl, self.gross_return)
        {
            r.kv(
                "mark_to_market",
                &format!(
                    "{}{pnl:.4} XP P&L at x={at:.4}  (gross return {gross:.4} XP)",
                    if pnl >= 0.0 { "+" } else { "" },
                ),
            );
        }
        if !self.profit_intervals.is_empty() {
            let ranges = self
                .profit_intervals
                .iter()
                .map(|(a, b)| format!("[{a:.4}, {b:.4}]"))
                .collect::<Vec<_>>()
                .join(" ∪ ");
            r.kv("profitable_if_settles_in", &ranges);
        }
        for leg in &self.legs {
            let state = if leg.settled {
                "settled"
            } else if leg.cancelled {
                "cancelled"
            } else {
                "active"
            };
            let entry = leg.entry.as_deref().unwrap_or("");
            println!(
                "  {} lot #{:<4} {} {}",
                r.dim("·"),
                leg.lot_id,
                r.dim(state),
                r.dim(entry)
            );
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "market: {}", self.market)?;
        writeln!(w, "trader: {}", self.trader)?;
        writeln!(w, "family: {}", self.family)?;
        writeln!(w, "exists: {}", self.exists)?;
        writeln!(w, "total_collateral: {}", self.total_collateral)?;
        writeln!(w, "leg_count: {}", self.leg_count)?;
        writeln!(w, "active_legs: {}", self.active_legs)?;
        writeln!(w, "claimed: {}", self.claimed)?;
        if let Some(settled) = self.market_settled {
            writeln!(w, "market_settled: {settled}")?;
        }
        if let Some(v) = self.settlement_value {
            writeln!(w, "settlement_value: {v}")?;
        }
        if let Some(at) = self.evaluated_at {
            writeln!(w, "evaluated_at: {at}")?;
        }
        if let Some(pnl) = self.unrealized_pnl {
            writeln!(w, "unrealized_pnl: {pnl}")?;
        }
        if let Some(gross) = self.gross_return {
            writeln!(w, "gross_return: {gross}")?;
        }
        for (a, b) in &self.profit_intervals {
            writeln!(w, "profit_interval: {a}..{b}")?;
        }
        for leg in &self.legs {
            writeln!(
                w,
                "leg: lot_id={} settled={} cancelled={} entry={}",
                leg.lot_id,
                leg.settled,
                leg.cancelled,
                leg.entry.as_deref().unwrap_or("-")
            )?;
        }
        Ok(())
    }
}

/// One valued leg row for `position value`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct LegValueRow {
    pub(crate) lot_id: u64,
    pub(crate) settled: bool,
    pub(crate) cancelled: bool,
    pub(crate) value_at: f64,
}

/// `position value`: settlement valuation and/or expected P&L under a belief.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PositionValueView {
    pub(crate) market: String,
    pub(crate) trader: String,
    pub(crate) family: &'static str,
    pub(crate) exists: bool,
    pub(crate) total_collateral: f64,
    /// Settlement outcome valued at (when a settlement is given).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) settlement: Option<SettlementPoint>,
    /// Σ leg value at `settlement` — the P&L if it settles there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) total_position_value: Option<f64>,
    /// total_collateral + total_position_value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gross_return: Option<f64>,
    /// Per-leg valuations (when a settlement is given).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) legs: Vec<LegValueRow>,
    /// Human-readable forecast/belief (family-specific), when `--belief…`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) belief: Option<String>,
    /// Expected P&L under the belief (when a belief is given).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) expected_pnl: Option<f64>,
}

impl Render for PositionValueView {
    fn render_pretty(&self, r: &Renderer) {
        r.header(&format!("Position value — {} market", self.family));
        r.kv("market", &self.market);
        r.kv("trader", &self.trader);
        if !self.exists {
            r.kv("position", "none (no legs to value)");
            return;
        }
        r.kv(
            "total_collateral",
            &format!("{:.6} XP", self.total_collateral),
        );
        if let (Some(x), Some(pv), Some(gr)) = (
            self.settlement,
            self.total_position_value,
            self.gross_return,
        ) {
            r.kv("if_settles_at", &x.label());
            r.kv("position_value (P&L)", &format!("{pv:+.6} XP"));
            r.kv("gross_return", &format!("{gr:.6} XP (collateral + P&L)"));
            for leg in &self.legs {
                let tag = if leg.settled {
                    " (settled)"
                } else if leg.cancelled {
                    " (cancelled)"
                } else {
                    ""
                };
                println!(
                    "  {} lot #{:<4} value {:+.6} XP{}",
                    r.dim("·"),
                    leg.lot_id,
                    leg.value_at,
                    r.dim(tag)
                );
            }
        }
        if let (Some(b), Some(ev)) = (&self.belief, self.expected_pnl) {
            println!();
            r.kv("belief", b);
            r.kv(
                "expected_pnl",
                &format!("{ev:+.6} XP (E[P&L] under belief)"),
            );
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> io::Result<()> {
        writeln!(w, "market: {}", self.market)?;
        writeln!(w, "trader: {}", self.trader)?;
        writeln!(w, "family: {}", self.family)?;
        writeln!(w, "exists: {}", self.exists)?;
        writeln!(w, "total_collateral: {}", self.total_collateral)?;
        if let (Some(x), Some(pv), Some(gr)) = (
            self.settlement,
            self.total_position_value,
            self.gross_return,
        ) {
            writeln!(w, "settlement: {}", x.label())?;
            writeln!(w, "total_position_value: {pv}")?;
            writeln!(w, "gross_return: {gr}")?;
            for leg in &self.legs {
                writeln!(
                    w,
                    "leg: lot_id={} value_at={} settled={} cancelled={}",
                    leg.lot_id, leg.value_at, leg.settled, leg.cancelled
                )?;
            }
        }
        if let (Some(b), Some(ev)) = (&self.belief, self.expected_pnl) {
            writeln!(w, "belief: {b}")?;
            writeln!(w, "expected_pnl: {ev}")?;
        }
        Ok(())
    }
}
