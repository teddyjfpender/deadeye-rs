//! Normal-distribution AMM market handle.
//!
//! Provides the "read-then-quote-then-execute" loop a market-maker drives
//! against a Gaussian AMM. The handle wraps [`deadeye_starknet`]'s
//! lower-level reader / writer pair and adds the off-chain collateral
//! solver — most callers want exactly this.
//!
//! ## Worked example — quote, execute, sell
//!
//! ```no_run
//! use deadeye_sdk::{
//!     core::{distribution::NormalDistributionRaw, sq128::Sq128Raw},
//!     normal::NormalMarket,
//!     starknet::{Felt, JsonRpcProvider, NormalMarketReader, NormalMarketWriter, OwnedAccount},
//! };
//! use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let rpc = JsonRpcClient::new(HttpTransport::new(
//!     "http://localhost:5050".parse::<url::Url>()?,
//! ));
//! let provider = JsonRpcProvider::new(rpc);
//! let (market, runtime): (Felt, Felt) = (Felt::ZERO, Felt::ZERO);
//!
//! let reader = NormalMarketReader::new(&provider, market);
//! let signer = OwnedAccount::from_signing_key(
//!     JsonRpcClient::new(HttpTransport::new(
//!         "http://localhost:5050".parse::<url::Url>()?,
//!     )),
//!     Felt::ZERO,
//!     Felt::ZERO,
//!     Felt::ZERO,
//! );
//! let writer = NormalMarketWriter::new(reader, signer);
//!
//! let candidate = NormalDistributionRaw {
//!     mean: Sq128Raw::ZERO,
//!     variance: Sq128Raw::ZERO,
//!     sigma: Sq128Raw::ZERO,
//! };
//! let quote = writer
//!     .reader()
//!     .quote_trade(
//!         runtime,
//!         candidate,
//!         Sq128Raw::ZERO,
//!         Sq128Raw::ZERO,
//!         Sq128Raw::ZERO,
//!     )
//!     .await?;
//! if quote.on_chain_will_accept {
//!     writer.execute_quote(quote).await?;
//! }
//! writer.sell_position(runtime, 0).await?;
//! # Ok(()) }
//! ```

use deadeye_collateral::{MinimizationPolicy, lambda as collateral_lambda, normal_collateral};
// Bring `Distribution::pdf` into scope for `λ_f f(x*) − λ_g g(x*)`
// computation below. Aliased to `_` to avoid name conflicts with the
// pub-use at the bottom of this module.
use deadeye_core::Distribution as _CollateralPdf;
use deadeye_core::{NormalDistribution, Sq128, distribution::NormalSqrtHintsRaw, sq128::Sq128Raw};
pub use deadeye_optimizer::normal_sigma_floor;
use deadeye_optimizer::{NormalOptimizationInput, optimize_normal_trade};
use deadeye_starknet::{
    Account, ExecutionReceipt, Felt, NormalMarketReader, NormalMarketWriter, NormalTradeQuote,
    Provider, TradeRejectionReason, types::normal::TradeInput,
};
use futures::future::join_all;
use tracing::instrument;

use crate::{
    error::{SdkError, SdkResult},
    legs::{LegInfo, LegValuation, PositionLegs, PositionValuation, SettlementPoint, belief_grid},
    quote::PreparedQuote,
};

/// Cairo `SQRT_PI_RAW` from
/// `the-situation/contracts/src/market/normal/constants.cairo`.
///
/// `sqrt(π) ≈ 1.7724538509055160272981674833...` floor-encoded into Q128.128.
/// This is the **exact same limb representation** the on-chain math runtime
/// uses when computing `compute_hints_view`, so any hints we derive from
/// it via `Sq128::checked_mul` + `Sq128::sqrt` are bit-identical to the
/// chain's output.
const SQRT_PI_RAW: Sq128Raw = Sq128Raw {
    limb0: 0xC3B0_520D_5DB9_383F,
    limb1: 0xC5BF_891B_4EF6_AA79,
    limb2: 0x1,
    limb3: 0x0,
    neg: false,
};

/// Returns the chain-bit-exact `√π` constant used inside `compute_hints_view`.
#[inline]
fn sqrt_pi() -> Sq128 {
    Sq128::from_raw(SQRT_PI_RAW)
}

/// Off-chain mirror of the Cairo `compute_hints_view(dist)` on a normal
/// math runtime.
///
/// Formulas (from `the-situation/contracts/src/market/normal/`):
/// * `l2_norm_denom = sqrt(mul_down(mul_down(2, σ), √π))`
/// * `backing_denom = sqrt(mul_down(σ, √π))`
///
/// `Sq128::checked_mul` matches `mul_down` (Q128.128 floor-truncating
/// product) and `Sq128::sqrt` matches `sqrt_verified`/`sqrt_unchecked`
/// for non-negative inputs (proven bit-exact in `docs/SQ128_SQRT.md`).
///
/// Returns `Err` only on `σ == 0` (degenerate distribution) or arithmetic
/// overflow — both of which the chain runtime also rejects with `Option::None`.
fn compute_normal_hints_offline(sigma: Sq128) -> SdkResult<NormalSqrtHintsRaw> {
    if sigma.is_zero() {
        return Err(deadeye_core::CoreError::invalid_input(
            "sigma",
            "must be > 0 for hint derivation",
        )
        .into());
    }
    let two = Sq128::from_i128(2);
    let two_sigma = two.checked_mul(sigma)?;
    let denom_sq_l2 = two_sigma.checked_mul(sqrt_pi())?;
    let l2_norm_denom = denom_sq_l2.sqrt()?;
    let sigma_sqrt_pi = sigma.checked_mul(sqrt_pi())?;
    let backing_denom = sigma_sqrt_pi.sqrt()?;
    Ok(NormalSqrtHintsRaw {
        l2_norm_denom: l2_norm_denom.to_raw(),
        backing_denom: backing_denom.to_raw(),
    })
}

/// Reject non-positive or non-finite `effective_k` overrides at the SDK
/// boundary.
///
/// The optimizer treats `effective_k <= 0` as a no-trade sentinel and
/// `NaN` / `±∞` would poison downstream Sq128 conversions with
/// `from_f64` errors. We'd rather surface a precise
/// `CoreError::InvalidInput` at the caller site than a less-targeted
/// numeric error several frames deep.
fn validate_effective_k_override(value: f64) -> SdkResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(deadeye_core::CoreError::invalid_input(
            "effective_k_override",
            "must be a finite, strictly positive value",
        )
        .into());
    }
    Ok(())
}

/// Live-state-derived `effective_k` mirroring the on-chain
/// `compute_effective_trade_k_view`:
/// `effective_k = max(base_k, mul_down(base_k, pool_backing) /
/// initial_backing)`.
///
/// **Convention (canonicalised by `REVIEW_ITEM5`; Cairo storage + on-chain
/// math runtime + TS indexer all agree). An earlier version of this
/// doc-comment had the two backings swapped — the function body has always
/// been correct, only the comment lied:**
///
/// * `base_k` := `params.k` (the AMM-stored invariant constant).
/// * `pool_backing` := `lp_info.total_backing_deposited` — the **live**
///   backing, growing as LPs deposit; this is the formula's numerator and the
///   only thing that can push `effective_k` above `base_k`.
/// * `initial_backing` := `params.backing` — the **immutable** backing the AMM
///   was deployed with. The on-chain `params` ABI field name is a leftover
///   misnomer: `params.backing` is *not* live pool backing.
///
/// In other words: `params.backing` is the historical reference (denominator),
/// `lp_info.total_backing_deposited` is the current pool size (numerator).
/// A swapped mapping silently floors `effective_k` to `base_k` whenever LP
/// backing has grown — see the test
/// `live_effective_k_convention_pool_is_current_initial_is_historical`.
///
/// Returns `base_k` as a fallback when `initial_backing == 0` (matches the
/// chain's `Option::None` path: no scaling possible, so no upgrade applied).
pub(crate) fn live_effective_k(
    params_k: Sq128,
    pool_backing: Sq128,
    initial_backing: Sq128,
) -> Sq128 {
    if initial_backing.is_zero() || initial_backing.is_negative() {
        return params_k;
    }
    let scaled = params_k
        .checked_mul(pool_backing)
        .and_then(|s| s.checked_div(initial_backing));
    match scaled {
        Ok(s) if s > params_k => s,
        _ => params_k,
    }
}

// ─── Fetch-once state snapshot (issue #14) ──────────────────────────────────

/// Serde mirror of [`Sq128Raw`] (limb decomposition + sign) so snapshots
/// round-trip the chain's fixed-point values bit-exactly through JSON.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Sq128RawWire {
    /// Least-significant 64-bit limb.
    pub limb0: u64,
    /// Second 64-bit limb.
    pub limb1: u64,
    /// Third 64-bit limb.
    pub limb2: u64,
    /// Most-significant 64-bit limb.
    pub limb3: u64,
    /// Sign flag (`true` = negative).
    pub neg: bool,
}

impl From<Sq128Raw> for Sq128RawWire {
    fn from(raw: Sq128Raw) -> Self {
        Self {
            limb0: raw.limb0,
            limb1: raw.limb1,
            limb2: raw.limb2,
            limb3: raw.limb3,
            neg: raw.neg,
        }
    }
}

impl From<Sq128RawWire> for Sq128Raw {
    fn from(wire: Sq128RawWire) -> Self {
        Self {
            limb0: wire.limb0,
            limb1: wire.limb1,
            limb2: wire.limb2,
            limb3: wire.limb3,
            neg: wire.neg,
        }
    }
}

/// One-fetch snapshot of everything a normal-market quote depends on.
///
/// Produced by [`NormalMarket::state_snapshot`] (3 view calls, once);
/// consumed by [`optimize_quote_from_state`] / [`quote_candidate_from_state`]
/// which are **pure** — exploring N candidates costs zero further RPC
/// (issue #14: fetch once, compute candidates offline).
///
/// Raw Sq128 limbs preserve chain-bit parity; the f64 mirrors are for humans
/// and are NOT used in the math.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalMarketStateSnapshot {
    /// Market family this snapshot was taken from (issue #38). Defaults to
    /// [`Family::Normal`] when absent so snapshot files that pre-date the
    /// field keep deserializing — only the normal path could legitimately
    /// produce them.
    #[serde(default = "family_default_normal")]
    pub family: crate::bulk::Family,
    /// Market contract address (hex).
    pub market: String,
    /// Distribution mean, chain-exact.
    pub mean_raw: Sq128RawWire,
    /// Distribution variance, chain-exact.
    pub variance_raw: Sq128RawWire,
    /// Distribution sigma, chain-exact.
    pub sigma_raw: Sq128RawWire,
    /// Immutable base liquidity constant `k`, chain-exact.
    pub base_k_raw: Sq128RawWire,
    /// Initial backing at deployment, chain-exact.
    pub initial_backing_raw: Sq128RawWire,
    /// Live pool backing (total deposited), chain-exact.
    pub pool_backing_raw: Sq128RawWire,
    /// Human mirror of the mean (display only).
    pub mean: f64,
    /// Human mirror of sigma (display only).
    pub sigma: f64,
    /// Human mirror of the live effective k (display only).
    pub effective_k: f64,
    /// Human mirror of the pool backing in XP (display only).
    pub pool_backing_xp: f64,
}

/// Serde default for [`NormalMarketStateSnapshot::family`].
fn family_default_normal() -> crate::bulk::Family {
    crate::bulk::Family::Normal
}

impl NormalMarketStateSnapshot {
    /// Rebuild the bit-exact current distribution.
    ///
    /// # Errors
    /// Returns [`SdkError::Core`] if the stored variance is negative.
    pub fn distribution(&self) -> SdkResult<NormalDistribution> {
        Ok(NormalDistribution::from_variance(
            Sq128::from_raw(self.mean_raw.into()),
            Sq128::from_raw(self.variance_raw.into()),
        )?)
    }

    /// Effective k derived from the stored raw pool figures (bit-exact).
    #[must_use]
    pub fn effective_k_exact(&self) -> f64 {
        let base_k = Sq128::from_raw(self.base_k_raw.into());
        let pool = Sq128::from_raw(self.pool_backing_raw.into());
        let initial = Sq128::from_raw(self.initial_backing_raw.into());
        live_effective_k(base_k, pool, initial).to_f64()
    }
}

/// Pure optimizer quote from a snapshot — zero RPC. See
/// [`NormalMarketStateSnapshot`].
///
/// # Errors
/// Same failure modes as `optimize_quote_offline_ev` minus the provider:
/// invalid snapshot values or an optimizer that finds no admissible candidate.
pub fn optimize_quote_from_state(
    snapshot: &NormalMarketStateSnapshot,
    belief_mean: f64,
    belief_sigma: f64,
    budget_xp: f64,
) -> SdkResult<(NormalTradeQuote, f64)> {
    let current = snapshot.distribution()?;
    optimize_quote_offline_inner(
        &current,
        belief_mean,
        belief_sigma,
        budget_xp,
        snapshot.effective_k_exact(),
    )
}

/// Pure fixed-candidate quote from a snapshot — zero RPC.
///
/// # Errors
/// Same failure modes as `quote_candidate_offline` minus the provider.
pub fn quote_candidate_from_state(
    snapshot: &NormalMarketStateSnapshot,
    candidate_mean: f64,
    candidate_variance: f64,
) -> SdkResult<NormalTradeQuote> {
    let current = snapshot.distribution()?;
    let pool = Sq128::from_raw(snapshot.pool_backing_raw.into());
    quote_candidate_offline_inner(
        &current,
        candidate_mean,
        candidate_variance,
        snapshot.effective_k_exact(),
        pool.to_f64(),
    )
}

/// Pure off-chain implementation behind
/// [`NormalMarket::optimize_quote_offline`] and
/// [`NormalMarket::optimize_quote_offline_with_override`].
///
/// Given the live market distribution and a fixed `effective_k`, runs
/// the optimizer + λ-scaled collateral solver and produces a
/// chain-bit-exact `NormalTradeQuote`. Does **not** touch the chain —
/// the caller is responsible for sourcing `current` and `effective_k`.
///
/// Extracted as a free function (rather than an associated function on
/// `NormalMarket<P>`) so it isn't monomorphized per `Provider` impl —
/// the body is entirely f64 + Sq128 math, with no `P`-dependence.
fn optimize_quote_offline_inner(
    current: &NormalDistribution,
    belief_mean: f64,
    belief_sigma: f64,
    budget_xp: f64,
    effective_k: f64,
) -> SdkResult<(NormalTradeQuote, f64)> {
    let market_mean = current.mean().to_f64();
    let market_sigma = current.sigma().to_f64();

    let opt = optimize_normal_trade(NormalOptimizationInput::new(
        budget_xp,
        belief_mean,
        belief_sigma,
        market_mean,
        market_sigma,
        effective_k,
    ));

    // Build the candidate distribution with the chain-bit-exact σ.
    // The optimizer reports `optimized_variance = optimized_sigma²`
    // in f64; we promote `variance` to Sq128 (round-trip via
    // `from_f64`) and then *re-derive* σ via `Sq128::sqrt`.
    // The optimizer's `optimized_sigma` field is discarded — it
    // was an f64 sqrt of the variance and would mis-satisfy
    // `sqrt_verified` for most variances.
    let cand_mean = Sq128::from_f64(opt.optimized_mean)?;
    let cand_variance = Sq128::from_f64(opt.optimized_variance)?;
    let candidate = NormalDistribution::from_variance(cand_mean, cand_variance)?;
    let candidate_sigma = candidate.sigma();
    let candidate_hints = compute_normal_hints_offline(candidate_sigma)?;

    // Derive the *real* stationary point `x*` and the chain-aligned
    // collateral via the audited `normal_collateral` λ-scaled
    // Newton solver. Two fixes vs. the previous heuristics:
    //
    // 1. **x_star is the actual stationary point**, not `cand_mean`. The chain's
    //    `check_trade_view` does **not** re-derive `x*`, it *verifies* that the
    //    supplied `x*` is at the stationary point of `d(x) = λ_g g(x) − λ_f f(x)`
    //    (`d'(x*) ≈ 0`, `d''(x*) > 0`). Using `cand_mean` blew the
    //    curvature/stationarity gates and surfaced as silent `is_valid=false` with
    //    `rejection=None` (see `docs/CHAIN_ACCEPTANCE_PARITY.md`).
    // 2. **Collateral is λ-scaled to match the chain.** `normal_collateral` returns
    //    the *unscaled* `−d_min` (max(0, f(x*) − g(x*))), but the chain's
    //    `computed_collateral` is the *λ-scaled* difference `max(0, λ_f f(x*) − λ_g
    //    g(x*))` with `λ = k / ‖p‖₂`. The unscaled value is ~200× too small at
    //    `k=50, σ≈10`, so the chain rejected with `coll_ok=false above_min=false`
    //    (the supplied was below `min_trade_collateral=1.0`). Cairo source:
    //    `helpers.cairo:190-230` / `scaled_verify_minimum_with_lambda`.
    //
    // When the audited solver fails (`Err`, non-finite, or zero
    // collateral), we *cannot* honestly report a chain-valid quote
    // — the previous fallback to `(cand_mean, opt.collateral_required)`
    // re-introduced the exact two bugs the fix above closes
    // (`x_star ≠ stationary point`, unscaled collateral). Instead,
    // surface "no trade" so the bot never submits a quote the chain
    // would reject. The identical-distribution short-circuit inside
    // `normal_collateral` already returns `(x_min = μ, collateral =
    // 0)` with `Ok`, so this branch only fires on genuine solver
    // failures (Newton divergence, verification failed).
    let sigma_f_f64 = current.sigma().to_f64();
    let sigma_g_f64 = candidate.sigma().to_f64();
    let (x_star, collateral_f64) =
        match normal_collateral(current, &candidate, MinimizationPolicy::standard()) {
            Ok(v) if v.collateral.is_finite() && v.collateral > 0.0 => {
                // Re-evaluate `λ_f f(x*) − λ_g g(x*)` so the returned
                // collateral matches the chain's scaling exactly.
                let lam_f = collateral_lambda(sigma_f_f64, effective_k);
                let lam_g = collateral_lambda(sigma_g_f64, effective_k);
                let x_q = Sq128::from_f64(v.x_min)?;
                let f_at = current.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                let g_at = candidate.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                let scaled = lam_f.mul_add(f_at, -(lam_g * g_at)).max(0.0);
                (x_q, scaled)
            },
            // Two no-trade sentinel paths sharing the same response:
            //   * `Ok(_)` — solver succeeded but produced zero / non- finite collateral (identical
            //     distributions).
            //   * `Err(_)` — solver failed outright (Newton non-convergence, verification failed).
            //     We refuse to claim; returning the unscaled optimizer value here would silently
            //     re-create the bug the λ-scaling fix closes.
            Ok(_) | Err(_) => (cand_mean, 0.0_f64),
        };

    let collateral_required = Sq128::from_f64(collateral_f64)?;
    let padded_collateral = collateral_required;

    // Decide acceptance: positive cost AND positive net EV. The
    // optimizer's "no-trade" sentinel (`collateral_required == 0`)
    // surfaces as `on_chain_will_accept = false` with an
    // informative rejection message.
    // `rejection` is the typed on-chain rejection enum; off-chain
    // we have no chain verdict to surface. Callers that want a
    // human-readable "no positive-EV trade" message read off the
    // pair `(on_chain_will_accept == false, required_collateral == 0)`.
    //
    // Accept whenever the move locks real collateral AND has positive expected
    // P&L under the belief. Collateral is a returned margin lock, not a cost,
    // so the bar is `EV > 0` — NOT `EV > collateral` (that wrongly treated the
    // lock as sunk and rejected every feasible move — issue #12).
    let has_positive_trade = collateral_f64 > 0.0 && opt.expected_value > 0.0;
    let rejection = None;

    Ok((
        NormalTradeQuote {
            candidate: candidate.to_raw(),
            candidate_hints,
            x_star: x_star.to_raw(),
            required_collateral: collateral_required.to_raw(),
            padded_collateral: padded_collateral.to_raw(),
            on_chain_will_accept: has_positive_trade,
            rejection,
        },
        opt.expected_value,
    ))
}

/// Pure offline quote for a **fixed** candidate `(μ, variance)` — no optimizer
/// and no math-runtime contract. Mirrors the chain's verify path:
/// chain-bit-exact σ (via [`NormalDistribution::from_variance`]), offline sqrt
/// hints, and the λ-scaled collateral at the true stationary point, plus the
/// backing-derived **σ-floor** gate: a candidate σ below
/// `normal_sigma_floor(effective_k, backing)` is rejected up front with
/// [`TradeRejectionReason::SigmaTooLow`] (exactly what the AMM would do), so we
/// never hand back a quote the chain would reject for `SIGMA_TOO_LOW`.
fn quote_candidate_offline_inner(
    current: &NormalDistribution,
    candidate_mean: f64,
    candidate_variance: f64,
    effective_k: f64,
    backing: f64,
) -> SdkResult<NormalTradeQuote> {
    let cand_mean = Sq128::from_f64(candidate_mean)?;
    let cand_variance = Sq128::from_f64(candidate_variance)?;
    let candidate = NormalDistribution::from_variance(cand_mean, cand_variance)?;
    let candidate_sigma = candidate.sigma();
    let candidate_hints = compute_normal_hints_offline(candidate_sigma)?;
    let cand_sigma_f64 = candidate_sigma.to_f64();

    // Backing-derived σ-floor: a tighter candidate pushes the scaled-PDF peak
    // above the pool backing → the AMM reverts with SIGMA_TOO_LOW.
    let sigma_floor = normal_sigma_floor(effective_k, backing);
    if sigma_floor > 0.0 && cand_sigma_f64 < sigma_floor {
        return Ok(NormalTradeQuote {
            candidate: candidate.to_raw(),
            candidate_hints,
            x_star: cand_mean.to_raw(),
            required_collateral: Sq128::ZERO.to_raw(),
            padded_collateral: Sq128::ZERO.to_raw(),
            on_chain_will_accept: false,
            rejection: Some(TradeRejectionReason::SigmaTooLow),
        });
    }

    let sigma_f_f64 = current.sigma().to_f64();
    let (x_star, collateral_f64) =
        match normal_collateral(current, &candidate, MinimizationPolicy::standard()) {
            Ok(v) if v.collateral.is_finite() && v.collateral > 0.0 => {
                let lam_f = collateral_lambda(sigma_f_f64, effective_k);
                let lam_g = collateral_lambda(cand_sigma_f64, effective_k);
                let x_q = Sq128::from_f64(v.x_min)?;
                let f_at = current.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                let g_at = candidate.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                let scaled = lam_f.mul_add(f_at, -(lam_g * g_at)).max(0.0);
                (x_q, scaled)
            },
            // Solver failure / zero collateral (identical distributions): no
            // trade rather than an unscaled (chain-rejected) claim.
            Ok(_) | Err(_) => (cand_mean, 0.0_f64),
        };

    let collateral_required = Sq128::from_f64(collateral_f64)?;
    Ok(NormalTradeQuote {
        candidate: candidate.to_raw(),
        candidate_hints,
        x_star: x_star.to_raw(),
        required_collateral: collateral_required.to_raw(),
        padded_collateral: collateral_required.to_raw(),
        on_chain_will_accept: collateral_f64 > 0.0,
        rejection: None,
    })
}

/// Handle to a deployed normal AMM market.
///
/// A handle bundles a reader (and optionally a writer) bound to a single
/// market address. It is the canonical entry point for off-chain quote
/// preparation + on-chain execution.
#[derive(Debug)]
pub struct NormalMarket<'p, P>
where
    P: Provider,
{
    reader: NormalMarketReader<&'p P>,
}

impl<'p, P> NormalMarket<'p, P>
where
    P: Provider,
{
    /// Construct a handle bound to `provider` and the on-chain `address`.
    pub fn new(provider: &'p P, address: Felt) -> Self {
        Self {
            reader: NormalMarketReader::new(provider, address),
        }
    }

    /// Underlying read-only reader.
    pub const fn reader(&self) -> &NormalMarketReader<&'p P> {
        &self.reader
    }

    /// Returns the contract address.
    pub const fn address(&self) -> Felt {
        self.reader.address()
    }

    /// Fetch the current market distribution.
    pub async fn distribution(&self) -> SdkResult<NormalDistribution> {
        Ok(self.reader.distribution().await?)
    }

    /// Prepares a quote that moves the market from its current distribution
    /// to `candidate`. Returns the [`PreparedQuote`] containing the off-chain
    /// `x*` and required collateral.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn prepare_quote(
        &self,
        candidate: NormalDistribution,
        policy: MinimizationPolicy,
    ) -> SdkResult<PreparedQuote> {
        let current = self.distribution().await?;
        let verified = normal_collateral(&current, &candidate, policy)?;
        Ok(PreparedQuote {
            x_star: Sq128::from_f64(verified.x_min)?,
            collateral: Sq128::from_f64(verified.collateral)?,
            iterations: verified.iterations,
        })
    }

    /// Pick the EV-maximizing candidate distribution given a belief +
    /// budget, then quote it against `runtime`.
    ///
    /// Use this when you have a directional view (the trader's belief
    /// about the true outcome) and a budget, but don't want to hand-pick
    /// the candidate `(μ_g, σ_g)`. The optimizer searches a fixed grid
    /// inside the policy region for the highest net-EV trade, then
    /// hands the winner to [`NormalMarketReader::quote_trade`] for a
    /// full chain preflight.
    ///
    /// # Inputs
    ///
    /// * `runtime` — math runtime contract address (used by `quote_trade`).
    /// * `belief_mean` / `belief_sigma` — the trader's belief about the true
    ///   outcome distribution.
    /// * `budget` — maximum collateral the trader is willing to risk.
    ///
    /// # Returns
    ///
    /// A [`NormalTradeQuote`]. Inspect `on_chain_will_accept`; if `false`
    /// the budget was too tight for any in-policy candidate (and the
    /// quote will carry the optimizer's "no trade" candidate equal to
    /// the current market distribution). If `true`, hand the quote to
    /// [`NormalMarketWriter::execute_quote`] from a signed handle.
    ///
    /// # Worked example
    ///
    /// ```no_run
    /// # use deadeye_sdk::normal::NormalMarket;
    /// # use deadeye_sdk::starknet::{Felt, JsonRpcProvider};
    /// # use deadeye_sdk::core::Sq128;
    /// # use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let rpc = JsonRpcClient::new(HttpTransport::new(
    ///     "http://localhost:5050".parse::<url::Url>()?,
    /// ));
    /// let provider = JsonRpcProvider::new(rpc);
    /// let market = NormalMarket::new(&provider, Felt::ZERO);
    /// // I believe outcome ≈ 50 (±2σ confidence), can spend up to 10 STRK.
    /// let quote = market.optimize_quote(Felt::ZERO, 50.0, 2.0, 10.0).await?;
    /// if quote.on_chain_will_accept {
    ///     // hand to a signed handle to execute
    /// }
    /// # Ok(()) }
    /// ```
    #[instrument(skip(self), fields(market = %self.reader.address(), runtime = %runtime))]
    pub async fn optimize_quote(
        &self,
        runtime: Felt,
        belief_mean: f64,
        belief_sigma: f64,
        budget: f64,
    ) -> SdkResult<NormalTradeQuote> {
        let params = self.reader.params().await?;
        // TODO(v1.1): scale `k` by `pool_backing / initial_backing` once
        // `lp_info` exposes the immutable initial backing distinctly from
        // `params.backing`. Today `params.k` reflects the AMM's stored k;
        // the on-chain verifier re-checks any candidate against the live
        // effective k, so a slightly-stale k here only widens the
        // optimizer's search region — never approves a bad trade.
        let effective_k = Sq128::from_raw(params.k).to_f64();
        self.optimize_quote_inner(runtime, belief_mean, belief_sigma, budget, effective_k)
            .await
    }

    /// `optimize_quote` variant that uses a **caller-supplied
    /// `effective_k`** instead of re-reading it from chain.
    ///
    /// Use this when the caller already knows the relevant `effective_k`
    /// and wants to avoid an extra view round-trip. Concrete use cases:
    ///
    /// * **Backtest** — replay an old position with the historical
    ///   `effective_k` captured at the time the trade was submitted (e.g. from
    ///   a journal snapshot or analytics warehouse).
    /// * **Simulation** — explore "what if `k` were `75`?" scenarios without
    ///   nudging LP balances on chain.
    /// * **Offline mode** — the cpi-bot reads `effective_k` once per quote
    ///   cycle for its observability banner; passing the same value through
    ///   here eliminates the SDK's redundant chain read.
    /// * **Testing** — unit tests fix `effective_k` to a known value without
    ///   standing up a chain reader.
    ///
    /// The on-chain verifier still re-validates the candidate against
    /// its own live `effective_k` at submit time, so a stale or
    /// hand-picked value here can only widen / narrow the optimizer's
    /// search region — it can never approve a bad trade.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Core`] with a `CoreError::InvalidInput` if
    /// `effective_k_override` is not strictly positive (NaN, ±∞, ≤ 0).
    /// The optimizer hard-rejects non-positive `k`, and panicking
    /// inside an `async fn` is poor SDK hygiene.
    #[instrument(skip(self), fields(market = %self.reader.address(), runtime = %runtime))]
    pub async fn optimize_quote_with_override(
        &self,
        runtime: Felt,
        belief_mean: f64,
        belief_sigma: f64,
        budget: f64,
        effective_k_override: f64,
    ) -> SdkResult<NormalTradeQuote> {
        validate_effective_k_override(effective_k_override)?;
        self.optimize_quote_inner(
            runtime,
            belief_mean,
            belief_sigma,
            budget,
            effective_k_override,
        )
        .await
    }

    /// Shared implementation behind [`optimize_quote`] and
    /// [`optimize_quote_with_override`]: runs the optimizer against a
    /// fixed `effective_k` and dispatches to `quote_trade` for the
    /// chain-faithful preflight.
    ///
    /// **P4 #67 fix.** Previously this passed `x_star = cand_mean` to
    /// the chain's `check_trade_view`. The deployed math runtime does
    /// **not** re-derive `x*`: it *verifies* that the supplied `x*`
    /// satisfies `d'(x*) ≈ 0` and `d''(x*) > 0` for
    /// `d(x) = λ_g · g(x) − λ_f · f(x)`. For σ-arb scenarios
    /// (`μ_b` ≈ `μ_m`, `σ_b` < `σ_m`), the stationary point is shifted
    /// away from `μ_g`, so `cand_mean` silently trips `stationary_valid`
    /// on devnet. FU2 fixed the same bug in
    /// `optimize_quote_offline_inner`; this brings the chain-runtime
    /// variant into parity. The two inner paths now produce
    /// byte-identical `(x_star, required_collateral)` — they only
    /// differ on the hints (chain bytes vs. `Sq128` mirror) and on
    /// whether the chain's verdict is surfaced via `quote_trade`.
    async fn optimize_quote_inner(
        &self,
        runtime: Felt,
        belief_mean: f64,
        belief_sigma: f64,
        budget: f64,
        effective_k: f64,
    ) -> SdkResult<NormalTradeQuote> {
        let current = self.distribution().await?;
        let market_mean = current.mean().to_f64();
        let market_sigma = current.sigma().to_f64();
        let opt = optimize_normal_trade(NormalOptimizationInput::new(
            budget,
            belief_mean,
            belief_sigma,
            market_mean,
            market_sigma,
            effective_k,
        ));
        // Construct the candidate via `from_variance` so σ is derived
        // via [`Sq128::sqrt`] (chain-bit-exact with `sqrt_verified`),
        // not via f64. This ensures the on-chain runtime's
        // `compute_hints_view` accepts the candidate by construction —
        // mirrors the path `optimize_quote_offline` uses below.
        let cand_mean = Sq128::from_f64(opt.optimized_mean)?;
        let cand_variance = Sq128::from_f64(opt.optimized_variance)?;
        let candidate = deadeye_core::NormalDistribution::from_variance(cand_mean, cand_variance)?;

        // Derive the real stationary point `x*` and the chain-aligned
        // λ-scaled collateral via the audited `normal_collateral`
        // solver — mirrors the FU2 fix in `optimize_quote_offline_inner`.
        // See that function's commentary for the full rationale; in
        // short, the chain *verifies* `x*` (it does not re-derive it),
        // and the chain's `computed_collateral` is the λ-scaled
        // difference `max(0, λ_f f(x*) − λ_g g(x*))`. Using
        // `cand_mean` + the optimizer's unscaled collateral leaks the
        // exact two bugs the offline path closed.
        //
        // No-trade sentinel: when the solver fails (`Err`, non-finite,
        // or zero collateral) we fall back to `(cand_mean, 0)`. The
        // chain will then reject (zero collateral < min_trade_collateral),
        // surfaced as `on_chain_will_accept=false` via `quote_trade`.
        let sigma_f_f64 = current.sigma().to_f64();
        let sigma_g_f64 = candidate.sigma().to_f64();
        let (x_star, collateral_f64) =
            match normal_collateral(&current, &candidate, MinimizationPolicy::standard()) {
                Ok(v) if v.collateral.is_finite() && v.collateral > 0.0 => {
                    let lam_f = collateral_lambda(sigma_f_f64, effective_k);
                    let lam_g = collateral_lambda(sigma_g_f64, effective_k);
                    let x_q = Sq128::from_f64(v.x_min)?;
                    let f_at = current.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                    let g_at = candidate.pdf(x_q).map(Sq128::to_f64).unwrap_or(0.0);
                    let scaled = lam_f.mul_add(f_at, -(lam_g * g_at)).max(0.0);
                    (x_q, scaled)
                },
                Ok(_) | Err(_) => (cand_mean, 0.0_f64),
            };

        let supplied = Sq128::from_f64(collateral_f64)?;
        let pad = supplied;
        Ok(self
            .reader
            .quote_trade(
                runtime,
                candidate.to_raw(),
                x_star.to_raw(),
                supplied.to_raw(),
                pad.to_raw(),
            )
            .await?)
    }

    /// Off-chain-only EV optimizer — chain-bit-exact σ + hints, no runtime
    /// round-trip.
    ///
    /// Use this when **no math-runtime instance is deployed** (mainnet
    /// today, where the normal AMM ships as a library-dispatch class
    /// hash with no separately deployed instance). The output is
    /// chain-bit-exact for σ (derived via [`Sq128::sqrt`]) and the
    /// hints (derived via the same Sq128 formulas the on-chain
    /// `compute_hints_view` runs), so the resulting candidate
    /// distribution and `candidate_hints` survive the on-chain
    /// `check_trade_view` `sqrt_verified` / `mul_down` invariants by
    /// construction.
    ///
    /// # Pipeline
    ///
    /// 1. Reads live market state from chain — `distribution`, `params`,
    ///    `lp_info`.
    /// 2. Derives `effective_k` from `(params.k, params.backing,
    ///    lp_info.total_backing_deposited)` using the exact chain formula
    ///    (`max(base_k, mul_down(base_k, pool_backing) / initial_backing)`).
    /// 3. Hands the live `(μ_market, σ_market, effective_k, budget)` to
    ///    [`deadeye_optimizer::optimize_normal_trade`] to pick the EV-optimal
    ///    `(μ_g, σ_g²)`.
    /// 4. **Replaces the optimizer's f64-derived σ** with the chain-bit-exact σ
    ///    obtained via [`NormalDistribution::from_variance`] (internally
    ///    `Sq128::sqrt`). This is the critical fix that prevents
    ///    `INVALID_DISTRIBUTION` rejections on submit — f64 sqrt rounds in the
    ///    last bit for any non-perfect Sq128 square, but `Sq128::sqrt` is
    ///    bit-exact with Cairo `u512_sqrt`.
    /// 5. Computes `(l2_norm_denom, backing_denom)` offline via the same
    ///    `sqrt(mul_down(...))` chain the on-chain runtime uses.
    ///
    /// # Returns
    ///
    /// A [`NormalTradeQuote`] with:
    /// * `candidate` — chain-bit-exact `(μ, variance, σ)` triple
    /// * `candidate_hints` — chain-bit-exact `(l2_norm_denom, backing_denom)`
    /// * `x_star` — the optimizer's `μ_g` (an in-support seed; the chain
    ///   re-derives the true stationary point at execute time)
    /// * `required_collateral` / `padded_collateral` — both set to the
    ///   off-chain collateral solver's output
    /// * `on_chain_will_accept` — `true` iff the optimizer found a positive-EV
    ///   trade (i.e. cost > 0 and EV > cost). Mind the caveat below.
    /// * `rejection` — `Some("off-chain optimizer found no positive-EV trade")`
    ///   when `!on_chain_will_accept`, else `None`.
    ///
    /// # Bit-exactness vs. chain acceptance
    ///
    /// **Off-chain accept ≠ chain guarantee.** The σ-derivation, hints
    /// and candidate distribution are bit-exact with what the chain
    /// would compute (proven against the deployed `normal_math_runtime`
    /// at 20/20 in [`docs/SQ128_SQRT.md`](../../../docs/SQ128_SQRT.md)),
    /// so `INVALID_DISTRIBUTION` and `INVALID_HINTS` are eliminated.
    /// What's still re-checked at execute time:
    /// * trader collateral balance + allowance
    /// * fresh-state nonce (no front-run between quote and execute)
    /// * the AMM's `check_trade_view` admissibility envelope (`min trade
    ///   collateral`, `tolerance`, policy region — these are conservative
    ///   side-constraints, not σ/hint correctness)
    ///
    /// In other words: this method removes the σ-precision footgun
    /// that the f64-fallback used to introduce. The chain still
    /// re-verifies the trade on submit, as it does for every path.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn optimize_quote_offline(
        &self,
        belief_mean: f64,
        belief_sigma: f64,
        budget_xp: f64,
    ) -> SdkResult<NormalTradeQuote> {
        self.optimize_quote_offline_ev(belief_mean, belief_sigma, budget_xp)
            .await
            .map(|(quote, _ev)| quote)
    }

    /// Like [`Self::optimize_quote_offline`], but also returns the optimizer's
    /// **expected value** (in XP) of the chosen candidate under the belief —
    /// the quantity the CLI surfaces as `expected_value` in `trade quote`.
    /// Same single optimizer pass; the EV is otherwise discarded.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn optimize_quote_offline_ev(
        &self,
        belief_mean: f64,
        belief_sigma: f64,
        budget_xp: f64,
    ) -> SdkResult<(NormalTradeQuote, f64)> {
        let current = self.distribution().await?;
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;

        let base_k = Sq128::from_raw(params.k);
        // Convention pin (matches the `live_effective_k` doc-comment and
        // `cpi-bot::effective_k::chain_read_effective_k`):
        //   * `pool_backing`   := `lp_info.total_backing_deposited` (the *live* pool —
        //     the formula's numerator).
        //   * `initial_backing` := `params.backing` (the *immutable* reference
        //     deposited at deploy — the formula's denominator; ABI-named `backing` but
        //     storage-named `self.initial_backing` per
        //     `onchain-normal-amm/contract.cairo:178`).
        // Swapping these silently floors `effective_k` to `base_k` whenever
        // LPs have grown the pool (see `REVIEW_ITEM5` §1 + the
        // `live_effective_k_convention_*` test).
        let pool_backing = Sq128::from_raw(lp_info.total_backing_deposited);
        let initial_backing = Sq128::from_raw(params.backing);
        let effective_k_sq = live_effective_k(base_k, pool_backing, initial_backing);
        let effective_k = effective_k_sq.to_f64();

        optimize_quote_offline_inner(&current, belief_mean, belief_sigma, budget_xp, effective_k)
    }

    /// `optimize_quote_offline` variant that uses a **caller-supplied
    /// `effective_k`** instead of deriving it from `(params.k,
    /// params.backing, lp_info.total_backing_deposited)`.
    ///
    /// Eliminates the `params` + `lp_info` chain reads — only the
    /// market distribution is fetched. Useful for:
    ///
    /// * **Backtest** — replay an old quote with the historical `effective_k`
    ///   from a journal snapshot.
    /// * **Simulation** — sweep `k` parameter space without LP movement.
    /// * **Offline mode** — cpi-bot already reads `effective_k` for its QUOTE
    ///   banner; passing it through avoids the redundant `params`+`lp_info`
    ///   read inside the SDK (~150ms saved per call when the indexer cache
    ///   misses).
    /// * **Testing** — fix `effective_k` without mocking the chain.
    ///
    /// All other chain-bit-exact behavior (Sq128-derived σ, hint
    /// derivation via the same `sqrt(mul_down(...))` chain) is
    /// preserved — the override only short-circuits the `k`-derivation
    /// step.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Core`] with a `CoreError::InvalidInput` if
    /// `effective_k_override` is not strictly positive.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn optimize_quote_offline_with_override(
        &self,
        belief_mean: f64,
        belief_sigma: f64,
        budget_xp: f64,
        effective_k_override: f64,
    ) -> SdkResult<NormalTradeQuote> {
        validate_effective_k_override(effective_k_override)?;
        let current = self.distribution().await?;
        optimize_quote_offline_inner(
            &current,
            belief_mean,
            belief_sigma,
            budget_xp,
            effective_k_override,
        )
        .map(|(quote, _ev)| quote)
    }

    /// Offline quote for a **fixed candidate** `(mean, variance)` — no
    /// optimizer, no math-runtime contract. Reads only the market
    /// distribution + params + LP info (all cheap views), derives the
    /// effective `k` and backing, then computes the chain-bit-exact candidate,
    /// hints, λ-scaled collateral, and the backing-derived σ-floor verdict
    /// locally. This is the zero-config path the CLI uses for
    /// `trade quote --mean --variance` so a read-only quote never needs a
    /// runtime address or an on-chain transaction.
    /// Fetch the one-shot state snapshot every quote depends on (3 view
    /// calls). Pair with [`optimize_quote_from_state`] /
    /// [`quote_candidate_from_state`] to explore candidates with ZERO further
    /// RPC (issue #14).
    ///
    /// # Errors
    /// Propagates provider/read failures from the three views.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn state_snapshot(&self) -> SdkResult<NormalMarketStateSnapshot> {
        let current = self.distribution().await?;
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let base_k = Sq128::from_raw(params.k);
        let pool_backing = Sq128::from_raw(lp_info.total_backing_deposited);
        let initial_backing = Sq128::from_raw(params.backing);
        let effective_k = live_effective_k(base_k, pool_backing, initial_backing).to_f64();
        Ok(NormalMarketStateSnapshot {
            family: crate::bulk::Family::Normal,
            market: format!("{:#x}", self.reader.address()),
            mean_raw: current.mean().to_raw().into(),
            variance_raw: current.variance().to_raw().into(),
            sigma_raw: current.sigma().to_raw().into(),
            base_k_raw: params.k.into(),
            initial_backing_raw: params.backing.into(),
            pool_backing_raw: lp_info.total_backing_deposited.into(),
            mean: current.mean().to_f64(),
            sigma: current.sigma().to_f64(),
            effective_k,
            pool_backing_xp: pool_backing.to_f64(),
        })
    }

    /// Fully client-side fixed-candidate quote — reads state once, then
    /// reproduces the AMM's accept/collateral math locally (no runtime, no
    /// gas). The default preflight behind `trade execute --mean/--variance`.
    ///
    /// # Errors
    /// Propagates provider/read failures and core math errors.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn quote_candidate_offline(
        &self,
        candidate_mean: f64,
        candidate_variance: f64,
    ) -> SdkResult<NormalTradeQuote> {
        let current = self.distribution().await?;
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let base_k = Sq128::from_raw(params.k);
        let pool_backing = Sq128::from_raw(lp_info.total_backing_deposited);
        let initial_backing = Sq128::from_raw(params.backing);
        let effective_k = live_effective_k(base_k, pool_backing, initial_backing).to_f64();
        quote_candidate_offline_inner(
            &current,
            candidate_mean,
            candidate_variance,
            effective_k,
            pool_backing.to_f64(),
        )
    }

    /// The backing-derived **σ-floor** for this market: the narrowest σ the
    /// pool backing can support. A trade whose candidate σ is below this is
    /// rejected on-chain with `SIGMA_TOO_LOW`. Surfaced in the quote so an
    /// agent can size variance above the floor before submitting.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn sigma_floor(&self) -> SdkResult<f64> {
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let base_k = Sq128::from_raw(params.k);
        let pool_backing = Sq128::from_raw(lp_info.total_backing_deposited);
        let initial_backing = Sq128::from_raw(params.backing);
        let effective_k = live_effective_k(base_k, pool_backing, initial_backing).to_f64();
        Ok(normal_sigma_floor(effective_k, pool_backing.to_f64()))
    }

    // ── Multi-leg (trade-lot) position tracking + valuation ─────────────

    /// Enumerate a trader's legs (lot ids + lifecycle flags) and read the
    /// position summary. Lifecycle flags are fetched concurrently.
    #[instrument(skip(self), fields(market = %self.reader.address(), %trader))]
    pub async fn legs(&self, trader: Felt) -> SdkResult<PositionLegs> {
        let summary = self.reader.position_summary(trader).await?;
        let ids = self.reader.trade_lot_ids(trader).await?;
        let legs = join_all(ids.iter().map(|&lot_id| async move {
            let settled = self.reader.trade_lot_settled(lot_id).await?;
            let cancelled = self.reader.trade_lot_cancelled(lot_id).await?;
            Ok::<LegInfo, SdkError>(LegInfo {
                lot_id,
                settled,
                cancelled,
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        Ok(PositionLegs {
            trader: format!("{trader:#x}"),
            legs,
            total_collateral: Sq128::from_raw(summary.total_collateral_locked).to_f64(),
            exists: summary.exists,
            claimed: summary.claimed,
            tracks_settlement_claim: summary.tracks_settlement_claim,
        })
    }

    /// Value a trader's whole position at a settlement outcome `x*`,
    /// authoritatively (each active leg via the on-chain `value_at` view).
    /// `total_position_value` is the position's P&L if the market settles at
    /// `x*`; `gross_return` adds back the locked collateral.
    #[instrument(skip(self), fields(market = %self.reader.address(), %trader, settlement))]
    pub async fn position_value_at(
        &self,
        trader: Felt,
        settlement: f64,
    ) -> SdkResult<PositionValuation> {
        let summary = self.reader.position_summary(trader).await?;
        let ids = self.reader.trade_lot_ids(trader).await?;
        let x_raw = Sq128::from_f64(settlement)?.to_raw();
        let legs = join_all(ids.iter().map(|&lot_id| async move {
            let settled = self.reader.trade_lot_settled(lot_id).await?;
            let cancelled = self.reader.trade_lot_cancelled(lot_id).await?;
            // Settled/cancelled legs have no future payout.
            let value_at = if settled || cancelled {
                0.0
            } else {
                Sq128::from_raw(self.reader.trade_lot_value_at(lot_id, x_raw).await?).to_f64()
            };
            Ok::<LegValuation, SdkError>(LegValuation {
                lot_id,
                settled,
                cancelled,
                value_at,
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let total_position_value: f64 = legs.iter().map(|l| l.value_at).sum();
        let total_collateral = Sq128::from_raw(summary.total_collateral_locked).to_f64();
        Ok(PositionValuation {
            trader: format!("{trader:#x}"),
            settlement: SettlementPoint::Scalar(settlement),
            legs,
            total_collateral,
            total_position_value,
            gross_return: total_collateral + total_position_value,
            exists: summary.exists,
            claimed: summary.claimed,
        })
    }

    /// Expected position value (P&L) under a Gaussian belief `N(μ, σ)`,
    /// integrating the on-chain leg value over the belief via a normal-pdf-
    /// weighted grid. Returns the expected P&L in XP. Costs `nodes × active
    /// legs` `value_at` reads (fanned out concurrently) — call when an agent
    /// wants its forward EV, not on every tick.
    #[instrument(skip(self), fields(market = %self.reader.address(), %trader, belief_mean, belief_sigma))]
    pub async fn expected_value_under_belief(
        &self,
        trader: Felt,
        belief_mean: f64,
        belief_sigma: f64,
    ) -> SdkResult<f64> {
        let ids = self.reader.trade_lot_ids(trader).await?;
        // Keep only claimable legs.
        let flags = join_all(ids.iter().map(|&lot_id| async move {
            let settled = self.reader.trade_lot_settled(lot_id).await?;
            let cancelled = self.reader.trade_lot_cancelled(lot_id).await?;
            Ok::<(u64, bool), SdkError>((lot_id, !settled && !cancelled))
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let active: Vec<u64> = flags
            .into_iter()
            .filter_map(|(id, ok)| ok.then_some(id))
            .collect();
        if active.is_empty() {
            return Ok(0.0);
        }
        // E[value] = Σ_i w_i · Σ_legs value_at(lot, x_i), weights summing to 1.
        let grid = belief_grid(belief_mean, belief_sigma, 4.0, 21);
        let mut futs = Vec::with_capacity(grid.len() * active.len());
        for (x, w) in &grid {
            let x_raw = Sq128::from_f64(*x)?.to_raw();
            for &lot_id in &active {
                let weight = *w;
                futs.push(async move {
                    let v = Sq128::from_raw(self.reader.trade_lot_value_at(lot_id, x_raw).await?)
                        .to_f64();
                    Ok::<f64, SdkError>(weight * v)
                });
            }
        }
        let parts = join_all(futs)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(parts.into_iter().sum())
    }

    /// Bind an [`Account`] to this market, returning a write-capable handle.
    ///
    /// The returned [`NormalMarketSigned`] holds the same reader plus the
    /// signer, so reads remain cheap and writes funnel through a single
    /// place.
    pub fn with_account<A>(self, account: A) -> NormalMarketSigned<'p, P, A>
    where
        A: Account,
    {
        NormalMarketSigned {
            writer: NormalMarketWriter::new(self.reader, account),
        }
    }
}

/// Account-bound companion to [`NormalMarket`].
#[derive(Debug)]
pub struct NormalMarketSigned<'p, P, A>
where
    P: Provider,
    A: Account,
{
    writer: NormalMarketWriter<&'p P, A>,
}

impl<P, A> NormalMarketSigned<'_, P, A>
where
    P: Provider,
    A: Account,
{
    /// Borrow the underlying writer.
    pub const fn writer(&self) -> &NormalMarketWriter<&P, A> {
        &self.writer
    }

    /// Returns the market address.
    pub const fn address(&self) -> Felt {
        self.writer.reader().address()
    }

    /// Fetch the current market distribution (read passthrough).
    pub async fn distribution(&self) -> SdkResult<NormalDistribution> {
        Ok(self.writer.reader().distribution().await?)
    }

    /// Execute a previously-prepared quote.
    #[instrument(skip(self, candidate), fields(market = %self.writer.reader().address()))]
    pub async fn execute(
        &self,
        candidate: NormalDistribution,
        quote: PreparedQuote,
    ) -> SdkResult<ExecutionReceipt> {
        // Build the on-chain trade input from the off-chain quote +
        // candidate. Hints default to zero — the on-chain math runtime
        // will fall back to its internal computation if the hints don't
        // validate. Callers that want deterministic gas should supply
        // their own hints via `writer().build_trade_call(...)`.
        let input = TradeInput {
            candidate: candidate.to_raw(),
            x_star: quote.x_star.to_raw(),
            supplied_collateral: quote.collateral.to_raw(),
            candidate_hints: deadeye_core::distribution::NormalSqrtHintsRaw {
                l2_norm_denom: Sq128::ZERO.to_raw(),
                backing_denom: Sq128::ZERO.to_raw(),
            },
        };
        Ok(self.writer.execute_trade(input).await?)
    }

    /// Pass-through to the underlying writer for advanced calldata building.
    pub const fn writer_ref(&self) -> &NormalMarketWriter<&P, A> {
        &self.writer
    }
}

// We re-export `Distribution` so callers that need to read `.mean()` /
// `.variance()` on the returned distribution can do so without an extra
// import.
pub use deadeye_core::Distribution;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {
        let _ = core::any::type_name::<T>();
    }

    #[test]
    fn normal_market_is_send_sync() {
        assert_send_sync::<NormalMarket<'_, MockProvider>>();
    }

    use async_trait::async_trait;
    use deadeye_starknet::ContractResult;
    use starknet_core::types::{BlockId, FunctionCall};

    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn call(&self, _call: FunctionCall, _block: BlockId) -> ContractResult<Vec<Felt>> {
            unreachable!("mock provider only used for trait wiring")
        }
    }

    // ─── Offline hint derivation parity ─────────────────────────────────
    //
    // These don't hit chain; they cross-check the offline formula against
    // its own mathematical specification — `l2_norm_denom == sqrt(2σ√π)`
    // and `backing_denom == sqrt(σ√π)` — at Sq128 precision. Devnet
    // bit-parity vs. `compute_hints_view` is covered by the integration
    // test `offline_optimize_quote_parity`.

    #[test]
    fn sqrt_pi_constant_round_trips() {
        // Reading the limbs into `Sq128` and back to raw must be the identity.
        let value = sqrt_pi();
        assert_eq!(value.to_raw(), SQRT_PI_RAW);
        // Value should be ≈ 1.7724538509...
        let approx = value.to_f64();
        let true_sqrt_pi = core::f64::consts::PI.sqrt();
        assert!(
            (approx - true_sqrt_pi).abs() < 1e-15,
            "sqrt_pi constant: got {approx}, expected ≈ {true_sqrt_pi}",
        );
    }

    #[test]
    fn offline_hints_satisfy_definition() {
        // For σ ∈ {0.1, 1.0, 2.5, 10.0}:
        //   l2_norm_denom² ≈ 2σ√π   (within Sq128 floor tolerance)
        //   backing_denom² ≈ σ√π    (within Sq128 floor tolerance)
        for &s in &[0.1_f64, 1.0_f64, 2.5_f64, 10.0_f64] {
            let sigma = Sq128::from_f64(s).expect("σ converts");
            let hints = compute_normal_hints_offline(sigma).expect("hints derive");

            let l2 = Sq128::from_raw(hints.l2_norm_denom);
            let backing = Sq128::from_raw(hints.backing_denom);

            // `l2² ≤ 2σ√π` (floor sqrt invariant)
            let two_sigma = Sq128::from_i128(2).checked_mul(sigma).unwrap();
            let two_sigma_sqrt_pi = two_sigma.checked_mul(sqrt_pi()).unwrap();
            let l2_sq = l2.checked_mul(l2).unwrap();
            assert!(
                l2_sq <= two_sigma_sqrt_pi,
                "σ={s}: l2² > 2σ√π violates sqrt floor invariant",
            );

            // `backing² ≤ σ√π`
            let sigma_sqrt_pi = sigma.checked_mul(sqrt_pi()).unwrap();
            let backing_sq = backing.checked_mul(backing).unwrap();
            assert!(
                backing_sq <= sigma_sqrt_pi,
                "σ={s}: backing² > σ√π violates sqrt floor invariant",
            );
        }
    }

    #[test]
    fn offline_hints_reject_zero_sigma() {
        let err = compute_normal_hints_offline(Sq128::ZERO).expect_err("must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("sigma"),
            "expected sigma-related error, got: {msg}",
        );
    }

    #[test]
    fn live_effective_k_falls_back_when_initial_backing_zero() {
        let base = Sq128::from_f64(75.07).unwrap();
        let pool = Sq128::from_f64(1000.0).unwrap();
        let initial = Sq128::ZERO;
        // initial_backing == 0 → fall back to base_k (no scaling).
        let k = live_effective_k(base, pool, initial);
        assert_eq!(k, base);
    }

    #[test]
    fn live_effective_k_uses_base_when_pool_below_initial() {
        // pool < initial → max(base, base*pool/initial) == base (smaller fraction).
        let base = Sq128::from_f64(75.07).unwrap();
        let pool = Sq128::from_f64(500.0).unwrap();
        let initial = Sq128::from_f64(1000.0).unwrap();
        let k = live_effective_k(base, pool, initial);
        assert_eq!(k, base);
    }

    #[test]
    fn live_effective_k_scales_up_when_pool_above_initial() {
        // pool > initial → max(base, base*pool/initial) > base.
        let base = Sq128::from_f64(75.07).unwrap();
        let pool = Sq128::from_f64(2000.0).unwrap();
        let initial = Sq128::from_f64(1000.0).unwrap();
        let k = live_effective_k(base, pool, initial);
        assert!(k > base, "expected scaled-up k, got {k} vs base {base}");
        // ≈ 2 × base, within Sq128 precision.
        let expected = Sq128::from_f64(150.14).unwrap();
        let diff = if k > expected {
            k.checked_sub(expected).unwrap()
        } else {
            expected.checked_sub(k).unwrap()
        };
        let one_ulp = Sq128::from_f64(1e-9).unwrap();
        assert!(
            diff < one_ulp,
            "expected k ≈ 150.14, got {} (diff {})",
            k.to_f64(),
            diff.to_f64()
        );
    }

    // ─── REVIEW_ITEM5 convention pin (Follow-up #3) ────────────────────
    //
    // The doc-comment on `live_effective_k` previously had the two
    // backings swapped (claimed `pool := params.backing`). The canonical
    // mapping per Cairo + indexer + on-chain math runtime is:
    //
    //   pool_backing    := lp_info.total_backing_deposited   (current)
    //   initial_backing := params.backing                    (historical)
    //
    // These tests pin the convention so future doc/code drift can't
    // re-introduce the inversion silently.

    /// Ground-truth round-trip from `REVIEW_ITEM5`: with `base_k=50`,
    /// `pool=20_000`, `initial=10_000`, `effective_k == 100` (= 2 × base).
    /// The inverse pairing (`pool=10_000, initial=20_000`) floors at 50
    /// — the asymmetry is what the convention pin guards.
    #[test]
    fn live_effective_k_convention_pool_is_current_initial_is_historical() {
        let base = Sq128::from_f64(50.0).unwrap();
        let pool_current = Sq128::from_f64(20_000.0).unwrap();
        let initial_historical = Sq128::from_f64(10_000.0).unwrap();

        let k = live_effective_k(base, pool_current, initial_historical);
        let one_ulp = Sq128::from_f64(1e-9).unwrap();
        let expected = Sq128::from_f64(100.0).unwrap();
        let diff = if k > expected {
            k.checked_sub(expected).unwrap()
        } else {
            expected.checked_sub(k).unwrap()
        };
        assert!(
            diff < one_ulp,
            "convention pin: pool=2×initial must give effective_k ≈ 100, got {}",
            k.to_f64(),
        );

        // Now swap the labels — pool below initial floors at base. This
        // is REVIEW_ITEM5 §1: a swapped mapping silently floors
        // `effective_k` to `base_k` whenever LP backing has grown.
        let k_swapped = live_effective_k(base, initial_historical, pool_current);
        assert_eq!(
            k_swapped,
            base,
            "swapped (pool, initial) must floor at base_k = 50, got {}",
            k_swapped.to_f64(),
        );
    }

    /// `REVIEW_ITEM5` §3 mainnet ratio: at CPI-YoY the live ratio was
    /// `pool ≈ 1.0009 × initial`, with `effective_k ≈ 75.068` for
    /// `base_k = 75`. A swapped mapping would floor at 75 — this test
    /// asserts the value rises above `base_k`.
    #[test]
    fn live_effective_k_mainnet_ratio_scales_above_base() {
        let base = Sq128::from_f64(75.0).unwrap();
        let pool = Sq128::from_f64(1_000.907_447_4).unwrap();
        let initial = Sq128::from_f64(1_000.0).unwrap();
        let k = live_effective_k(base, pool, initial);
        assert!(
            k > base,
            "mainnet ratio: effective_k must rise above base when LPs have deposited; got {} vs base {}",
            k.to_f64(),
            base.to_f64(),
        );
        let expected = Sq128::from_f64(75.068_058_55).unwrap();
        let diff = if k > expected {
            k.checked_sub(expected).unwrap()
        } else {
            expected.checked_sub(k).unwrap()
        };
        assert!(
            diff < Sq128::from_f64(1e-3).unwrap(),
            "mainnet ratio: expected ≈ 75.068, got {} (diff {})",
            k.to_f64(),
            diff.to_f64(),
        );
    }

    /// Call-site wiring pin: `optimize_quote_offline` reads
    /// `params.backing` and `lp_info.total_backing_deposited`, then feeds
    /// them as `(pool_backing, initial_backing)` into [`live_effective_k`].
    /// The previous wiring had these swapped — the test below recreates
    /// the call-site mapping with hand-pinned values and asserts the
    /// `effective_k` matches the chain's formula. If anyone re-swaps the
    /// assignments in `optimize_quote_offline`, this test fires.
    ///
    /// Scenario: post-LP-grow market.
    ///   * `params.backing`                  = 10 000 (immutable / denominator)
    ///   * `lp_info.total_backing_deposited` = 20 000 (live / numerator)
    ///   * `base_k`                          = 50
    ///   * Chain: `effective_k = max(50, 50 × 20000 / 10000) = 100`.
    ///
    /// A swapped mapping would compute `max(50, 50 × 10000 / 20000) = 50`.
    #[test]
    fn live_effective_k_call_site_wiring_matches_chain_after_lp_grow() {
        // Simulate the values the SDK reads from chain (with chain-side
        // naming, *not* the SDK's local-var naming, to mimic the call-site).
        let params_backing = Sq128::from_f64(10_000.0).unwrap();
        let lp_total_backing_deposited = Sq128::from_f64(20_000.0).unwrap();
        let base_k = Sq128::from_f64(50.0).unwrap();

        // Reproduce the call-site assignment direction:
        // `pool_backing := lp_info.total_backing_deposited`
        // `initial_backing := params.backing`
        let pool_backing = lp_total_backing_deposited;
        let initial_backing = params_backing;

        let k = live_effective_k(base_k, pool_backing, initial_backing);
        let expected = Sq128::from_f64(100.0).unwrap();
        let one_ulp = Sq128::from_f64(1e-9).unwrap();
        let diff = if k > expected {
            k.checked_sub(expected).unwrap()
        } else {
            expected.checked_sub(k).unwrap()
        };
        assert!(
            diff < one_ulp,
            "call-site wiring: post-LP-grow effective_k must scale 2×; got {} (expected ≈ 100)",
            k.to_f64(),
        );
        assert!(
            k > base_k,
            "wiring regression: effective_k floored to base — call-site likely swapped",
        );
    }

    // ─── λ-scaled collateral parity (Item 3 review) ────────────────────
    //
    // Closed-form sanity check on the formula Driver B's fix applies:
    //   computed_collateral = max(0, λ_f · f(x*) − λ_g · g(x*))
    //
    // At μ_f=42, σ=8, μ_g=43, σ=8, x* = midpoint = 42.5 (symmetric case),
    // with k=50:
    //   λ_f = λ_g = 50 · √(2 · 8 · √π) = 50 · √(16 · √π)
    //   f(42.5) = g(42.5) (PDFs equal at midpoint for same σ)
    //   ⇒ scaled = 0
    // So the formula returns 0 collateral for the trivial case.
    // For a μ-only shift the actual chain `x_star` sits slightly outside
    // the midpoint, but the optimizer's `normal_collateral` finds it; the
    // collateral is non-zero and λ-scaled here.
    #[test]
    fn lambda_scaled_collateral_zero_at_symmetric_midpoint() {
        use deadeye_collateral::lambda as coll_lambda;
        let mean_f = Sq128::from_f64(42.0).unwrap();
        let mean_g = Sq128::from_f64(43.0).unwrap();
        let var = Sq128::from_f64(64.0).unwrap(); // σ=8
        let f = NormalDistribution::from_variance(mean_f, var).unwrap();
        let g = NormalDistribution::from_variance(mean_g, var).unwrap();
        let k = 50.0_f64;
        let lam_f = coll_lambda(8.0, k);
        let lam_g = coll_lambda(8.0, k);
        // Equal-σ trade: λ_f == λ_g.
        assert!((lam_f - lam_g).abs() < 1e-12);
        let mid = Sq128::from_f64(42.5).unwrap();
        let f_at = f.pdf(mid).unwrap().to_f64();
        let g_at = g.pdf(mid).unwrap().to_f64();
        // At the midpoint, f == g for equal-σ ⇒ scaled diff = 0.
        let scaled = lam_f.mul_add(f_at, -(lam_g * g_at));
        assert!(scaled.abs() < 1e-9, "expected 0 at midpoint, got {scaled}");
    }

    /// Concrete check: at k=50, σ=8, λ ≈ 266.
    /// The doc claimed `λ ≈ 266` — protect that with a regression test
    /// so a future change to `lambda` doesn't silently move it.
    #[test]
    fn lambda_at_k50_sigma8_matches_doc() {
        use deadeye_collateral::lambda as coll_lambda;
        let l = coll_lambda(8.0, 50.0);
        // l2_norm = 1/√(2σ√π) = 1/√(16√π) ≈ 0.18804
        // λ = 50 / 0.18804 ≈ 265.92
        assert!(
            (l - 265.92).abs() < 0.5,
            "λ at k=50, σ=8 expected ≈ 265.92, got {l}"
        );
    }

    // ─── effective_k override (Follow-up #6) ───────────────────────────
    //
    // The new `*_with_override` variants let backtests / sims pass a
    // known `effective_k` and skip the SDK's internal chain re-read.
    // We can exercise the offline inner directly without a Provider,
    // so the tests below assert (a) the override flows into the
    // optimizer, (b) two `effective_k` values produce different
    // λ-scaled collateral, and (c) the override = chain-read parity
    // assumption holds when the chain reports the same value.

    fn make_market_dist(mean: f64, sigma: f64) -> NormalDistribution {
        let m = Sq128::from_f64(mean).unwrap();
        let v = Sq128::from_f64(sigma * sigma).unwrap();
        NormalDistribution::from_variance(m, v).unwrap()
    }

    #[test]
    fn validate_effective_k_override_rejects_zero() {
        let err = validate_effective_k_override(0.0).expect_err("zero must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("effective_k_override"),
            "expected effective_k_override-related error, got: {msg}",
        );
    }

    #[test]
    fn validate_effective_k_override_rejects_negative() {
        assert!(validate_effective_k_override(-1.0).is_err());
        assert!(validate_effective_k_override(-50.0).is_err());
    }

    #[test]
    fn validate_effective_k_override_rejects_non_finite() {
        assert!(validate_effective_k_override(f64::NAN).is_err());
        assert!(validate_effective_k_override(f64::INFINITY).is_err());
        assert!(validate_effective_k_override(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn validate_effective_k_override_accepts_positive() {
        validate_effective_k_override(1.0).unwrap();
        validate_effective_k_override(75.07).unwrap();
        validate_effective_k_override(1e9).unwrap();
        validate_effective_k_override(f64::MIN_POSITIVE).unwrap();
    }

    /// The offline inner must produce a positive-collateral quote when
    /// the override is the chain-typical `k = 50` and the trader's
    /// belief differs meaningfully from the market.
    #[test]
    fn offline_inner_with_override_produces_positive_collateral() {
        let current = make_market_dist(42.0, 8.0);
        let (q, _ev) = optimize_quote_offline_inner(&current, 50.0, 2.0, 20.0, 50.0)
            .expect("inner returns Ok");
        let coll = Sq128::from_raw(q.required_collateral).to_f64();
        assert!(coll >= 0.0, "collateral must be non-negative, got {coll}");
        // For a belief 1σ above market with k=50, σ=8, σ_b=2 and a
        // budget of 20, the optimizer reliably finds a non-trivial
        // positive-EV trade. If this regresses, the upstream optimizer
        // changed — investigate before adjusting the threshold.
        assert!(coll > 0.0, "expected positive collateral, got {coll}");
    }

    /// **Probe**: distinct `effective_k` inputs produce distinct
    /// λ-scaled collateral — proves the override actually threads
    /// through to the λ-scaling math. The optimizer's collateral grid
    /// itself depends on `k`, but more importantly the λ-scaling
    /// rescales the final reported `required_collateral` by `√(k …)`.
    /// If we accidentally hard-coded `k` somewhere, doubling `k`
    /// wouldn't change the output.
    #[test]
    fn offline_inner_collateral_responds_to_effective_k() {
        let current = make_market_dist(42.0, 8.0);
        let (q_k50, _) =
            optimize_quote_offline_inner(&current, 50.0, 2.0, 20.0, 50.0).expect("k=50 inner Ok");
        let (q_k200, _) =
            optimize_quote_offline_inner(&current, 50.0, 2.0, 20.0, 200.0).expect("k=200 inner Ok");
        let coll_k50 = Sq128::from_raw(q_k50.required_collateral).to_f64();
        let coll_k200 = Sq128::from_raw(q_k200.required_collateral).to_f64();
        // Higher k → larger λ → larger collateral. We don't need a
        // tight scaling factor; any *strict* inequality is enough to
        // prove the override is wired through.
        assert!(
            coll_k200 > coll_k50,
            "k=200 collateral ({coll_k200}) should exceed k=50 collateral ({coll_k50})",
        );
    }

    /// Property: for any fixed `(belief, σ_b, budget, current)`, the
    /// offline inner is deterministic in `effective_k` — calling twice
    /// with the same `k` produces byte-for-byte identical quotes. This
    /// is what justifies the "override == chain-read returning the
    /// same k" parity: the chain-reading variant just supplies the
    /// `effective_k` from `params + lp_info`; the math downstream is
    /// the same code path.
    #[test]
    fn offline_inner_is_deterministic_in_effective_k() {
        let current = make_market_dist(42.0, 8.0);
        let (a, _) =
            optimize_quote_offline_inner(&current, 50.0, 2.0, 20.0, 75.07).expect("call 1");
        let (b, _) =
            optimize_quote_offline_inner(&current, 50.0, 2.0, 20.0, 75.07).expect("call 2");
        assert_eq!(a.candidate, b.candidate, "candidate distribution diverged");
        assert_eq!(a.candidate_hints, b.candidate_hints, "hints diverged");
        assert_eq!(a.x_star, b.x_star, "x_star diverged");
        assert_eq!(
            a.required_collateral, b.required_collateral,
            "collateral diverged"
        );
        assert_eq!(a.on_chain_will_accept, b.on_chain_will_accept);
    }

    /// Sanity: zero / non-finite overrides surface a typed
    /// `CoreError::InvalidInput` from the public `_with_override`
    /// async surface (i.e. we don't fall through into the optimizer's
    /// numeric guards). We exercise the validation via the public
    /// helper rather than the full async method to avoid needing a
    /// Provider mock that returns a real distribution.
    #[test]
    fn override_validation_short_circuits_before_chain_read() {
        // The `*_with_override` methods call `validate_effective_k_override`
        // before *any* await — so a bad value never touches the
        // network. We assert that contract here.
        assert!(validate_effective_k_override(0.0).is_err());
        assert!(validate_effective_k_override(-75.07).is_err());
        assert!(validate_effective_k_override(f64::NAN).is_err());
    }

    // ─── P4 #67 regression pin ──────────────────────────────────────────
    //
    // Pre-P4, both inner paths emitted `x_star = cand_mean` (the
    // optimizer's grid-search μ_g), which mis-satisfied the on-chain
    // `stationary_valid` gate of `check_trade_view` whenever the true
    // stationary point of `d(x) = λ_g g(x) − λ_f f(x)` was shifted away
    // from μ_g — i.e. on the σ-arb scenarios that drive the chain-runtime
    // parity test (Driver B's `optimize_quote_chain_runtime_must_be_
    // accepted_by_chain`). FU2 fixed `optimize_quote_offline_inner`; P4
    // mirrored that fix into `optimize_quote_inner`. The two inner paths
    // now run byte-identical math (same `optimize_normal_trade` call,
    // same `Sq128::from_f64` conversions, same `normal_collateral` solver
    // with `MinimizationPolicy::standard()`, same λ-scaling, same
    // `(cand_mean, 0.0)` no-trade fallback). This unit test pins the
    // contract on the offline inner; the chain-runtime inner is the same
    // math wrapped around a `quote_trade` call (it forwards the
    // optimizer-derived `x_star` to the chain unchanged), so a regression
    // in either path is caught here.

    /// **P4 contract.** A σ-tightening scenario (`σ_b < σ_m`, `μ ≈ μ_m`)
    /// has a stationary point of the chain's payoff functional
    /// `d(x) = λ_g g(x) − λ_f f(x)` that is **not** `μ_g`. This test
    /// constructs that scenario and asserts the emitted `x_star`
    /// matches the audited `normal_collateral` solver's `x_min` — i.e.,
    /// the post-FU2 / P4 contract — and is *strictly distinct* from
    /// `cand_mean`. Pre-FU2 / pre-P4, both inner paths returned
    /// `x_star = cand_mean`; this regression pin fires if anyone reverts
    /// that.
    #[test]
    fn offline_inner_x_star_matches_normal_collateral_not_cand_mean() {
        // μ-shift + σ-tightening scenario: the chain's stationary
        // point of `d(x) = λ_g g(x) − λ_f f(x)` is *not* μ_g whenever
        // (σ_f, σ_g) differ — the second-derivative balance shifts the
        // root of `d'(x) = 0` away from the candidate mean. We reuse
        // the parameters of `offline_inner_with_override_produces_
        // positive_collateral` since they're already pinned as a
        // positive-EV configuration.
        let current = make_market_dist(42.0, 8.0);
        let belief_mean = 50.0_f64;
        let belief_sigma = 2.0_f64;
        let budget = 20.0_f64;
        let effective_k = 50.0_f64;

        let (quote, _ev) =
            optimize_quote_offline_inner(&current, belief_mean, belief_sigma, budget, effective_k)
                .expect("offline inner returns Ok");

        // Sanity: a positive-EV trade was found (otherwise the fallback
        // `(cand_mean, 0.0)` path fires and the pin below is vacuous).
        assert!(
            quote.on_chain_will_accept,
            "scenario should produce a positive-EV trade; \
             if the optimizer changed, pick another σ-arb scenario"
        );

        // Re-run the same math the inner uses so we can compare `x_star`
        // against the audited solver's `x_min` independently. If the
        // inner regresses to `cand_mean`, `x_star` will diverge from
        // `x_min` and we'll catch it.
        let cand_mean_q = Sq128::from_raw(quote.candidate.mean);
        let cand_variance_q = Sq128::from_raw(quote.candidate.variance);
        let candidate = NormalDistribution::from_variance(cand_mean_q, cand_variance_q)
            .expect("candidate rebuild from quote raws");
        let v = normal_collateral(&current, &candidate, MinimizationPolicy::standard())
            .expect("normal_collateral converges on the σ-arb scenario");
        let expected_x_star = Sq128::from_f64(v.x_min).expect("x_min round-trips to Sq128");

        assert_eq!(
            quote.x_star,
            expected_x_star.to_raw(),
            "x_star must be the normal_collateral solver's x_min (P4 contract); \
             pre-fix this was `cand_mean`"
        );

        // And — for the σ-arb class — `x_min` is strictly distinct from
        // `cand_mean`, which is what makes this pin non-vacuous. Without
        // this assertion, a future bug that re-set `x_star = cand_mean`
        // would still pass the equality above if the solver happened to
        // also return `cand_mean` (it doesn't for σ-arb, but pin it).
        assert_ne!(
            quote.x_star, quote.candidate.mean,
            "σ-arb scenario must have x* ≠ μ_g — otherwise this test \
             cannot distinguish the P4 fix from the pre-fix behaviour"
        );
    }

    /// **Companion pin.** When the solver fails or returns a zero /
    /// non-finite collateral (identical-distributions / Newton
    /// divergence), the inner falls back to `(cand_mean, 0.0)` — the
    /// "no-trade" sentinel. The chain-runtime inner uses the exact same
    /// fallback (lines 533-547 of this file mirror lines 252-275). This
    /// test pins the sentinel by feeding the inner a belief that matches
    /// the current market — `normal_collateral` short-circuits with
    /// `collateral = 0` for identical distributions, tripping the
    /// fallback branch.
    #[test]
    fn offline_inner_no_trade_fallback_is_cand_mean_zero_collateral() {
        // Belief == market → identical distributions → solver returns
        // `Ok(v)` with `v.collateral == 0`, which trips the `Ok(_)`
        // fallback arm.
        let current = make_market_dist(42.0, 8.0);
        let (quote, _ev) = optimize_quote_offline_inner(&current, 42.0, 8.0, 60.0, 50.0)
            .expect("offline inner returns Ok on belief == market");

        // `on_chain_will_accept` is `false` (no positive-EV trade), and
        // `required_collateral` is 0 — the canonical no-trade sentinel.
        assert!(
            !quote.on_chain_will_accept,
            "belief == market must produce no-trade"
        );
        assert_eq!(
            quote.required_collateral,
            Sq128::ZERO.to_raw(),
            "no-trade fallback must report zero required_collateral"
        );

        // And `x_star == cand_mean` per the fallback arm. This is the
        // same fallback the chain-runtime inner uses; matching here
        // pins the parity.
        assert_eq!(
            quote.x_star, quote.candidate.mean,
            "no-trade fallback must report x_star == cand_mean"
        );
    }

    // ─── Snapshot family stamping (issue #38) ───────────────────────────

    fn make_snapshot() -> NormalMarketStateSnapshot {
        let zero = Sq128RawWire::from(Sq128::ZERO.to_raw());
        NormalMarketStateSnapshot {
            family: crate::bulk::Family::Normal,
            market: "0x1".to_owned(),
            mean_raw: zero,
            variance_raw: zero,
            sigma_raw: zero,
            base_k_raw: zero,
            initial_backing_raw: zero,
            pool_backing_raw: zero,
            mean: 0.0,
            sigma: 0.0,
            effective_k: 0.0,
            pool_backing_xp: 0.0,
        }
    }

    #[test]
    fn snapshot_serializes_family_normal() {
        let json = serde_json::to_value(make_snapshot()).unwrap();
        assert_eq!(
            json.get("family").and_then(|v| v.as_str()),
            Some("normal"),
            "snapshot JSON must carry the family tag",
        );
    }

    #[test]
    fn legacy_snapshot_without_family_defaults_to_normal() {
        // Files written before issue #38 lack the field; only the normal
        // path could legitimately produce them.
        let mut json = serde_json::to_value(make_snapshot()).unwrap();
        json.as_object_mut().unwrap().remove("family");
        let snap: NormalMarketStateSnapshot = serde_json::from_value(json).unwrap();
        assert_eq!(snap.family, crate::bulk::Family::Normal);
    }

    #[test]
    fn snapshot_family_lognormal_round_trips() {
        let mut json = serde_json::to_value(make_snapshot()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("family".into(), serde_json::json!("lognormal"));
        let snap: NormalMarketStateSnapshot = serde_json::from_value(json).unwrap();
        assert_eq!(snap.family, crate::bulk::Family::Lognormal);
    }
}
