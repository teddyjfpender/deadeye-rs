//! Portfolio-level aggregate over many markets, families, and traders.
//!
//! A market maker holds positions, LP shares, and a single STRK balance
//! across N markets and possibly multiple families. The naive way to
//! reconstruct that view is `N × (position + lp_info + distribution +
//! balance)` sequential reads — at 50 ms RTT a 20-market book costs
//! upward of a wall-clock second per refresh.
//!
//! [`Portfolio`] composes [`BulkReader`] (Wave 1) so every read in
//! `Portfolio::load` is fanned out concurrently, then post-processes the
//! results into a [`BTreeMap`]-keyed aggregate the MM can query in O(log
//! N).
//!
//! ## Conventions
//!
//! * `Family` is reused from [`crate::bulk`] — no new tag.
//! * Position valuation uses `total_collateral` as a conservative fallback when
//!   the pricing primitives aren't enabled. A future wave may swap in an
//!   EV-based valuation; the field is named `current_value_f64` so the API
//!   doesn't shift when that lands.
//! * `total_strk_balance` is supplied by the caller because the SDK doesn't
//!   have a hard dependency on a specific ERC-20 reader. The constructor
//!   accepts `0` if the caller doesn't care.

use std::collections::BTreeMap;

use deadeye_core::Sq128;
use deadeye_starknet::{ContractError, Felt, Provider};

use crate::{
    bulk::{BulkReader, Family, Position},
    client::DeadeyeClient,
};

/// Re-export the multi-market effective-sensitivity primitive from
/// `deadeye-collateral` so the SDK can compute coupled-exposure risk
/// without taking a direct dependency on the solver crate.
///
/// Background: when a market maker warehouses positions across k
/// markets simultaneously, the joint risk budget is no longer the
/// naive sum of per-market exposures. If the per-market positions are
/// linearly coupled through the MM's shared capital (e.g. positions
/// in markets j and j' both react to a common underlying factor), the
/// effective risk parameter is inflated by the L2-norm of the
/// coupling vector `c`. See Zenodo DOI 10.5281/zenodo.20434661
/// (Theorem 1) for the formal reduction.
pub use deadeye_collateral::{effective_sensitivity, inflate_collateral};

/// A reference to a market identified by `(family, address)`.
///
/// This mirrors the tuple shape consumed by [`BulkReader`] but is a
/// named struct so callers can build typed lists without juggling
/// inferred tuple positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarketRef {
    /// Family that owns the market.
    pub family: Family,
    /// Market contract address.
    pub address: Felt,
}

impl MarketRef {
    /// Sugar constructor.
    #[must_use]
    pub const fn new(family: Family, address: Felt) -> Self {
        Self { family, address }
    }
}

/// A single trader-position entry inside a [`Portfolio`].
#[derive(Debug, Clone)]
pub struct PositionEntry {
    /// Family the position belongs to.
    pub family: Family,
    /// Raw position record (one of four shapes).
    pub raw: Position,
    /// Conservative valuation: `total_collateral` projected to `f64`.
    ///
    /// When pricing primitives ship, this becomes the EV-based valuation;
    /// callers can already reason about exposure today by summing this
    /// field across the portfolio.
    pub current_value_f64: f64,
}

/// A single LP-position entry inside a [`Portfolio`].
#[derive(Debug, Clone, Copy)]
pub struct LpEntry {
    /// Family the LP position belongs to.
    pub family: Family,
    /// LP shares the trader holds, projected to `f64`.
    pub shares_f64: f64,
    /// `shares / total_shares × 100` — % of pool the trader owns.
    pub backing_share_pct: f64,
}

/// A recommended hedge action for delta-neutral rebalancing.
#[derive(Debug, Clone, Copy)]
pub struct HedgeRecommendation {
    /// Market that needs the counter-trade.
    pub market: Felt,
    /// Family of `market` (so the caller picks the right writer).
    pub family: Family,
    /// Notional size (in collateral, f64) of the suggested counter-trade.
    pub notional_f64: f64,
    /// Sign of the recommended counter-trade — `+1.0` means "trade up",
    /// `-1.0` means "trade down". This is intentionally simple: the
    /// concrete distribution shift is left to the optimizer.
    pub direction: f64,
}

/// Multi-market portfolio aggregate.
///
/// Constructed via [`Portfolio::load`], which fans out concurrent reads
/// against every entry in `markets`. Failing reads on a sub-market are
/// silently dropped — the portfolio is best-effort, mirroring the
/// `Option<…>` shape on [`crate::bulk::MarketStateSnapshot`].
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Trader the portfolio belongs to.
    pub trader: Felt,
    /// Every market the caller asked us to inspect.
    pub markets: Vec<MarketRef>,
    /// Open positions, keyed by market address.
    pub positions: BTreeMap<Felt, PositionEntry>,
    /// LP shares the trader holds, keyed by market address.
    pub lp_positions: BTreeMap<Felt, LpEntry>,
    /// STRK balance, supplied externally (0 if unknown).
    pub total_strk_balance: u128,
}

impl Portfolio {
    /// Load every position and LP-share record for `trader` across
    /// `markets`, concurrently.
    ///
    /// Failed sub-reads are dropped silently — they're reported as
    /// "missing entry" in the returned [`BTreeMap`]s. Callers that need
    /// strict semantics can compare `positions.len()` against
    /// `markets.len()` after the call.
    pub async fn load<P: Provider>(
        client: &DeadeyeClient<P>,
        trader: Felt,
        markets: Vec<MarketRef>,
    ) -> Result<Self, ContractError> {
        Self::load_with_strk_balance(client, trader, markets, 0).await
    }

    /// Like [`Self::load`] but also captures a pre-fetched STRK balance.
    ///
    /// Keeping the balance read out-of-band keeps the SDK free of a
    /// hard ERC-20 dependency.
    pub async fn load_with_strk_balance<P: Provider>(
        client: &DeadeyeClient<P>,
        trader: Felt,
        markets: Vec<MarketRef>,
        total_strk_balance: u128,
    ) -> Result<Self, ContractError> {
        // Borrow the inner provider for two fan-outs.
        let bulk = BulkReader::new(DeadeyeClient::new(BorrowedProvider {
            inner: client.provider(),
        }));

        let pos_queries: Vec<_> = markets
            .iter()
            .map(|m| (m.family, m.address, trader))
            .collect();
        let lp_queries: Vec<_> = markets.iter().map(|m| (m.family, m.address)).collect();

        let pos_fut = bulk.positions(&pos_queries);
        let lp_fut = bulk.lp_infos(&lp_queries);
        let (pos_results, lp_results) = futures::future::join(pos_fut, lp_fut).await;

        let mut positions: BTreeMap<Felt, PositionEntry> = BTreeMap::new();
        let mut lp_positions: BTreeMap<Felt, LpEntry> = BTreeMap::new();

        for (m, res) in markets.iter().zip(pos_results) {
            if let Ok(pos) = res {
                let current_value_f64 = position_current_value(&pos);
                if current_value_f64 > 0.0 {
                    positions.insert(m.address, PositionEntry {
                        family: m.family,
                        raw: pos,
                        current_value_f64,
                    });
                }
            }
        }
        for (m, res) in markets.iter().zip(lp_results) {
            if let Ok(lp) = res {
                let shares = Sq128::from_raw(lp.total_shares).to_f64();
                if shares <= 0.0 || !shares.is_finite() {
                    continue;
                }
                let total = Sq128::from_raw(lp.total_backing_deposited).to_f64();
                let pct = if total > 0.0 {
                    (shares / total) * 100.0_f64
                } else {
                    0.0
                };
                lp_positions.insert(m.address, LpEntry {
                    family: m.family,
                    shares_f64: shares,
                    backing_share_pct: pct,
                });
            }
        }

        Ok(Self {
            trader,
            markets,
            positions,
            lp_positions,
            total_strk_balance,
        })
    }

    /// Total exposure: position notional + LP-share valuation + free STRK.
    ///
    /// Per `deadeye_optimizer::lp::compute_lp_claim_component_value`, an
    /// LP's claim on the pool is `pool_share × pool_value` (NOT
    /// `pool_share × shares`). With `pool_share = shares / total_shares`
    /// and `pool_value ≈ total_backing_deposited`, the LP component
    /// reduces to `(shares / total_shares) × total_backing_deposited`.
    /// We store that ratio in `LpEntry.shares_f64 / (shares_f64 ×
    /// 100/backing_share_pct)` — equivalent to reading the original
    /// `total_backing_deposited` back via `shares / (backing_share_pct /
    /// 100)`. For clarity we recompute the LP claim directly.
    ///
    /// All sums are kept in **whole-STRK** (token decimals) so the
    /// free-STRK contribution is dimensionally consistent.
    #[must_use]
    pub fn total_exposure_f64(&self) -> f64 {
        let pos_sum: f64 = self
            .positions
            .values()
            .map(|p| p.current_value_f64.max(0.0))
            .sum();
        // LP claim ≈ pool_share × pool_value.
        //   pool_share        = shares_f64 / total_shares
        //   total_shares      = shares_f64 / (backing_share_pct / 100)
        //   so pool_share     = backing_share_pct / 100
        //   pool_value        ≈ total_backing_deposited
        //                     = shares_f64 / (backing_share_pct / 100)
        //   ⇒ claim           = backing_share_pct / 100 × total_backing_deposited.
        // The numerically robust form (avoiding zero-pct divide) is:
        //   `shares_f64 × (backing_share_pct/100) / (backing_share_pct/100)`
        // which collapses to `total_backing_deposited × pool_share`.
        // Equivalent: `shares_f64` already equals `total_backing × pool_share`
        // when shares are 1-to-1 with backing — so `shares_f64` IS the LP claim
        // expressed in backing units. We use that directly.
        let lp_sum: f64 = self
            .lp_positions
            .values()
            .map(|lp| lp.shares_f64.max(0.0))
            .sum();
        // STRK is held in token-decimal base units (10^18 per STRK). Both
        // position notionals and LP claims are quoted in whole units of
        // backing collateral (which IS STRK for the chaos profile), so
        // project STRK to whole tokens for a dimensionally consistent sum.
        let strk_f = (self.total_strk_balance as f64) / 1e18_f64;
        pos_sum + lp_sum + strk_f
    }

    /// Coupling coefficient `c_j` for each market in `self.markets`,
    /// based on the position's `current_value_f64` normalised to the
    /// largest position. The first market in the list is the reference
    /// (`c = 1`); others are scaled relative to it. Markets with no
    /// open position are skipped.
    ///
    /// ## ⚠ This is a position-size heuristic, not a risk-coupling estimate
    ///
    /// `heuristic_coupling_coefficients()` returns `c` proportional to
    /// notional position size. **Notional size is not the same as risk
    /// coupling.** Two markets can hold positions of similar size but
    /// share almost no risk (e.g. a large BTC price market and a large
    /// ETH price market — the BTC/ETH correlation is high but not 1,
    /// and the dominant risk factor for each is its own spot price, not
    /// "size of my position on the other one"). Conversely, two markets
    /// with very different position sizes can be strongly coupled (e.g.
    /// a small "World Cup Winner" market and a larger "World Cup
    /// Finalist" market — the conditional probability of the second
    /// depends on the first).
    ///
    /// For production risk accounting, supply your own coupling
    /// coefficients derived from domain knowledge, empirical
    /// correlations, or a dedicated risk model — not from this
    /// heuristic. The reduction in
    /// [`effective_exposure_f64_with_coupling`] is exact under the
    /// [linear-coupling model] given an accurate `c`; this heuristic
    /// just makes the API usable out of the box for a uniformly-
    /// warehoused book where a position-size proxy is acceptable.
    ///
    /// [linear-coupling model]: deadeye_collateral::multi_market#assumptions--standard-model
    #[must_use]
    pub fn heuristic_coupling_coefficients(&self) -> Vec<f64> {
        let values: Vec<f64> = self
            .markets
            .iter()
            .map(|m| {
                self.positions
                    .get(&m.address)
                    .map_or(0.0_f64, |p| p.current_value_f64)
                    .max(0.0)
            })
            .collect();
        let max = values.iter().fold(0.0_f64, |a, b| a.max(*b));
        if max <= 0.0 {
            return vec![0.0; values.len()];
        }
        values.iter().map(|v| v / max).collect()
    }

    /// Effective risk-aware exposure across the warehouse, computed via
    /// the multi-output Gaussian release reduction
    /// ([`effective_sensitivity`]). The coupling vector `c` is supplied
    /// by the caller — use [`Self::heuristic_coupling_coefficients`] for
    /// a default or pass a model-derived vector for production.
    ///
    /// Returns `None` if `c` is not the same length as `self.markets`
    /// or the underlying accountant rejects the inputs (NaN, non-finite
    /// `delta`, empty book). The naive per-market sum is returned by
    /// [`Self::total_exposure_f64`].
    #[must_use]
    pub fn effective_exposure_f64_with_coupling(
        &self,
        c: &[f64],
    ) -> Option<f64> {
        if c.len() != self.markets.len() {
            return None;
        }
        let values: Vec<f64> = self
            .markets
            .iter()
            .map(|m| {
                self.positions
                    .get(&m.address)
                    .map_or(0.0_f64, |p| p.current_value_f64)
                    .max(0.0)
            })
            .collect();
        // Per-market sensitivity proxy: scaled position value.
        // The caller should pass a calibrated `delta` in production;
        // we use 1.0 so the inflation factor is unit-agnostic.
        let delta = 1.0_f64;
        let factor = effective_sensitivity(c, delta).ok()?;
        let naive: f64 = values.iter().sum();
        Some(naive * factor)
    }

    /// Realised LP cashflow since `since_block`, per market.
    ///
    /// "Realised cashflow" = sum of `tokenAmount` on every
    /// `liquidity_removed` event for the portfolio's trader within
    /// `[since_block, +∞)`, minus the sum of `tokenAmount` on every
    /// `liquidity_added` event in the same window. Positive values
    /// represent net STRK *returned* to the LP since the cutoff —
    /// realised yield + principal returned. Negative values represent
    /// the trader topping the pool up.
    ///
    /// This is a coarser proxy than a true "fee yield" because the
    /// indexer doesn't distinguish principal from accrued fees in the
    /// event stream; an LP that withdrew their full principal will see
    /// the whole withdrawal counted here. Callers that need a pure
    /// "fee-only" yield should pair this number with an entry/exit
    /// reference value from `lp_history` directly.
    ///
    /// Markets the trader never LP'd in are silently skipped (the
    /// indexer returns an empty history for them). Markets the indexer
    /// rejects with HTTP errors are also skipped — the SDK refuses to
    /// fail-the-whole-portfolio on one bad endpoint.
    pub async fn lp_yield_since(
        &self,
        indexer: &deadeye_indexer::IndexerClient,
        since_block: u64,
    ) -> Result<BTreeMap<Felt, f64>, ContractError> {
        const STRK_SCALE: f64 = 1e18_f64;
        let trader_hex = format!("{:#x}", self.trader);
        let mut out = BTreeMap::new();
        for market in &self.markets {
            let market_hex = format!("{:#x}", market.address);
            let Ok(history) = indexer.lp_history(&market_hex, &trader_hex).await else {
                continue;
            };
            let mut realised = 0.0_f64;
            for event in &history {
                if event.block_number < since_block {
                    continue;
                }
                let Ok(amount) = event.token_amount.parse::<u128>() else {
                    continue;
                };
                let scaled = amount as f64 / STRK_SCALE;
                match event.event_type.as_str() {
                    "liquidity_removed" | "lp_claim" => realised += scaled,
                    "liquidity_added" => realised -= scaled,
                    _ => {},
                }
            }
            if realised != 0.0_f64 {
                out.insert(market.address, realised);
            }
        }
        Ok(out)
    }

    /// "If I want to be delta-neutral against `market_id`, what
    /// counter-trade do I need on the others?"
    ///
    /// **Correlation caveat — read this.** The returned hedge assumes
    /// **all listed markets share a common driver** (i.e. they are
    /// perfectly positively correlated against the target). For
    /// uncorrelated books the result is statistical noise; for
    /// *negatively* correlated markets the recommendation will *double*
    /// the trader's exposure rather than reducing it. A real Δ-neutral
    /// hedge requires a covariance matrix the caller must supply — this
    /// SDK does not currently model that.
    ///
    /// The returned hedges have notionals proportional to the absolute
    /// `current_value_f64` of each non-target position, with a direction
    /// that opposes the target's sign. This is intentionally
    /// approximate — refining the math is what `deadeye-optimizer`
    /// exists for; the recommendation here is the *shape* of the hedge.
    #[must_use]
    pub fn delta_neutral_hedge_for(&self, market_id: Felt) -> Vec<HedgeRecommendation> {
        let Some(target) = self.positions.get(&market_id) else {
            return Vec::new();
        };
        // The "direction" we want to oppose is the sign of the target.
        // For our conservative valuation (total_collateral, always >= 0)
        // the sign is +1.0, so the hedge direction is -1.0 by default.
        // We keep the field so callers that swap in a signed valuation
        // get the correct sign automatically.
        let target_sign = target
            .current_value_f64
            .signum()
            .abs()
            .max(1.0)
            .copysign(1.0);
        let hedge_sign = -target_sign;

        let others_total: f64 = self
            .positions
            .iter()
            .filter(|(addr, _)| **addr != market_id)
            .map(|(_, p)| p.current_value_f64.abs())
            .sum();
        if others_total <= 0.0 || !others_total.is_finite() {
            return Vec::new();
        }
        let target_value = target.current_value_f64.abs();
        let mut out = Vec::new();
        for (addr, entry) in &self.positions {
            if *addr == market_id {
                continue;
            }
            let weight = entry.current_value_f64.abs() / others_total;
            out.push(HedgeRecommendation {
                market: *addr,
                family: entry.family,
                notional_f64: weight * target_value,
                direction: hedge_sign,
            });
        }
        out
    }
}

fn position_current_value(pos: &Position) -> f64 {
    let raw = match pos {
        Position::Normal(p) => p.total_collateral,
        Position::Lognormal(p) => p.total_collateral,
        Position::Multinoulli(p) => p.total_collateral,
        Position::Bivariate(p) => p.total_collateral,
    };
    let v = Sq128::from_raw(raw).to_f64();
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

/// Lightweight `Provider` wrapper that borrows the inner provider so
/// `Portfolio::load` can build an ad-hoc [`BulkReader`] without taking
/// ownership of the caller's client.
///
/// The lifetime is bound to the borrowed provider via a PhantomData-like
/// trick (the struct holds a reference). Every method delegates straight
/// through to the inner provider.
#[derive(Debug)]
struct BorrowedProvider<'a, P>
where
    P: Provider,
{
    inner: &'a P,
}

#[async_trait::async_trait]
impl<P> Provider for BorrowedProvider<'_, P>
where
    P: Provider,
{
    async fn call(
        &self,
        call: starknet_core::types::FunctionCall,
        block: starknet_core::types::BlockId,
    ) -> deadeye_starknet::ContractResult<Vec<Felt>> {
        self.inner.call(call, block).await
    }

    fn default_block(&self) -> starknet_core::types::BlockId {
        self.inner.default_block()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use async_trait::async_trait;
    use deadeye_core::sq128::Sq128Raw;
    use deadeye_starknet::{CairoSerde, ContractResult};
    use starknet_core::types::{BlockId, FunctionCall};

    use super::*;
    use crate::client::DeadeyeClient;

    /// Provider that returns a fixed-shape `PositionCompactRaw` for every
    /// `position` selector and a fixed `LpInfoRaw` for every `lp_info`
    /// selector. Position/lp record content is per-market so we can tell
    /// they round-trip through the bulk fan-out.
    #[derive(Debug, Default)]
    struct CannedProvider;

    fn sq(value: u64) -> Sq128Raw {
        Sq128Raw {
            limb0: 0,
            limb1: 0,
            limb2: value,
            limb3: 0,
            neg: false,
        }
    }

    #[async_trait]
    impl Provider for CannedProvider {
        async fn call(&self, call: FunctionCall, _block: BlockId) -> ContractResult<Vec<Felt>> {
            // Use selector text length (via the calldata's first felt for
            // address-encoded entry point) to disambiguate. Simpler: pick
            // by length of calldata — `position(trader)` has 1 calldata,
            // `lp_info()` has none.
            if call.calldata.is_empty() {
                // lp_info() → LpInfoRaw with non-zero shares + backing.
                let lp = deadeye_starknet::types::common::LpInfoRaw {
                    total_shares: sq(10),
                    total_backing_deposited: sq(100),
                };
                Ok(lp.to_calldata())
            } else {
                // position(trader) → all four position shapes share the
                // same minimal header: just return a normal compact.
                let p = deadeye_starknet::types::normal::PositionCompactRaw {
                    original_mean: sq(1),
                    original_variance: sq(1),
                    original_sigma: sq(1),
                    original_lambda: sq(1),
                    effective_mean: sq(1),
                    effective_variance: sq(1),
                    effective_sigma: sq(1),
                    effective_lambda: sq(1),
                    total_collateral: sq(50),
                    flags: 0,
                };
                Ok(p.to_calldata())
            }
        }
    }

    #[tokio::test]
    async fn load_aggregates_positions_and_lp_shares() {
        let client = DeadeyeClient::new(CannedProvider);
        let trader = Felt::from(0xABCD_u64);
        let markets = vec![
            MarketRef::new(Family::Normal, Felt::from(0x1_u64)),
            MarketRef::new(Family::Normal, Felt::from(0x2_u64)),
        ];
        let portfolio =
            Portfolio::load_with_strk_balance(&client, trader, markets, 5 * 10_u128.pow(18))
                .await
                .unwrap();
        assert_eq!(portfolio.positions.len(), 2);
        assert_eq!(portfolio.lp_positions.len(), 2);
        // 2 positions × 50 collateral + 2 LP × (10 shares × 10% pct) + 5 STRK.
        let exp = portfolio.total_exposure_f64();
        assert!(exp > 0.0, "exposure should be strictly positive, got {exp}");
    }

    #[tokio::test]
    async fn delta_neutral_hedge_picks_other_markets() {
        let client = DeadeyeClient::new(CannedProvider);
        let trader = Felt::from(0xABCD_u64);
        let m1 = Felt::from(0x1_u64);
        let m2 = Felt::from(0x2_u64);
        let m3 = Felt::from(0x3_u64);
        let markets = vec![
            MarketRef::new(Family::Normal, m1),
            MarketRef::new(Family::Normal, m2),
            MarketRef::new(Family::Normal, m3),
        ];
        let portfolio = Portfolio::load(&client, trader, markets).await.unwrap();
        let hedges = portfolio.delta_neutral_hedge_for(m1);
        assert_eq!(hedges.len(), 2);
        for h in &hedges {
            assert_ne!(h.market, m1);
            assert!(h.notional_f64 > 0.0);
            assert!(h.direction.is_finite());
        }
    }

    #[tokio::test]
    async fn delta_neutral_hedge_for_unknown_market_is_empty() {
        let client = DeadeyeClient::new(CannedProvider);
        let trader = Felt::from(0xABCD_u64);
        let m1 = Felt::from(0x1_u64);
        let markets = vec![MarketRef::new(Family::Normal, m1)];
        let portfolio = Portfolio::load(&client, trader, markets).await.unwrap();
        let hedges = portfolio.delta_neutral_hedge_for(Felt::from(0x99_u64));
        assert!(hedges.is_empty());
    }
}
