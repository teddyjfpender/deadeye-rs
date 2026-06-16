//! Bivariate-normal AMM market handle.
//!
//! Two correlated Gaussians (μ₁, μ₂, σ₁², σ₂², ρ). The constructor's
//! hint check requires byte-exact Sq128 derivations of σ₁, σ₂,
//! `1/(1−ρ²)`, and the joint normalization. The writer's
//! `quote_trade` runs `expand_distribution_core_view` first so f64
//! inputs are safely promoted to chain-exact full distributions.
//!
//! ## Worked example
//!
//! ```no_run
//! use deadeye_sdk::{
//!     core::{
//!         bivariate::{BivariateNormalDistributionCoreRaw, BivariatePointRaw},
//!         sq128::Sq128Raw,
//!     },
//!     starknet::{
//!         BivariateMarketReader, BivariateMarketWriter, Felt, JsonRpcProvider, OwnedAccount,
//!     },
//! };
//! use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let rpc = JsonRpcClient::new(HttpTransport::new(
//!     "http://localhost:5050".parse::<url::Url>()?,
//! ));
//! let provider = JsonRpcProvider::new(rpc);
//! let (market, runtime): (Felt, Felt) = (Felt::ZERO, Felt::ZERO);
//! let reader = BivariateMarketReader::new(&provider, market);
//! let signer = OwnedAccount::from_signing_key(
//!     JsonRpcClient::new(HttpTransport::new(
//!         "http://localhost:5050".parse::<url::Url>()?,
//!     )),
//!     Felt::ZERO,
//!     Felt::ZERO,
//!     Felt::ZERO,
//! );
//! let writer = BivariateMarketWriter::new(reader, signer);
//!
//! let core = BivariateNormalDistributionCoreRaw {
//!     mu1: Sq128Raw::ZERO,
//!     mu2: Sq128Raw::ZERO,
//!     variance1: Sq128Raw::ZERO,
//!     variance2: Sq128Raw::ZERO,
//!     rho: Sq128Raw::ZERO,
//! };
//! let x_star = BivariatePointRaw {
//!     x1: Sq128Raw::ZERO,
//!     x2: Sq128Raw::ZERO,
//! };
//! let quote = writer
//!     .reader()
//!     .quote_trade(runtime, core, x_star, Sq128Raw::ZERO)
//!     .await?;
//! writer.execute_quote(quote).await?;
//! writer.sell_position(runtime, 0).await?;
//! # Ok(()) }
//! ```

use deadeye_collateral::{BivariateOptions, BivariateVerifiedMinimum, bivariate_collateral};
use deadeye_core::{
    BivariateNormalDistribution, Sq128,
    bivariate::{BivariateNormalDistributionRaw, BivariateNormalSqrtHintsRaw, BivariatePointRaw},
    sq128::Sq128Raw,
};
use deadeye_starknet::{
    Account, BivariateMarketReader, BivariateMarketWriter, ExecutionReceipt, Felt, Provider,
    types::bivariate::{BivariateNormalSellExecutionGuardsRaw, BivariateTradeInput},
};
use futures::future::join_all;
use tracing::instrument;

use crate::{
    error::{SdkError, SdkResult},
    legs::{LegInfo, LegValuation, PositionLegs, PositionValuation, SettlementPoint, belief_grid},
};

/// Handle to a deployed bivariate AMM market.
#[derive(Debug)]
pub struct BivariateMarket<'p, P>
where
    P: Provider,
{
    reader: BivariateMarketReader<&'p P>,
}

impl<'p, P> BivariateMarket<'p, P>
where
    P: Provider,
{
    /// Construct a handle.
    pub fn new(provider: &'p P, address: Felt) -> Self {
        Self {
            reader: BivariateMarketReader::new(provider, address),
        }
    }

    /// Underlying read-only reader.
    pub const fn reader(&self) -> &BivariateMarketReader<&'p P> {
        &self.reader
    }

    /// Contract address.
    pub const fn address(&self) -> Felt {
        self.reader.address()
    }

    /// Reads the current market distribution.
    pub async fn distribution(&self) -> SdkResult<BivariateNormalDistribution> {
        Ok(self.reader.distribution().await?)
    }

    /// Prepares an off-chain quote for moving the market from its current
    /// state to `candidate`.
    #[instrument(skip(self, candidate), fields(market = %self.reader.address()))]
    pub async fn prepare_quote(
        &self,
        candidate: &BivariateNormalDistribution,
        opts: BivariateOptions,
    ) -> SdkResult<BivariateVerifiedMinimum> {
        let current = self.distribution().await?;
        Ok(bivariate_collateral(&current, candidate, opts)?)
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

    /// Value a trader's whole position at a settlement point `(x1, x2)`,
    /// authoritatively (each active leg via the on-chain `value_at` view).
    /// `total_position_value` is the position's P&L if the market settles at
    /// `(x1, x2)`; `gross_return` adds back the locked collateral.
    #[instrument(skip(self), fields(market = %self.reader.address(), %trader, x1, x2))]
    pub async fn position_value_at(
        &self,
        trader: Felt,
        x1: f64,
        x2: f64,
    ) -> SdkResult<PositionValuation> {
        let summary = self.reader.position_summary(trader).await?;
        let ids = self.reader.trade_lot_ids(trader).await?;
        let point = BivariatePointRaw {
            x1: Sq128::from_f64(x1)?.to_raw(),
            x2: Sq128::from_f64(x2)?.to_raw(),
        };
        let legs = join_all(ids.iter().map(|&lot_id| async move {
            let settled = self.reader.trade_lot_settled(lot_id).await?;
            let cancelled = self.reader.trade_lot_cancelled(lot_id).await?;
            // Settled/cancelled legs have no future payout.
            let value_at = if settled || cancelled {
                0.0
            } else {
                Sq128::from_raw(self.reader.trade_lot_value_at(lot_id, point).await?).to_f64()
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
            settlement: SettlementPoint::Point { x1, x2 },
            legs,
            total_collateral,
            total_position_value,
            gross_return: total_collateral + total_position_value,
            exists: summary.exists,
            claimed: summary.claimed,
        })
    }

    /// Expected position value (P&L) under a bivariate-normal belief
    /// `N₂(μ₁, μ₂, σ₁², σ₂², ρ)`, integrating the on-chain leg value over the
    /// joint belief via a tensor-product normal-pdf-weighted grid. Returns the
    /// expected P&L in XP.
    ///
    /// The grid is the outer product of two per-axis [`belief_grid`]s with
    /// **N = 11** nodes each (121 joint nodes). N = 11 is a deliberate cost
    /// trade-off: a 2-D grid costs `nodes² × active legs` `value_at` reads, so
    /// the per-axis 21 nodes the univariate path uses would explode to 441
    /// joint nodes per leg and exhaust the RPC connection pool. Eleven nodes
    /// per axis keeps the midpoint quadrature within a few percent while
    /// bounding the read count.
    ///
    /// ### Correlation handling
    ///
    /// [`belief_grid`] yields per-axis weights that are the **marginal**
    /// Gaussian densities (each axis normalised to sum to 1). Their outer
    /// product `w₁·w₂` is therefore the *independent* joint density (ρ = 0).
    /// To recover the correlated joint we multiply each node by the
    /// bivariate-normal correlation correction
    /// `exp(ρ/(1−ρ²) · z₁·z₂)` with `zᵢ = (xᵢ−μᵢ)/σᵢ`, then **renormalise** the
    /// full grid to sum to 1. (The `exp(−ρ²/(2(1−ρ²))·(z₁²+z₂²))` term in the
    /// exact joint density is already captured by the marginal weights up to a
    /// constant, which the final renormalisation absorbs.) When ρ = 0 the
    /// correction is identically 1 and the grid reduces to the plain tensor
    /// product.
    ///
    /// Costs `121 × active legs` `value_at` reads, fanned out concurrently in
    /// batches of ~200 futures to avoid connection exhaustion — call when an
    /// agent wants its forward EV, not on every tick.
    #[instrument(skip(self), fields(market = %self.reader.address(), %trader, mu1, mu2, sigma1, sigma2, rho))]
    pub async fn expected_value_under_belief(
        &self,
        trader: Felt,
        mu1: f64,
        mu2: f64,
        sigma1: f64,
        sigma2: f64,
        rho: f64,
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
        // Tensor-product belief grid. N = 11 per axis (121 joint nodes); a
        // larger N would multiply the read count quadratically.
        let axis1 = belief_grid(mu1, sigma1, 4.0, 11);
        let axis2 = belief_grid(mu2, sigma2, 4.0, 11);
        // Build the (x1, x2, weight) nodes with the ρ-correction applied. The
        // correction is 1 when ρ == 0; otherwise it tilts the independent
        // tensor weights into the correlated joint. We renormalise after the
        // full grid is built so the corrected weights sum to 1.
        let rho_factor = if rho == 0.0 {
            0.0
        } else {
            // `1 − ρ²` via `mul_add` (ρ·(−ρ) + 1) to satisfy `suboptimal_flops`.
            rho / rho.mul_add(-rho, 1.0)
        };
        let mut nodes: Vec<(f64, f64, f64)> = Vec::with_capacity(axis1.len() * axis2.len());
        let mut wsum = 0.0;
        for &(x1, w1) in &axis1 {
            for &(x2, w2) in &axis2 {
                let mut weight = w1 * w2;
                if rho_factor != 0.0 {
                    let z1 = (x1 - mu1) / sigma1;
                    let z2 = (x2 - mu2) / sigma2;
                    weight *= (rho_factor * z1 * z2).exp();
                }
                wsum += weight;
                nodes.push((x1, x2, weight));
            }
        }
        if wsum > 0.0 {
            for node in &mut nodes {
                node.2 /= wsum;
            }
        }
        // E[value] = Σ_nodes w · Σ_legs value_at(lot, (x1, x2)). Fan out the
        // per-(node, leg) reads concurrently, chunked at ~200 futures so a
        // large position never exhausts the connection pool.
        let mut weighted = 0.0;
        let mut futs = Vec::with_capacity(200);
        for &(x1, x2, weight) in &nodes {
            let point = BivariatePointRaw {
                x1: Sq128::from_f64(x1)?.to_raw(),
                x2: Sq128::from_f64(x2)?.to_raw(),
            };
            for &lot_id in &active {
                futs.push(async move {
                    let v = Sq128::from_raw(self.reader.trade_lot_value_at(lot_id, point).await?)
                        .to_f64();
                    Ok::<f64, SdkError>(weight * v)
                });
                if futs.len() >= 200 {
                    let batch = core::mem::take(&mut futs);
                    let parts = join_all(batch)
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;
                    weighted += parts.into_iter().sum::<f64>();
                }
            }
        }
        if !futs.is_empty() {
            let parts = join_all(futs)
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            weighted += parts.into_iter().sum::<f64>();
        }
        Ok(weighted)
    }

    /// Bind an account for writes.
    pub fn with_account<A>(self, account: A) -> BivariateMarketSigned<'p, P, A>
    where
        A: Account,
    {
        BivariateMarketSigned {
            writer: BivariateMarketWriter::new(self.reader, account),
        }
    }
}

/// Account-bound companion.
#[derive(Debug)]
pub struct BivariateMarketSigned<'p, P, A>
where
    P: Provider,
    A: Account,
{
    writer: BivariateMarketWriter<&'p P, A>,
}

impl<P, A> BivariateMarketSigned<'_, P, A>
where
    P: Provider,
    A: Account,
{
    /// Borrow the underlying writer.
    pub const fn writer(&self) -> &BivariateMarketWriter<&P, A> {
        &self.writer
    }

    /// Contract address.
    pub const fn address(&self) -> Felt {
        self.writer.reader().address()
    }

    /// Execute a trade.
    pub async fn execute_trade(&self, input: BivariateTradeInput) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.execute_trade(input).await?)
    }

    /// Submit a guarded sell. ABI:
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    pub async fn sell_position_guarded(
        &self,
        candidate: BivariateNormalDistributionRaw,
        x_star: BivariatePointRaw,
        candidate_hints: BivariateNormalSqrtHintsRaw,
        guards: BivariateNormalSellExecutionGuardsRaw,
    ) -> SdkResult<ExecutionReceipt> {
        Ok(self
            .writer
            .sell_position_guarded(candidate, x_star, candidate_hints, guards)
            .await?)
    }

    /// Claim the caller's settled position.
    pub async fn claim(&self) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.claim().await?)
    }

    /// Claim a settled position on behalf of `trader`.
    pub async fn claim_for(&self, trader: Felt) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.claim_for(trader).await?)
    }

    /// Add liquidity to the bivariate pool. ABI takes `share_amount` only.
    pub async fn add_liquidity(&self, share_amount: Sq128Raw) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.add_liquidity(share_amount).await?)
    }

    /// Remove a fraction of the caller's liquidity. ABI takes
    /// `share_amount` only.
    pub async fn remove_liquidity(&self, share_amount: Sq128Raw) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.remove_liquidity(share_amount).await?)
    }

    /// Settle the bivariate market at `settlement_point`.
    pub async fn settle(&self, settlement_point: BivariatePointRaw) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.settle(settlement_point).await?)
    }
}
