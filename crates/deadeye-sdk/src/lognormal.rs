//! Lognormal AMM market handle.
//!
//! Same shape as [`crate::normal`] but parameterised on the lognormal
//! distribution: μ + σ² live in log-space, the chain expects the dist's
//! σ as a separate field, and `x*` is always positive. Use this family
//! for markets whose payouts are non-negative and skewed (yield curves,
//! mcap targets, etc).
//!
//! ## Worked example — quote + `execute_quote` + sell
//!
//! ```no_run
//! use deadeye_sdk::{
//!     core::{distribution::LognormalDistributionRaw, sq128::Sq128Raw},
//!     starknet::{
//!         Felt, JsonRpcProvider, LognormalMarketReader, LognormalMarketWriter, OwnedAccount,
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
//! let reader = LognormalMarketReader::new(&provider, market);
//! let signer = OwnedAccount::from_signing_key(
//!     JsonRpcClient::new(HttpTransport::new(
//!         "http://localhost:5050".parse::<url::Url>()?,
//!     )),
//!     Felt::ZERO,
//!     Felt::ZERO,
//!     Felt::ZERO,
//! );
//! let writer = LognormalMarketWriter::new(reader, signer);
//!
//! let candidate = LognormalDistributionRaw {
//!     mu: Sq128Raw::ZERO,
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
//! writer.execute_quote(quote).await?;
//! writer.sell_position(runtime, 0).await?;
//! # Ok(()) }
//! ```

use deadeye_collateral::{LognormalOptions, LognormalVerifiedMinimum, lognormal_collateral};
use deadeye_core::{
    LognormalDistribution, Sq128, distribution::LognormalDistributionRaw, sq128::Sq128Raw,
};
pub use deadeye_optimizer::LognormalOptimizationResult;
use deadeye_starknet::{
    Account, ExecutionReceipt, Felt, LognormalMarketReader, LognormalMarketWriter, Provider,
    types::lognormal::{
        LognormalSellExecutionGuardsRaw, LognormalSqrtHintsRaw, LognormalTradeInput,
    },
};
use futures::future::join_all;
use tracing::instrument;

use crate::{
    error::{SdkError, SdkResult},
    legs::{LegInfo, LegValuation, PositionLegs, PositionValuation, SettlementPoint, belief_grid},
};

/// Handle to a deployed lognormal AMM market.
#[derive(Debug)]
pub struct LognormalMarket<'p, P>
where
    P: Provider,
{
    reader: LognormalMarketReader<&'p P>,
}

impl<'p, P> LognormalMarket<'p, P>
where
    P: Provider,
{
    /// Construct a handle.
    pub fn new(provider: &'p P, address: Felt) -> Self {
        Self {
            reader: LognormalMarketReader::new(provider, address),
        }
    }

    /// Underlying read-only reader.
    pub const fn reader(&self) -> &LognormalMarketReader<&'p P> {
        &self.reader
    }

    /// Contract address.
    pub const fn address(&self) -> Felt {
        self.reader.address()
    }

    /// Reads the current market distribution.
    pub async fn distribution(&self) -> SdkResult<LognormalDistribution> {
        Ok(self.reader.distribution().await?)
    }

    /// Prepares an off-chain quote for moving the market from its current
    /// state to `candidate`.
    #[instrument(skip(self, candidate), fields(market = %self.reader.address()))]
    pub async fn prepare_quote(
        &self,
        candidate: &LognormalDistribution,
        opts: LognormalOptions,
    ) -> SdkResult<LognormalVerifiedMinimum> {
        let current = self.distribution().await?;
        Ok(lognormal_collateral(&current, candidate, opts)?)
    }

    /// EV-max trade under a **log-space** belief `N(μ, σ²)` and a budget —
    /// the lognormal twin of the normal market's `optimize_quote_offline_ev`.
    /// Fully client-side: one fetch of distribution + params + `lp_info`, then
    /// the grid optimizer with the audited Newton collateral minimiser per
    /// candidate. Returns the optimizer result (log-space candidate,
    /// collateral at the live effective k, audited x*, EV in tokens).
    ///
    /// # Errors
    /// Propagates provider/read failures; the optimizer itself is total and
    /// returns a no-trade result when nothing in the policy region has
    /// positive EV under the budget.
    #[instrument(skip(self), fields(market = %self.reader.address()))]
    pub async fn optimize_quote_offline_ev(
        &self,
        belief_mu: f64,
        belief_sigma: f64,
        budget_xp: f64,
    ) -> SdkResult<deadeye_optimizer::LognormalOptimizationResult> {
        let current = self.distribution().await?;
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let base_k = Sq128::from_raw(params.k);
        // Convention pin — identical to the normal market's
        // `optimize_quote_offline_ev` (see that doc comment): live pool over
        // immutable initial backing, floored at base k.
        let pool_backing = Sq128::from_raw(lp_info.total_backing_deposited);
        let initial_backing = Sq128::from_raw(params.backing);
        let effective_k =
            crate::normal::live_effective_k(base_k, pool_backing, initial_backing).to_f64();
        Ok(deadeye_optimizer::optimize_lognormal_trade(
            deadeye_optimizer::LognormalOptimizationInput::new(
                budget_xp,
                belief_mu,
                belief_sigma,
                current.mu().to_f64(),
                deadeye_core::Distribution::sigma(&current).to_f64(),
                effective_k,
            ),
        ))
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
    ///
    /// `settlement` is the lognormal scalar outcome `x*` in **log-space**
    /// (`μ_log` coordinates) — the same space the lognormal AMM stores its
    /// distribution in and the on-chain `value_at` view expects.
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

    /// Expected position value (P&L) under a **log-space** belief
    /// `N(`μ_log`, `σ_log`)`, integrating the on-chain leg value over the
    /// belief via a normal-pdf-weighted grid. Returns the expected P&L in
    /// XP.
    ///
    /// `belief_mean` / `belief_sigma` are the belief's parameters in
    /// **log-space** (`μ_log`, `σ_log`) — the same space the lognormal AMM
    /// stores its distribution in. The quadrature grid is built directly on
    /// those log-space coordinates and the on-chain `value_at` view is
    /// sampled at each node. A wide `σ_log` can push the outer grid nodes
    /// outside the on-chain feasible domain (the small log-domain that the
    /// side law can resolve); such a node's `value_at` read errors, and its
    /// contribution is treated as `0.0` (skipped) rather than failing the
    /// whole call. Costs `nodes × active legs` `value_at` reads (fanned out
    /// concurrently) — call when an agent wants its forward EV, not on
    /// every tick.
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
                    // A grid node outside the on-chain feasible domain makes
                    // `value_at` error (the small log-domain can't resolve the
                    // far node). Treat that node's contribution as 0 rather
                    // than failing the whole quadrature.
                    let v = match self.reader.trade_lot_value_at(lot_id, x_raw).await {
                        Ok(raw) => Sq128::from_raw(raw).to_f64(),
                        Err(_) => 0.0,
                    };
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

    /// Bind an account for writes.
    pub fn with_account<A>(self, account: A) -> LognormalMarketSigned<'p, P, A>
    where
        A: Account,
    {
        LognormalMarketSigned {
            writer: LognormalMarketWriter::new(self.reader, account),
        }
    }
}

/// Account-bound companion.
#[derive(Debug)]
pub struct LognormalMarketSigned<'p, P, A>
where
    P: Provider,
    A: Account,
{
    writer: LognormalMarketWriter<&'p P, A>,
}

impl<P, A> LognormalMarketSigned<'_, P, A>
where
    P: Provider,
    A: Account,
{
    /// Borrow the underlying writer.
    pub const fn writer(&self) -> &LognormalMarketWriter<&P, A> {
        &self.writer
    }

    /// Contract address.
    pub const fn address(&self) -> Felt {
        self.writer.reader().address()
    }

    /// Execute a previously-prepared trade (caller supplies a full
    /// [`LognormalTradeInput`] — typically constructed from the
    /// [`LognormalVerifiedMinimum`]).
    pub async fn execute_trade(&self, input: LognormalTradeInput) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.execute_trade(input).await?)
    }

    /// Submit a guarded sell. ABI:
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    pub async fn sell_position_guarded(
        &self,
        candidate: LognormalDistributionRaw,
        x_star: Sq128Raw,
        candidate_hints: LognormalSqrtHintsRaw,
        guards: &LognormalSellExecutionGuardsRaw,
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

    /// Add liquidity to the lognormal pool. ABI takes `share_amount` only.
    pub async fn add_liquidity(&self, share_amount: Sq128Raw) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.add_liquidity(share_amount).await?)
    }

    /// Remove a fraction of the caller's liquidity. ABI takes
    /// `share_amount` only.
    pub async fn remove_liquidity(&self, share_amount: Sq128Raw) -> SdkResult<ExecutionReceipt> {
        Ok(self.writer.remove_liquidity(share_amount).await?)
    }
}
