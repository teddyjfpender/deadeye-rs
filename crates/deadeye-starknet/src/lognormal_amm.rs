//! View + write clients for Deadeye's lognormal AMM.

use deadeye_core::{
    Distribution, LognormalDistribution, distribution::LognormalDistributionRaw, sq128::Sq128Raw,
};
use starknet_core::types::{Felt, FunctionCall};
use tracing::instrument;

use crate::{
    account::Account,
    cairo_serde::CairoSerde,
    error::{
        ContractError, ContractResult, TradeError, TradeRejectionReason, TradeResult,
        VerificationSubReason,
    },
    execution::{Call, ExecutionReceipt},
    provider::Provider,
    runtime::{check_lognormal_trade, compute_lognormal_hints},
    selectors::amm,
    types::{
        common::{AmmConfigRaw, AmmParamsRaw, FeeConfigRaw, LpInfoRaw, TradeRejection},
        lognormal::{
            LognormalDistributionCoreRaw, LognormalPositionCompactRaw,
            LognormalSellExecutionGuardsRaw, LognormalSqrtHintsRaw, LognormalTradeCheckRaw,
            LognormalTradeInput,
        },
        normal::PositionSummaryRaw,
    },
};

/// Typed reader for a lognormal AMM contract.
#[derive(Debug)]
pub struct LognormalMarketReader<P>
where
    P: Provider,
{
    provider: P,
    address: Felt,
}

impl<P> LognormalMarketReader<P>
where
    P: Provider,
{
    /// Bind a reader to a market address.
    pub const fn new(provider: P, address: Felt) -> Self {
        Self { provider, address }
    }

    /// Contract address.
    pub const fn address(&self) -> Felt {
        self.address
    }

    /// Borrow the provider.
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Reads the current lognormal distribution.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn distribution(&self) -> ContractResult<LognormalDistribution> {
        let raw = self
            .call_view::<LognormalDistributionRaw>("get_distribution", amm::get_distribution(), &[])
            .await?;
        LognormalDistribution::from_raw(raw).map_err(ContractError::Core)
    }

    /// Reads market status (initialised / paused / settled). Same view and
    /// wire shape as the normal AMM's `get_market_status`.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn market_status(&self) -> ContractResult<crate::normal_amm::MarketStatus> {
        self.call_view::<crate::normal_amm::MarketStatus>(
            "get_market_status",
            amm::get_market_status(),
            &[],
        )
        .await
    }

    /// Reads market parameters.
    pub async fn params(&self) -> ContractResult<AmmParamsRaw> {
        self.call_view::<AmmParamsRaw>("get_params", amm::get_params(), &[])
            .await
    }

    /// Reads the full configuration.
    pub async fn config(&self) -> ContractResult<AmmConfigRaw> {
        self.call_view::<AmmConfigRaw>("get_config", amm::get_config(), &[])
            .await
    }

    /// Reads fee configuration.
    pub async fn fee_config(&self) -> ContractResult<FeeConfigRaw> {
        self.call_view::<FeeConfigRaw>("get_fee_config", amm::get_fee_config(), &[])
            .await
    }

    /// Reads LP info.
    pub async fn lp_info(&self) -> ContractResult<LpInfoRaw> {
        self.call_view::<LpInfoRaw>("get_lp_info", amm::get_lp_info(), &[])
            .await
    }

    /// Reads the chain's canonical sqrt hints for the *current* distribution
    /// (`get_distribution_hints`) — the byte-exact values the verifier uses.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn distribution_hints(&self) -> ContractResult<LognormalSqrtHintsRaw> {
        self.call_view::<LognormalSqrtHintsRaw>(
            "get_distribution_hints",
            amm::get_distribution_hints(),
            &[],
        )
        .await
    }

    /// Reads the class hash of the math runtime this market library-calls
    /// (`get_runtime_class_hash`) — the class the chain-probe refinement
    /// deploys inside a gas-free simulation.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn runtime_class_hash(&self) -> ContractResult<Felt> {
        self.call_view::<Felt>("get_runtime_class_hash", amm::get_runtime_class_hash(), &[])
            .await
    }

    /// Reads a trader's position summary (shape shared with normal AMM).
    pub async fn position_summary(&self, trader: Felt) -> ContractResult<PositionSummaryRaw> {
        self.call_view::<PositionSummaryRaw>(
            "get_position_summary",
            amm::get_position_summary(),
            &[trader],
        )
        .await
    }

    /// Reads a trader's compact position.
    pub async fn position(&self, trader: Felt) -> ContractResult<LognormalPositionCompactRaw> {
        self.call_view::<LognormalPositionCompactRaw>(
            "get_position_compact",
            amm::get_position_compact(),
            &[trader],
        )
        .await
    }

    // ── Trade-lot (multi-leg) read surface ──────────────────────────────
    //
    // Each market-moving trade stores an explicit *trade lot* (leg). A
    // trader accumulates one lot per admitted delta; legs are valued and
    // settled independently. These low-level views mirror the on-chain lot
    // entrypoints; the SDK aggregates them into a `MultiLegPosition`.

    /// Number of trade lots (legs) this trader holds in the market.
    #[instrument(skip(self), fields(market = %self.address, %trader))]
    pub async fn trade_lot_count(&self, trader: Felt) -> ContractResult<u64> {
        self.call_view::<u64>(
            "get_trader_trade_lot_count",
            amm::get_trader_trade_lot_count(),
            &[trader],
        )
        .await
    }

    /// The `lot_id` at `index` in this trader's lot list (`index < count`).
    #[instrument(skip(self), fields(market = %self.address, %trader, index))]
    pub async fn trade_lot_id(&self, trader: Felt, index: u64) -> ContractResult<u64> {
        self.call_view::<u64>(
            "get_trader_trade_lot_id",
            amm::get_trader_trade_lot_id(),
            &[trader, Felt::from(index)],
        )
        .await
    }

    /// Whether a lot has already been settled (paid out).
    #[instrument(skip(self), fields(market = %self.address, lot_id))]
    pub async fn trade_lot_settled(&self, lot_id: u64) -> ContractResult<bool> {
        self.call_view::<bool>("get_trade_lot_settled", amm::get_trade_lot_settled(), &[
            Felt::from(lot_id),
        ])
        .await
    }

    /// Whether a lot was cancelled (terminalised without payout — collateral
    /// forfeit to the LP because it could not be valued at settlement).
    #[instrument(skip(self), fields(market = %self.address, lot_id))]
    pub async fn trade_lot_cancelled(&self, lot_id: u64) -> ContractResult<bool> {
        self.call_view::<bool>(
            "get_trade_lot_cancelled",
            amm::get_trade_lot_cancelled(),
            &[Felt::from(lot_id)],
        )
        .await
    }

    /// The lot's **signed** scoring-rule value at a settlement outcome `x*`
    /// (`to_λ·pdf(x*; to) − from_λ·pdf(x*; from)`), read authoritatively from
    /// the chain. This is the per-leg position value; gross payout for the leg
    /// is `collateral_locked + value_at(x*)`.
    #[instrument(skip(self, settlement), fields(market = %self.address, lot_id))]
    pub async fn trade_lot_value_at(
        &self,
        lot_id: u64,
        settlement: Sq128Raw,
    ) -> ContractResult<Sq128Raw> {
        let mut calldata = vec![Felt::from(lot_id)];
        settlement.encode(&mut calldata);
        self.call_view::<Sq128Raw>(
            "get_trade_lot_value_at",
            amm::get_trade_lot_value_at(),
            &calldata,
        )
        .await
    }

    /// Enumerate **all** of a trader's `lot_id`s (count → id-by-index). One
    /// RPC for the count plus one per lot; the SDK wraps this with concurrent
    /// fan-out for bulk valuation.
    #[instrument(skip(self), fields(market = %self.address, %trader))]
    pub async fn trade_lot_ids(&self, trader: Felt) -> ContractResult<Vec<u64>> {
        let count = self.trade_lot_count(trader).await?;
        let mut ids = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for index in 0..count {
            ids.push(self.trade_lot_id(trader, index).await?);
        }
        Ok(ids)
    }

    /// Preflight a lognormal trade against the current market — see
    /// [`crate::NormalMarketReader::quote_trade`] for the full contract.
    #[instrument(skip(self), fields(market = %self.address, runtime = %runtime))]
    pub async fn quote_trade(
        &self,
        runtime: Felt,
        candidate: LognormalDistributionRaw,
        x_star: Sq128Raw,
        supplied_collateral: Sq128Raw,
        collateral_pad: Sq128Raw,
    ) -> TradeResult<LognormalTradeQuote> {
        let current = self.distribution().await?.to_raw();
        let params = self.params().await?;
        let current_hints = compute_lognormal_hints(&self.provider, runtime, current).await?;
        let candidate_hints = compute_lognormal_hints(&self.provider, runtime, candidate).await?;
        let check = check_lognormal_trade(
            &self.provider,
            runtime,
            current,
            candidate,
            x_star,
            supplied_collateral,
            params.k,
            params.backing,
            params.tolerance,
            params.min_trade_collateral,
            current_hints,
            candidate_hints,
        )
        .await?;
        let rejection = map_lognormal_rejection(check.rejection_reason, &check);
        Ok(LognormalTradeQuote {
            candidate,
            candidate_hints,
            x_star,
            required_collateral: check.verification.computed_collateral,
            padded_collateral: max_sq_lognormal(supplied_collateral, collateral_pad),
            on_chain_will_accept: check.is_valid,
            rejection,
        })
    }

    /// Closed-form payout at `x_star` for a lognormal candidate.
    ///
    /// See [`crate::NormalMarketReader::payout_at`] for the contract.
    #[inline]
    #[must_use]
    pub fn payout_at(
        &self,
        x_star: f64,
        candidate: &deadeye_core::LognormalDistribution,
        k: f64,
    ) -> f64 {
        crate::pricing::payout_at_lognormal(candidate, k, x_star)
    }

    /// Estimate collateral for a μ_log-shift of `delta_mu`.
    ///
    /// See [`crate::NormalMarketReader::impact_for_mu_shift`] for the
    /// contract.
    pub async fn impact_for_mu_shift(
        &self,
        delta_mu: f64,
    ) -> ContractResult<crate::pricing::ImpactEstimate> {
        let current = self.distribution().await?;
        let mu_current = current.mu().to_f64();
        let sigma = current.sigma().to_f64();
        let target_mu = mu_current + delta_mu;
        let mu_q = deadeye_core::Sq128::from_f64(target_mu).map_err(ContractError::Core)?;
        let sigma_q = deadeye_core::Sq128::from_f64(sigma).map_err(ContractError::Core)?;
        let candidate = deadeye_core::LognormalDistribution::from_sigma(mu_q, sigma_q)
            .map_err(ContractError::Core)?;
        let solver = deadeye_collateral::lognormal_collateral(
            &current,
            &candidate,
            deadeye_collateral::LognormalOptions::default(),
        )
        .map_err(|e| ContractError::InvalidResponse {
            call: "impact_for_mu_shift",
            message: format!("lognormal solver: {e}"),
        })?;
        Ok(crate::pricing::ImpactEstimate {
            delta_mu,
            x_star: solver.x_star,
            required_collateral: solver.collateral,
            iterations: solver.iterations,
        })
    }

    /// Numerical greeks for `candidate` at `x_star`.
    #[inline]
    #[must_use]
    pub fn sensitivities_at(
        &self,
        candidate: &deadeye_core::LognormalDistribution,
        x_star: f64,
        k: f64,
    ) -> crate::pricing::LognormalSensitivities {
        crate::pricing::sensitivities_lognormal(candidate, k, x_star)
    }

    async fn call_view<T>(
        &self,
        call_name: &'static str,
        selector: Felt,
        calldata: &[Felt],
    ) -> ContractResult<T>
    where
        T: CairoSerde,
    {
        let response = self
            .provider
            .call(
                FunctionCall {
                    contract_address: self.address,
                    entry_point_selector: selector,
                    calldata: calldata.to_vec(),
                },
                self.provider.default_block(),
            )
            .await?;
        let (value, rest) = T::decode(&response)?;
        if !rest.is_empty() {
            return Err(ContractError::UnexpectedReturnSize {
                call: call_name,
                actual: response.len(),
                expected: response.len() - rest.len(),
            });
        }
        Ok(value)
    }
}

/// Preflighted lognormal-AMM trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LognormalTradeQuote {
    /// Candidate distribution.
    pub candidate: LognormalDistributionRaw,
    /// Chain-correct sqrt hints.
    pub candidate_hints: LognormalSqrtHintsRaw,
    /// `x_star` for the minimum.
    pub x_star: Sq128Raw,
    /// Chain-computed required collateral.
    pub required_collateral: Sq128Raw,
    /// Padded supplied collateral.
    pub padded_collateral: Sq128Raw,
    /// Verdict from `check_trade_view`.
    pub on_chain_will_accept: bool,
    /// Typed rejection reason if not accepted.
    pub rejection: Option<TradeRejectionReason>,
}

fn max_sq_lognormal(a: Sq128Raw, b: Sq128Raw) -> Sq128Raw {
    let lhs = deadeye_core::Sq128::from_raw(a);
    let rhs = deadeye_core::Sq128::from_raw(b);
    if lhs.cmp_signed(rhs) == core::cmp::Ordering::Less {
        b
    } else {
        a
    }
}

fn map_lognormal_rejection(
    raw: TradeRejection,
    check: &LognormalTradeCheckRaw,
) -> Option<TradeRejectionReason> {
    match raw {
        TradeRejection::None => None,
        TradeRejection::InvalidDistribution => Some(TradeRejectionReason::InvalidDistribution),
        TradeRejection::InvalidHints => Some(TradeRejectionReason::InvalidHints),
        TradeRejection::BackingFail => Some(TradeRejectionReason::BackingFail),
        TradeRejection::SigmaTooLow => Some(TradeRejectionReason::SigmaTooLow),
        TradeRejection::LowCollateral => Some(TradeRejectionReason::LowCollateral),
        TradeRejection::VerificationFailed => {
            let sub = if !check.verification.side_valid {
                Some(VerificationSubReason::SideInvalid)
            } else if !check.verification.stationary_valid {
                Some(VerificationSubReason::StationaryInvalid)
            } else if !check.verification.curvature_valid {
                Some(VerificationSubReason::CurvatureInvalid)
            } else if !check.verification.collateral_sufficient {
                Some(VerificationSubReason::CollateralInsufficient)
            } else {
                None
            };
            Some(TradeRejectionReason::VerificationFailed { sub_reason: sub })
        },
    }
}

/// Write-capable companion.
#[derive(Debug)]
pub struct LognormalMarketWriter<P, A>
where
    P: Provider,
    A: Account,
{
    reader: LognormalMarketReader<P>,
    account: A,
}

impl<P, A> LognormalMarketWriter<P, A>
where
    P: Provider,
    A: Account,
{
    /// Pair a reader with an account.
    pub const fn new(reader: LognormalMarketReader<P>, account: A) -> Self {
        Self { reader, account }
    }

    /// Borrow the reader.
    pub const fn reader(&self) -> &LognormalMarketReader<P> {
        &self.reader
    }

    /// Borrow the account.
    pub const fn account(&self) -> &A {
        &self.account
    }

    /// Build a Call for `execute_trade`.
    pub fn build_trade_call(&self, input: LognormalTradeInput) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::execute_trade(),
            calldata: input.to_calldata(),
        }
    }

    /// Build a Call for `sell_position_guarded`.
    ///
    /// ABI signature (`lognormal_amm.abi.json`):
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    /// All four fields are encoded into the calldata in that order.
    pub fn build_sell_call(
        &self,
        candidate: LognormalDistributionRaw,
        x_star: deadeye_core::sq128::Sq128Raw,
        candidate_hints: LognormalSqrtHintsRaw,
        guards: &LognormalSellExecutionGuardsRaw,
    ) -> Call {
        let mut calldata = Vec::with_capacity(32);
        candidate.encode(&mut calldata);
        x_star.encode(&mut calldata);
        candidate_hints.encode(&mut calldata);
        guards.encode(&mut calldata);
        Call {
            to: self.reader.address(),
            selector: amm::sell_position_guarded(),
            calldata,
        }
    }

    /// Build a Call for `claim`.
    pub fn build_claim_call(&self) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::claim(),
            calldata: Vec::new(),
        }
    }

    /// Build a Call for `claim_for(trader)`.
    pub fn build_claim_for_call(&self, trader: Felt) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::claim_for(),
            calldata: vec![trader],
        }
    }

    /// Build the [`Call`] for `add_liquidity(share_amount)`.
    ///
    /// The lognormal ABI takes only `share_amount`. Starknet rejects
    /// trailing calldata, so nothing else is appended.
    pub fn build_add_liquidity_call(&self, share_amount: deadeye_core::sq128::Sq128Raw) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::add_liquidity(),
            calldata: share_amount.to_calldata(),
        }
    }

    /// Build the [`Call`] for `remove_liquidity(share_amount)`.
    ///
    /// Same shape as [`Self::build_add_liquidity_call`].
    pub fn build_remove_liquidity_call(&self, share_amount: deadeye_core::sq128::Sq128Raw) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::remove_liquidity(),
            calldata: share_amount.to_calldata(),
        }
    }

    /// Execute a trade.
    #[instrument(skip(self, input), fields(market = %self.reader.address(), family = "lognormal", kind = "trade"))]
    pub async fn execute_trade(
        &self,
        input: LognormalTradeInput,
    ) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_trade_call(input)])
            .await
    }

    /// Submit a guarded sell. Mirrors the ABI:
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    #[instrument(skip(self, candidate, x_star, candidate_hints, guards), fields(market = %self.reader.address(), family = "lognormal", kind = "sell"))]
    pub async fn sell_position_guarded(
        &self,
        candidate: LognormalDistributionRaw,
        x_star: deadeye_core::sq128::Sq128Raw,
        candidate_hints: LognormalSqrtHintsRaw,
        guards: &LognormalSellExecutionGuardsRaw,
    ) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_sell_call(
                candidate,
                x_star,
                candidate_hints,
                guards,
            )])
            .await
    }

    /// Claim the caller's settled position.
    #[instrument(skip(self), fields(market = %self.reader.address(), family = "lognormal", kind = "claim"))]
    pub async fn claim(&self) -> ContractResult<ExecutionReceipt> {
        self.account.execute(vec![self.build_claim_call()]).await
    }

    /// Claim a settled position on behalf of `trader`.
    #[instrument(skip(self), fields(market = %self.reader.address(), family = "lognormal", kind = "claim_for", %trader))]
    pub async fn claim_for(&self, trader: Felt) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_claim_for_call(trader)])
            .await
    }

    /// Add liquidity to the pool. ABI takes a single `share_amount`.
    #[instrument(skip(self, share_amount), fields(market = %self.reader.address(), family = "lognormal", kind = "lp_add"))]
    pub async fn add_liquidity(
        &self,
        share_amount: deadeye_core::sq128::Sq128Raw,
    ) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_add_liquidity_call(share_amount)])
            .await
    }

    /// Remove a fraction of the caller's liquidity. ABI takes a single
    /// `share_amount`.
    #[instrument(skip(self, share_amount), fields(market = %self.reader.address(), family = "lognormal", kind = "lp_remove"))]
    pub async fn remove_liquidity(
        &self,
        share_amount: deadeye_core::sq128::Sq128Raw,
    ) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_remove_liquidity_call(share_amount)])
            .await
    }

    /// Build the atomic `[approve, trade]` multicall a quote submits, without
    /// sending it. See [`crate::NormalMarketWriter::build_trade_calls`].
    pub async fn build_trade_calls(&self, quote: &LognormalTradeQuote) -> TradeResult<Vec<Call>> {
        if !quote.on_chain_will_accept {
            return Err(TradeError::Rejected {
                reason: quote.rejection.unwrap_or(TradeRejectionReason::Other {
                    raw: "quote not acceptable on-chain",
                }),
                source: ContractError::InvalidResponse {
                    call: "execute_quote",
                    message: "quote.on_chain_will_accept = false".into(),
                },
            });
        }
        // Bundle the ERC-20 collateral `approve` into the same atomic
        // multicall as the trade so `transfer_from` succeeds (issue #13).
        let config = self
            .reader
            .config()
            .await
            .map_err(TradeError::from_contract)?;
        let human = deadeye_core::Sq128::from_raw(quote.padded_collateral).to_f64();
        // 5% allowance margin (matches the webapp's approve buffer).
        let amount =
            crate::collateral::collateral_allowance_base_units(human, config.token_decimals, 5);
        let approve = crate::collateral::build_erc20_approve_call(
            config.collateral_token,
            self.reader.address(),
            amount,
        );
        let input = LognormalTradeInput {
            candidate: quote.candidate,
            x_star: quote.x_star,
            supplied_collateral: quote.padded_collateral,
            candidate_hints: quote.candidate_hints,
        };
        Ok(vec![approve, self.build_trade_call(input)])
    }

    /// Submit a quote previously prepared by
    /// [`LognormalMarketReader::quote_trade`].
    #[instrument(skip(self, quote), fields(market = %self.reader.address(), family = "lognormal", kind = "trade", accepts = quote.on_chain_will_accept))]
    pub async fn execute_quote(&self, quote: LognormalTradeQuote) -> TradeResult<ExecutionReceipt> {
        self.execute_quote_bundled(quote, Vec::new()).await
    }

    /// [`Self::execute_quote`] with extra `leading` calls prepended to the
    /// `[approve, trade]` multicall — e.g. a `claim_initial_grant()` that
    /// bootstraps a fresh wallet atomically with its first trade. The same
    /// simulate-before-submit gate applies to the full bundle.
    #[instrument(skip(self, quote, leading), fields(market = %self.reader.address(), family = "lognormal", kind = "trade", leading = leading.len()))]
    pub async fn execute_quote_bundled(
        &self,
        quote: LognormalTradeQuote,
        leading: Vec<Call>,
    ) -> TradeResult<ExecutionReceipt> {
        let mut calls = leading;
        calls.extend(self.build_trade_calls(&quote).await?);
        // Gas-free pre-flight: catch a reverting trade before it burns a fee.
        if let Some(sim) = self
            .account
            .simulate(&calls)
            .await
            .map_err(TradeError::from_contract)?
            && let Some(reason) = sim.revert_reason
        {
            return Err(TradeError::Rejected {
                reason: TradeRejectionReason::Other {
                    raw: "on-chain simulation reverted",
                },
                source: ContractError::InvalidResponse {
                    call: "execute_trade(simulated)",
                    message: reason,
                },
            });
        }
        self.account
            .execute(calls)
            .await
            .map_err(TradeError::from_contract)
    }

    /// Close out the caller's full lognormal position. See
    /// [`crate::NormalMarketWriter::sell_position`] for the contract.
    pub async fn sell_position(
        &self,
        runtime: Felt,
        min_token_out: u128,
    ) -> TradeResult<ExecutionReceipt> {
        let market_dist = self.reader.distribution().await?.to_raw();
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let candidate_hints =
            compute_lognormal_hints(self.reader.provider(), runtime, market_dist).await?;
        let guards = LognormalSellExecutionGuardsRaw {
            expected_market_dist: LognormalDistributionCoreRaw {
                mu: market_dist.mu,
                variance: market_dist.variance,
            },
            expected_backing: lp_info.total_backing_deposited,
            expected_tolerance: params.tolerance,
            expected_min_trade_collateral: params.min_trade_collateral,
            min_token_out,
        };
        let call = self.build_sell_call(market_dist, market_dist.mu, candidate_hints, &guards);
        self.account
            .execute(vec![call])
            .await
            .map_err(TradeError::from_contract)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use std::{sync::Mutex, vec::Vec};

    use async_trait::async_trait;
    use deadeye_core::{LognormalDistribution, Sq128};
    use starknet_core::types::{BlockId, FunctionCall};

    use super::*;

    struct CannedProvider {
        responses: Mutex<Vec<Vec<Felt>>>,
    }

    #[async_trait]
    impl Provider for CannedProvider {
        async fn call(&self, _call: FunctionCall, _block: BlockId) -> ContractResult<Vec<Felt>> {
            self.responses
                .lock()
                .expect("mutex")
                .pop()
                .ok_or_else(|| ContractError::Provider("no canned response".into()))
        }
    }

    fn lognormal_dist(mu: f64, var: f64) -> LognormalDistribution {
        LognormalDistribution::from_variance(
            Sq128::from_f64(mu).unwrap(),
            Sq128::from_f64(var).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn payout_at_returns_finite_positive() {
        let provider = CannedProvider {
            responses: Mutex::new(Vec::new()),
        };
        let reader = LognormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let dist = lognormal_dist(0.0, 1.0);
        let p = reader.payout_at(1.0, &dist, 1.0);
        assert!(p > 0.0 && p.is_finite(), "got {p}");
    }

    #[test]
    fn sensitivities_at_finite() {
        let provider = CannedProvider {
            responses: Mutex::new(Vec::new()),
        };
        let reader = LognormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let dist = lognormal_dist(0.0, 1.0);
        let s = reader.sensitivities_at(&dist, 1.5, 1.0);
        assert!(s.d_payout_d_mu.is_finite());
        assert!(s.d_payout_d_sigma.is_finite());
    }

    #[tokio::test]
    async fn impact_for_mu_shift_returns_collateral() {
        // Canned: provide distribution (LN(0,1)) then params; impact_for_mu_shift
        // reads dist once (no params() in lognormal version, but let's be safe).
        let raw = LognormalDistributionRaw {
            mu: Sq128::ZERO.to_raw(),
            variance: Sq128::from_f64(1.0).unwrap().to_raw(),
            sigma: Sq128::from_f64(1.0).unwrap().to_raw(),
        };
        let provider = CannedProvider {
            responses: Mutex::new(vec![raw.to_calldata()]),
        };
        let reader = LognormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let estimate = reader.impact_for_mu_shift(0.2_f64).await.unwrap();
        assert!(estimate.required_collateral >= 0.0, "got {estimate:?}");
    }
}
