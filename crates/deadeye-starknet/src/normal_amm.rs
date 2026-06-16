//! View + write clients for Deadeye's normal (Gaussian) AMM contract.

use deadeye_core::{
    Distribution,
    distribution::{NormalDistribution, NormalDistributionRaw, NormalSqrtHintsRaw},
    sq128::Sq128Raw,
};
use starknet_core::types::{Felt, FunctionCall};
use tracing::instrument;

use crate::{
    account::Account,
    cairo_serde::CairoSerde,
    error::{ContractError, ContractResult, TradeError, TradeRejectionReason, TradeResult},
    execution::{Call, ExecutionReceipt},
    provider::Provider,
    runtime::compute_normal_hints,
    selectors::amm,
    types::{
        common::{AmmConfigRaw, AmmParamsRaw, FeeConfigRaw, LpInfoRaw, TradeRejection},
        normal::{
            PositionCompactRaw, PositionSummaryRaw, SellExecutionGuardsRaw, TradeCheckRaw,
            TradeInput,
        },
    },
};

/// Typed read accessors for a deployed normal AMM contract.
#[derive(Debug)]
pub struct NormalMarketReader<P>
where
    P: Provider,
{
    provider: P,
    address: Felt,
}

impl<P> NormalMarketReader<P>
where
    P: Provider,
{
    /// Construct a reader bound to a specific market address.
    pub const fn new(provider: P, address: Felt) -> Self {
        Self { provider, address }
    }

    /// Contract address this reader targets.
    pub const fn address(&self) -> Felt {
        self.address
    }

    /// Borrow the underlying [`Provider`].
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Reads the current market distribution.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn distribution(&self) -> ContractResult<NormalDistribution> {
        let raw = self
            .call_view::<NormalDistributionRaw>("get_distribution", amm::get_distribution(), &[])
            .await?;
        NormalDistribution::from_raw(raw).map_err(ContractError::Core)
    }

    /// Reads market status (initialised / paused / settled).
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn market_status(&self) -> ContractResult<MarketStatus> {
        self.call_view::<MarketStatus>("get_market_status", amm::get_market_status(), &[])
            .await
    }

    /// Reads the AMM parameters (k, backing, tolerance, …).
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn params(&self) -> ContractResult<AmmParamsRaw> {
        self.call_view::<AmmParamsRaw>("get_params", amm::get_params(), &[])
            .await
    }

    /// Reads the full AMM configuration (params + collateral token + decimals).
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn config(&self) -> ContractResult<AmmConfigRaw> {
        self.call_view::<AmmConfigRaw>("get_config", amm::get_config(), &[])
            .await
    }

    /// Reads the fee configuration.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn fee_config(&self) -> ContractResult<FeeConfigRaw> {
        self.call_view::<FeeConfigRaw>("get_fee_config", amm::get_fee_config(), &[])
            .await
    }

    /// Reads the chain's **canonical** sqrt hints for the *current* market
    /// distribution (`get_distribution_hints`).
    ///
    /// These are the byte-exact `(l2_norm_denom, backing_denom)` the on-chain
    /// verifier derives for the live σ. Comparing them against the off-chain
    /// [`crate::runtime::compute_normal_hints`] / SDK offline hints is the
    /// ground-truth parity check for a `VERIFICATION_FAILED` trade revert.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn distribution_hints(&self) -> ContractResult<NormalSqrtHintsRaw> {
        self.call_view::<NormalSqrtHintsRaw>(
            "get_distribution_hints",
            amm::get_distribution_hints(),
            &[],
        )
        .await
    }

    /// Reads the class hash of the math runtime this market library-calls for
    /// trade verification (`get_runtime_class_hash`).
    ///
    /// The chain-probe refinement deploys exactly this class inside a gas-free
    /// simulation so the `check_trade_view` verdicts it reads are guaranteed to
    /// match what `execute_trade` will enforce.
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn runtime_class_hash(&self) -> ContractResult<Felt> {
        self.call_view::<Felt>("get_runtime_class_hash", amm::get_runtime_class_hash(), &[])
            .await
    }

    /// Reads the LP pool info (total shares + total backing).
    #[instrument(skip(self), fields(market = %self.address))]
    pub async fn lp_info(&self) -> ContractResult<LpInfoRaw> {
        self.call_view::<LpInfoRaw>("get_lp_info", amm::get_lp_info(), &[])
            .await
    }

    /// Reads a trader's position summary.
    #[instrument(skip(self), fields(market = %self.address, %trader))]
    pub async fn position_summary(&self, trader: Felt) -> ContractResult<PositionSummaryRaw> {
        self.call_view::<PositionSummaryRaw>(
            "get_position_summary",
            amm::get_position_summary(),
            &[trader],
        )
        .await
    }

    /// Reads a trader's compact position record.
    ///
    /// NOTE: the live contract (trade-lot model) no longer exposes
    /// `get_position_compact`; a trader now holds one or more *trade lots*.
    /// Use [`Self::trade_lot_ids`] / [`Self::trade_lot_value_at`] /
    /// [`Self::position_summary`] for the current model. This method remains
    /// for older deployments that still telescope to a single compact position.
    #[instrument(skip(self), fields(market = %self.address, %trader))]
    pub async fn position(&self, trader: Felt) -> ContractResult<PositionCompactRaw> {
        self.call_view::<PositionCompactRaw>(
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

    /// Quote a candidate trade against the **current** market.
    ///
    /// This is the preflight a production market-maker takes before
    /// every submission. The reader:
    /// 1. Reads the current distribution + AMM params.
    /// 2. Fetches chain-correct sqrt hints for `candidate` from `runtime`'s
    ///    `compute_hints_view`.
    /// 3. Calls `check_trade_view` to obtain the chain's verdict (`is_valid`,
    ///    `rejection_reason`, computed collateral).
    ///
    /// The returned [`NormalTradeQuote`] carries everything
    /// [`NormalMarketWriter::execute_quote`] needs to submit the trade —
    /// no further chain round-trips required.
    #[instrument(skip(self), fields(market = %self.address, runtime = %runtime))]
    pub async fn quote_trade(
        &self,
        runtime: Felt,
        candidate: NormalDistributionRaw,
        x_star: Sq128Raw,
        supplied_collateral: Sq128Raw,
        collateral_pad: Sq128Raw,
    ) -> TradeResult<NormalTradeQuote> {
        let current = self.distribution().await?.to_raw();
        let params = self.params().await?;
        let current_hints = compute_normal_hints(&self.provider, runtime, current).await?;
        let candidate_hints = compute_normal_hints(&self.provider, runtime, candidate).await?;
        let check = crate::runtime::check_normal_trade(
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
        let rejection = map_trade_rejection(check.rejection_reason, &check);
        Ok(NormalTradeQuote {
            candidate,
            candidate_hints,
            x_star,
            required_collateral: check.verification.computed_collateral,
            padded_collateral: max_sq(supplied_collateral, collateral_pad),
            on_chain_will_accept: check.is_valid,
            rejection,
        })
    }

    /// Closed-form payout at observation point `x_star` for the given
    /// candidate distribution.
    ///
    /// Returns `λ · pdf(x*; μ_eff, σ_eff)` in f64 (no chain round-trip,
    /// no Sq128 arithmetic — this is the hot-path pricing call). The
    /// caller supplies `k` so the same reader can re-price the same
    /// distribution at multiple `k` values without re-fetching params.
    ///
    /// Pair with [`Self::params`] to fetch the live `k` once, then
    /// price many candidates against it.
    #[inline]
    #[must_use]
    pub fn payout_at(
        &self,
        x_star: f64,
        candidate: &deadeye_core::NormalDistribution,
        k: f64,
    ) -> f64 {
        crate::pricing::payout_at_normal(candidate, k, x_star)
    }

    /// Estimate the collateral required to shift the market mean by
    /// `delta_mu`, keeping σ constant.
    ///
    /// Reads current distribution + params, builds the candidate
    /// `(μ_current + Δ, σ_current)`, and runs the off-chain
    /// [`deadeye_collateral::normal_collateral`] solver. Returns the
    /// `x*` and the implied collateral — exactly what a strategy needs
    /// to decide whether the move is in budget.
    ///
    /// This is f64 throughout; the on-chain verifier re-checks the
    /// result with Q128.128 arithmetic at trade submission time.
    pub async fn impact_for_mu_shift(
        &self,
        delta_mu: f64,
    ) -> ContractResult<crate::pricing::ImpactEstimate> {
        let current = self.distribution().await?;
        let params = self.params().await?;
        let mu_current = current.mean().to_f64();
        let sigma = current.sigma().to_f64();
        let target_mu = mu_current + delta_mu;
        let mu_q = deadeye_core::Sq128::from_f64(target_mu).map_err(ContractError::Core)?;
        let sigma_q = deadeye_core::Sq128::from_f64(sigma).map_err(ContractError::Core)?;
        let candidate = deadeye_core::NormalDistribution::from_sigma(mu_q, sigma_q)
            .map_err(ContractError::Core)?;
        let _ = params; // params kept available for callers via reader.params().
        let solver = deadeye_collateral::normal_collateral(
            &current,
            &candidate,
            deadeye_collateral::MinimizationPolicy::standard(),
        )
        .map_err(|e| ContractError::InvalidResponse {
            call: "impact_for_mu_shift",
            message: format!("collateral solver: {e}"),
        })?;
        Ok(crate::pricing::ImpactEstimate {
            delta_mu,
            x_star: solver.x_min,
            required_collateral: solver.collateral,
            iterations: solver.iterations,
        })
    }

    /// Numerical greeks for `candidate` at `x_star`.
    ///
    /// Pure off-chain; safe to call inside a tight loop (every greek is
    /// a constant number of [`Self::payout_at`] evaluations).
    #[inline]
    #[must_use]
    pub fn sensitivities_at(
        &self,
        candidate: &deadeye_core::NormalDistribution,
        x_star: f64,
        k: f64,
    ) -> crate::pricing::NormalSensitivities {
        crate::pricing::sensitivities_normal(candidate, k, x_star)
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
        tracing::debug!(
            target: "deadeye::rpc",
            contract = %self.address,
            entrypoint = call_name,
            calldata_len = calldata.len(),
            "starknet_call",
        );
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
        tracing::debug!(
            target: "deadeye::rpc",
            entrypoint = call_name,
            felts_returned = response.len(),
            "starknet_call returned",
        );
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

/// A preflighted normal-AMM trade.
///
/// Carries the chain-correct hints + `x_star` + collateral the writer
/// will submit. Construct via [`NormalMarketReader::quote_trade`]; execute
/// via [`NormalMarketWriter::execute_quote`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormalTradeQuote {
    /// Candidate distribution the MM wants to move the market to.
    pub candidate: NormalDistributionRaw,
    /// Chain-correct sqrt hints for `candidate`, ready for submission.
    pub candidate_hints: NormalSqrtHintsRaw,
    /// `x_star` the off-chain solver (or the caller) picked.
    pub x_star: Sq128Raw,
    /// Collateral the chain computed as required for this trade.
    pub required_collateral: Sq128Raw,
    /// Collateral the writer should actually supply (≥ required, with
    /// pad).
    pub padded_collateral: Sq128Raw,
    /// `true` iff the chain's `check_trade_view` accepted the trade.
    pub on_chain_will_accept: bool,
    /// Typed rejection reason if `!on_chain_will_accept`.
    pub rejection: Option<TradeRejectionReason>,
}

/// Pick the larger of two `Sq128Raw` values via signed comparison.
fn max_sq(a: Sq128Raw, b: Sq128Raw) -> Sq128Raw {
    let lhs = deadeye_core::Sq128::from_raw(a);
    let rhs = deadeye_core::Sq128::from_raw(b);
    if lhs.cmp_signed(rhs) == core::cmp::Ordering::Less {
        b
    } else {
        a
    }
}

fn map_trade_rejection(raw: TradeRejection, check: &TradeCheckRaw) -> Option<TradeRejectionReason> {
    use crate::error::VerificationSubReason as Sub;
    match raw {
        TradeRejection::None => None,
        TradeRejection::InvalidDistribution => Some(TradeRejectionReason::InvalidDistribution),
        TradeRejection::InvalidHints => Some(TradeRejectionReason::InvalidHints),
        TradeRejection::BackingFail => Some(TradeRejectionReason::BackingFail),
        TradeRejection::SigmaTooLow => Some(TradeRejectionReason::SigmaTooLow),
        TradeRejection::LowCollateral => Some(TradeRejectionReason::LowCollateral),
        TradeRejection::VerificationFailed => {
            // Drill into the embedded verification flags to recover the
            // refined sub-reason without round-tripping a revert string.
            let sub = if !check.verification.side_valid {
                Some(Sub::SideInvalid)
            } else if !check.verification.stationary_valid {
                Some(Sub::StationaryInvalid)
            } else if !check.verification.curvature_valid {
                Some(Sub::CurvatureInvalid)
            } else if !check.verification.collateral_sufficient {
                Some(Sub::CollateralInsufficient)
            } else {
                None
            };
            Some(TradeRejectionReason::VerificationFailed { sub_reason: sub })
        },
    }
}

/// Decoded market status (`get_market_status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketStatus {
    /// Whether `initialize()` has been called.
    pub is_initialised: bool,
    /// Whether the market is currently paused.
    pub is_paused: bool,
    /// Whether the market has been settled.
    pub is_settled: bool,
    /// Settlement value (only meaningful if `is_settled`).
    pub settlement_value: deadeye_core::sq128::Sq128Raw,
}

impl CairoSerde for MarketStatus {
    fn encode(&self, out: &mut Vec<Felt>) {
        self.is_initialised.encode(out);
        self.is_paused.encode(out);
        self.is_settled.encode(out);
        self.settlement_value.encode(out);
    }
    fn decode(slice: &[Felt]) -> Result<(Self, &[Felt]), crate::cairo_serde::CairoSerdeError> {
        let (is_initialised, slice) = bool::decode(slice)?;
        let (is_paused, slice) = bool::decode(slice)?;
        let (is_settled, slice) = bool::decode(slice)?;
        let (settlement_value, slice) = deadeye_core::sq128::Sq128Raw::decode(slice)?;
        Ok((
            Self {
                is_initialised,
                is_paused,
                is_settled,
                settlement_value,
            },
            slice,
        ))
    }
}

// ─── Writer ──────────────────────────────────────────────────────────────────

/// Write-capable companion to [`NormalMarketReader`]. Pairs the reader with
/// an [`Account`] so the same handle can run pre-flight reads, build
/// calldata, and submit the trade in one place.
#[derive(Debug)]
pub struct NormalMarketWriter<P, A>
where
    P: Provider,
    A: Account,
{
    reader: NormalMarketReader<P>,
    account: A,
}

impl<P, A> NormalMarketWriter<P, A>
where
    P: Provider,
    A: Account,
{
    /// Construct a writer from a reader and an account.
    pub const fn new(reader: NormalMarketReader<P>, account: A) -> Self {
        Self { reader, account }
    }

    /// Borrow the underlying reader.
    pub const fn reader(&self) -> &NormalMarketReader<P> {
        &self.reader
    }

    /// Borrow the underlying account.
    pub const fn account(&self) -> &A {
        &self.account
    }

    /// Build the [`Call`] for an `execute_trade` invocation without submitting.
    /// Useful when the caller wants to bundle additional calls (e.g. an
    /// ERC20 `approve`) into a single multi-call transaction.
    pub fn build_trade_call(&self, input: TradeInput) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::execute_trade(),
            calldata: input.to_calldata(),
        }
    }

    /// Build the [`Call`] for `sell_position_guarded`.
    ///
    /// On-chain signature (per `normal_amm.abi.json`):
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    /// All four are forwarded as concatenated calldata in that order.
    pub fn build_sell_call(
        &self,
        candidate: NormalDistributionRaw,
        x_star: deadeye_core::sq128::Sq128Raw,
        candidate_hints: deadeye_core::distribution::NormalSqrtHintsRaw,
        guards: SellExecutionGuardsRaw,
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

    /// Build the [`Call`] for `claim` (the caller's own position).
    pub fn build_claim_call(&self) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::claim(),
            calldata: Vec::new(),
        }
    }

    /// Build the [`Call`] for `claim_for(trader)`.
    pub fn build_claim_for_call(&self, trader: Felt) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::claim_for(),
            calldata: vec![trader],
        }
    }

    /// Build the [`Call`] for `add_liquidity(share_amount)`.
    ///
    /// The ABI declares only `share_amount`; nothing else is encoded into the
    /// calldata. Starknet rejects extra calldata, so the prior "trail the
    /// hints" shape would always revert.
    pub fn build_add_liquidity_call(&self, share_amount: deadeye_core::sq128::Sq128Raw) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::add_liquidity(),
            calldata: share_amount.to_calldata(),
        }
    }

    /// Build the [`Call`] for `remove_liquidity(share_amount)`.
    ///
    /// The ABI declares only `share_amount`; see
    /// [`Self::build_add_liquidity_call`].
    pub fn build_remove_liquidity_call(&self, share_amount: deadeye_core::sq128::Sq128Raw) -> Call {
        Call {
            to: self.reader.address(),
            selector: amm::remove_liquidity(),
            calldata: share_amount.to_calldata(),
        }
    }

    /// Execute a trade.
    #[instrument(skip(self, input), fields(market = %self.reader.address(), family = "normal", kind = "trade"))]
    pub async fn execute_trade(&self, input: TradeInput) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_trade_call(input)])
            .await
    }

    /// Execute a guarded sell. The four arguments mirror the ABI:
    /// `sell_position_guarded(candidate, x_star, candidate_hints, guards)`.
    #[instrument(skip(self, candidate, x_star, candidate_hints, guards), fields(market = %self.reader.address(), family = "normal", kind = "sell"))]
    pub async fn sell_position_guarded(
        &self,
        candidate: NormalDistributionRaw,
        x_star: deadeye_core::sq128::Sq128Raw,
        candidate_hints: deadeye_core::distribution::NormalSqrtHintsRaw,
        guards: SellExecutionGuardsRaw,
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
    #[instrument(skip(self), fields(market = %self.reader.address(), family = "normal", kind = "claim"))]
    pub async fn claim(&self) -> ContractResult<ExecutionReceipt> {
        self.account.execute(vec![self.build_claim_call()]).await
    }

    /// Claim a settled position on behalf of `trader`.
    #[instrument(skip(self), fields(market = %self.reader.address(), family = "normal", kind = "claim_for", %trader))]
    pub async fn claim_for(&self, trader: Felt) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_claim_for_call(trader)])
            .await
    }

    /// Add liquidity to the pool. ABI takes a single `share_amount`.
    #[instrument(skip(self, share_amount), fields(market = %self.reader.address(), family = "normal", kind = "lp_add"))]
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
    #[instrument(skip(self, share_amount), fields(market = %self.reader.address(), family = "normal", kind = "lp_remove"))]
    pub async fn remove_liquidity(
        &self,
        share_amount: deadeye_core::sq128::Sq128Raw,
    ) -> ContractResult<ExecutionReceipt> {
        self.account
            .execute(vec![self.build_remove_liquidity_call(share_amount)])
            .await
    }

    /// Submit a quote previously prepared by
    /// [`NormalMarketReader::quote_trade`].
    ///
    /// Returns [`TradeError::Submission`] if the writer refuses the quote
    /// (`on_chain_will_accept = false`) or if the on-chain call reverts;
    /// reverts get promoted into a typed [`TradeError::Rejected`] arm so
    /// callers can branch on `TradeRejectionReason`.
    /// Build the atomic `[approve, trade]` multicall a quote submits, without
    /// sending it.
    ///
    /// Reads the market's collateral config (token address + decimals) to size
    /// an ERC-20 `approve` of the collateral token to the market — the leading
    /// call that lets the AMM's `transfer_from` of the trader's collateral
    /// succeed (issue #13) — then appends the `execute_trade` call. Exposed so
    /// callers can **simulate** the calls gas-free (e.g. a `--dry-run`) before
    /// deciding to submit.
    pub async fn build_trade_calls(&self, quote: &NormalTradeQuote) -> TradeResult<Vec<Call>> {
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
        let trade = self.build_trade_call(TradeInput {
            candidate: quote.candidate,
            x_star: quote.x_star,
            supplied_collateral: quote.padded_collateral,
            candidate_hints: quote.candidate_hints,
        });
        Ok(vec![approve, trade])
    }

    /// Submit a quote previously prepared by
    /// [`NormalMarketReader::quote_trade`].
    ///
    /// Builds the `[approve, trade]` multicall, runs a **gas-free** chain
    /// simulation first, and refuses to submit if the sequencer reports the
    /// call would revert — surfacing the raw Cairo reason as a typed rejection
    /// instead of burning a fee on a reverted transaction (issue #13).
    #[instrument(skip(self, quote), fields(market = %self.reader.address(), family = "normal", kind = "trade", accepts = quote.on_chain_will_accept))]
    pub async fn execute_quote(&self, quote: NormalTradeQuote) -> TradeResult<ExecutionReceipt> {
        self.execute_quote_bundled(quote, Vec::new()).await
    }

    /// [`Self::execute_quote`] with extra `leading` calls prepended to the
    /// `[approve, trade]` multicall — e.g. a `claim_initial_grant()` that
    /// bootstraps a fresh wallet's collateral atomically with its first trade.
    /// The same simulate-before-submit gate applies to the full bundle.
    #[instrument(skip(self, quote, leading), fields(market = %self.reader.address(), family = "normal", kind = "trade", leading = leading.len()))]
    pub async fn execute_quote_bundled(
        &self,
        quote: NormalTradeQuote,
        leading: Vec<Call>,
    ) -> TradeResult<ExecutionReceipt> {
        let mut calls = leading;
        calls.extend(self.build_trade_calls(&quote).await?);
        // Gas-free pre-flight: refuse a doomed submission *before* it burns a
        // fee. If the account can simulate and the sequencer says the multicall
        // reverts, surface the raw Cairo reason as a typed rejection instead of
        // letting an on-chain `Result::unwrap failed` cost the trader gas.
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

    /// Close out the caller's full position with a single high-level
    /// call.
    ///
    /// Internally:
    /// * reads live `distribution`, `params`, `lp_info` (so the on-chain
    ///   `expected_*` guards see byte-exact values);
    /// * fetches chain-correct sqrt hints for the current dist;
    /// * builds [`SellExecutionGuardsRaw`] using live LP backing (the AMM
    ///   guards against `get_pool_backing()` (not `params.backing`, which is
    ///   wrong here);
    /// * submits `sell_position_guarded(current_dist, x* = current.mean,
    ///   current_hints, guards)`.
    ///
    /// The caller specifies only the minimum-token-out slippage floor.
    #[instrument(skip(self), fields(market = %self.reader.address(), family = "normal", kind = "sell", runtime = %runtime))]
    pub async fn sell_position(
        &self,
        runtime: Felt,
        min_token_out: u128,
    ) -> TradeResult<ExecutionReceipt> {
        let market_dist = self.reader.distribution().await?.to_raw();
        let params = self.reader.params().await?;
        let lp_info = self.reader.lp_info().await?;
        let candidate_hints =
            compute_normal_hints(self.reader.provider(), runtime, market_dist).await?;
        let guards = SellExecutionGuardsRaw {
            expected_market_dist: market_dist,
            expected_backing: lp_info.total_backing_deposited,
            expected_tolerance: params.tolerance,
            expected_min_trade_collateral: params.min_trade_collateral,
            min_token_out,
        };
        let call = self.build_sell_call(market_dist, market_dist.mean, candidate_hints, guards);
        self.account
            .execute(vec![call])
            .await
            .map_err(TradeError::from_contract)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use std::{sync::Mutex, vec, vec::Vec};

    use async_trait::async_trait;
    use deadeye_core::{Distribution, sq128::Sq128Raw};
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

    #[tokio::test]
    async fn distribution_decodes_correctly() {
        let raw = NormalDistributionRaw {
            mean: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 100,
                limb3: 0,
                neg: false,
            },
            variance: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 4,
                limb3: 0,
                neg: false,
            },
            sigma: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 2,
                limb3: 0,
                neg: false,
            },
        };
        let canned = raw.to_calldata();
        let provider = CannedProvider {
            responses: Mutex::new(vec![canned]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let dist = reader.distribution().await.unwrap();
        assert_eq!(dist.mean().to_raw(), raw.mean);
        assert_eq!(dist.variance().to_raw(), raw.variance);
    }

    #[tokio::test]
    async fn market_status_decodes_correctly() {
        let status = MarketStatus {
            is_initialised: true,
            is_paused: false,
            is_settled: true,
            settlement_value: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 42,
                limb3: 0,
                neg: false,
            },
        };
        let canned = status.to_calldata();
        let provider = CannedProvider {
            responses: Mutex::new(vec![canned]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let observed = reader.market_status().await.unwrap();
        assert_eq!(observed, status);
    }

    struct MockAccount;
    #[async_trait]
    impl Account for MockAccount {
        fn address(&self) -> Felt {
            Felt::from(0x0_u64)
        }
        async fn execute(&self, _calls: Vec<Call>) -> ContractResult<ExecutionReceipt> {
            unreachable!()
        }
    }

    #[test]
    fn build_trade_call_uses_correct_selector() {
        let provider = CannedProvider {
            responses: Mutex::new(vec![]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let writer = NormalMarketWriter::new(reader, MockAccount);
        let input = TradeInput {
            candidate: NormalDistributionRaw {
                mean: Sq128Raw {
                    limb0: 0,
                    limb1: 0,
                    limb2: 100,
                    limb3: 0,
                    neg: false,
                },
                variance: Sq128Raw {
                    limb0: 0,
                    limb1: 0,
                    limb2: 4,
                    limb3: 0,
                    neg: false,
                },
                sigma: Sq128Raw {
                    limb0: 0,
                    limb1: 0,
                    limb2: 2,
                    limb3: 0,
                    neg: false,
                },
            },
            x_star: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 101,
                limb3: 0,
                neg: false,
            },
            supplied_collateral: Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: 5,
                limb3: 0,
                neg: false,
            },
            candidate_hints: deadeye_core::distribution::NormalSqrtHintsRaw {
                l2_norm_denom: Sq128Raw {
                    limb0: 0,
                    limb1: 0,
                    limb2: 7,
                    limb3: 0,
                    neg: false,
                },
                backing_denom: Sq128Raw {
                    limb0: 0,
                    limb1: 0,
                    limb2: 9,
                    limb3: 0,
                    neg: false,
                },
            },
        };
        let call = writer.build_trade_call(input);
        assert_eq!(call.to, Felt::from(0x1234_u64));
        assert_eq!(call.selector, amm::execute_trade());
        // calldata = 5 felts/Sq128Raw * 3 distribution + 5 + 5 + 2*5 hints = 15 + 5 + 5
        // + 10 = 35
        assert_eq!(call.calldata.len(), 35);
    }

    /// Captures the `Vec<Call>` submitted to `execute` so a test can inspect
    /// the multicall the writer builds.
    struct RecordingAccount {
        recorded: std::sync::Arc<std::sync::Mutex<Vec<Call>>>,
    }
    #[async_trait]
    impl Account for RecordingAccount {
        fn address(&self) -> Felt {
            Felt::from(0xab_u64)
        }
        async fn execute(&self, calls: Vec<Call>) -> ContractResult<ExecutionReceipt> {
            *self.recorded.lock().unwrap() = calls;
            Ok(ExecutionReceipt::new(Felt::from(0x7a_u64), 2))
        }
    }

    /// #13 — `execute_quote` must submit an atomic `[approve, trade]`
    /// multicall: an ERC-20 `approve` of the collateral token to the market
    /// (so the AMM's `transfer_from` succeeds) followed by the trade. Without
    /// the leading approve, the trade reverts with `Result::unwrap failed`.
    #[tokio::test]
    async fn execute_quote_bundles_collateral_approve_before_trade() {
        use crate::types::common::{AmmConfigRaw, AmmParamsRaw};
        fn sq(n: u64) -> Sq128Raw {
            Sq128Raw {
                limb0: 0,
                limb1: 0,
                limb2: n,
                limb3: 0,
                neg: false,
            }
        }
        let token = Felt::from(0x1d77_u64);
        let market = Felt::from(0x1234_u64);
        // Canned `get_config`: collateral token + 18 decimals.
        let config = AmmConfigRaw {
            collateral_token: token,
            token_decimals: 18,
            internal_decimals: 6,
            decimal_shift: 12,
            params: AmmParamsRaw {
                k: sq(200),
                backing: sq(1000),
                tolerance: sq(1),
                min_trade_collateral: sq(1),
            },
        };
        let provider = CannedProvider {
            responses: Mutex::new(vec![config.to_calldata()]),
        };
        let reader = NormalMarketReader::new(provider, market);
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Call>::new()));
        let writer = NormalMarketWriter::new(reader, RecordingAccount {
            recorded: std::sync::Arc::clone(&recorded),
        });
        let quote = NormalTradeQuote {
            candidate: NormalDistributionRaw {
                mean: sq(4),
                variance: sq(1),
                sigma: sq(1),
            },
            candidate_hints: NormalSqrtHintsRaw {
                l2_norm_denom: sq(1),
                backing_denom: sq(1),
            },
            x_star: sq(4),
            required_collateral: sq(5),
            padded_collateral: sq(5),
            on_chain_will_accept: true,
            rejection: None,
        };
        writer.execute_quote(quote).await.unwrap();

        let calls = recorded.lock().unwrap().clone();
        assert_eq!(
            calls.len(),
            2,
            "expected an [approve, trade] multicall, got {} call(s)",
            calls.len()
        );
        // call[0] = approve(market, amount) on the collateral token.
        assert_eq!(
            calls[0].to, token,
            "approve must target the collateral token"
        );
        assert_eq!(
            calls[0].selector,
            starknet_core::utils::get_selector_from_name("approve").unwrap(),
        );
        assert_eq!(calls[0].calldata[0], market, "spender must be the market");
        assert_ne!(
            calls[0].calldata[1],
            Felt::ZERO,
            "approve amount must be > 0"
        );
        // call[1] = the trade on the market.
        assert_eq!(calls[1].to, market, "trade must target the market");
        assert_eq!(calls[1].selector, amm::execute_trade());
    }

    // ─── Pricing primitives (Wave 2 Item 8) ─────────────────────────

    #[test]
    fn payout_at_matches_closed_form() {
        // N(0, 1): payout = λ · pdf(0). For k=1, σ=1 →
        // λ = √(2·1·√π); pdf(0) = 1/√(2π); product = √(1/√π) ≈ 0.7511
        let provider = CannedProvider {
            responses: Mutex::new(vec![]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let dist = deadeye_core::NormalDistribution::from_sigma(
            deadeye_core::Sq128::ZERO,
            deadeye_core::Sq128::from_f64(1.0).unwrap(),
        )
        .unwrap();
        let payout = reader.payout_at(0.0, &dist, 1.0);
        let expected = (1.0_f64 / core::f64::consts::PI.sqrt()).sqrt();
        assert!(
            (payout - expected).abs() < 1e-9,
            "got {payout}, expected {expected}"
        );
    }

    #[tokio::test]
    async fn impact_for_mu_shift_returns_positive_collateral() {
        // Canned provider returns N(0,1) for `get_distribution` and an
        // AmmParams blob for `get_params`. impact_for_mu_shift(1.0) must
        // yield positive collateral (equal-σ μ-shift).
        let dist_raw = NormalDistributionRaw {
            mean: deadeye_core::Sq128::ZERO.to_raw(),
            variance: deadeye_core::Sq128::from_f64(1.0).unwrap().to_raw(),
            sigma: deadeye_core::Sq128::from_f64(1.0).unwrap().to_raw(),
        };
        // Params: k, backing, tolerance, min_trade_collateral
        let params = crate::types::common::AmmParamsRaw {
            k: deadeye_core::Sq128::from_f64(1.0).unwrap().to_raw(),
            backing: deadeye_core::Sq128::from_f64(1000.0).unwrap().to_raw(),
            tolerance: deadeye_core::Sq128::from_f64(1.0).unwrap().to_raw(),
            min_trade_collateral: deadeye_core::Sq128::from_f64(0.001).unwrap().to_raw(),
        };
        // Mutex pops responses LIFO — push params *first* then dist so
        // distribution() (first await) gets dist, and params() gets params.
        let provider = CannedProvider {
            responses: Mutex::new(vec![params.to_calldata(), dist_raw.to_calldata()]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let estimate = reader.impact_for_mu_shift(1.0_f64).await.unwrap();
        assert!(
            estimate.required_collateral > 0.0,
            "expected positive collateral, got {estimate:?}",
        );
        assert!(estimate.x_star.is_finite());
    }

    #[test]
    fn sensitivities_at_finite_and_signed() {
        let provider = CannedProvider {
            responses: Mutex::new(vec![]),
        };
        let reader = NormalMarketReader::new(provider, Felt::from(0x1234_u64));
        let dist = deadeye_core::NormalDistribution::from_sigma(
            deadeye_core::Sq128::ZERO,
            deadeye_core::Sq128::from_f64(1.0).unwrap(),
        )
        .unwrap();
        let s = reader.sensitivities_at(&dist, 0.5, 1.0);
        assert!(s.d_payout_d_mu.is_finite());
        assert!(s.d_payout_d_sigma.is_finite());
        // At x* = 0.5 > μ = 0, raising μ moves it closer to x*, raising payout.
        assert!(s.d_payout_d_mu > 0.0, "got {s:?}");
    }

    /// DIAGNOSTIC: deploy a normal-math-runtime instance inside a simulation
    /// and call `check_trade_view` for the live CPI market to read the exact
    /// `TradeCheckRaw` sub-flags (no gas, no real signature). Run with:
    /// `DEADEYE_LIVE_SIM=1 cargo test -p deadeye-starknet --all-features \
    ///   live_check_trade_flags -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live mainnet simulation — run manually with DEADEYE_LIVE_SIM=1"]
    #[expect(
        clippy::print_stderr,
        clippy::panic,
        clippy::suboptimal_flops,
        reason = "diagnostic tool"
    )]
    async fn live_check_trade_flags() {
        use deadeye_core::{Distribution as _, Sq128};
        use starknet_accounts::{Account as _, ExecutionEncoding, SingleOwnerAccount};
        use starknet_core::{
            types::{
                BlockId, BlockTag, Call, ExecuteInvocation, FunctionInvocation, TransactionTrace,
            },
            utils::{UdcUniqueness, get_selector_from_name, get_udc_deployed_address},
        };
        use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
        use starknet_signers::{LocalWallet, SigningKey};

        fn env_f64(key: &str, default: f64) -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        if std::env::var("DEADEYE_LIVE_SIM").is_err() {
            eprintln!("skipped (set DEADEYE_LIVE_SIM=1)");
            return;
        }
        let rpc = "https://api.zan.top/public/starknet-mainnet/rpc/v0_10";
        let market =
            Felt::from_hex("0x00ba9a4b3eee835820fa0d0a5470876068a9fd423d2fe2ac9b1ff7b1e6d6c1ff")
                .unwrap();
        let runtime_class =
            Felt::from_hex("0x112f893233ffdfcd3ed8e41af8e3d08c901362a8deef80983fe4d36e3cd824f")
                .unwrap();
        let udc =
            Felt::from_hex("0x041a78e741e5af2fec34b695679bc6891742439f7afb8484ecd7766661ad02bf")
                .unwrap();
        let deployer =
            Felt::from_hex("0x77b277249a962ad47b04cc60bda09625c1258bbcc3dbab613d23a68833d8f0")
                .unwrap();
        let url = url::Url::parse(rpc).unwrap();
        let read_provider =
            crate::JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(url.clone())));
        let reader = NormalMarketReader::new(&read_provider, market);
        let current = reader.distribution().await.unwrap().to_raw();
        let params = reader.params().await.unwrap();
        let current_hints = reader.distribution_hints().await.unwrap();

        // Candidate σ-tighten + offline hints.
        let cand_dist = deadeye_core::NormalDistribution::from_variance(
            Sq128::from_f64(env_f64("DEADEYE_CAND_MU", 4.174)).unwrap(),
            Sq128::from_f64(env_f64("DEADEYE_CAND_VAR", 0.0144)).unwrap(),
        )
        .unwrap();
        let candidate = cand_dist.to_raw();
        let sigma_g = cand_dist.sigma().to_f64();
        let sqrt_pi = core::f64::consts::PI.sqrt();
        let candidate_hints = NormalSqrtHintsRaw {
            l2_norm_denom: Sq128::from_f64((2.0 * sigma_g * sqrt_pi).sqrt())
                .unwrap()
                .to_raw(),
            backing_denom: Sq128::from_f64((sigma_g * sqrt_pi).sqrt())
                .unwrap()
                .to_raw(),
        };
        let x_star_f64 = env_f64("DEADEYE_XSTAR", 4.405_769_206_389_262);
        let x_star = Sq128::from_f64(x_star_f64).unwrap().to_raw();
        let supplied = Sq128::from_f64(500.0).unwrap().to_raw();

        // Encode check_trade_view(current, candidate, x_star, supplied, k, backing,
        // tolerance, min_trade_collateral, current_hints, candidate_hints).
        let mut cd = Vec::new();
        current.encode(&mut cd);
        candidate.encode(&mut cd);
        x_star.encode(&mut cd);
        supplied.encode(&mut cd);
        params.k.encode(&mut cd);
        params.backing.encode(&mut cd);
        params.tolerance.encode(&mut cd);
        params.min_trade_collateral.encode(&mut cd);
        current_hints.encode(&mut cd);
        candidate_hints.encode(&mut cd);

        let salt = Felt::from(987_654_321_u64);
        let deployed =
            get_udc_deployed_address(salt, runtime_class, &UdcUniqueness::NotUnique, &[]);
        let deploy_call = Call {
            to: udc,
            selector: get_selector_from_name("deployContract").unwrap(),
            calldata: vec![runtime_class, salt, Felt::ZERO, Felt::ZERO],
        };
        let check_sel = get_selector_from_name("check_trade_view").unwrap();
        let lambda_sel = get_selector_from_name("compute_lambda_view").unwrap();
        let mut lam_cd_f = Vec::new();
        current.encode(&mut lam_cd_f);
        params.k.encode(&mut lam_cd_f);
        let mut lam_cd_g = Vec::new();
        candidate.encode(&mut lam_cd_g);
        params.k.encode(&mut lam_cd_g);

        // x* sweep offsets around the f64 solver's value.
        let base_x = x_star_f64;
        let off_env = std::env::var("DEADEYE_SWEEP").unwrap_or_default();
        let offsets: Vec<f64> = if off_env == "pos" {
            vec![5e-8, 1e-7, 2e-7, 4e-7, 6e-7, 8e-7, 1.2e-6]
        } else if off_env == "neg" {
            vec![-1.2e-6, -8e-7, -6e-7, -4e-7, -2e-7, -1e-7, -5e-8]
        } else if off_env == "pos2" {
            vec![2e-6, 4e-6, 8e-6, 2e-5, 6e-5, 2e-4, 6e-4]
        } else {
            vec![-1e-2, -1e-3, -3e-7, -1e-7, 0.0, 1e-7, 3e-7, 1e-3, 1e-2]
        };
        // pdf oracle: a lot with from_λ=0, to_λ=1 makes
        // compute_trade_lot_value_view(lot, x) return the chain's pdf_to(x).
        let lot_sel = get_selector_from_name("compute_trade_lot_value_view").unwrap();
        let one = Sq128::from_i128(1).to_raw();
        let zero = Sq128::ZERO.to_raw();
        let pdf_probe = |dist: &NormalDistributionRaw, x: Sq128Raw| -> Call {
            let mut c = Vec::with_capacity(53);
            c.push(Felt::ZERO); // lot_id: u64
            c.push(Felt::ZERO); // trader
            zero.encode(&mut c); // collateral_locked
            dist.mean.encode(&mut c); // from_* (λ=0 ⇒ ignored value-wise)
            dist.variance.encode(&mut c);
            dist.sigma.encode(&mut c);
            zero.encode(&mut c); // from_lambda = 0
            dist.mean.encode(&mut c); // to_*
            dist.variance.encode(&mut c);
            dist.sigma.encode(&mut c);
            one.encode(&mut c); // to_lambda = 1
            c.push(Felt::ONE); // flags: u8 = NORMAL_LOT_FLAG_EXISTS
            x.encode(&mut c);
            Call {
                to: deployed,
                selector: lot_sel,
                calldata: c,
            }
        };
        let x0_raw = Sq128::from_f64(base_x).unwrap().to_raw();
        let mut calls = vec![
            deploy_call,
            Call {
                to: deployed,
                selector: lambda_sel,
                calldata: lam_cd_f,
            },
            Call {
                to: deployed,
                selector: lambda_sel,
                calldata: lam_cd_g,
            },
            pdf_probe(&current, x0_raw),
            pdf_probe(&candidate, x0_raw),
        ];
        for off in &offsets {
            let xs = Sq128::from_f64(base_x + off).unwrap().to_raw();
            let mut c = Vec::new();
            current.encode(&mut c);
            candidate.encode(&mut c);
            xs.encode(&mut c);
            supplied.encode(&mut c);
            params.k.encode(&mut c);
            params.backing.encode(&mut c);
            params.tolerance.encode(&mut c);
            params.min_trade_collateral.encode(&mut c);
            current_hints.encode(&mut c);
            candidate_hints.encode(&mut c);
            calls.push(Call {
                to: deployed,
                selector: check_sel,
                calldata: c,
            });
        }
        let _ = cd;

        let acct_provider = JsonRpcClient::new(HttpTransport::new(url));
        let signer = LocalWallet::from(SigningKey::from_secret_scalar(Felt::from(2_u32)));
        let chain_id = Felt::from_hex("0x534e5f4d41494e").unwrap();
        let mut account = SingleOwnerAccount::new(
            acct_provider,
            signer,
            deployer,
            chain_id,
            ExecutionEncoding::New,
        );
        account.set_block_id(BlockId::Tag(BlockTag::PreConfirmed));

        let sim = account
            .execute_v3(calls)
            .simulate(true, true)
            .await
            .expect("simulate");
        let TransactionTrace::Invoke(inv) = sim.transaction_trace else {
            panic!("not an invoke trace");
        };
        let top = match inv.execute_invocation {
            ExecuteInvocation::Success(f) => f,
            ExecuteInvocation::Reverted(r) => panic!("execute reverted: {}", r.revert_reason),
        };
        // top.calls are the multicall entries in order: [deploy, λf, λg, check×N].
        let inner: Vec<&FunctionInvocation> = top.calls.iter().collect();
        let decode_lambda = |li: &FunctionInvocation| -> f64 {
            // Option<Sq128Raw> → [0]=Some tag (0), then Sq128.
            let r = &li.result;
            if r.first() == Some(&Felt::ZERO) {
                let (sq, _) = deadeye_core::sq128::Sq128Raw::decode(&r[1..]).unwrap();
                Sq128::from_raw(sq).to_f64()
            } else {
                f64::NAN
            }
        };
        let dl = |s: f64| -> f64 { 200.0 * (2.0 * s * core::f64::consts::PI.sqrt()).sqrt() };
        eprintln!(
            "market mu       = {:.17}",
            Sq128::from_raw(current.mean).to_f64()
        );
        eprintln!(
            "market sigma    = {:.17}",
            Sq128::from_raw(current.sigma).to_f64()
        );
        eprintln!(
            "market variance = {:.17}",
            Sq128::from_raw(current.variance).to_f64()
        );
        eprintln!(
            "tolerance       = {:.10e}",
            Sq128::from_raw(params.tolerance).to_f64()
        );
        eprintln!(
            "cand mu={:.6} var={:.8} sigma={:.12}",
            Sq128::from_raw(candidate.mean).to_f64(),
            Sq128::from_raw(candidate.variance).to_f64(),
            sigma_g
        );
        let lambda_f = decode_lambda(inner[1]);
        let lambda_g = decode_lambda(inner[2]);
        eprintln!("chain lambda_f = {lambda_f:.8}");
        eprintln!("chain lambda_g = {lambda_g:.8}");
        eprintln!(
            "deadeye lambda_f = {:.8}",
            dl(Sq128::from_raw(current.sigma).to_f64())
        );
        eprintln!("deadeye lambda_g = {:.8}", dl(sigma_g));
        // pdf parity: chain pdf via the lot-value oracle vs true (f64) pdf.
        let decode_sq = |li: &FunctionInvocation| -> f64 {
            // Option<Sq128Raw>: first felt is the Some(0)/None(1) tag.
            assert_eq!(
                li.result.first(),
                Some(&Felt::ZERO),
                "lot view returned None"
            );
            let (sq, _) = deadeye_core::sq128::Sq128Raw::decode(&li.result[1..]).unwrap();
            Sq128::from_raw(sq).to_f64()
        };
        let chain_pdf_f = decode_sq(inner[3]);
        let chain_pdf_g = decode_sq(inner[4]);
        let true_pdf = |mu: f64, s: f64, x: f64| -> f64 {
            let z = (x - mu) / s;
            (-0.5 * z * z).exp() / (s * (2.0 * core::f64::consts::PI).sqrt())
        };
        let mu_f = Sq128::from_raw(current.mean).to_f64();
        let s_f = Sq128::from_raw(current.sigma).to_f64();
        let mu_g = Sq128::from_raw(candidate.mean).to_f64();
        let tf = true_pdf(mu_f, s_f, base_x);
        let tg = true_pdf(mu_g, sigma_g, base_x);
        eprintln!(
            "pdf_f  chain={chain_pdf_f:.17}  true={tf:.17}  rel_err={:.3e}",
            (chain_pdf_f - tf) / tf
        );
        eprintln!(
            "pdf_g  chain={chain_pdf_g:.17}  true={tg:.17}  rel_err={:.3e}",
            (chain_pdf_g - tg) / tg
        );
        // Reconstruct the chain's d'(x0) from its own pdf values:
        // d' = λg·(-(x-μg)/σg²)·pdf_g − λf·(-(x-μf)/σf²)·pdf_f
        let var_f = Sq128::from_raw(current.variance).to_f64();
        let var_g = Sq128::from_raw(candidate.variance).to_f64();
        let chain_dprime = lambda_g * (-(base_x - mu_g) / var_g) * chain_pdf_g
            - lambda_f * (-(base_x - mu_f) / var_f) * chain_pdf_f;
        let true_dprime =
            lambda_g * (-(base_x - mu_g) / var_g) * tg - lambda_f * (-(base_x - mu_f) / var_f) * tf;
        eprintln!(
            "chain d'(x0) ≈ {chain_dprime:.6e}   true d'(x0) = {true_dprime:.6e}   (tol 1e-3)"
        );
        eprintln!("--- stationary sweep around x*={base_x:.12} ---");
        for (i, off) in offsets.iter().enumerate() {
            let r = &inner[5 + i].result;
            let (check, _) = crate::types::normal::TradeCheckRaw::decode(r).unwrap();
            eprintln!(
                "off={off:+.1e}  stationary={}  side={}  curv={}  collat_suff={}  computed_collat={:.4}",
                check.verification.stationary_valid,
                check.verification.side_valid,
                check.verification.curvature_valid,
                check.verification.collateral_sufficient,
                Sq128::from_raw(check.verification.computed_collateral).to_f64(),
            );
        }
    }
}
