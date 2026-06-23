//! Account abstraction for write paths.
//!
//! [`Account`] is the minimal trait every signing context must satisfy.
//! When the `account` feature is enabled, [`OwnedAccount`] wraps the
//! `starknet-accounts` `SingleOwnerAccount` so MMs can sign with a
//! locally-held private key with zero ceremony.
//!
//! ## Design
//!
//! * The trait is intentionally narrow — `address()` and `execute(calls)`. This
//!   keeps it implementable by mock signers, multi-sig services, MPC relayers,
//!   etc. without dragging in the upstream `Account` super-trait surface (which
//!   is sized + tied to `ExecutionEncoder`).
//! * Returned [`ExecutionReceipt`] carries the transaction hash only; we do not
//!   block on inclusion. Callers that need a confirmed receipt poll via the
//!   [`Provider`](crate::Provider).

use async_trait::async_trait;

use crate::{
    error::ContractResult,
    execution::{Call, ExecutionReceipt, SimOutcome},
};

/// Returns whether an account contract is deployed at `address`.
///
/// This wraps Starknet's `get_class_hash_at(latest, address)` and treats
/// `ContractNotFound` as `Ok(false)` instead of a provider error. That is the
/// precondition check write-path bots need before submitting through a freshly
/// derived account.
#[cfg(feature = "provider")]
pub async fn is_account_deployed<P>(
    provider: &P,
    address: starknet_core::types::Felt,
) -> ContractResult<bool>
where
    P: starknet_providers::Provider + Sync,
{
    use starknet_core::types::{BlockId, BlockTag, StarknetError};
    use starknet_providers::ProviderError;

    match provider
        .get_class_hash_at(BlockId::Tag(BlockTag::Latest), address)
        .await
    {
        Ok(_) => Ok(true),
        Err(ProviderError::StarknetError(StarknetError::ContractNotFound)) => Ok(false),
        Err(e) => Err(crate::error::ContractError::Provider(format!(
            "get_class_hash_at: {e}"
        ))),
    }
}

/// Minimum surface every signing context must satisfy.
#[async_trait]
pub trait Account: Send + Sync {
    /// Returns the account's on-chain address.
    fn address(&self) -> starknet_core::types::Felt;

    /// Signs and submits the given calls as a single INVOKE v3 transaction.
    /// Does **not** wait for confirmation — the returned receipt carries
    /// only the transaction hash.
    async fn execute(&self, calls: Vec<Call>) -> ContractResult<ExecutionReceipt>;

    /// **Gas-free** chain simulation of `calls` (skip-validate + skip-fee),
    /// used to catch a reverting transaction *before* it is submitted and
    /// burns gas.
    ///
    /// Returns `Ok(Some(outcome))` when the account can simulate (inspect
    /// [`SimOutcome::revert_reason`]), or `Ok(None)` for account types that
    /// cannot simulate (mocks / signers without a provider) — callers treat
    /// `None` as "unknown, proceed". The default impl returns `None`; concrete
    /// provider-backed accounts override it.
    async fn simulate(&self, _calls: &[Call]) -> ContractResult<Option<SimOutcome>> {
        Ok(None)
    }
}

#[async_trait]
impl<A> Account for &A
where
    A: Account + ?Sized,
{
    fn address(&self) -> starknet_core::types::Felt {
        (*self).address()
    }

    async fn execute(&self, calls: Vec<Call>) -> ContractResult<ExecutionReceipt> {
        (*self).execute(calls).await
    }

    async fn simulate(&self, calls: &[Call]) -> ContractResult<Option<SimOutcome>> {
        (*self).simulate(calls).await
    }
}

#[cfg(feature = "account")]
pub use owned::{
    AccountWithNonceManager, FeeBumpPolicy, FeeEstimate, GasParams, OwnedAccount, PriceUnit,
};

#[cfg(feature = "account")]
mod owned {
    //! Concrete signer backed by `starknet-accounts::SingleOwnerAccount`.

    use std::{sync::Arc, time::Duration};

    use async_trait::async_trait;
    use starknet_accounts::{
        Account as _, ConnectedAccount as _, ExecutionEncoding, SingleOwnerAccount,
    };
    use starknet_core::types::{BlockId, BlockTag, ExecuteInvocation, Felt, TransactionTrace};
    use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
    use tracing::instrument;

    use crate::{
        account::Account,
        error::{ContractError, ContractResult},
        execution::{Call, ExecutionReceipt, SimOutcome},
        nonce::{NonceGuard, NonceManager},
        signer::{DeadeyeSigner, LocalSigner, SignerAdapter},
    };

    /// A signing-capable account.
    ///
    /// `OwnedAccount` owns its own JSON-RPC connection so a single
    /// process can hold many [`OwnedAccount`]s pointing at the same
    /// network without forcing the consumer to deal with `Arc`/cloning.
    /// The underlying signer is pluggable: local in-memory wallets,
    /// remote HSM/KMS gateways, or any custom [`DeadeyeSigner`]
    /// implementation work transparently.
    #[derive(Debug)]
    pub struct OwnedAccount {
        inner: SingleOwnerAccount<JsonRpcClient<HttpTransport>, SignerAdapter>,
        signer: Arc<dyn DeadeyeSigner>,
    }

    impl OwnedAccount {
        /// Construct an account from a raw signing-key felt (back-compat).
        ///
        /// Internally wraps the felt in a [`LocalSigner`]. For HSM / KMS
        /// signing prefer [`Self::with_signer`].
        #[must_use]
        pub fn from_signing_key(
            client: JsonRpcClient<HttpTransport>,
            address: Felt,
            signing_key: Felt,
            chain_id: Felt,
        ) -> Self {
            let signer: Arc<dyn DeadeyeSigner> =
                Arc::new(LocalSigner::from_signing_key(signing_key));
            Self::with_signer(client, address, signer, chain_id)
        }

        /// Construct an address-only account for skip-validation simulations.
        ///
        /// The account can run [`Self::simulate_calls`] and chain-probe
        /// helpers, but any real submission fails before broadcast
        /// because the signer intentionally cannot produce signatures.
        #[must_use]
        pub fn for_simulation(
            client: JsonRpcClient<HttpTransport>,
            address: Felt,
            chain_id: Felt,
        ) -> Self {
            let signer: Arc<dyn DeadeyeSigner> =
                Arc::new(crate::signer::SimulationSigner::new(Felt::ONE));
            Self::with_signer(client, address, signer, chain_id)
        }

        /// Construct an account from any [`DeadeyeSigner`] implementation.
        ///
        /// Use this for HSM/KMS-backed signers, MPC threshold signers,
        /// or any custodian service exposed via
        /// [`crate::signer::RemoteSigner`].
        #[must_use]
        pub fn with_signer(
            client: JsonRpcClient<HttpTransport>,
            address: Felt,
            signer: Arc<dyn DeadeyeSigner>,
            chain_id: Felt,
        ) -> Self {
            let adapter = SignerAdapter::new(Arc::clone(&signer));
            let mut inner =
                SingleOwnerAccount::new(client, adapter, address, chain_id, ExecutionEncoding::New);
            // Use Pre-Confirmed for nonce + fee estimation so MM loops see their
            // own tx-in-flight when computing the next nonce.
            let _ = inner.set_block_id(BlockId::Tag(BlockTag::PreConfirmed));
            Self { inner, signer }
        }

        /// Borrow the underlying [`SingleOwnerAccount`] (e.g. for
        /// `declare_v3`).
        #[must_use]
        pub fn inner(&self) -> &SingleOwnerAccount<JsonRpcClient<HttpTransport>, SignerAdapter> {
            &self.inner
        }

        /// Borrow the wrapped [`DeadeyeSigner`].
        #[must_use]
        pub fn signer(&self) -> &Arc<dyn DeadeyeSigner> {
            &self.signer
        }

        /// Returns the underlying chain id.
        #[must_use]
        pub fn chain_id(&self) -> Felt {
            <SingleOwnerAccount<_, _> as starknet_accounts::Account>::chain_id(&self.inner)
        }

        /// Reads the next available nonce from the upstream provider.
        pub async fn nonce(&self) -> ContractResult<Felt> {
            self.inner
                .get_nonce()
                .await
                .map_err(|e| ContractError::Provider(format!("nonce: {e}")))
        }

        /// Estimate the fee for a multi-call.
        ///
        /// Returns L1/L2 gas consumed plus the per-unit prices the
        /// sequencer quoted. The returned [`FeeEstimate`] is what
        /// [`Self::execute_with_bump`] uses as its starting budget.
        #[instrument(skip(self, calls), fields(call_count = calls.len()))]
        pub async fn estimate_fee(&self, calls: &[Call]) -> ContractResult<FeeEstimate> {
            let est = self
                .inner
                .execute_v3(calls.to_vec())
                .estimate_fee()
                .await
                .map_err(|e| ContractError::Provider(format!("estimate_fee: {e}")))?;
            Ok(FeeEstimate {
                l1_gas_consumed: est.l1_gas_consumed,
                l1_gas_price: est.l1_gas_price,
                l2_gas_consumed: est.l2_gas_consumed,
                l2_gas_price: est.l2_gas_price,
                l1_data_gas_consumed: est.l1_data_gas_consumed,
                l1_data_gas_price: est.l1_data_gas_price,
                overall_fee: est.overall_fee,
                unit: PriceUnit::Fri,
            })
        }

        /// **Gas-free** chain simulation of `calls`.
        ///
        /// Runs the multicall through the sequencer with `skip_validate = true`
        /// (no signature needed — the in-memory signer's signature is ignored)
        /// and `skip_fee_charge = true` (no STRK balance needed). The returned
        /// [`SimOutcome`] reports whether the call path would **revert** (and
        /// the raw Cairo reason) plus the fee the real submission would pay.
        ///
        /// This is the pre-flight the trade write-paths run before
        /// [`Self::execute`], so a doomed trade (e.g. a hint mismatch that
        /// makes the AMM panic with `Result::unwrap failed`) is caught
        /// here for free instead of costing a fee on a reverted
        /// on-chain transaction.
        #[instrument(skip(self, calls), fields(call_count = calls.len()))]
        pub async fn simulate_calls(&self, calls: &[Call]) -> ContractResult<SimOutcome> {
            let sim = self
                .inner
                .execute_v3(calls.to_vec())
                .simulate(true, true)
                .await
                .map_err(|e| ContractError::Provider(format!("simulate: {e}")))?;
            let revert_reason = match sim.transaction_trace {
                TransactionTrace::Invoke(trace) => match trace.execute_invocation {
                    ExecuteInvocation::Reverted(reverted) => Some(reverted.revert_reason),
                    ExecuteInvocation::Success(_) => None,
                },
                // Non-INVOKE traces never arise from `execute_v3`; treat as clean.
                TransactionTrace::DeployAccount(_)
                | TransactionTrace::L1Handler(_)
                | TransactionTrace::Declare(_) => None,
            };
            Ok(SimOutcome {
                revert_reason,
                estimated_fee: sim.fee_estimation.overall_fee,
            })
        }

        /// Submit `calls` with automatic fee-bumping retry on stuck-tx
        /// signatures.
        ///
        /// The flow:
        ///
        /// 1. Estimate fee against the current chain state.
        /// 2. Reserve a nonce (via the wrapped manager) or read the current
        ///    chain nonce directly.
        /// 3. Submit with `policy.initial_tip`.
        /// 4. If the submission times out or returns a stuck-tx error, multiply
        ///    the tip by `policy.tip_multiplier` and resubmit with the **same
        ///    nonce** (replacement tx).
        /// 5. Repeat up to `policy.max_attempts`.
        ///
        /// The same nonce is reused across attempts — Starknet sequencers
        /// accept replacement transactions when the higher-tip one
        /// outbids the predecessor in their mempool.
        #[instrument(skip(self, calls, policy), fields(call_count = calls.len(), max_attempts = policy.max_attempts))]
        pub async fn execute_with_bump(
            &self,
            calls: Vec<Call>,
            policy: FeeBumpPolicy,
        ) -> ContractResult<ExecutionReceipt> {
            let call_count = calls.len();
            let est = self.estimate_fee(&calls).await?;

            // Anchor the nonce so each attempt uses the same value.
            let nonce_felt = self
                .inner
                .get_nonce()
                .await
                .map_err(|e| ContractError::Provider(format!("nonce: {e}")))?;

            let mut tip = policy.initial_tip;
            let mut last_err: Option<String> = None;
            for attempt in 0..policy.max_attempts {
                metrics::counter!(
                    "deadeye.tx.submitted",
                    "kind" => "execute_with_bump",
                    "attempt" => attempt.to_string(),
                )
                .increment(1);
                let exec = self
                    .inner
                    .execute_v3(calls.clone())
                    .nonce(nonce_felt)
                    .l1_gas(est.l1_gas_consumed.saturating_mul(2))
                    .l1_gas_price(est.l1_gas_price.saturating_mul(2))
                    .l2_gas(est.l2_gas_consumed.saturating_mul(2))
                    .l2_gas_price(est.l2_gas_price.saturating_mul(2))
                    .l1_data_gas(est.l1_data_gas_consumed.saturating_mul(2))
                    .l1_data_gas_price(est.l1_data_gas_price.saturating_mul(2))
                    .tip(saturating_cast_u128_to_u64(tip));
                match tokio::time::timeout(policy.attempt_timeout, exec.send()).await {
                    Ok(Ok(rcpt)) => {
                        metrics::counter!(
                            "deadeye.tx.accepted",
                            "kind" => "execute_with_bump",
                        )
                        .increment(1);
                        return Ok(ExecutionReceipt::new(rcpt.transaction_hash, call_count));
                    },
                    Ok(Err(e)) => {
                        let msg = format!("{e}");
                        let is_stuck = is_stuck_tx_signature(&msg);
                        metrics::counter!(
                            "deadeye.tx.rejected",
                            "kind" => "execute_with_bump",
                            "reason" => if is_stuck { "stuck" } else { "other" }.to_owned(),
                        )
                        .increment(1);
                        last_err = Some(msg);
                        if !is_stuck {
                            break;
                        }
                    },
                    Err(_) => {
                        metrics::counter!(
                            "deadeye.tx.rejected",
                            "kind" => "execute_with_bump",
                            "reason" => "timeout".to_owned(),
                        )
                        .increment(1);
                        last_err = Some(format!(
                            "submission timed out after {:?}",
                            policy.attempt_timeout,
                        ));
                    },
                }
                let bumped = (tip as f64) * policy.tip_multiplier;
                tip = bumped.max(tip.saturating_add(1) as f64) as u128;
            }
            Err(ContractError::Provider(format!(
                "execute_with_bump exhausted {n} attempts: {err}",
                n = policy.max_attempts,
                err = last_err.unwrap_or_else(|| "unknown".into()),
            )))
        }

        /// Pair this account with a [`NonceManager`] so submissions can run
        /// concurrently across many in-flight transactions.
        ///
        /// The returned [`AccountWithNonceManager`] consumes this
        /// `OwnedAccount`. Its [`Account::execute`] impl reserves a nonce,
        /// submits the tx, and commits the reservation on success (or
        /// drops it on failure, freeing the nonce for the next caller).
        /// Callers that already hold a [`NonceGuard`] can submit via
        /// [`AccountWithNonceManager::execute_managed`] for explicit
        /// control.
        #[must_use]
        pub fn with_nonce_manager(self, manager: NonceManager) -> AccountWithNonceManager {
            AccountWithNonceManager {
                inner: self,
                manager,
            }
        }
    }

    /// Fee estimate returned by [`OwnedAccount::estimate_fee`].
    #[derive(Debug, Clone, Copy)]
    pub struct FeeEstimate {
        /// L1 gas units consumed.
        pub l1_gas_consumed: u64,
        /// L1 gas price (FRI per unit).
        pub l1_gas_price: u128,
        /// L2 gas units consumed.
        pub l2_gas_consumed: u64,
        /// L2 gas price (FRI per unit).
        pub l2_gas_price: u128,
        /// L1 data-gas units consumed.
        pub l1_data_gas_consumed: u64,
        /// L1 data-gas price (FRI per unit).
        pub l1_data_gas_price: u128,
        /// Total fee in the unit of `unit` (FRI for V3 transactions).
        pub overall_fee: u128,
        /// Denomination of fee + per-unit prices.
        pub unit: PriceUnit,
    }

    /// Denomination of a [`FeeEstimate`]. INVOKE-v3 always pays in
    /// [`PriceUnit::Fri`] (STRK); legacy v1 paid in Wei (ETH).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PriceUnit {
        /// Fee paid in Wei (Ether). Used by legacy v1 invokes.
        Wei,
        /// Fee paid in FRI (STRK). Default for v3 invokes.
        Fri,
    }

    /// Retry policy for [`OwnedAccount::execute_with_bump`].
    #[derive(Debug, Clone, Copy)]
    pub struct FeeBumpPolicy {
        /// Tip in FRI for the first attempt.
        pub initial_tip: u128,
        /// Multiplier applied to the tip on each retry. `1.5` is a
        /// sensible default.
        pub tip_multiplier: f64,
        /// Maximum number of total attempts (including the initial).
        pub max_attempts: u32,
        /// Per-attempt timeout. The retry triggers once we miss this
        /// window without a chain-side response.
        pub attempt_timeout: Duration,
    }

    impl FeeBumpPolicy {
        /// Sensible defaults: start at 0 tip, 1.5× bumps, up to 5
        /// attempts with a 10-second window each.
        #[must_use]
        pub const fn market_maker_defaults() -> Self {
            Self {
                initial_tip: 0,
                tip_multiplier: 1.5,
                max_attempts: 5,
                attempt_timeout: Duration::from_secs(10),
            }
        }
    }

    impl Default for FeeBumpPolicy {
        fn default() -> Self {
            Self::market_maker_defaults()
        }
    }

    fn is_stuck_tx_signature(msg: &str) -> bool {
        msg.contains("InsufficientResources")
            || msg.contains("InsufficientFee")
            || msg.contains("InsufficientTip")
            || msg.contains("InsufficientMaxFee")
            || msg.contains("MaxFeeExceedsBalance")
            || msg.contains("tx replaced")
            || msg.contains("FailedToReceiveTransaction")
    }

    const fn saturating_cast_u128_to_u64(v: u128) -> u64 {
        if v > u64::MAX as u128 {
            u64::MAX
        } else {
            v as u64
        }
    }

    /// An [`OwnedAccount`] paired with a [`NonceManager`].
    ///
    /// Provides the same [`Account`] surface as the bare `OwnedAccount`
    /// but routes every `execute` call through the manager, enabling
    /// fan-out execution from a single wallet without re-querying the
    /// chain on every submission.
    #[derive(Debug)]
    pub struct AccountWithNonceManager {
        inner: OwnedAccount,
        manager: NonceManager,
    }

    impl AccountWithNonceManager {
        /// Borrow the underlying owned account.
        pub const fn inner_account(&self) -> &OwnedAccount {
            &self.inner
        }

        /// Borrow the nonce manager (for diagnostics, resync, …).
        pub const fn manager(&self) -> &NonceManager {
            &self.manager
        }

        /// Submit `calls` using the supplied [`NonceGuard`]. On success
        /// the guard is committed; on failure the guard is dropped
        /// (releasing the nonce). Use this when the caller owns the
        /// reservation lifecycle (e.g. an outer orchestrator already
        /// pulled the guard).
        ///
        /// The default flow estimates fees on every submission. Under
        /// high concurrency, the on-chain validator may reject a
        /// fee-estimation with a future nonce because pre-confirmed
        /// state hasn't propagated yet. Either pre-set gas via
        /// [`Self::execute_managed_with_gas`] or accept the occasional
        /// retry — the manager will free the dropped nonce for reuse.
        pub async fn execute_managed(
            &self,
            calls: Vec<Call>,
            guard: NonceGuard,
        ) -> ContractResult<ExecutionReceipt> {
            self.execute_managed_inner(calls, guard, None).await
        }

        /// Submit using `guard` with explicit gas parameters, bypassing
        /// the upstream fee-estimation call. Required for high-concurrency
        /// fan-out where the fee-estimate-with-future-nonce path is
        /// rejected by the node's validator.
        pub async fn execute_managed_with_gas(
            &self,
            calls: Vec<Call>,
            guard: NonceGuard,
            gas: GasParams,
        ) -> ContractResult<ExecutionReceipt> {
            self.execute_managed_inner(calls, guard, Some(gas)).await
        }

        #[instrument(skip(self, calls, guard, gas), fields(call_count = calls.len(), nonce = guard.raw()))]
        async fn execute_managed_inner(
            &self,
            calls: Vec<Call>,
            guard: NonceGuard,
            gas: Option<GasParams>,
        ) -> ContractResult<ExecutionReceipt> {
            // Retry on transient nonce-validation failures. Some sequencers
            // (notably starknet-devnet-rs) don't queue future nonces — a
            // submission with nonce N+1 is rejected if N hasn't landed
            // yet. We retry a few times with backoff so the high-concurrency
            // fan-out absorbs that without manual sleep on the caller.
            const MAX_RETRIES: u32 = 8;
            let nonce = guard.value();
            let call_count = calls.len();
            let mut backoff_ms = 25_u64;
            let mut last_err: Option<String> = None;
            metrics::counter!(
                "deadeye.tx.submitted",
                "kind" => "managed",
            )
            .increment(1);
            for _ in 0..=MAX_RETRIES {
                let mut exec = self.inner.inner().execute_v3(calls.clone()).nonce(nonce);
                if let Some(g) = gas {
                    exec = exec
                        .l1_gas(g.l1_gas)
                        .l1_gas_price(g.l1_gas_price)
                        .l2_gas(g.l2_gas)
                        .l2_gas_price(g.l2_gas_price)
                        .l1_data_gas(g.l1_data_gas)
                        .l1_data_gas_price(g.l1_data_gas_price)
                        .tip(g.tip);
                }
                match exec.send().await {
                    Ok(rcpt) => {
                        guard.commit();
                        metrics::counter!(
                            "deadeye.tx.accepted",
                            "kind" => "managed",
                        )
                        .increment(1);
                        return Ok(ExecutionReceipt::new(rcpt.transaction_hash, call_count));
                    },
                    Err(e) => {
                        let msg = format!("{e}");
                        let is_transient = msg.contains("InvalidTransactionNonce")
                            || msg.contains("Account transaction nonce is invalid")
                            || msg.contains("nonce");
                        last_err = Some(msg);
                        if !is_transient {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(500);
                    },
                }
            }
            // Drop the guard implicitly — the nonce slot is
            // returned to the queue so the next submission can
            // reuse it.
            drop(guard);
            metrics::counter!(
                "deadeye.tx.rejected",
                "kind" => "managed",
                "reason" => "nonce_validation".to_owned(),
            )
            .increment(1);
            Err(ContractError::Provider(format!(
                "execute_v3: {}",
                last_err.unwrap_or_else(|| "unknown".into()),
            )))
        }
    }

    /// Pre-set gas parameters for
    /// [`AccountWithNonceManager::execute_managed_with_gas`].
    ///
    /// Use these when you don't want the SDK to perform a fee estimation
    /// (e.g. when fanning out many concurrent submissions). Reasonable
    /// MM defaults are returned by [`Self::generous_defaults`].
    #[derive(Debug, Clone, Copy)]
    pub struct GasParams {
        /// L1 gas budget.
        pub l1_gas: u64,
        /// L1 gas price (FRI per unit).
        pub l1_gas_price: u128,
        /// L2 gas budget.
        pub l2_gas: u64,
        /// L2 gas price (FRI per unit).
        pub l2_gas_price: u128,
        /// L1 data gas budget.
        pub l1_data_gas: u64,
        /// L1 data gas price (FRI per unit).
        pub l1_data_gas_price: u128,
        /// Tip in FRI.
        pub tip: u64,
    }

    impl GasParams {
        /// Generous defaults that cover most calls. Excess gas is refunded.
        #[must_use]
        pub const fn generous_defaults() -> Self {
            Self {
                l1_gas: 10_000,
                l1_gas_price: 1_000_000_000_000,
                l2_gas: 100_000_000,
                l2_gas_price: 100_000_000_000,
                l1_data_gas: 10_000,
                l1_data_gas_price: 1_000_000_000_000,
                tip: 0,
            }
        }
    }

    #[async_trait]
    impl Account for AccountWithNonceManager {
        fn address(&self) -> Felt {
            self.inner.address()
        }

        async fn execute(&self, calls: Vec<Call>) -> ContractResult<ExecutionReceipt> {
            let guard = self.manager.reserve().await;
            self.execute_managed(calls, guard).await
        }

        async fn simulate(&self, calls: &[Call]) -> ContractResult<Option<SimOutcome>> {
            self.inner.simulate_calls(calls).await.map(Some)
        }
    }

    #[async_trait]
    impl Account for OwnedAccount {
        fn address(&self) -> Felt {
            <SingleOwnerAccount<_, _> as starknet_accounts::Account>::address(&self.inner)
        }

        #[instrument(skip(self, calls), fields(call_count = calls.len(), address = %<SingleOwnerAccount<_, _> as starknet_accounts::Account>::address(&self.inner)))]
        async fn execute(&self, calls: Vec<Call>) -> ContractResult<ExecutionReceipt> {
            let call_count = calls.len();
            metrics::counter!(
                "deadeye.tx.submitted",
                "kind" => "execute",
            )
            .increment(1);
            let result = self.inner.execute_v3(calls).send().await.map_err(|e| {
                metrics::counter!(
                    "deadeye.tx.rejected",
                    "kind" => "execute",
                    "reason" => "submission".to_owned(),
                )
                .increment(1);
                ContractError::Provider(format!("execute_v3: {e}"))
            })?;
            metrics::counter!(
                "deadeye.tx.accepted",
                "kind" => "execute",
            )
            .increment(1);
            Ok(ExecutionReceipt::new(result.transaction_hash, call_count))
        }

        async fn simulate(&self, calls: &[Call]) -> ContractResult<Option<SimOutcome>> {
            Self::simulate_calls(self, calls).await.map(Some)
        }
    }
}
