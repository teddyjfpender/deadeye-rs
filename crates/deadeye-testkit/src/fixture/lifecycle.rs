//! High-level lifecycle helpers used by chaos tests.
//!
//! Sits one layer above the bootstrap and provides:
//! * `upsert_normal_profile_for_test` — installs a reusable normal-market
//!   profile.
//! * `deploy_normal_market_with_event` — deploys + extracts the market address
//!   from `MarketDeployed`.
//! * `wait_for_receipt` — polls `get_transaction_receipt` until inclusion.
//! * `MarketCalldataBuilder` — sq128-friendly calldata helpers (init dist +
//!   hints + zero overrides).

use std::time::Duration;

use deadeye_core::{
    Sq128,
    bivariate::{
        BivariateNormalDistributionCoreRaw, BivariateNormalDistributionRaw,
        BivariateNormalSqrtHintsRaw,
    },
    categorical::{CategoricalDistributionRaw, CategoricalL2HintRaw},
    distribution::{LognormalDistributionRaw, NormalDistributionRaw, NormalSqrtHintsRaw},
    sq128::Sq128Raw,
};
use deadeye_starknet::{CairoSerde, types::lognormal::LognormalSqrtHintsRaw};
use starknet_accounts::{Account, ConnectedAccount};
use starknet_core::{
    types::{
        BlockId, BlockTag, Call, ExecutionResult, Felt, FunctionCall, TransactionReceipt,
        TransactionReceiptWithBlockInfo,
    },
    utils::get_selector_from_name,
};
use starknet_providers::{Provider, ProviderError};
use thiserror::Error;

use crate::fixture::{
    erc20::{Erc20Error, approve},
    factory_setup::{DeployProfileParams, FactorySetupError, MarketKind, upsert_deploy_profile},
};

/// Initializes a freshly-deployed market.
///
/// The AMM's `initialize()` reads the per-profile backing from storage
/// and `transferFrom`s the corresponding token amount from the caller.
/// We pre-approve a generous allowance so the call always has enough.
///
/// **IMPORTANT (`u256_sub` Overflow gotcha):** The caller (`account`) **must**
/// hold at least `backing × 10^token_decimals` base units of `collateral`
/// or the underlying ERC20's `transferFrom` will revert with
/// `u256_sub Overflow` while computing `balance - amount`. The error is
/// emitted by the token contract, not by the AMM. On devnet with the
/// predeployed STRK each account starts with only **1000 STRK**, and the
/// bootstrap burns roughly 30 STRK in declare/deploy gas — keep the
/// per-profile `backing` at or below ~500 STRK for the predeployed admin
/// account.
pub async fn initialize_market<A>(
    account: &A,
    market: Felt,
    collateral: Felt,
    approve_amount: u128,
) -> Result<Felt, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    approve(account.clone(), collateral, market, approve_amount).await?;
    let call = Call {
        to: market,
        selector: get_selector_from_name("initialize")
            .map_err(|e| LifecycleError::Provider(format!("invalid selector: {e}")))?,
        calldata: Vec::new(),
    };
    let result = account
        .execute_v3(vec![call])
        .send()
        .await
        .map_err(|e| LifecycleError::Submit(format!("initialize: {e}")))?;
    let _ = wait_for_receipt(account, result.transaction_hash).await?;
    Ok(result.transaction_hash)
}

/// Errors emitted by the lifecycle helpers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LifecycleError {
    /// Setup call failed.
    #[error(transparent)]
    Setup(#[from] FactorySetupError),
    /// ERC20 helper failed.
    #[error(transparent)]
    Erc20(#[from] Erc20Error),
    /// Provider call failed.
    #[error("provider failure: {0}")]
    Provider(String),
    /// Submission failed.
    #[error("submission failed: {0}")]
    Submit(String),
    /// Receipt indicates the transaction reverted.
    #[error("transaction reverted: {0}")]
    Reverted(String),
    /// Expected event was not present in the receipt.
    #[error("MarketDeployed event not found in receipt")]
    NoMarketDeployedEvent,
}

/// Construct a `Sq128Raw` from a finite `f64`.
fn sq(v: f64) -> Sq128Raw {
    Sq128::from_f64(v).expect("finite f64").to_raw()
}

/// Installs a normal-market deploy profile suitable for chaos tests.
///
/// Defaults: `k=50`, `backing=50`, `tolerance=1e-3`,
/// `min_trade_collateral=1.0`, fees=0, no extension.
///
/// **Why `backing=50` and not the production `1000`:** the devnet
/// predeployed STRK token funds each account with **1000 STRK**, and the
/// bootstrap declare+deploy sequence burns ~30 STRK of gas before the
/// admin reaches `initialize()`. A profile backing of `1000` would request
/// `1000 × 10^18` base units in `transferFrom(admin, market, …)` against
/// an admin balance of ~970 STRK — that subtraction underflows inside
/// OZ ERC20 (`balance - amount`) and surfaces as a confusing
/// `u256_sub Overflow` revert on `initialize()`.
pub async fn upsert_normal_profile_for_test<A>(
    account: A,
    factory: Felt,
    collateral: Felt,
    profile_id: u32,
) -> Result<(), LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    let params = DeployProfileParams {
        market_type: MarketKind::Normal,
        collateral_token: collateral,
        token_decimals: 18,
        // Internal precision must be < token decimals; 6 mirrors the
        // upstream production profile.
        internal_decimals: 6,
        k: sq(50.0),
        backing: sq(50.0),
        // Loose tolerance (1.0 instead of 0.001) so the on-chain stationarity
        // check `|d'(x*)| ≤ tolerance · scale` accepts the f64-precision
        // off-chain Newton x_star. The Cairo tests precompute x_star with
        // Python `Decimal(prec=60)` for bit-exact Sq128 — see
        // `the-situation/packages/onchain-normal-amm/src/tests/test_amm_contract.cairo:70-77`.
        // For the Rust chaos suite we trade strict tolerance for usability.
        tolerance: sq(1.0),
        min_trade_collateral: sq(1.0),
        fee_config: deadeye_starknet::types::common::FeeConfigRaw {
            lp_fee_bps: 0,
            protocol_fee_bps: 0,
            settlement_fee_bps: 0,
        },
        extension: Felt::ZERO,
        extension_call_points: 0,
    };
    upsert_deploy_profile(account, factory, profile_id, params).await?;
    Ok(())
}

/// Installs a normal-market deploy profile with caller-supplied parameters.
///
/// Use this when the default `upsert_normal_profile_for_test` values
/// (`k=50`, `backing=50`, `internal_decimals=6`, `token_decimals=18`)
/// don't match the scenario.
///
/// **Gotcha:** keep `backing × 10^token_decimals` ≤ caller's balance on
/// `collateral`. On devnet that's ~970 STRK = `9.7 × 10^20` base units.
#[allow(
    clippy::too_many_arguments,
    reason = "every argument maps to a distinct knob on the on-chain `DeployProfileParams` struct — collapsing them into a builder would force callers to construct it themselves"
)]
pub async fn upsert_normal_profile_for_test_with_params<A>(
    account: A,
    factory: Felt,
    collateral: Felt,
    profile_id: u32,
    k: f64,
    backing: f64,
    internal_decimals: u8,
    token_decimals: u8,
) -> Result<(), LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    let params = DeployProfileParams {
        market_type: MarketKind::Normal,
        collateral_token: collateral,
        token_decimals,
        internal_decimals,
        k: sq(k),
        backing: sq(backing),
        tolerance: sq(0.001),
        min_trade_collateral: sq(1.0),
        fee_config: deadeye_starknet::types::common::FeeConfigRaw {
            lp_fee_bps: 0,
            protocol_fee_bps: 0,
            settlement_fee_bps: 0,
        },
        extension: Felt::ZERO,
        extension_call_points: 0,
    };
    upsert_deploy_profile(account, factory, profile_id, params).await?;
    Ok(())
}

/// Default-zero `MarketDeployOverridesRaw` — keeps every field from the
/// profile.
fn encode_zero_overrides(out: &mut Vec<Felt>) {
    out.push(Felt::ZERO); // mask
    out.push(Felt::ZERO); // collateral_token
    out.push(Felt::ZERO); // token_decimals
    out.push(Felt::ZERO); // internal_decimals
    let zero = Sq128Raw {
        limb0: 0,
        limb1: 0,
        limb2: 0,
        limb3: 0,
        neg: false,
    };
    zero.encode(out); // k
    zero.encode(out); // backing
    zero.encode(out); // tolerance
    zero.encode(out); // min_trade_collateral
    out.push(Felt::ZERO); // fee_config.lp_fee_bps
    out.push(Felt::ZERO); // fee_config.protocol_fee_bps
    out.push(Felt::ZERO); // fee_config.settlement_fee_bps
    out.push(Felt::ZERO); // extension
    out.push(Felt::ZERO); // extension_call_points
}

/// Deploys a normal market and returns the deployed contract address by
/// parsing the `MarketDeployed` event from the transaction receipt.
pub async fn deploy_normal_market_with_event<A>(
    account: &A,
    factory: Felt,
    profile_id: u32,
    salt: Felt,
    metadata_hash: Felt,
    initial_dist: NormalDistributionRaw,
    initial_hints: NormalSqrtHintsRaw,
) -> Result<Felt, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(64);
    profile_id.encode(&mut calldata);
    calldata.push(salt);
    calldata.push(metadata_hash);
    initial_dist.encode(&mut calldata);
    initial_hints.encode(&mut calldata);
    encode_zero_overrides(&mut calldata);

    let call = Call {
        to: factory,
        selector: get_selector_from_name("deploy_normal_market_from_profile")
            .expect("selector valid"),
        calldata,
    };
    let result =
        account.execute_v3(vec![call]).send().await.map_err(|e| {
            LifecycleError::Submit(format!("deploy_normal_market_from_profile: {e}"))
        })?;

    let receipt = wait_for_receipt(account, result.transaction_hash).await?;
    extract_market_address_from_events(&receipt)
}

/// Polls `get_transaction_receipt` until the transaction is included or a
/// timeout elapses.
pub async fn wait_for_receipt<A>(
    account: &A,
    tx_hash: Felt,
) -> Result<TransactionReceiptWithBlockInfo, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match account.provider().get_transaction_receipt(tx_hash).await {
            Ok(r) => {
                if let ExecutionResult::Reverted { reason } = r.receipt.execution_result() {
                    return Err(LifecycleError::Reverted(reason.clone()));
                }
                return Ok(r);
            },
            Err(ProviderError::StarknetError(
                starknet_core::types::StarknetError::TransactionHashNotFound,
            )) => {
                // Mempool latency, keep polling.
            },
            Err(other) => return Err(LifecycleError::Provider(format!("{other}"))),
        }
        if std::time::Instant::now() >= deadline {
            return Err(LifecycleError::Provider(format!(
                "timed out waiting for receipt {tx_hash:#x}"
            )));
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

fn extract_market_address_from_events(
    receipt: &TransactionReceiptWithBlockInfo,
) -> Result<Felt, LifecycleError> {
    let events = match &receipt.receipt {
        TransactionReceipt::Invoke(r) => &r.events,
        TransactionReceipt::Deploy(r) => &r.events,
        TransactionReceipt::DeployAccount(r) => &r.events,
        TransactionReceipt::Declare(r) => &r.events,
        TransactionReceipt::L1Handler(r) => &r.events,
    };
    let market_deployed_key = get_selector_from_name("MarketDeployed")
        .map_err(|e| LifecycleError::Provider(format!("invalid selector: {e}")))?;
    for event in events {
        if event.keys.first() == Some(&market_deployed_key) {
            // The market address is typically keys[1] (indexed first key
            // beyond the event-name selector).
            if let Some(addr) = event.keys.get(1) {
                return Ok(*addr);
            }
            // Some contracts emit the address as the first data entry instead.
            if let Some(addr) = event.data.first() {
                return Ok(*addr);
            }
        }
    }
    Err(LifecycleError::NoMarketDeployedEvent)
}

/// Calls `compute_hints_view(dist)` on a deployed math-runtime instance
/// and returns the chain-correct sqrt hints byte-for-byte.
pub async fn fetch_normal_hints<P: Provider + Sync>(
    provider: &P,
    runtime: Felt,
    dist: NormalDistributionRaw,
) -> Result<NormalSqrtHintsRaw, LifecycleError> {
    let mut calldata = Vec::with_capacity(15);
    dist.encode(&mut calldata);
    let selector = get_selector_from_name("compute_hints_view")
        .map_err(|e| LifecycleError::Provider(format!("invalid selector name: {e}")))?;
    let response = provider
        .call(
            FunctionCall {
                contract_address: runtime,
                entry_point_selector: selector,
                calldata,
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| LifecycleError::Provider(format!("compute_hints_view: {e}")))?;
    if response.first() != Some(&Felt::ZERO) {
        return Err(LifecycleError::Provider(
            "compute_hints_view returned None — distribution rejected by runtime".into(),
        ));
    }
    let tail = response
        .get(1..)
        .ok_or_else(|| LifecycleError::Provider("compute_hints_view: response too short".into()))?;
    let (hints, _rest) = NormalSqrtHintsRaw::decode(tail)
        .map_err(|e| LifecycleError::Provider(format!("decode hints: {e}")))?;
    Ok(hints)
}

/// Construct the [`NormalDistributionRaw`] for an initial market state
/// (mean + variance). Sigma is computed via f64 sqrt — for chain-correct
/// hints, fetch them separately via [`fetch_normal_hints`].
#[must_use]
pub fn build_initial_normal_inputs(
    mean: f64,
    variance: f64,
    _backing: f64,
) -> (NormalDistributionRaw, NormalSqrtHintsRaw) {
    let sigma = variance.sqrt();
    let dist = NormalDistributionRaw {
        mean: sq(mean),
        variance: sq(variance),
        sigma: sq(sigma),
    };
    let placeholder = NormalSqrtHintsRaw {
        l2_norm_denom: sq(0.0),
        backing_denom: sq(0.0),
    };
    (dist, placeholder)
}

// ---------------------------------------------------------------------------
// Lognormal
// ---------------------------------------------------------------------------

/// Installs a lognormal-market deploy profile suitable for chaos tests.
pub async fn upsert_lognormal_profile_for_test<A>(
    account: A,
    factory: Felt,
    collateral: Felt,
    profile_id: u32,
) -> Result<(), LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    let params = DeployProfileParams {
        market_type: MarketKind::Lognormal,
        collateral_token: collateral,
        token_decimals: 18,
        internal_decimals: 6,
        k: sq(50.0),
        backing: sq(50.0),
        tolerance: sq(1.0),
        min_trade_collateral: sq(0.1),
        fee_config: deadeye_starknet::types::common::FeeConfigRaw {
            lp_fee_bps: 0,
            protocol_fee_bps: 0,
            settlement_fee_bps: 0,
        },
        extension: Felt::ZERO,
        extension_call_points: 0,
    };
    upsert_deploy_profile(account, factory, profile_id, params).await?;
    Ok(())
}

/// Construct initial lognormal distribution inputs from `(mu, variance)`
/// in log-space. Sigma is computed off-chain via f64 sqrt; fetch chain-correct
/// hints separately via [`fetch_lognormal_hints`].
#[must_use]
pub fn build_initial_lognormal_inputs(mu: f64, variance: f64) -> LognormalDistributionRaw {
    let sigma = variance.sqrt();
    LognormalDistributionRaw {
        mu: sq(mu),
        variance: sq(variance),
        sigma: sq(sigma),
    }
}

/// Calls `compute_hints_view(dist)` on a deployed lognormal math-runtime
/// instance and returns the chain-correct sqrt hints byte-for-byte.
pub async fn fetch_lognormal_hints<P: Provider + Sync>(
    provider: &P,
    runtime: Felt,
    dist: LognormalDistributionRaw,
) -> Result<LognormalSqrtHintsRaw, LifecycleError> {
    let mut calldata = Vec::with_capacity(15);
    dist.encode(&mut calldata);
    let selector = get_selector_from_name("compute_hints_view")
        .map_err(|e| LifecycleError::Provider(format!("invalid selector name: {e}")))?;
    let response = provider
        .call(
            FunctionCall {
                contract_address: runtime,
                entry_point_selector: selector,
                calldata,
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| LifecycleError::Provider(format!("compute_hints_view: {e}")))?;
    if response.first() != Some(&Felt::ZERO) {
        return Err(LifecycleError::Provider(
            "compute_hints_view returned None — distribution rejected by runtime".into(),
        ));
    }
    let tail = response
        .get(1..)
        .ok_or_else(|| LifecycleError::Provider("compute_hints_view: response too short".into()))?;
    let (hints, _rest) = LognormalSqrtHintsRaw::decode(tail)
        .map_err(|e| LifecycleError::Provider(format!("decode hints: {e}")))?;
    Ok(hints)
}

/// Deploys a lognormal market via the factory and returns the deployed
/// market address parsed from the `MarketDeployed` event.
pub async fn deploy_lognormal_market_with_event<A>(
    account: &A,
    factory: Felt,
    profile_id: u32,
    salt: Felt,
    metadata_hash: Felt,
    initial_dist: LognormalDistributionRaw,
    initial_hints: LognormalSqrtHintsRaw,
) -> Result<Felt, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(64);
    profile_id.encode(&mut calldata);
    calldata.push(salt);
    calldata.push(metadata_hash);
    initial_dist.encode(&mut calldata);
    initial_hints.encode(&mut calldata);
    encode_zero_overrides(&mut calldata);

    let call = Call {
        to: factory,
        selector: get_selector_from_name("deploy_lognormal_market_from_profile")
            .expect("selector valid"),
        calldata,
    };
    let result = account.execute_v3(vec![call]).send().await.map_err(|e| {
        LifecycleError::Submit(format!("deploy_lognormal_market_from_profile: {e}"))
    })?;
    let receipt = wait_for_receipt(account, result.transaction_hash).await?;
    extract_market_address_from_events(&receipt)
}

// ---------------------------------------------------------------------------
// Multinoulli
// ---------------------------------------------------------------------------

/// Installs a multinoulli-market deploy profile suitable for chaos tests.
pub async fn upsert_multinoulli_profile_for_test<A>(
    account: A,
    factory: Felt,
    collateral: Felt,
    profile_id: u32,
) -> Result<(), LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    // Chain enforces `tolerance ≤ 2^MAX_TOLERANCE_SHIFT = 2^-20 ≈ 9.54e-7`
    // for multinoulli markets (Cairo: `factory/src/contract.cairo:416`,
    // `categorical-types/src/types.cairo::MAX_TOLERANCE_SHIFT = -20`).
    // The previous `sq(1.0)` was 1e6× over the cap and panicked with
    // 'tolerance too large' at upsert time. Use a value just under the
    // cap.
    let tolerance_value = 2.0_f64.powi(-20);
    let params = DeployProfileParams {
        market_type: MarketKind::Multinoulli,
        collateral_token: collateral,
        token_decimals: 18,
        internal_decimals: 6,
        k: sq(50.0),
        backing: sq(50.0),
        tolerance: sq(tolerance_value),
        min_trade_collateral: sq(0.1),
        fee_config: deadeye_starknet::types::common::FeeConfigRaw {
            lp_fee_bps: 0,
            protocol_fee_bps: 0,
            settlement_fee_bps: 0,
        },
        extension: Felt::ZERO,
        extension_call_points: 0,
    };
    upsert_deploy_profile(account, factory, profile_id, params).await?;
    Ok(())
}

/// Calls `compute_hint_view(dist)` on a deployed multinoulli math-runtime
/// instance and returns the L2-norm hint byte-for-byte.
pub async fn fetch_multinoulli_hint<P: Provider + Sync>(
    provider: &P,
    runtime: Felt,
    dist: &CategoricalDistributionRaw,
) -> Result<CategoricalL2HintRaw, LifecycleError> {
    let mut calldata = Vec::with_capacity(2 + 5 * dist.probs.len());
    dist.encode(&mut calldata);
    // Try `compute_hint_view` first; fall back to `compute_hints_view`.
    for sel_name in ["compute_hint_view", "compute_hints_view"] {
        let selector = get_selector_from_name(sel_name)
            .map_err(|e| LifecycleError::Provider(format!("invalid selector: {e}")))?;
        let call_res = provider
            .call(
                FunctionCall {
                    contract_address: runtime,
                    entry_point_selector: selector,
                    calldata: calldata.clone(),
                },
                BlockId::Tag(BlockTag::PreConfirmed),
            )
            .await;
        if let Ok(response) = call_res {
            if response.first() != Some(&Felt::ZERO) {
                return Err(LifecycleError::Provider(
                    "compute_hint_view returned None".into(),
                ));
            }
            let tail = response.get(1..).ok_or_else(|| {
                LifecycleError::Provider("compute_hint_view: response too short".into())
            })?;
            let (hint, _rest) = CategoricalL2HintRaw::decode(tail)
                .map_err(|e| LifecycleError::Provider(format!("decode hint: {e}")))?;
            return Ok(hint);
        }
    }
    Err(LifecycleError::Provider(
        "no compute_hint(s)_view selector accepted".into(),
    ))
}

/// Deploys a multinoulli market via the factory.
pub async fn deploy_multinoulli_market_with_event<A>(
    account: &A,
    factory: Felt,
    profile_id: u32,
    salt: Felt,
    metadata_hash: Felt,
    initial_dist: &CategoricalDistributionRaw,
    initial_hint: CategoricalL2HintRaw,
) -> Result<Felt, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(64);
    profile_id.encode(&mut calldata);
    calldata.push(salt);
    calldata.push(metadata_hash);
    initial_dist.encode(&mut calldata);
    initial_hint.encode(&mut calldata);
    // Chain expects `matrix_constraints: MultinoulliMatrixConstraintsRaw`
    // as param #6 (Cairo:
    // `factory/src/contract.cairo::deploy_multinoulli_market_from_profile:1515`).
    // The previous calldata omitted this entirely and the chain failed
    // 'Failed to deserialize param #7'. Use the `Disabled` default
    // (mode=0, rows=0, cols=0) — chaos suite does not exercise matrix
    // constraints.
    let matrix_constraints =
        deadeye_starknet::types::multinoulli::MultinoulliMatrixConstraintsRaw {
            mode: deadeye_starknet::types::multinoulli::MultinoulliMatrixConstraintMode::Disabled,
            row_count: 0,
            col_count: 0,
        };
    matrix_constraints.encode(&mut calldata);
    encode_zero_overrides(&mut calldata);

    let call = Call {
        to: factory,
        selector: get_selector_from_name("deploy_multinoulli_market_from_profile")
            .expect("selector valid"),
        calldata,
    };
    let result = account.execute_v3(vec![call]).send().await.map_err(|e| {
        LifecycleError::Submit(format!("deploy_multinoulli_market_from_profile: {e}"))
    })?;
    let receipt = wait_for_receipt(account, result.transaction_hash).await?;
    extract_market_address_from_events(&receipt)
}

// ---------------------------------------------------------------------------
// Bivariate
// ---------------------------------------------------------------------------

/// Installs a bivariate-market deploy profile suitable for chaos tests.
pub async fn upsert_bivariate_profile_for_test<A>(
    account: A,
    factory: Felt,
    collateral: Felt,
    profile_id: u32,
) -> Result<(), LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send + Clone,
{
    let params = DeployProfileParams {
        market_type: MarketKind::BivariateNormal,
        collateral_token: collateral,
        token_decimals: 18,
        internal_decimals: 6,
        k: sq(50.0),
        backing: sq(50.0),
        tolerance: sq(1.0),
        min_trade_collateral: sq(0.1),
        fee_config: deadeye_starknet::types::common::FeeConfigRaw {
            lp_fee_bps: 0,
            protocol_fee_bps: 0,
            settlement_fee_bps: 0,
        },
        extension: Felt::ZERO,
        extension_call_points: 0,
    };
    upsert_deploy_profile(account, factory, profile_id, params).await?;
    Ok(())
}

/// Calls `expand_distribution_core_view(core_dist)` on a deployed bivariate
/// math runtime.
///
/// Returns the chain-correct full distribution byte-for-byte. The full
/// distribution carries the chain-derived `sigma1`, `sigma2`,
/// `inv_one_minus_rho_sq` and `normalization` fields — without this the
/// constructor's `inv_one_minus_rho_sq_hint != expected_inv` check
/// (Cairo: `dist-bivariate-normal/src/lib.cairo::new:370`) rejects any
/// f64-derived initial distribution because the f64 derivations differ
/// from the chain's Sq128 derivations in low limbs. `compute_hints_view`
/// returns `None` for f64-derived inputs as a direct consequence —
/// fetch this FIRST, then feed the result into `fetch_bivariate_hints`.
pub async fn expand_bivariate_distribution<P: Provider + Sync>(
    provider: &P,
    runtime: Felt,
    core_dist: BivariateNormalDistributionCoreRaw,
) -> Result<BivariateNormalDistributionRaw, LifecycleError> {
    let mut calldata = Vec::with_capacity(32);
    core_dist.encode(&mut calldata);
    let selector = get_selector_from_name("expand_distribution_core_view")
        .map_err(|e| LifecycleError::Provider(format!("invalid selector name: {e}")))?;
    let response = provider
        .call(
            FunctionCall {
                contract_address: runtime,
                entry_point_selector: selector,
                calldata,
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| LifecycleError::Provider(format!("expand_distribution_core_view: {e}")))?;
    if response.first() != Some(&Felt::ZERO) {
        return Err(LifecycleError::Provider(
            "expand_distribution_core_view returned None — core dist rejected".into(),
        ));
    }
    let tail = response.get(1..).ok_or_else(|| {
        LifecycleError::Provider("expand_distribution_core_view: response too short".into())
    })?;
    let (dist, _rest) = BivariateNormalDistributionRaw::decode(tail)
        .map_err(|e| LifecycleError::Provider(format!("decode full dist: {e}")))?;
    Ok(dist)
}

/// Calls `compute_hints_view(dist)` on a deployed bivariate math-runtime
/// instance and returns the chain-correct sqrt hints byte-for-byte.
pub async fn fetch_bivariate_hints<P: Provider + Sync>(
    provider: &P,
    runtime: Felt,
    dist: BivariateNormalDistributionRaw,
) -> Result<BivariateNormalSqrtHintsRaw, LifecycleError> {
    let mut calldata = Vec::with_capacity(64);
    dist.encode(&mut calldata);
    let selector = get_selector_from_name("compute_hints_view")
        .map_err(|e| LifecycleError::Provider(format!("invalid selector name: {e}")))?;
    let response = provider
        .call(
            FunctionCall {
                contract_address: runtime,
                entry_point_selector: selector,
                calldata,
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| LifecycleError::Provider(format!("compute_hints_view: {e}")))?;
    if response.first() != Some(&Felt::ZERO) {
        return Err(LifecycleError::Provider(
            "compute_hints_view returned None".into(),
        ));
    }
    let tail = response
        .get(1..)
        .ok_or_else(|| LifecycleError::Provider("compute_hints_view: response too short".into()))?;
    let (hints, _rest) = BivariateNormalSqrtHintsRaw::decode(tail)
        .map_err(|e| LifecycleError::Provider(format!("decode hints: {e}")))?;
    Ok(hints)
}

/// Build an initial bivariate distribution from `(μ₁, μ₂, σ₁², σ₂², ρ)`.
///
/// Uses f64 derivations for `sigma_i`, `inv_one_minus_rho_sq`, and
/// `normalization`; for chain-byte-exact deploys fetch a sibling hints
/// vector from the math runtime and pre-validate the derived fields via
/// the runtime's `derive_full_distribution_view` if available.
#[must_use]
pub fn build_initial_bivariate_inputs(
    mu1: f64,
    mu2: f64,
    variance1: f64,
    variance2: f64,
    rho: f64,
) -> BivariateNormalDistributionRaw {
    let sigma1 = variance1.sqrt();
    let sigma2 = variance2.sqrt();
    let one_minus_rho_sq = rho.mul_add(-rho, 1.0_f64);
    let inv_one_minus_rho_sq = if one_minus_rho_sq > 0.0_f64 {
        1.0_f64 / one_minus_rho_sq
    } else {
        0.0_f64
    };
    let normalization = if sigma1 > 0.0_f64 && sigma2 > 0.0_f64 && one_minus_rho_sq > 0.0_f64 {
        1.0_f64 / (2.0_f64 * core::f64::consts::PI * sigma1 * sigma2 * one_minus_rho_sq.sqrt())
    } else {
        0.0_f64
    };
    BivariateNormalDistributionRaw {
        mu1: sq(mu1),
        mu2: sq(mu2),
        variance1: sq(variance1),
        variance2: sq(variance2),
        sigma1: sq(sigma1),
        sigma2: sq(sigma2),
        rho: sq(rho),
        inv_one_minus_rho_sq: sq(inv_one_minus_rho_sq),
        normalization: sq(normalization),
    }
}

/// Deploys a bivariate-normal market via the factory.
pub async fn deploy_bivariate_market_with_event<A>(
    account: &A,
    factory: Felt,
    profile_id: u32,
    salt: Felt,
    metadata_hash: Felt,
    initial_dist: BivariateNormalDistributionRaw,
    initial_hints: BivariateNormalSqrtHintsRaw,
) -> Result<Felt, LifecycleError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(64);
    profile_id.encode(&mut calldata);
    calldata.push(salt);
    calldata.push(metadata_hash);
    initial_dist.encode(&mut calldata);
    initial_hints.encode(&mut calldata);
    encode_zero_overrides(&mut calldata);

    let call = Call {
        to: factory,
        selector: get_selector_from_name("deploy_bivariate_normal_market_from_profile")
            .expect("selector valid"),
        calldata,
    };
    let result = account.execute_v3(vec![call]).send().await.map_err(|e| {
        LifecycleError::Submit(format!("deploy_bivariate_normal_market_from_profile: {e}"))
    })?;
    let receipt = wait_for_receipt(account, result.transaction_hash).await?;
    extract_market_address_from_events(&receipt)
}
