//! Factory configuration helpers — `configure_market_type` +
//! `upsert_deploy_profile`.

use deadeye_core::sq128::Sq128Raw;
use deadeye_starknet::{CairoSerde, types::common::FeeConfigRaw};
use starknet_accounts::{Account, ConnectedAccount};
use starknet_core::{
    types::{Call, Felt},
    utils::get_selector_from_name,
};
use thiserror::Error;

/// Errors emitted by the factory setup helpers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FactorySetupError {
    /// Submission failed.
    #[error("factory setup failed: {0}")]
    Submit(String),
}

/// Market-family discriminants. Must match the on-chain Cairo `MarketKind`
/// values: Normal=1, Multinoulli=2, Lognormal=4, BivariateNormal=5 (see
/// `discriminant()` below; 3 is unassigned, 6 is skew-normal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketKind {
    /// Normal-distribution AMM.
    Normal,
    /// Lognormal-distribution AMM.
    Lognormal,
    /// Bivariate-normal AMM.
    BivariateNormal,
    /// Multinoulli AMM.
    Multinoulli,
}

impl MarketKind {
    /// Discriminant byte for the on-chain enum.
    ///
    /// Source: `the-situation/packages/onchain-core/src/common.cairo:783-789`.
    /// Values are non-contiguous (3 is reserved for skew-normal, omitted here).
    #[must_use]
    pub const fn discriminant(self) -> u8 {
        match self {
            Self::Normal => 1,
            Self::Multinoulli => 2,
            Self::Lognormal => 4,
            Self::BivariateNormal => 5,
        }
    }
}

/// Configure a market type on the factory: register the AMM + runtime class
/// hashes and the plugin contract address.
pub async fn configure_market_type<A>(
    account: A,
    factory: Felt,
    kind: MarketKind,
    amm_class_hash: Felt,
    runtime_class_hash: Felt,
    plugin_address: Felt,
    enabled: bool,
) -> Result<(), FactorySetupError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(6);
    kind.discriminant().encode(&mut calldata);
    calldata.push(amm_class_hash);
    calldata.push(runtime_class_hash);
    calldata.push(plugin_address);
    enabled.encode(&mut calldata);
    let call = Call {
        to: factory,
        selector: get_selector_from_name("configure_market_type").expect("selector valid"),
        calldata,
    };
    account
        .execute_v3(vec![call])
        .send()
        .await
        .map_err(|e| FactorySetupError::Submit(format!("configure_market_type: {e}")))?;
    Ok(())
}

/// Parameters for a deploy profile.
#[derive(Debug, Clone, Copy)]
pub struct DeployProfileParams {
    /// Market family discriminant.
    pub market_type: MarketKind,
    /// Collateral token contract.
    pub collateral_token: Felt,
    /// Token decimals.
    pub token_decimals: u8,
    /// Internal precision used by the AMM.
    pub internal_decimals: u8,
    /// AMM `k`.
    pub k: Sq128Raw,
    /// Initial backing.
    pub backing: Sq128Raw,
    /// Tolerance.
    pub tolerance: Sq128Raw,
    /// Min trade collateral.
    pub min_trade_collateral: Sq128Raw,
    /// Fee config.
    pub fee_config: FeeConfigRaw,
    /// Extension contract.
    pub extension: Felt,
    /// Extension call-points bitfield.
    pub extension_call_points: u16,
}

/// Upsert a deploy profile on the factory.
pub async fn upsert_deploy_profile<A>(
    account: A,
    factory: Felt,
    profile_id: u32,
    profile: DeployProfileParams,
) -> Result<(), FactorySetupError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(32);
    profile_id.encode(&mut calldata);
    profile.market_type.discriminant().encode(&mut calldata);
    calldata.push(profile.collateral_token);
    profile.token_decimals.encode(&mut calldata);
    profile.internal_decimals.encode(&mut calldata);
    profile.k.encode(&mut calldata);
    profile.backing.encode(&mut calldata);
    profile.tolerance.encode(&mut calldata);
    profile.min_trade_collateral.encode(&mut calldata);
    profile.fee_config.encode(&mut calldata);
    calldata.push(profile.extension);
    profile.extension_call_points.encode(&mut calldata);

    let call = Call {
        to: factory,
        selector: get_selector_from_name("upsert_deploy_profile").expect("selector valid"),
        calldata,
    };
    account
        .execute_v3(vec![call])
        .send()
        .await
        .map_err(|e| FactorySetupError::Submit(format!("upsert_deploy_profile: {e}")))?;
    Ok(())
}

/// Enable a deploy profile.
pub async fn set_profile_enabled<A>(
    account: A,
    factory: Felt,
    profile_id: u32,
    enabled: bool,
) -> Result<(), FactorySetupError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(2);
    profile_id.encode(&mut calldata);
    enabled.encode(&mut calldata);
    let call = Call {
        to: factory,
        selector: get_selector_from_name("set_profile_enabled").expect("selector valid"),
        calldata,
    };
    account
        .execute_v3(vec![call])
        .send()
        .await
        .map_err(|e| FactorySetupError::Submit(format!("set_profile_enabled: {e}")))?;
    Ok(())
}

/// Enable a market type.
pub async fn set_market_type_enabled<A>(
    account: A,
    factory: Felt,
    kind: MarketKind,
    enabled: bool,
) -> Result<(), FactorySetupError>
where
    A: Account + ConnectedAccount + Sync + Send,
{
    let mut calldata = Vec::with_capacity(2);
    kind.discriminant().encode(&mut calldata);
    enabled.encode(&mut calldata);
    let call = Call {
        to: factory,
        selector: get_selector_from_name("set_market_type_enabled").expect("selector valid"),
        calldata,
    };
    account
        .execute_v3(vec![call])
        .send()
        .await
        .map_err(|e| FactorySetupError::Submit(format!("set_market_type_enabled: {e}")))?;
    Ok(())
}
