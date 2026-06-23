//! Starknet contract bindings for Deadeye markets.
//!
//! This crate sits between [`deadeye-core`](::deadeye_core) and the
//! [`starknet`](https://github.com/software-mansion/starknet-rust) ecosystem.
//! It provides:
//!
//! 1. **Cairo Serde encoders** — the [`CairoSerde`] trait and concrete
//!    implementations for `Sq128Raw`, `NormalDistributionRaw`, primitive
//!    integers, and `bool`.
//! 2. **Entry-point selectors** — pre-computed [`starknet_core::types::Felt`]
//!    constants for every AMM and Factory function we call.
//! 3. **View-call clients** — typed wrappers around an arbitrary [`Provider`]
//!    (the trait abstracts over `starknet-providers::JsonRpcClient`, mocks,
//!    multi-RPC racers, etc.).
//!
//! Write-paths (transactions, signatures) live in `deadeye-sdk`; this
//! crate is deliberately read-only so
//! it can be reused inside read-only indexers, dashboards, and watch-tower
//! bots without ever holding key material.

#![doc(html_no_source)]

pub mod account;
pub mod bivariate_amm;
pub mod cairo_serde;
#[cfg(feature = "account")]
pub mod chain_probe;
pub mod collateral;
pub mod error;
pub mod execution;
pub mod factory;
pub mod lognormal_amm;
#[cfg(feature = "provider")]
pub mod multi_rpc;
pub mod multinoulli_amm;
pub mod nonce;
pub mod normal_amm;
pub mod oracle;
pub mod pricing;
pub mod provider;
pub mod retry;
pub mod runtime;
pub mod selectors;
#[cfg(feature = "account")]
pub mod signer;
pub mod types;
#[cfg(feature = "account")]
pub mod wallet_pool;

pub use account::Account;
#[cfg(feature = "account")]
pub use account::{
    AccountWithNonceManager, FeeBumpPolicy, FeeEstimate, GasParams, OwnedAccount, PriceUnit,
};
pub use bivariate_amm::{BivariateMarketReader, BivariateMarketWriter, BivariateTradeQuote};
pub use cairo_serde::{CairoSerde, CairoSerdeError};
#[cfg(feature = "account")]
pub use chain_probe::{
    LognormalProbeOutcome, ProbeOutcome, refine_lognormal_quote, refine_normal_quote,
};
pub use collateral::{
    CollateralTokenReader, CollateralTokenWriter, MAINNET_XP_TOKEN_ADDRESS, U256Value,
    build_claim_initial_grant_call, build_erc20_approve_call, collateral_allowance_base_units,
};
pub use error::{
    ContractError, ContractResult, TradeError, TradeRejectionReason, TradeResult,
    VerificationSubReason, parse_revert_reason,
};
pub use execution::{Call, ExecutionReceipt, SimOutcome};
pub use factory::{FactoryReader, FactoryWriter};
pub use lognormal_amm::{LognormalMarketReader, LognormalMarketWriter, LognormalTradeQuote};
#[cfg(feature = "provider")]
pub use multi_rpc::{EndpointHealth, EndpointHealthState, MultiRpcProvider, RpcConfig};
pub use multinoulli_amm::{
    MultinoulliMarketReader, MultinoulliMarketWriter, MultinoulliTradeQuote,
};
pub use nonce::{NonceError, NonceFetcher, NonceGuard, NonceManager, NonceSnapshot};
pub use normal_amm::{NormalMarketReader, NormalMarketWriter, NormalTradeQuote};
pub use oracle::OracleClient;
#[cfg(feature = "provider")]
pub use provider::JsonRpcProvider;
pub use provider::Provider;
#[cfg(feature = "account")]
pub use signer::{
    DeadeyeSigner, LocalSigner, RemoteSigner, RemoteSignerConfig, SignerAdapter, SignerError,
    SimulationSigner,
};
pub use starknet_core::types::Felt;
#[cfg(feature = "account")]
pub use wallet_pool::{PoolSelector, WalletLease, WalletPool, WalletPoolError};
