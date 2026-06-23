//! High-level Rust SDK for Deadeye prediction markets.
//!
//! Designed for **market makers**: the API is synchronous-leaning, allocation-
//! aware, and exposes the same primitives the on-chain verifier uses so that
//! a MM can build a quote, validate it locally, and submit it without
//! round-tripping for confirmation.
//!
//! ## Architecture
//!
//! ```text
//! deadeye-sdk          ← high-level facade (this crate)
//!   ├── deadeye-collateral   ← off-chain numeric primitives
//!   ├── deadeye-starknet     ← view calls, calldata encoders
//!   ├── deadeye-artifacts    ← embedded ABIs
//!   └── deadeye-core         ← Sq128, distributions, errors
//! ```
//!
//! Each layer is independently usable. A latency-critical MM that prefers
//! to drive [`deadeye_starknet`] directly can do so; the SDK is
//! convenience, not a wall.
//!
//! ## Getting started — a 30-line worked example
//!
//! End-to-end: bootstrap an RPC client + signer, read the market, quote
//! a trade, execute it, inspect the position, sell, then claim after
//! settlement.
//!
//! ```no_run
//! use deadeye_sdk::{
//!     core::{distribution::NormalDistributionRaw, sq128::Sq128Raw},
//!     starknet::{
//!         BivariateMarketReader, BivariateMarketWriter, Felt, JsonRpcProvider,
//!         NormalMarketReader, NormalMarketWriter, OwnedAccount, TradeError, TradeRejectionReason,
//!     },
//! };
//! use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let rpc = JsonRpcClient::new(HttpTransport::new(
//!     "http://localhost:5050".parse::<url::Url>()?,
//! ));
//! let provider = JsonRpcProvider::new(rpc);
//!
//! let (market, runtime): (Felt, Felt) = (Felt::ZERO, Felt::ZERO);
//! let chain_id = Felt::ZERO;
//! let signing_key = Felt::ZERO;
//! let address = Felt::ZERO;
//!
//! let reader = NormalMarketReader::new(&provider, market);
//! let signer = OwnedAccount::from_signing_key(
//!     JsonRpcClient::new(HttpTransport::new(
//!         "http://localhost:5050".parse::<url::Url>()?,
//!     )),
//!     address,
//!     signing_key,
//!     chain_id,
//! );
//! let writer = NormalMarketWriter::new(reader, signer);
//!
//! // 1) Preflight: chain-correct hints + check_trade_view in one call.
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
//!
//! // 2) Branch on the typed verdict.
//! if !quote.on_chain_will_accept {
//!     eprintln!("trade rejected: {:?}", quote.rejection);
//!     return Ok(());
//! }
//!
//! // 3) Submit.
//! match writer.execute_quote(quote).await {
//!     Ok(receipt) => println!("trade tx: {:#x}", receipt.transaction_hash),
//!     Err(TradeError::Rejected {
//!         reason: TradeRejectionReason::StaleState { .. },
//!         ..
//!     }) => {
//!         // typical MM retry: re-read, re-quote, re-submit
//!     },
//!     Err(e) => return Err(e.into()),
//! }
//!
//! // 4) Read the resulting position.
//! let _pos = writer.reader().position(address).await?;
//!
//! // 5) Close it out — guards + hints built internally.
//! let _ = writer.sell_position(runtime, 0).await?;
//!
//! // 6) After settlement: claim. `claim_for(trader)` returns the
//! //    decoded payout (a `ClaimResultRaw`). Settlement is driven by
//! //    a market admin via `FactoryWriter::settle_normal_market`.
//! let _claim = writer.claim().await?;
//! # Ok(()) }
//! ```
//!
//! The same shape applies to every family — see
//! [`normal`], [`lognormal`], [`multinoulli`], [`bivariate`] for the
//! family-specific idioms.

#![doc(html_no_source)]

pub mod backtest;
pub mod bulk;
pub mod client;
pub mod error;
pub mod journal;
pub mod legs;
pub mod portfolio;
pub mod quote;
pub mod stream;

#[cfg(feature = "normal-market")]
pub mod normal;

#[cfg(feature = "lognormal-market")]
pub mod lognormal;

#[cfg(feature = "multinoulli-market")]
pub mod multinoulli;

#[cfg(feature = "bivariate-market")]
pub mod bivariate;

pub mod factory;
pub mod oracle;
pub mod read_models;

pub use backtest::{
    BacktestEngine, BacktestResult, EventDistribution, MarketEvent, MarketState, SimDistribution,
    Strategy, StrategyAction,
};
pub use bulk::{BulkReader, DistributionSnapshot, Family, MarketStateSnapshot, Position};
pub use client::DeadeyeClient;
pub use deadeye_artifacts as artifacts;
pub use deadeye_collateral as collateral;
pub use deadeye_core as core;
pub use deadeye_starknet as starknet;
#[cfg(feature = "jsonrpc")]
pub use deadeye_starknet::is_account_deployed;
pub use error::{SdkError, SdkResult};
pub use journal::{
    EntryKind, JournalEntry, JournalError, JournalSink, JournalledBivariateWriter,
    JournalledLognormalWriter, JournalledMultinoulliWriter, JournalledNormalWriter, TradeJournal,
};
pub use legs::{LegInfo, LegValuation, PositionLegs, PositionValuation, SettlementPoint};
pub use portfolio::{HedgeRecommendation, LpEntry, MarketRef, Portfolio, PositionEntry};
pub use stream::{
    BlockNumberSource, CandidateQuote, MarketStateStream, MarketStateUpdate, QuoteSnapshot,
    StarknetBlockSource, StreamConfig,
};
