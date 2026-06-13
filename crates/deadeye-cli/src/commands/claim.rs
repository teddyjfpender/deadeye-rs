//! `deadeye claim` — settled-position claim (self or admin-on-behalf).

use anyhow::Result;
use deadeye_sdk::bulk::Family;
use deadeye_starknet::{
    BivariateMarketReader, BivariateMarketWriter, LognormalMarketReader, LognormalMarketWriter,
    MultinoulliMarketReader, MultinoulliMarketWriter, NormalMarketReader, NormalMarketWriter,
};

use crate::{
    cli::ClaimArgs,
    commands::{
        render_helpers::{submission_from_receipt, submission_from_trade_error},
        runtime_resolver::{build_owned_account, build_provider, parse_felt},
    },
    context::AppContext,
};

pub(crate) async fn run(args: ClaimArgs, ctx: &AppContext) -> Result<()> {
    let market = parse_felt("market address", &args.market)?;
    let trader_override = match args.trader {
        Some(s) => Some(parse_felt("trader address", &s)?),
        None => None,
    };
    let provider = build_provider(ctx)?;
    let client = deadeye_sdk::DeadeyeClient::new(provider);
    let family = match args.family {
        Some(f) => f.as_sdk(),
        None => super::runtime_resolver::detect_family(ctx, &client, market).await?,
    };

    let account = build_owned_account(ctx)?;
    let writer_provider = build_provider(ctx)?;
    let market_hex = format!("{market:#x}");

    // Map `claim` (no position / no claim) into a friendly "no-op" rather
    // than an error. Any other revert is surfaced verbatim.
    let result = match family {
        Family::Normal => {
            let writer =
                NormalMarketWriter::new(NormalMarketReader::new(&writer_provider, market), account);
            let outcome = if let Some(t) = trader_override {
                writer.claim_for(t).await
            } else {
                writer.claim().await
            };
            match outcome {
                Ok(receipt) => submission_from_receipt("claim", market_hex, receipt),
                Err(e) => submission_from_trade_error(
                    "claim",
                    market_hex,
                    &deadeye_starknet::TradeError::from_contract(e),
                ),
            }
        },
        Family::Lognormal => {
            let writer = LognormalMarketWriter::new(
                LognormalMarketReader::new(&writer_provider, market),
                account,
            );
            let outcome = if let Some(t) = trader_override {
                writer.claim_for(t).await
            } else {
                writer.claim().await
            };
            match outcome {
                Ok(receipt) => submission_from_receipt("claim", market_hex, receipt),
                Err(e) => submission_from_trade_error(
                    "claim",
                    market_hex,
                    &deadeye_starknet::TradeError::from_contract(e),
                ),
            }
        },
        Family::Multinoulli => {
            let writer = MultinoulliMarketWriter::new(
                MultinoulliMarketReader::new(&writer_provider, market),
                account,
            );
            let outcome = if let Some(t) = trader_override {
                writer.claim_for(t).await
            } else {
                writer.claim().await
            };
            match outcome {
                Ok(receipt) => submission_from_receipt("claim", market_hex, receipt),
                Err(e) => submission_from_trade_error(
                    "claim",
                    market_hex,
                    &deadeye_starknet::TradeError::from_contract(e),
                ),
            }
        },
        Family::Bivariate => {
            let writer = BivariateMarketWriter::new(
                BivariateMarketReader::new(&writer_provider, market),
                account,
            );
            let outcome = if let Some(t) = trader_override {
                writer.claim_for(t).await
            } else {
                writer.claim().await
            };
            match outcome {
                Ok(receipt) => submission_from_receipt("claim", market_hex, receipt),
                Err(e) => submission_from_trade_error(
                    "claim",
                    market_hex,
                    &deadeye_starknet::TradeError::from_contract(e),
                ),
            }
        },
    };
    ctx.renderer.print(&result)
}
