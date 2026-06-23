//! `deadeye claim` — settled-position claim (self or admin-on-behalf).

use anyhow::Result;
use deadeye_sdk::bulk::Family;
use deadeye_starknet::{
    Account, BivariateMarketReader, BivariateMarketWriter, Call, LognormalMarketReader,
    LognormalMarketWriter, MultinoulliMarketReader, MultinoulliMarketWriter, NormalMarketReader,
    NormalMarketWriter,
};

use crate::{
    cli::ClaimArgs,
    commands::{
        call_bundle::{
            CalldataInput, WriteMode, build_write_account, calldata_result, labels,
            print_calldata_json, simulate_calls, submit_external_calls,
        },
        render_helpers::{submission_from_receipt, submission_from_trade_error},
        runtime_resolver::{build_provider, parse_felt},
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

    let mode = if args.emit_calldata {
        WriteMode::EmitCalldata
    } else {
        WriteMode::Submit
    };
    let claim_entrypoint = if trader_override.is_some() {
        "claim_for"
    } else {
        "claim"
    };
    let account = build_write_account(ctx, mode)?;
    let writer_provider = build_provider(ctx)?;

    // Map `claim` (no position / no claim) into a friendly "no-op" rather
    // than an error. Any other revert is surfaced verbatim.
    match family {
        Family::Normal => {
            let writer =
                NormalMarketWriter::new(NormalMarketReader::new(&writer_provider, market), account);
            let calls = if let Some(t) = trader_override {
                vec![writer.build_claim_for_call(t)]
            } else {
                vec![writer.build_claim_call()]
            };
            finish_claim_calls(ctx, market, writer.account(), calls, mode, claim_entrypoint).await
        },
        Family::Lognormal => {
            let writer = LognormalMarketWriter::new(
                LognormalMarketReader::new(&writer_provider, market),
                account,
            );
            let calls = if let Some(t) = trader_override {
                vec![writer.build_claim_for_call(t)]
            } else {
                vec![writer.build_claim_call()]
            };
            finish_claim_calls(ctx, market, writer.account(), calls, mode, claim_entrypoint).await
        },
        Family::Multinoulli => {
            let writer = MultinoulliMarketWriter::new(
                MultinoulliMarketReader::new(&writer_provider, market),
                account,
            );
            let calls = if let Some(t) = trader_override {
                vec![writer.build_claim_for_call(t)]
            } else {
                vec![writer.build_claim_call()]
            };
            finish_claim_calls(ctx, market, writer.account(), calls, mode, claim_entrypoint).await
        },
        Family::Bivariate => {
            let writer = BivariateMarketWriter::new(
                BivariateMarketReader::new(&writer_provider, market),
                account,
            );
            let calls = if let Some(t) = trader_override {
                vec![writer.build_claim_for_call(t)]
            } else {
                vec![writer.build_claim_call()]
            };
            finish_claim_calls(ctx, market, writer.account(), calls, mode, claim_entrypoint).await
        },
    }
}

async fn finish_claim_calls<A>(
    ctx: &AppContext,
    market: starknet_core::types::Felt,
    account: &A,
    calls: Vec<Call>,
    mode: WriteMode,
    entrypoint: &'static str,
) -> Result<()>
where
    A: Account,
{
    let call_labels = labels([entrypoint]);
    if mode == WriteMode::EmitCalldata {
        let sim = simulate_calls("claim(preflight)", market, account, &calls).await;
        let result = calldata_result(CalldataInput {
            action: "claim(emit-calldata)",
            family: None,
            target: market,
            account: account.address(),
            calls: &calls,
            labels: &call_labels,
            simulation: &sim,
            computed_collateral: None,
        });
        return print_calldata_json(&result);
    }
    let result = if ctx.config.uses_external_signer() {
        let sim = simulate_calls("claim(preflight)", market, account, &calls).await;
        if sim.accepted {
            submit_external_calls(
                ctx,
                "claim",
                market,
                account.address(),
                &calls,
                &call_labels,
            )
            .await?
        } else {
            sim
        }
    } else {
        match account.execute(calls).await {
            Ok(receipt) => submission_from_receipt("claim", format!("{market:#x}"), receipt),
            Err(e) => submission_from_trade_error(
                "claim",
                format!("{market:#x}"),
                &deadeye_starknet::TradeError::from_contract(e),
            ),
        }
    };
    ctx.renderer.print(&result)
}
