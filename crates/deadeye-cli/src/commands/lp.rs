//! `deadeye lp …` — add / remove liquidity (Driver B).

use anyhow::{Context as _, Result};
use deadeye_core::Sq128;
use deadeye_sdk::bulk::Family;
use deadeye_starknet::{
    Account, BivariateMarketReader, BivariateMarketWriter, Call, Felt, LognormalMarketReader,
    LognormalMarketWriter, MultinoulliMarketReader, MultinoulliMarketWriter, NormalMarketReader,
    NormalMarketWriter, build_erc20_approve_call, collateral_allowance_base_units,
    types::common::AmmConfigRaw,
};

use crate::{
    cli::{LpAddArgs, LpCmd, LpRemoveArgs},
    commands::{
        call_bundle::{
            CalldataInput, WriteMode, build_write_account, calldata_result, labels,
            print_calldata_json, simulate_calls, submit_external_calls,
        },
        render_helpers::{submission_from_receipt, submission_from_trade_error},
        runtime_resolver::{build_provider, family_label, parse_felt},
    },
    context::AppContext,
    output::OutputMode,
};

pub(crate) async fn run(action: LpCmd, ctx: &AppContext, confirm: bool) -> Result<()> {
    match action {
        LpCmd::Add(args) => add(ctx, args, confirm).await,
        LpCmd::Remove(args) => remove(ctx, args, confirm).await,
    }
}

async fn add(ctx: &AppContext, args: LpAddArgs, confirm: bool) -> Result<()> {
    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider(ctx)?;
    let client = deadeye_sdk::DeadeyeClient::new(provider);
    let family = match args.family {
        Some(f) => f.as_sdk(),
        None => super::runtime_resolver::detect_family(ctx, &client, market).await?,
    };
    let label = family_label(family);

    if !confirm
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && ctx.renderer.mode() != OutputMode::Json
    {
        eprintln!(
            "About to add {} LP shares to {label} market {market:#x}.",
            args.amount
        );
        super::confirm_or_bail("Continue?")?;
    }

    let mode = if args.emit_calldata {
        WriteMode::EmitCalldata
    } else {
        WriteMode::Submit
    };
    let account = build_write_account(ctx, mode)?;
    let writer_provider = build_provider(ctx)?;
    let share_amount = Sq128::from_f64(args.amount)?.to_raw();

    // The deposit pulls collateral via `transfer_from`, so the multicall must
    // lead with an ERC-20 approve of the collateral token to the market —
    // exactly like `trade execute` (issue #29: a bare `add_liquidity` call
    // reverts on `Result::unwrap failed` with zero allowance).
    let approve_for = |config: &AmmConfigRaw| -> Call {
        // 5% allowance margin (matches the trade path / webapp buffer).
        let amount = collateral_allowance_base_units(args.amount, config.token_decimals, 5);
        build_erc20_approve_call(config.collateral_token, market, amount)
    };

    match family {
        Family::Normal => {
            let writer =
                NormalMarketWriter::new(NormalMarketReader::new(&writer_provider, market), account);
            let config = writer
                .reader()
                .config()
                .await
                .context("reading market config for the collateral approve")?;
            let calls = vec![
                approve_for(&config),
                writer.build_add_liquidity_call(share_amount),
            ];
            return finish_lp_calls(ctx, "lp_add", market, writer.account(), calls, mode).await;
        },
        Family::Lognormal => {
            let writer = LognormalMarketWriter::new(
                LognormalMarketReader::new(&writer_provider, market),
                account,
            );
            let config = writer
                .reader()
                .config()
                .await
                .context("reading market config for the collateral approve")?;
            let calls = vec![
                approve_for(&config),
                writer.build_add_liquidity_call(share_amount),
            ];
            return finish_lp_calls(ctx, "lp_add", market, writer.account(), calls, mode).await;
        },
        Family::Multinoulli => {
            let _ = (writer_provider, account, share_amount);
            let _ = MultinoulliMarketReader::<&deadeye_sdk::starknet::JsonRpcProvider>::new;
            let _ = MultinoulliMarketWriter::<
                &deadeye_sdk::starknet::JsonRpcProvider,
                deadeye_starknet::OwnedAccount,
            >::new;
            anyhow::bail!(
                "lp_add is not yet wired for multinoulli markets (no `add_liquidity` on the \
                 writer). The AMM's add_liquidity entrypoint is byte-identical across \
                 families, so `--family normal` works as a stopgap"
            );
        },
        Family::Bivariate => {
            let writer = BivariateMarketWriter::new(
                BivariateMarketReader::new(&writer_provider, market),
                account,
            );
            let config = writer
                .reader()
                .config()
                .await
                .context("reading market config for the collateral approve")?;
            let calls = vec![
                approve_for(&config),
                writer.build_add_liquidity_call(share_amount),
            ];
            return finish_lp_calls(ctx, "lp_add", market, writer.account(), calls, mode).await;
        },
    }
}

async fn remove(ctx: &AppContext, args: LpRemoveArgs, confirm: bool) -> Result<()> {
    if !(args.fraction > 0.0 && args.fraction <= 1.0) {
        anyhow::bail!("`--fraction` must be in (0, 1]");
    }
    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider(ctx)?;
    let client = deadeye_sdk::DeadeyeClient::new(provider);
    let family = match args.family {
        Some(f) => f.as_sdk(),
        None => super::runtime_resolver::detect_family(ctx, &client, market).await?,
    };
    let label = family_label(family);

    let writer_provider = build_provider(ctx)?;
    let share_amount_f64 = match family {
        Family::Normal => {
            let reader = NormalMarketReader::new(&writer_provider, market);
            let lp = reader.lp_info().await.context("reading LP info")?;
            Sq128::from_raw(lp.total_shares).to_f64() * args.fraction
        },
        Family::Lognormal => {
            let reader = LognormalMarketReader::new(&writer_provider, market);
            let lp = reader.lp_info().await.context("reading LP info")?;
            Sq128::from_raw(lp.total_shares).to_f64() * args.fraction
        },
        Family::Multinoulli => {
            let reader = MultinoulliMarketReader::new(&writer_provider, market);
            let lp = reader.lp_info().await.context("reading LP info")?;
            Sq128::from_raw(lp.total_shares).to_f64() * args.fraction
        },
        Family::Bivariate => {
            let reader = BivariateMarketReader::new(&writer_provider, market);
            let lp = reader.lp_info().await.context("reading LP info")?;
            Sq128::from_raw(lp.total_shares).to_f64() * args.fraction
        },
    };

    if !confirm
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && ctx.renderer.mode() != OutputMode::Json
    {
        eprintln!(
            "About to remove {share_amount_f64:.6} LP shares ({:.2}%) from {label} market {market:#x}.",
            args.fraction * 100.0
        );
        super::confirm_or_bail("Continue?")?;
    }

    let mode = if args.emit_calldata {
        WriteMode::EmitCalldata
    } else {
        WriteMode::Submit
    };
    let account = build_write_account(ctx, mode)?;
    let share_amount = Sq128::from_f64(share_amount_f64)?.to_raw();
    let writer_provider_for_write = build_provider(ctx)?;

    match family {
        Family::Normal => {
            let writer = NormalMarketWriter::new(
                NormalMarketReader::new(&writer_provider_for_write, market),
                account,
            );
            let calls = vec![writer.build_remove_liquidity_call(share_amount)];
            return finish_lp_calls(ctx, "lp_remove", market, writer.account(), calls, mode).await;
        },
        Family::Lognormal => {
            let writer = LognormalMarketWriter::new(
                LognormalMarketReader::new(&writer_provider_for_write, market),
                account,
            );
            let calls = vec![writer.build_remove_liquidity_call(share_amount)];
            return finish_lp_calls(ctx, "lp_remove", market, writer.account(), calls, mode).await;
        },
        Family::Multinoulli => {
            let _ = (writer_provider_for_write, account, share_amount);
            anyhow::bail!(
                "lp_remove is not yet wired for multinoulli markets (no `remove_liquidity` on \
                 the writer). The AMM's remove_liquidity entrypoint is byte-identical across \
                 families, so `--family normal` works as a stopgap"
            );
        },
        Family::Bivariate => {
            let writer = BivariateMarketWriter::new(
                BivariateMarketReader::new(&writer_provider_for_write, market),
                account,
            );
            let calls = vec![writer.build_remove_liquidity_call(share_amount)];
            return finish_lp_calls(ctx, "lp_remove", market, writer.account(), calls, mode).await;
        },
    }
}

async fn finish_lp_calls<A>(
    ctx: &AppContext,
    action: &'static str,
    market: Felt,
    account: &A,
    calls: Vec<Call>,
    mode: WriteMode,
) -> Result<()>
where
    A: Account,
{
    let call_labels = match action {
        "lp_add" => labels(["approve", "add_liquidity"]),
        "lp_remove" => labels(["remove_liquidity"]),
        _ => labels(["unknown"]),
    };
    if mode == WriteMode::EmitCalldata {
        let sim = simulate_calls("lp(preflight)", market, account, &calls).await;
        let result = calldata_result(CalldataInput {
            action: "lp(emit-calldata)",
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
        let sim = simulate_calls("lp(preflight)", market, account, &calls).await;
        if sim.accepted {
            submit_external_calls(ctx, action, market, account.address(), &calls, &call_labels)
                .await?
        } else {
            sim
        }
    } else {
        match account.execute(calls).await {
            Ok(receipt) => submission_from_receipt(action, format!("{market:#x}"), receipt),
            Err(e) => submission_from_trade_error(
                action,
                format!("{market:#x}"),
                &deadeye_starknet::TradeError::from_contract(e),
            ),
        }
    };
    ctx.renderer.print(&result)
}
