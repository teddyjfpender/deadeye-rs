//! `deadeye watch <market>` — live block-driven stream.
//!
//! Subscribes via the SDK's [`MarketStateStream`] and renders one
//! update per observed block height transition. JSON mode emits one
//! JSON object per line (NDJSON-friendly).
//!
//! Non-TTY behaviour: append-only line per update. TTY behaviour: same
//! for now (in-place cursor positioning is deferred to v1.1).

use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use deadeye_core::Sq128;
use deadeye_sdk::{
    DeadeyeClient,
    bulk::{DistributionSnapshot, Family},
    starknet::JsonRpcProvider,
    stream::{
        MarketStateStream, MarketStateUpdate, QuoteSnapshot, StarknetBlockSource, StreamConfig,
    },
};
use deadeye_starknet::Felt;
use futures::StreamExt as _;
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
use tokio::sync::Notify;
use url::Url;

use crate::{
    cli::WatchArgs,
    commands::{
        render_helpers::WatchUpdate,
        runtime_resolver::{family_label, parse_felt},
    },
    context::AppContext,
    output::OutputMode,
};

pub(crate) async fn run(args: WatchArgs, ctx: &AppContext) -> Result<()> {
    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider_owned(ctx)?;
    let client = DeadeyeClient::new(provider);
    let family = match args.family {
        Some(f) => f.as_sdk(),
        None => super::runtime_resolver::detect_family(ctx, &client, market).await?,
    };
    let label = family_label(family);

    // We need a separate provider for the block-number source because
    // the stream's block source owns its inner provider.
    let block_url = Url::parse(&ctx.config.rpc_url)?;
    let block_rpc = JsonRpcClient::new(HttpTransport::new(block_url));
    let block_source = StarknetBlockSource::new(block_rpc);

    let candidate_quote = args
        .show_quote_for
        .as_deref()
        .map(|s| parse_quote_spec(s, family))
        .transpose()?;
    // Resolve runtime once if a quote is requested.
    let runtime: Option<Felt> = if candidate_quote.is_some() {
        Some(crate::commands::runtime_resolver::resolve_runtime(
            args.runtime.as_deref(),
            family,
        )?)
    } else {
        None
    };
    let config = StreamConfig {
        poll_interval: Duration::from_millis(args.poll_interval_ms),
        include_distribution: true,
        include_lp_info: true,
        include_quote_for_candidate: candidate_quote
            .zip(runtime)
            .map(|(spec, rt)| build_candidate_quote(family, market, rt, spec)),
    };

    let mut stream = MarketStateStream::subscribe(client, block_source, family, market, config);

    // SIGINT handling: notify the loop to break.
    let stop = Arc::new(Notify::new());
    let stop_for_signal = Arc::<Notify>::clone(&stop);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        stop_for_signal.notify_one();
    });

    let mut count: u32 = 0;
    let max_updates = args.max_updates.unwrap_or(u32::MAX);
    loop {
        tokio::select! {
            () = stop.notified() => {
                ctx.renderer.warning("interrupted — shutting down stream");
                break;
            }
            update = stream.next() => {
                let Some(update) = update else { break; };
                emit(ctx, label, family, &update);
                count = count.saturating_add(1);
                if count >= max_updates {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn build_provider_owned(ctx: &AppContext) -> Result<crate::context::CliProvider> {
    let url = Url::parse(&ctx.config.rpc_url)?;
    Ok(deadeye_starknet::retry::RetryingProvider::new(
        JsonRpcProvider::new(JsonRpcClient::new(HttpTransport::new(url))),
    ))
}

/// Local mirror of the parsed `--show-quote-for` spec.
#[derive(Debug, Clone)]
struct QuoteSpec {
    mean: f64,
    variance: f64,
    rho: Option<f64>,
    mu2: Option<f64>,
}

fn parse_quote_spec(s: &str, family: Family) -> Result<QuoteSpec> {
    let mut mean = None;
    let mut variance = None;
    let mut rho = None;
    let mut mu2 = None;
    for kv in s.split(',') {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--show-quote-for expects `key=value` pairs"))?;
        let value: f64 = v.trim().parse()?;
        match k.trim() {
            "mean" | "mu" | "mu1" => mean = Some(value),
            "variance" | "var" => variance = Some(value),
            "rho" => rho = Some(value),
            "mu2" => mu2 = Some(value),
            other => anyhow::bail!("unrecognised key `{other}` in --show-quote-for"),
        }
    }
    let mean = mean.ok_or_else(|| anyhow::anyhow!("--show-quote-for missing `mean`"))?;
    let variance =
        variance.ok_or_else(|| anyhow::anyhow!("--show-quote-for missing `variance`"))?;
    let _ = family;
    Ok(QuoteSpec {
        mean,
        variance,
        rho,
        mu2,
    })
}

fn build_candidate_quote(
    family: Family,
    _market: Felt,
    runtime: Felt,
    spec: QuoteSpec,
) -> deadeye_sdk::stream::CandidateQuote {
    use deadeye_sdk::stream::CandidateQuote;
    let supplied = Sq128::ZERO.to_raw();
    let zero = Sq128::ZERO.to_raw();
    match family {
        Family::Normal => CandidateQuote::Normal {
            runtime,
            // Raw derived via from_variance so (σ, σ²) stays Sq128-exact —
            // the runtime rejects inconsistent encodings (issue #36).
            candidate: deadeye_core::NormalDistribution::from_variance(
                Sq128::from_f64(spec.mean).unwrap_or(Sq128::ZERO),
                Sq128::from_f64(spec.variance).unwrap_or(Sq128::ZERO),
            )
            .map(|d| deadeye_core::Distribution::to_raw(&d))
            .unwrap_or_else(|_| deadeye_core::distribution::NormalDistributionRaw {
                mean: Sq128::ZERO.to_raw(),
                variance: Sq128::ZERO.to_raw(),
                sigma: Sq128::ZERO.to_raw(),
            }),
            x_star: Sq128::from_f64(spec.mean).unwrap_or(Sq128::ZERO).to_raw(),
            supplied_collateral: supplied,
            collateral_pad: zero,
        },
        Family::Lognormal => CandidateQuote::Lognormal {
            runtime,
            // Same Sq128-exact (σ, σ²) requirement as the normal arm.
            candidate: deadeye_core::LognormalDistribution::from_variance(
                Sq128::from_f64(spec.mean).unwrap_or(Sq128::ZERO),
                Sq128::from_f64(spec.variance).unwrap_or(Sq128::ZERO),
            )
            .map(|d| deadeye_core::Distribution::to_raw(&d))
            .unwrap_or_else(|_| deadeye_core::distribution::LognormalDistributionRaw {
                mu: Sq128::ZERO.to_raw(),
                variance: Sq128::ZERO.to_raw(),
                sigma: Sq128::ZERO.to_raw(),
            }),
            x_star: Sq128::from_f64(spec.mean).unwrap_or(Sq128::ZERO).to_raw(),
            supplied_collateral: supplied,
            collateral_pad: zero,
        },
        Family::Multinoulli | Family::Bivariate => {
            // Unsupported in this driver; fall back to a no-op normal quote
            // — the stream will skip it gracefully.
            let _ = (spec.rho, spec.mu2);
            CandidateQuote::Normal {
                runtime,
                candidate: deadeye_core::distribution::NormalDistributionRaw {
                    mean: zero,
                    variance: zero,
                    sigma: zero,
                },
                x_star: zero,
                supplied_collateral: supplied,
                collateral_pad: zero,
            }
        },
    }
}

fn emit(ctx: &AppContext, label: &'static str, family: Family, update: &MarketStateUpdate) {
    let (mean, sigma, variance) = match &update.distribution {
        Some(DistributionSnapshot::Normal(d)) => (
            Some(Sq128::from_raw(d.mean).to_f64()),
            Some(Sq128::from_raw(d.sigma).to_f64()),
            Some(Sq128::from_raw(d.variance).to_f64()),
        ),
        Some(DistributionSnapshot::Lognormal(d)) => (
            Some(Sq128::from_raw(d.mu).to_f64()),
            Some(Sq128::from_raw(d.sigma).to_f64()),
            Some(Sq128::from_raw(d.variance).to_f64()),
        ),
        _ => (None, None, None),
    };
    let (lp_b, lp_s) = match update.lp_info {
        Some(lp) => (
            Some(Sq128::from_raw(lp.total_backing_deposited).to_f64()),
            Some(Sq128::from_raw(lp.total_shares).to_f64()),
        ),
        None => (None, None),
    };
    let (q_acc, q_rc, q_xs) = match &update.quote {
        Some(QuoteSnapshot::Normal(q)) => (
            Some(q.on_chain_will_accept),
            Some(Sq128::from_raw(q.required_collateral).to_f64()),
            Some(Sq128::from_raw(q.x_star).to_f64()),
        ),
        Some(QuoteSnapshot::Lognormal(q)) => (
            Some(q.on_chain_will_accept),
            Some(Sq128::from_raw(q.required_collateral).to_f64()),
            Some(Sq128::from_raw(q.x_star).to_f64()),
        ),
        _ => (None, None, None),
    };
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let item = WatchUpdate {
        family: label,
        market: format!("{:#x}", update.market),
        block_number: update.block_number,
        timestamp_unix_ms: ts_ms,
        mean,
        sigma,
        variance,
        lp_total_backing: lp_b,
        lp_total_shares: lp_s,
        quote_accepts: q_acc,
        quote_required_collateral: q_rc,
        quote_x_star: q_xs,
    };
    let _ = family;
    match ctx.renderer.mode() {
        OutputMode::Json => {
            // NDJSON: one compact JSON object per line.
            if let Ok(s) = serde_json::to_string(&item) {
                println!("{s}");
            }
        },
        OutputMode::Pretty | OutputMode::Plain => {
            let _ = ctx.renderer.print(&item);
        },
    }
}
