//! `deadeye markets …` — list / show / info read paths.

use anyhow::{Context as _, Result};
use deadeye_core::{Distribution, Sq128};
use deadeye_sdk::{DeadeyeClient, bulk::Family};
use deadeye_starknet::{
    BivariateMarketReader, ContractResult, Felt, LognormalMarketReader, MultinoulliMarketReader,
    NormalMarketReader,
    types::common::{AmmParamsRaw, FeeConfigRaw, LpInfoRaw},
};
use serde_json::json;

use crate::{
    cli::{FamilyArg, MarketsCmd},
    context::{AppContext, CliProvider, parse_address},
    render::{
        MarketFeeConfigView, MarketInfoView, MarketLpInfoView, MarketParamsView, MarketRow,
        MarketShowView, MarketStatusView,
    },
};

pub(crate) async fn run(action: MarketsCmd, ctx: &AppContext) -> Result<()> {
    match action {
        MarketsCmd::Snapshot { address, family } => snapshot(ctx, &address, family).await,
        MarketsCmd::List { family, limit } => list(ctx, family, limit).await,
        MarketsCmd::Show { address, family } => show(ctx, &address, family).await,
        MarketsCmd::Info { address } => info(ctx, &address).await,
    }
}

/// One-shot quote state snapshot for `trade quote --from-state` (issue #14):
/// fetch the three views a quote depends on ONCE, emit them bit-exactly.
///
/// Family-gated (issue #38): normal and lognormal AMMs answer the same
/// selectors with the same wire shapes, so snapshotting a lognormal market
/// through the normal reader "succeeds" while silently emitting log-space
/// values. Resolve the family first and refuse anything but normal until a
/// lognormal snapshot type exists.
async fn snapshot(ctx: &AppContext, address: &str, family: Option<FamilyArg>) -> Result<()> {
    let market = parse_address(address)?;
    let client = ctx.deadeye_client()?;
    // The snapshot file carries an authoritative `family` stamp that
    // `trade quote --from-state` later trusts, so an explicit --family flag
    // is verified against detection whenever detection can conclude — the
    // flag is only an escape hatch when every detection rung is unavailable,
    // never a way to launder a wrong guess into a stamped file.
    let family = match (
        family,
        super::runtime_resolver::detect_family(ctx, &client, market).await,
    ) {
        (Some(flag), Ok(detected)) => {
            let flagged = flag.as_sdk();
            anyhow::ensure!(
                flagged == detected,
                "--family {} contradicts detection: {address} is a {} market",
                super::runtime_resolver::family_label(flagged),
                super::runtime_resolver::family_label(detected),
            );
            detected
        },
        (Some(flag), Err(detect_err)) => {
            ctx.renderer.warning(&format!(
                "family detection inconclusive ({detect_err:#}); trusting --family {} — \
                 the snapshot will be stamped with it",
                super::runtime_resolver::family_label(flag.as_sdk()),
            ));
            flag.as_sdk()
        },
        (None, Ok(detected)) => detected,
        (None, Err(e)) => {
            return Err(e.context(format!("could not auto-detect family for market {address}")));
        },
    };
    if family != Family::Normal {
        anyhow::bail!(
            "markets snapshot supports normal markets only today; {address} is a {} market — \
             quote it live instead: `deadeye trade quote {address} --family {} --belief … --budget …`",
            super::runtime_resolver::family_label(family),
            super::runtime_resolver::family_label(family),
        );
    }
    let snapshot = client
        .normal_market(market)
        .state_snapshot()
        .await
        .map_err(|e| anyhow::anyhow!("state_snapshot: {e}"))?;
    ctx.renderer.print(&SnapshotView(snapshot))
}

/// Render wrapper: pretty/plain output shows the human mirrors; `--output
/// json` serializes the full bit-exact snapshot for `--from-state`.
struct SnapshotView(deadeye_sdk::normal::NormalMarketStateSnapshot);

impl serde::Serialize for SnapshotView {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl crate::output::Render for SnapshotView {
    fn render_pretty(&self, r: &crate::output::Renderer) {
        r.kv("market", &self.0.market);
        r.kv("mean", &format!("{:.6}", self.0.mean));
        r.kv("sigma", &format!("{:.6}", self.0.sigma));
        r.kv("effective_k", &format!("{:.6}", self.0.effective_k));
        r.kv("pool_backing_xp", &format!("{:.6}", self.0.pool_backing_xp));
    }

    fn render_plain(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(w, "market: {}", self.0.market)?;
        writeln!(w, "mean: {:.6}", self.0.mean)?;
        writeln!(w, "sigma: {:.6}", self.0.sigma)?;
        writeln!(w, "effective_k: {:.6}", self.0.effective_k)?;
        writeln!(w, "pool_backing_xp: {:.6}", self.0.pool_backing_xp)
    }
}

async fn list(ctx: &AppContext, family: Option<FamilyArg>, limit: usize) -> Result<()> {
    let indexer = ctx.indexer_client()?;
    let mut markets = indexer.markets().await.with_context(|| {
        format!(
            "indexer GET /api/markets ({}) failed",
            ctx.config.indexer_url
        )
    })?;
    if let Some(f) = family {
        let slug = f.as_indexer_slug();
        markets.retain(|m| m.market_type == slug);
    }
    markets.truncate(limit);
    let rows: Vec<MarketRow> = markets.iter().map(MarketRow::from_summary).collect();
    if rows.is_empty() {
        ctx.renderer.warning("no markets matched the filter");
    }
    ctx.renderer.print_table(&rows)
}

async fn show(ctx: &AppContext, address: &str, family_override: Option<FamilyArg>) -> Result<()> {
    let market = parse_address(address)?;
    let client = ctx.deadeye_client()?;

    let family = if let Some(f) = family_override {
        f.as_sdk()
    } else {
        super::runtime_resolver::detect_family(ctx, &client, market)
            .await
            .with_context(|| format!("could not auto-detect family for market {address}"))?
    };

    let view = match family {
        Family::Normal => show_normal(&client, market, address).await?,
        Family::Lognormal => show_lognormal(&client, market, address).await?,
        Family::Multinoulli => show_multinoulli(&client, market, address).await?,
        Family::Bivariate => show_bivariate(&client, market, address).await?,
    };
    ctx.renderer.print(&view)
}

async fn info(ctx: &AppContext, address: &str) -> Result<()> {
    let indexer = ctx.indexer_client()?;
    let summary = indexer
        .market(address)
        .await
        .with_context(|| format!("indexer GET /api/markets/{address} failed"))?;
    ctx.renderer.print(&MarketInfoView { summary })
}

fn params_view(p: AmmParamsRaw) -> MarketParamsView {
    MarketParamsView {
        k: Sq128::from_raw(p.k).to_f64(),
        backing: Sq128::from_raw(p.backing).to_f64(),
        tolerance: Sq128::from_raw(p.tolerance).to_f64(),
        min_trade_collateral: Sq128::from_raw(p.min_trade_collateral).to_f64(),
    }
}

fn lp_view(lp: LpInfoRaw) -> MarketLpInfoView {
    MarketLpInfoView {
        total_shares: Sq128::from_raw(lp.total_shares).to_f64(),
        total_backing_deposited: Sq128::from_raw(lp.total_backing_deposited).to_f64(),
    }
}

fn fee_view(fee: FeeConfigRaw) -> MarketFeeConfigView {
    MarketFeeConfigView {
        lp_fee_bps: fee.lp_fee_bps,
        protocol_fee_bps: fee.protocol_fee_bps,
        settlement_fee_bps: fee.settlement_fee_bps,
        total_bps: fee.total_bps(),
    }
}

async fn show_normal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    address: &str,
) -> Result<MarketShowView> {
    let reader = NormalMarketReader::new(client.provider(), market);
    let (dist_r, params_r, lp_r, fees_r, status_r) = futures::join!(
        reader.distribution(),
        reader.params(),
        reader.lp_info(),
        reader.fee_config(),
        reader.market_status(),
    );
    let dist: deadeye_core::NormalDistribution = wrap("distribution", dist_r)?;
    let params = wrap("params", params_r)?;
    let lp = wrap("lp_info", lp_r)?;
    let fees = wrap("fee_config", fees_r)?;
    let status = status_r.ok();
    // Hint parity: the chain's canonical sqrt hints for the live σ vs the
    // off-chain closed form. A divergence here is the systemic cause of a
    // `VERIFICATION_FAILED` trade revert.
    let chain_hints = reader.distribution_hints().await.ok();
    let sigma = dist.sigma().to_f64();
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let l2_offline = (2.0 * sigma * sqrt_pi).sqrt();
    let backing_offline = (sigma * sqrt_pi).sqrt();
    let hints_json = chain_hints.map(|h| {
        json!({
            "chain_l2_norm_denom": Sq128::from_raw(h.l2_norm_denom).to_f64(),
            "chain_backing_denom": Sq128::from_raw(h.backing_denom).to_f64(),
            "offline_l2_norm_denom": l2_offline,
            "offline_backing_denom": backing_offline,
        })
    });
    Ok(MarketShowView {
        address: address.to_owned(),
        family: "normal".to_owned(),
        distribution: json!({
            "mu": dist.mean().to_f64(),
            "sigma": dist.sigma().to_f64(),
            "variance": dist.variance().to_f64(),
            // P5/P25/P50/P75/P95 — issue #20: see the shape, not two numbers.
            "quantiles": super::render_helpers::normal_quantiles(
                dist.mean().to_f64(),
                dist.sigma().to_f64(),
            ),
            "hints": hints_json,
        }),
        params: params_view(params),
        lp_info: lp_view(lp),
        fee_config: fee_view(fees),
        status: status.map(|s| MarketStatusView {
            is_initialised: s.is_initialised,
            is_paused: s.is_paused,
            is_settled: s.is_settled,
            settlement_value: Sq128::from_raw(s.settlement_value).to_f64(),
        }),
    })
}

async fn show_lognormal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    address: &str,
) -> Result<MarketShowView> {
    let reader = LognormalMarketReader::new(client.provider(), market);
    let (dist_r, params_r, lp_r, fees_r) = futures::join!(
        reader.distribution(),
        reader.params(),
        reader.lp_info(),
        reader.fee_config(),
    );
    let dist: deadeye_core::LognormalDistribution = wrap("distribution", dist_r)?;
    let params = wrap("params", params_r)?;
    let lp = wrap("lp_info", lp_r)?;
    let fees = wrap("fee_config", fees_r)?;
    let raw = dist.to_raw();
    Ok(MarketShowView {
        address: address.to_owned(),
        family: "lognormal".to_owned(),
        distribution: json!({
            "mu": Sq128::from_raw(raw.mu).to_f64(),
            "variance": Sq128::from_raw(raw.variance).to_f64(),
            "sigma": Sq128::from_raw(raw.sigma).to_f64(),
        }),
        params: params_view(params),
        lp_info: lp_view(lp),
        fee_config: fee_view(fees),
        status: None,
    })
}

async fn show_multinoulli(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    address: &str,
) -> Result<MarketShowView> {
    let reader = MultinoulliMarketReader::new(client.provider(), market);
    let (dist_r, params_r, lp_r, fees_r) = futures::join!(
        reader.distribution(),
        reader.params(),
        reader.lp_info(),
        reader.fee_config(),
    );
    let dist: deadeye_core::CategoricalDistribution = wrap("distribution", dist_r)?;
    let params = wrap("params", params_r)?;
    let lp = wrap("lp_info", lp_r)?;
    let fees = wrap("fee_config", fees_r)?;
    let probs: Vec<f64> = dist.probs().to_vec();
    Ok(MarketShowView {
        address: address.to_owned(),
        family: "multinoulli".to_owned(),
        distribution: json!({
            "outcome_count": probs.len(),
            "probs": probs,
        }),
        params: params_view(params),
        lp_info: lp_view(lp),
        fee_config: fee_view(fees),
        status: None,
    })
}

async fn show_bivariate(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    address: &str,
) -> Result<MarketShowView> {
    let reader = BivariateMarketReader::new(client.provider(), market);
    let (dist_r, params_r, lp_r, fees_r) = futures::join!(
        reader.distribution_raw(),
        reader.params(),
        reader.lp_info(),
        reader.fee_config(),
    );
    let dist: deadeye_core::BivariateNormalDistributionRaw = wrap("distribution", dist_r)?;
    let params = wrap("params", params_r)?;
    let lp = wrap("lp_info", lp_r)?;
    let fees = wrap("fee_config", fees_r)?;
    Ok(MarketShowView {
        address: address.to_owned(),
        family: "bivariate".to_owned(),
        distribution: json!({
            "mu1": Sq128::from_raw(dist.mu1).to_f64(),
            "mu2": Sq128::from_raw(dist.mu2).to_f64(),
            "variance1": Sq128::from_raw(dist.variance1).to_f64(),
            "variance2": Sq128::from_raw(dist.variance2).to_f64(),
            "sigma1": Sq128::from_raw(dist.sigma1).to_f64(),
            "sigma2": Sq128::from_raw(dist.sigma2).to_f64(),
            "rho": Sq128::from_raw(dist.rho).to_f64(),
        }),
        params: params_view(params),
        lp_info: lp_view(lp),
        fee_config: fee_view(fees),
        status: None,
    })
}

fn wrap<T>(name: &str, r: ContractResult<T>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("reading {name} failed: {e}"))
}
