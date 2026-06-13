//! `deadeye doctor` — readiness preflight.
//!
//! One command that answers "can I trade right now?" before an agent
//! burns a turn discovering it can't. Each probe is independent and
//! degrades gracefully: a check that can't run (no address, no RPC)
//! reports *why* and what to do, rather than aborting the whole report.
//!
//! Checks, in order:
//!   1. profile + address resolved
//!   2. RPC reachable (`chain_id`)
//!   3. account contract deployed
//!   4. gas (STRK) balance > 0
//!   5. XP balance / initial grant claimed
//!   6. indexer reachable
//!   7. (with `--market`) market active, initialised, not settled, and its
//!      family is on-chain readable
//!
//! Exit code is non-zero when any check fails, so it composes in `&&`
//! chains and CI gates. Output honours `--output {pretty,plain,json}`.

use std::io::Write;

use anyhow::{Result, bail};
use deadeye_starknet::{CollateralTokenReader, MAINNET_XP_TOKEN_ADDRESS};
use serde::Serialize;
use starknet_core::types::Felt;
use starknet_providers::{JsonRpcClient, Provider as _, jsonrpc::HttpTransport};
use url::Url;

use crate::{
    cli::DoctorArgs,
    commands::{
        account::{is_deployed, strk_balance},
        collateral::u256_to_human_18,
        runtime_resolver::{build_provider, detect_family},
    },
    context::AppContext,
    output::{Render, Renderer},
};

/// One readiness probe and its verdict.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DoctorCheck {
    /// Stable, human-readable name (e.g. `"account deployed"`).
    name: &'static str,
    /// Whether the probe passed.
    ok: bool,
    /// Short status detail (the observed value or error).
    detail: String,
    /// Remediation hint, present only when `ok == false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    fix: Option<String>,
}

impl DoctorCheck {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            ok: true,
            detail: detail.into(),
            fix: None,
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            name,
            ok: false,
            detail: detail.into(),
            fix: Some(fix.into()),
        }
    }
}

/// Full readiness report — all probes plus the rolled-up verdict.
#[derive(Debug, Serialize)]
pub(crate) struct DoctorReport {
    /// True iff every check passed.
    all_ok: bool,
    /// The probes, in execution order.
    checks: Vec<DoctorCheck>,
}

pub(crate) async fn run(args: DoctorArgs, ctx: &AppContext) -> Result<()> {
    let mut checks: Vec<DoctorCheck> = Vec::new();

    // 1. profile + address ------------------------------------------------
    let address = ctx
        .config
        .address
        .as_deref()
        .and_then(|s| Felt::from_hex(s).ok());
    checks.push(match (&ctx.config.address, address) {
        (Some(a), Some(_)) => DoctorCheck::pass("profile + address", a.clone()),
        _ => DoctorCheck::fail(
            "profile + address",
            "no address resolved",
            "run `deadeye onboard` (or pass --address / --profile)",
        ),
    });

    // 2. RPC reachable ----------------------------------------------------
    let raw_provider = raw_provider(ctx);
    let chain_ok = match &raw_provider {
        Some(p) => p.chain_id().await.is_ok(),
        None => false,
    };
    checks.push(if chain_ok {
        DoctorCheck::pass("rpc reachable", ctx.config.rpc_url.clone())
    } else {
        DoctorCheck::fail(
            "rpc reachable",
            ctx.config.rpc_url.clone(),
            "point at a JSON-RPC v0_8+ endpoint: \
             `deadeye config init --rpc-url <url>` (the CLI needs the pre_confirmed tag)",
        )
    });

    // 3/4. account deployed + gas ----------------------------------------
    if let (Some(addr), Some(p)) = (address, raw_provider.as_ref()) {
        match is_deployed(p, addr).await {
            Ok(true) => checks.push(DoctorCheck::pass("account deployed", "deployed")),
            Ok(false) => checks.push(DoctorCheck::fail(
                "account deployed",
                "not deployed",
                "fund the address with STRK, then `deadeye account deploy`",
            )),
            Err(e) => checks.push(DoctorCheck::fail(
                "account deployed",
                format!("probe failed: {e}"),
                "check the RPC endpoint, then retry",
            )),
        }

        let strk = strk_balance(p, &ctx.config.strk_token, addr)
            .await
            .unwrap_or(0);
        #[allow(clippy::cast_precision_loss, reason = "display-only STRK amount")]
        let strk_h = (strk as f64) / 1.0e18;
        checks.push(if strk > 0 {
            DoctorCheck::pass("gas (STRK) balance", format!("{strk_h:.6} STRK"))
        } else {
            DoctorCheck::fail(
                "gas (STRK) balance",
                "0 STRK",
                "send a little STRK to the account — every tx (deploy, claim, trade) pays gas",
            )
        });
    }

    // 5. XP balance + initial grant --------------------------------------
    if let Some(addr) = address {
        match build_provider(ctx) {
            Ok(dp) => {
                let reader = CollateralTokenReader::new(&dp, MAINNET_XP_TOKEN_ADDRESS);
                let claimed = reader
                    .has_claimed_initial_grant(addr)
                    .await
                    .unwrap_or(false);
                let xp = match reader.balance_of(addr).await {
                    Ok(raw) => u256_to_human_18(raw),
                    Err(_) => 0.0,
                };
                checks.push(if claimed || xp > 0.0 {
                    DoctorCheck::pass(
                        "XP balance + grant",
                        format!("{xp:.4} XP (grant claimed: {claimed})"),
                    )
                } else {
                    DoctorCheck::fail(
                        "XP balance + grant",
                        "0 XP, grant unclaimed",
                        "deploy the account, then `deadeye collateral claim-grant --execute`",
                    )
                });
            },
            Err(e) => checks.push(DoctorCheck::fail(
                "XP balance + grant",
                format!("provider error: {e}"),
                "fix the RPC endpoint, then retry",
            )),
        }
    }

    // 6. indexer reachable ------------------------------------------------
    let idx_ok = match ctx.indexer_client() {
        Ok(c) => c.health().await.is_ok() || c.markets().await.is_ok(),
        Err(_) => false,
    };
    checks.push(if idx_ok {
        DoctorCheck::pass("indexer reachable", ctx.config.indexer_url.clone())
    } else {
        DoctorCheck::fail(
            "indexer reachable",
            ctx.config.indexer_url.clone(),
            "set --indexer-url or DEADEYE_INDEXER_URL to a reachable indexer \
             (used for discovery, positions, and market state)",
        )
    });

    // 7. market readiness (optional) -------------------------------------
    if let Some(market) = &args.market {
        market_checks(ctx, market, &mut checks).await;
    }

    let all_ok = checks.iter().all(|c| c.ok);
    let report = DoctorReport { all_ok, checks };
    ctx.renderer.print(&report)?;
    if !all_ok {
        bail!("readiness checks failed — see the fixes above");
    }
    Ok(())
}

/// Probe a specific market: indexer state (active / not settled) and that
/// the AMM family is on-chain readable.
async fn market_checks(ctx: &AppContext, market: &str, checks: &mut Vec<DoctorCheck>) {
    // Indexer view: active + lifecycle flags.
    match ctx.indexer_client() {
        Ok(c) => match c.market(market).await {
            Ok(summary) => {
                let settled = summary.state.as_ref().is_some_and(|s| s.is_settled);
                let paused = summary.state.as_ref().is_some_and(|s| s.is_paused);
                let init = summary.state.as_ref().is_some_and(|s| s.is_initialised);
                if settled {
                    checks.push(DoctorCheck::fail(
                        "market tradeable",
                        format!("{} is settled", summary.market_type),
                        "pick a live market — this one has settled (claim XP via \
                         `deadeye claim` instead)",
                    ));
                } else if paused {
                    checks.push(DoctorCheck::fail(
                        "market tradeable",
                        "market is paused",
                        "wait for the operator to unpause, or pick another market",
                    ));
                } else if !init && summary.state.is_some() {
                    checks.push(DoctorCheck::fail(
                        "market tradeable",
                        "market not initialised",
                        "wait for the AMM to be initialised before trading",
                    ));
                } else if summary.is_active {
                    checks.push(DoctorCheck::pass(
                        "market tradeable",
                        format!("{} active", summary.market_type),
                    ));
                } else {
                    checks.push(DoctorCheck::fail(
                        "market tradeable",
                        "indexer reports market inactive",
                        "pick a market with isActive=true (`deadeye markets list`)",
                    ));
                }
            },
            Err(e) => checks.push(DoctorCheck::fail(
                "market tradeable",
                format!("not on indexer: {e}"),
                "confirm the address with `deadeye markets list`",
            )),
        },
        Err(_) => checks.push(DoctorCheck::fail(
            "market tradeable",
            "indexer unavailable",
            "set a reachable --indexer-url, then retry",
        )),
    }

    // Two separate probes (issue #38 review): raw readability first —
    // `get_params` is wire-identical across all four families, so one call
    // proves the address is a Deadeye AMM even when the family cannot be
    // classified (off-mainnet, no indexer, unknown class hash).
    match (build_provider(ctx).ok(), Felt::from_hex(market)) {
        (Some(_), Ok(market_felt)) => match ctx.deadeye_client() {
            Ok(client) => {
                match deadeye_sdk::starknet::NormalMarketReader::new(
                    client.provider(),
                    market_felt,
                )
                .params()
                .await
                {
                    Ok(_) => checks.push(DoctorCheck::pass(
                        "market readable on-chain",
                        "get_params answered — the address is a Deadeye AMM".to_owned(),
                    )),
                    Err(e) => checks.push(DoctorCheck::fail(
                        "market readable on-chain",
                        format!("get_params probe failed: {e}"),
                        "confirm the address is a Deadeye AMM contract on this chain",
                    )),
                }
                // Family classification via the detection ladder (indexer
                // metadata → class hash → factory registration).
                match detect_family(ctx, &client, market_felt).await {
                    Ok(family) => checks.push(DoctorCheck::pass(
                        "market family detected",
                        format!("family: {family:?} — math runs client-side, no runtime needed"),
                    )),
                    Err(e) => checks.push(DoctorCheck::fail(
                        "market family detected",
                        format!("{e:#}"),
                        "pass --family explicitly, or make a detection source reachable \
                         (indexer URL, DEADEYE_FACTORY_ADDR)",
                    )),
                }
            },
            Err(e) => checks.push(DoctorCheck::fail(
                "market readable on-chain",
                format!("client error: {e}"),
                "fix the RPC endpoint, then retry",
            )),
        },
        (_, Err(_)) => checks.push(DoctorCheck::fail(
            "market readable on-chain",
            "market address is not a felt",
            "pass a 0x… contract address to --market",
        )),
        (None, _) => checks.push(DoctorCheck::fail(
            "market readable on-chain",
            "no provider",
            "fix the RPC endpoint, then retry",
        )),
    }
}

/// Build a bare `JsonRpcClient` for the deploy/balance probes (which take
/// the starknet-rs provider directly). Returns `None` if the URL is bad —
/// the RPC check then reports the failure.
fn raw_provider(ctx: &AppContext) -> Option<JsonRpcClient<HttpTransport>> {
    let url = Url::parse(&ctx.config.rpc_url).ok()?;
    Some(JsonRpcClient::new(HttpTransport::new(url)))
}

impl Render for DoctorReport {
    fn render_pretty(&self, r: &Renderer) {
        r.header("deadeye doctor — readiness");
        for c in &self.checks {
            let mark = if c.ok {
                r.highlight("✓")
            } else {
                "✗".to_owned()
            };
            println!("  {mark} {:<24} {}", c.name, r.dim(&c.detail));
            if let Some(fix) = &c.fix {
                println!("      {} {}", r.dim("→ fix:"), fix);
            }
        }
        println!();
        if self.all_ok {
            r.success("all checks passed — ready to trade");
        } else {
            let n = self.checks.iter().filter(|c| !c.ok).count();
            r.warning(&format!("{n} check(s) failed — see fixes above"));
        }
    }

    fn render_plain(&self, w: &mut dyn Write) -> std::io::Result<()> {
        for c in &self.checks {
            let status = if c.ok { "ok" } else { "FAIL" };
            writeln!(w, "{}: {status} ({})", c.name, c.detail)?;
            if let Some(fix) = &c.fix {
                writeln!(w, "  fix: {fix}")?;
            }
        }
        writeln!(w, "all_ok: {}", self.all_ok)?;
        Ok(())
    }
}
