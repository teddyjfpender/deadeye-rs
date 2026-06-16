//! Helpers for resolving common command inputs:
//!
//! * Felt parsing with friendly error messages.
//! * Math-runtime address resolution (CLI flag → family-specific env var).
//! * Market family auto-detect (issue #38) via a layered ladder: explicit
//!   `--family` flag → indexer `marketType` metadata → the market's class hash
//!   compared against the pinned deployment manifest → the factory's
//!   `market_type_for_market` registration. Never guesses: when every rung
//!   fails the caller gets an error asking for `--family`, because normal and
//!   lognormal AMMs are wire-identical on every shared view and a wrong guess
//!   silently produces wrong math.
//! * Owned-account construction from the resolved config + private-key env var.
//!
//! All resolution failures are surfaced as `anyhow::Error` with a hint
//! about which flag / env var would fix it.

use anyhow::{Context as _, Result, bail};
use deadeye_deployer::{ClassHashes, Deployment};
use deadeye_sdk::{DeadeyeClient, bulk::Family};
use deadeye_starknet::{FactoryReader, JsonRpcProvider, OwnedAccount};
use starknet_core::types::{BlockId, BlockTag, Felt, StarknetError};
use starknet_providers::{JsonRpcClient, Provider as _, ProviderError, jsonrpc::HttpTransport};
use url::Url;

use crate::{cli::FamilyArg, context::AppContext};

/// Parse a hex felt with a clear context message.
pub(crate) fn parse_felt(label: &str, raw: &str) -> Result<Felt> {
    Felt::from_hex(raw).with_context(|| format!("{label} `{raw}` is not a valid hex felt"))
}

/// Convert [`FamilyArg`] → SDK [`Family`].
pub(crate) const fn family_from_arg(arg: FamilyArg) -> Family {
    match arg {
        FamilyArg::Normal => Family::Normal,
        FamilyArg::Lognormal => Family::Lognormal,
        FamilyArg::Multinoulli => Family::Multinoulli,
        FamilyArg::Bivariate => Family::Bivariate,
    }
}

/// Pretty label for a family (used in rendered output).
pub(crate) const fn family_label(family: Family) -> &'static str {
    match family {
        Family::Normal => "normal",
        Family::Lognormal => "lognormal",
        Family::Multinoulli => "multinoulli",
        Family::Bivariate => "bivariate",
    }
}

/// Resolve the math-runtime contract address for `family`.
///
/// Priority: `--runtime` flag → `DEADEYE_<FAMILY>_RUNTIME_ADDR` env var.
pub(crate) fn resolve_runtime(cli_runtime: Option<&str>, family: Family) -> Result<Felt> {
    let env_var = runtime_env_var(family);
    resolve_runtime_opt(cli_runtime, family)?.with_context(|| {
        format!(
            "math runtime address required: pass `--runtime 0x...` or set `{env_var}` \
             (normal-family quotes resolve this automatically — no runtime needed)"
        )
    })
}

/// Resolution precedence for a math runtime: `--runtime` flag, then
/// `DEADEYE_<FAMILY>_RUNTIME_ADDR`. Returns `None` (rather than erroring) when
/// neither is set, so callers with a client-side path can fall back to offline.
pub(crate) fn resolve_runtime_opt(
    cli_runtime: Option<&str>,
    family: Family,
) -> Result<Option<Felt>> {
    if let Some(raw) = cli_runtime {
        return Ok(Some(parse_felt("runtime address", raw)?));
    }
    match std::env::var(runtime_env_var(family)) {
        Ok(s) if !s.trim().is_empty() => Ok(Some(parse_felt("runtime address", s.trim())?)),
        _ => Ok(None),
    }
}

const fn runtime_env_var(family: Family) -> &'static str {
    match family {
        Family::Normal => "DEADEYE_NORMAL_RUNTIME_ADDR",
        Family::Lognormal => "DEADEYE_LOGNORMAL_RUNTIME_ADDR",
        Family::Multinoulli => "DEADEYE_MULTINOULLI_RUNTIME_ADDR",
        Family::Bivariate => "DEADEYE_BIVARIATE_RUNTIME_ADDR",
    }
}

/// Map an indexer `marketType` slug to a [`Family`]. Inverse of
/// [`FamilyArg::as_indexer_slug`].
pub(crate) fn family_from_slug(slug: &str) -> Option<Family> {
    match slug {
        "normal" => Some(Family::Normal),
        "lognormal" => Some(Family::Lognormal),
        "multinoulli" => Some(Family::Multinoulli),
        "bivariate" => Some(Family::Bivariate),
        _ => None,
    }
}

/// Match an AMM class hash against the pinned deployment manifest.
///
/// Sierra class hashes are intrinsic to the compiled class, not the network,
/// so this works on devnet too as long as the declared contracts match the
/// manifest's contracts version.
fn family_from_amm_class_hash(hashes: &ClassHashes, hash: Felt) -> Option<Family> {
    let matches = |s: &str| Felt::from_hex(s).is_ok_and(|h| h == hash);
    if matches(&hashes.normal_amm) {
        Some(Family::Normal)
    } else if matches(&hashes.lognormal_amm) {
        Some(Family::Lognormal)
    } else if matches(&hashes.multinoulli_amm) {
        Some(Family::Multinoulli)
    } else if matches(&hashes.bivariate_amm) {
        Some(Family::Bivariate)
    } else {
        None
    }
}

/// Map the factory's on-chain `MarketKind` discriminant to a [`Family`].
///
/// Source of truth: `the-situation/packages/onchain-core` `MarketKind`
/// (mirrored in `deadeye-testkit`'s `MarketKind::discriminant`). Values are
/// non-contiguous: `0` means "no record", `3` is unassigned, `6` is
/// skew-normal (unsupported here).
fn family_from_factory_discriminant(discriminant: u8) -> Option<Family> {
    match discriminant {
        1 => Some(Family::Normal),
        2 => Some(Family::Multinoulli),
        4 => Some(Family::Lognormal),
        5 => Some(Family::Bivariate),
        _ => None,
    }
}

/// Outcome of probing the market's class hash on chain.
enum ClassHashProbe {
    /// The contract exists and reported this class hash.
    Found(Felt),
    /// No contract is deployed at the address — definitive, nothing else
    /// can succeed either.
    NotDeployed,
    /// Transient provider failure — inconclusive, keep going.
    Unavailable(String),
}

/// Read the market's class hash via a raw JSON-RPC client (the
/// [`deadeye_starknet::Provider`] trait only exposes `call`).
async fn market_class_hash(ctx: &AppContext, market: Felt) -> ClassHashProbe {
    let url = match Url::parse(&ctx.config.rpc_url) {
        Ok(u) => u,
        Err(e) => return ClassHashProbe::Unavailable(format!("invalid rpc_url: {e}")),
    };
    let rpc = JsonRpcClient::new(HttpTransport::new(url));
    match rpc
        .get_class_hash_at(BlockId::Tag(BlockTag::PreConfirmed), market)
        .await
    {
        Ok(hash) => ClassHashProbe::Found(hash),
        Err(ProviderError::StarknetError(StarknetError::ContractNotFound)) => {
            ClassHashProbe::NotDeployed
        },
        Err(e) => ClassHashProbe::Unavailable(format!("get_class_hash_at: {e}")),
    }
}

/// Resolve the factory address: `DEADEYE_FACTORY_ADDR` env var, else the
/// bundled mainnet manifest when the configured chain is mainnet.
fn resolve_factory_addr(ctx: &AppContext, manifest: Option<&Deployment>) -> Option<Felt> {
    if let Ok(s) = std::env::var("DEADEYE_FACTORY_ADDR")
        && !s.trim().is_empty()
        && let Ok(f) = Felt::from_hex(s.trim())
    {
        return Some(f);
    }
    if ctx.config.chain_id == crate::config::MAINNET_CHAIN_ID {
        return manifest.and_then(|m| m.factory_felt().ok());
    }
    None
}

/// Family auto-detection ladder (issue #38).
///
/// Normal and lognormal AMMs expose the **same selectors with the same wire
/// shapes** on every shared view, so probing reader calls can never tell
/// them apart — the old probe always concluded "normal" on lognormal
/// markets and quoted wrong math. Instead, consult sources that know the
/// family *semantically*, cheapest first:
///
/// 1. indexer `GET /api/markets/{address}` → `marketType` slug;
/// 2. the market's class hash vs the pinned deployment manifest;
/// 3. the factory's `market_type_for_market` registration (cross-validated
///    against the manifest via `market_type_config`, falling back to the
///    documented `MarketKind` discriminants).
///
/// When every rung fails this **errors** instead of guessing.
pub(crate) async fn detect_family<P>(
    ctx: &AppContext,
    client: &DeadeyeClient<P>,
    market: Felt,
) -> Result<Family>
where
    P: deadeye_starknet::Provider,
{
    let mut attempts: Vec<String> = Vec::new();

    // 1. Indexer metadata — authoritative for every registered market and zero RPC
    //    against rate-limited public nodes.
    match ctx.indexer_client() {
        Ok(indexer) => match indexer.market(&format!("{market:#x}")).await {
            Ok(summary) => match family_from_slug(&summary.market_type) {
                Some(family) => return Ok(family),
                None => bail!(
                    "indexer reports market family `{}` for {market:#x}, which this CLI \
                     does not support — upgrade deadeye (`deadeye update`) or pass --family",
                    summary.market_type
                ),
            },
            Err(e) => attempts.push(format!("indexer market lookup: {e}")),
        },
        Err(e) => attempts.push(format!("indexer client: {e}")),
    }

    let manifest = Deployment::mainnet().ok();

    // 2. Class hash vs the pinned deployment manifest.
    let market_class = market_class_hash(ctx, market).await;
    match &market_class {
        ClassHashProbe::Found(hash) => {
            if let Some(m) = &manifest {
                if let Some(family) = family_from_amm_class_hash(&m.class_hashes, *hash) {
                    return Ok(family);
                }
                attempts.push(format!(
                    "class hash {hash:#x} matches no AMM class in the bundled deployment \
                     manifest (contracts {})",
                    m.contracts_version
                ));
            }
        },
        ClassHashProbe::NotDeployed => {
            // Don't conclude from one un-retried replica: load-balanced RPC
            // pools can answer ContractNotFound for a just-deployed market
            // one backend hasn't seen yet. The factory rung (retrying
            // provider) gets its say; the final error lists this attempt.
            attempts.push(
                "get_class_hash_at found no contract at the address (typo'd address, or a \
                 lagging RPC replica for a fresh deployment)"
                    .to_owned(),
            );
        },
        ClassHashProbe::Unavailable(e) => attempts.push(e.clone()),
    }

    // 3. Factory registration.
    if let Some(factory) = resolve_factory_addr(ctx, manifest.as_ref()) {
        let reader = FactoryReader::new(client.provider(), factory);
        match reader.market_type_for_market(market).await {
            Ok(0) => attempts.push(format!(
                "factory {factory:#x} has no market-type record for this address"
            )),
            Ok(t) => {
                // Prefer the registered AMM class hash over the raw
                // discriminant — it survives discriminant renumbering.
                if let Some(m) = &manifest
                    && let Ok(cfg) = reader.market_type_config(t).await
                    && let Some(family) =
                        family_from_amm_class_hash(&m.class_hashes, cfg.amm_class_hash)
                {
                    return Ok(family);
                }
                if let Some(family) = family_from_factory_discriminant(t) {
                    return Ok(family);
                }
                attempts.push(format!(
                    "factory market-type discriminant {t} is unknown to this CLI"
                ));
            },
            Err(e) => attempts.push(format!("factory market-type lookup: {e}")),
        }
    } else {
        attempts.push(
            "no factory address available (set DEADEYE_FACTORY_ADDR; on mainnet the bundled \
             manifest provides it)"
                .to_owned(),
        );
    }

    bail!(
        "could not determine the market family for {market:#x} — pass \
         `--family normal|lognormal|multinoulli|bivariate`.\nTried:\n  - {}",
        attempts.join("\n  - ")
    )
}

/// Decide the family: explicit flag wins, else auto-detect.
pub(crate) async fn resolve_family<P>(
    ctx: &AppContext,
    client: &DeadeyeClient<P>,
    market: Felt,
    cli_family: Option<FamilyArg>,
) -> Result<Family>
where
    P: deadeye_starknet::Provider,
{
    if let Some(f) = cli_family {
        return Ok(family_from_arg(f));
    }
    detect_family(ctx, client, market).await
}

/// Build an [`OwnedAccount`] for the caller, sourcing the private key
/// from `DEADEYE_PRIVATE_KEY`.
pub(crate) fn build_owned_account(ctx: &AppContext) -> Result<OwnedAccount> {
    let raw_key = ctx.config.private_key.as_deref().context(
        "this command requires a private key; run `deadeye onboard`, or set DEADEYE_PRIVATE_KEY",
    )?;
    let address = ctx.resolved_address_felt()?;
    let key = parse_felt("private key", raw_key.trim())?;
    let chain_id = parse_felt("chain_id", &ctx.config.chain_id)?;

    let url = Url::parse(&ctx.config.rpc_url)
        .with_context(|| format!("invalid rpc_url: {}", ctx.config.rpc_url))?;
    let rpc = JsonRpcClient::new(HttpTransport::new(url));
    Ok(OwnedAccount::from_signing_key(rpc, address, key, chain_id))
}

/// Build a fresh provider-only client. Each call constructs its own HTTP
/// connection so concurrent code paths don't share state. Wrapped in
/// [`deadeye_starknet::retry::RetryingProvider`] so throttled endpoints back
/// off instead of surfacing raw parse errors (issue #14).
pub(crate) fn build_provider(ctx: &AppContext) -> Result<crate::context::CliProvider> {
    let url = Url::parse(&ctx.config.rpc_url)
        .with_context(|| format!("invalid rpc_url: {}", ctx.config.rpc_url))?;
    tracing::debug!(target: "deadeye::rpc", rpc_url = %url, "resolved RPC endpoint");
    let rpc = JsonRpcClient::new(HttpTransport::new(url));
    Ok(deadeye_starknet::retry::RetryingProvider::new(
        JsonRpcProvider::new(rpc),
    ))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests panic on construction failure"
)]
mod tests {
    use super::*;

    #[test]
    fn slug_mapping_is_inverse_of_indexer_slugs() {
        for arg in [
            FamilyArg::Normal,
            FamilyArg::Lognormal,
            FamilyArg::Multinoulli,
            FamilyArg::Bivariate,
        ] {
            assert_eq!(
                family_from_slug(arg.as_indexer_slug()),
                Some(family_from_arg(arg)),
                "slug round-trip failed for {arg:?}",
            );
        }
        assert_eq!(family_from_slug("skew_normal"), None);
        assert_eq!(family_from_slug(""), None);
    }

    #[test]
    fn manifest_class_hashes_map_to_their_families() {
        // The bundled mainnet manifest is the source of truth the detector
        // compares against — every AMM hash must resolve to its own family
        // and the hashes must be pairwise distinct (a collision would make
        // detection ambiguous).
        let manifest = Deployment::mainnet().expect("bundled manifest parses");
        let hashes = &manifest.class_hashes;
        let cases = [
            (&hashes.normal_amm, Family::Normal),
            (&hashes.lognormal_amm, Family::Lognormal),
            (&hashes.multinoulli_amm, Family::Multinoulli),
            (&hashes.bivariate_amm, Family::Bivariate),
        ];
        for (raw, want) in cases {
            let felt = Felt::from_hex(raw).expect("manifest hash parses");
            assert_eq!(family_from_amm_class_hash(hashes, felt), Some(want));
        }
        assert_eq!(
            family_from_amm_class_hash(hashes, Felt::from_hex("0xdead").unwrap()),
            None,
            "unknown hashes must not match",
        );
    }

    #[test]
    fn factory_discriminants_follow_onchain_market_kind() {
        // Mirrors `MarketKind` in the-situation onchain-core: values are
        // non-contiguous (3 = skew-normal, unsupported; 0 = no record).
        assert_eq!(family_from_factory_discriminant(1), Some(Family::Normal));
        assert_eq!(
            family_from_factory_discriminant(2),
            Some(Family::Multinoulli)
        );
        assert_eq!(family_from_factory_discriminant(4), Some(Family::Lognormal));
        assert_eq!(family_from_factory_discriminant(5), Some(Family::Bivariate));
        for unknown in [0u8, 3, 6, 255] {
            assert_eq!(family_from_factory_discriminant(unknown), None);
        }
    }
}
