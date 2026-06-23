//! `deadeye account …` — read account / profile state, deploy the account.

use anyhow::{Context as _, Result, bail};
use deadeye_starknet::Provider as _;
use starknet_core::{
    types::{BlockId, BlockTag, Felt, FunctionCall},
    utils::get_selector_from_name,
};
use starknet_providers::{JsonRpcClient, Provider as _, jsonrpc::HttpTransport};
use url::Url;

use crate::{
    cli::{AccountCmd, AccountDeriveArgs},
    commands::{confirm_or_bail, onboard},
    config,
    context::AppContext,
    output::OutputMode,
    render::AccountView,
    wallet,
};

pub(crate) async fn run(action: AccountCmd, ctx: &AppContext, confirm: bool) -> Result<()> {
    match action {
        AccountCmd::Show => show(ctx).await,
        AccountCmd::List => list(ctx),
        AccountCmd::Deploy => deploy(ctx, confirm).await,
        AccountCmd::Derive(args) => derive(ctx, args),
    }
}

/// Derive accounts from an existing profile's mnemonic (`deadeye/hd/v1`,
/// issue #37): one seed → many independent on-chain accounts, ideal for
/// per-domain / per-strategy budget isolation. Idempotent — re-running
/// reports existing derivations instead of overwriting anything.
fn derive(ctx: &AppContext, args: AccountDeriveArgs) -> Result<()> {
    let mut cfg = config::load()?;
    let parent_name = args
        .from_profile
        .unwrap_or_else(|| ctx.config.profile_name.clone());
    let parent = cfg.profiles.get(&parent_name).cloned().with_context(|| {
        let known = cfg.profiles.keys().cloned().collect::<Vec<_>>().join(", ");
        format!("no profile named `{parent_name}` — saved profiles: [{known}]")
    })?;
    let phrase = parent.mnemonic.clone().with_context(|| {
        format!(
            "profile `{parent_name}` has no stored recovery phrase, so nothing can be \
             derived from it. Re-onboard it with \
             `deadeye onboard --import --profile {parent_name}` (same phrase, same \
             address) to store the phrase, then retry."
        )
    })?;
    let class_hash_hex = parent
        .account_class_hash
        .clone()
        .unwrap_or_else(|| wallet::DEFAULT_OZ_ACCOUNT_CLASS_HASH.to_owned());
    let class_hash =
        Felt::from_hex(&class_hash_hex).context("stored account class hash is not a felt")?;

    // Resolve the target (index, profile-name) list.
    let targets: Vec<(u32, String)> = match (args.index, args.count) {
        (Some(0), _) => bail!(
            "index 0 is the parent account itself (`{parent_name}`) — derived accounts \
             start at --index 1"
        ),
        (Some(i), None) => {
            let name = args.profile.unwrap_or_else(|| format!("{parent_name}-{i}"));
            vec![(i, name)]
        },
        (None, Some(0)) => bail!("--count must be ≥ 1"),
        (None, Some(n)) => {
            let prefix = args.prefix.unwrap_or_else(|| parent_name.clone());
            (1..=n).map(|i| (i, format!("{prefix}-{i}"))).collect()
        },
        (None, None) => bail!(
            "pass --index <n> for one account or --count <N> to bulk-derive 1..=N \
             (e.g. `deadeye account derive --count 5`)"
        ),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
    };

    #[derive(serde::Serialize)]
    struct DerivedRow {
        profile: String,
        index: u32,
        address: String,
        status: &'static str,
        derived_from: String,
    }
    let mut rows: Vec<DerivedRow> = Vec::with_capacity(targets.len());

    for (index, name) in targets {
        let w = wallet::import_at_index(&phrase, class_hash, index)?;
        let address = format!("{:#066x}", w.address);
        if let Some(existing) = cfg.profiles.get(&name) {
            // Idempotency: the same name holding the same account is fine;
            // a different account under that name must never be clobbered.
            if existing.address.as_deref() == Some(address.as_str()) {
                rows.push(DerivedRow {
                    profile: name,
                    index,
                    address,
                    status: "exists",
                    derived_from: parent_name.clone(),
                });
                continue;
            }
            bail!(
                "profile `{name}` already holds a different account \
                 ({existing}) — pick another name with --profile/--prefix",
                existing = existing.address.as_deref().unwrap_or("address unknown"),
            );
        }
        cfg.profiles.insert(name.clone(), config::ProfileConfig {
            rpc_url: parent.rpc_url.clone(),
            indexer_url: parent.indexer_url.clone(),
            chain_id: parent.chain_id.clone(),
            address: Some(address.clone()),
            strk_token: parent.strk_token.clone(),
            private_key: Some(format!("{:#066x}", w.private_key)),
            // The parent's phrase recovers this account — don't duplicate
            // seed material across profiles.
            mnemonic: None,
            account_class_hash: Some(class_hash_hex.clone()),
            account_deployed: false,
            derivation_index: Some(index),
            derived_from: Some(parent_name.clone()),
        });
        rows.push(DerivedRow {
            profile: name,
            index,
            address,
            status: "created",
            derived_from: parent_name.clone(),
        });
    }
    config::save(&cfg)?;

    if ctx.renderer.mode() == OutputMode::Json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    let created = rows.iter().filter(|r| r.status == "created").count();
    println!(
        "Derived {created} new account(s) from `{parent_name}` (deadeye/hd/v1; {} already existed):\n",
        rows.len() - created,
    );
    for r in &rows {
        println!(
            "  [{}] #{} {:<16} {}",
            r.status, r.index, r.profile, r.address
        );
    }
    println!(
        "\nAll of these are recoverable from `{parent_name}`'s recovery phrase alone\n\
         (re-derive anywhere with `deadeye account derive` or \
         `deadeye onboard --import --account-index <n>`).\n\n\
         Next steps, per account:\n  \
         1. fund the address with a little STRK for gas\n  \
         2. deadeye account deploy --profile <name>\n  \
         3. deadeye collateral claim-grant --execute --profile <name>\n\n\
         Then pass `--profile <name>` to any command to act as that account\n\
         (`deadeye account list` shows the whole fleet)."
    );
    Ok(())
}

/// Deploy the active profile's account contract so it can send transactions.
async fn deploy(ctx: &AppContext, confirm: bool) -> Result<()> {
    let cfg = &ctx.config;
    let pk_hex = cfg.private_key.as_deref().context(
        "no wallet key on the active profile — run `deadeye onboard` (or pass --profile)",
    )?;
    let private_key = Felt::from_hex(pk_hex.trim()).context("stored private key is not a felt")?;

    // Class hash: the profile's recorded class, else the default OZ class.
    let file = config::load()?;
    let class_hash_hex = file
        .profiles
        .get(&cfg.profile_name)
        .and_then(|p| p.account_class_hash.clone())
        .unwrap_or_else(|| wallet::DEFAULT_OZ_ACCOUNT_CLASS_HASH.to_owned());
    let class_hash =
        Felt::from_hex(&class_hash_hex).context("stored account class hash is not a felt")?;

    let w = wallet::from_private_key(private_key, class_hash);

    let url =
        Url::parse(&cfg.rpc_url).with_context(|| format!("invalid rpc_url: {}", cfg.rpc_url))?;
    let provider = JsonRpcClient::new(HttpTransport::new(url));
    onboard::verify_rpc_reachable(&provider, &cfg.rpc_url).await?;

    println!("Account : {:#066x}", w.address);
    if is_deployed(&provider, w.address).await? {
        println!("Already deployed — nothing to do.");
        mark_deployed(&cfg.profile_name)?;
        return Ok(());
    }

    onboard::verify_class_declared(&provider, class_hash).await?;

    // Must have gas to pay for the deploy.
    let bal = strk_balance(&provider, &cfg.strk_token, w.address)
        .await
        .unwrap_or(0);
    if bal == 0 {
        bail!(
            "account {:#x} has 0 STRK — fund it with a little STRK for gas, then re-run \
             `deadeye account deploy`",
            w.address
        );
    }
    println!("Balance : {:.6} STRK", (bal as f64) / 1e18_f64);

    if !confirm {
        confirm_or_bail(&format!(
            "Deploy the account contract for {:#066x}? Gas is paid from this address.",
            w.address
        ))?;
    }
    let tx = onboard::deploy_account(&provider, &cfg.chain_id, &w).await?;
    println!("\nAccount deployed. deploy_account tx: {tx:#066x}");
    mark_deployed(&cfg.profile_name)?;
    println!("\nNow you can: deadeye collateral claim-grant --execute");
    Ok(())
}

/// Whether an account contract exists at `address` (deployed) on-chain.
pub(crate) async fn is_deployed(
    provider: &JsonRpcClient<HttpTransport>,
    address: Felt,
) -> Result<bool> {
    deadeye_starknet::is_account_deployed(provider, address)
        .await
        .map_err(|e| anyhow::anyhow!("could not check deployment status: {e}"))
}

/// Read the STRK balance (low u128 limb) for `holder`.
pub(crate) async fn strk_balance(
    provider: &JsonRpcClient<HttpTransport>,
    token_hex: &str,
    holder: Felt,
) -> Result<u128> {
    let token = Felt::from_hex(token_hex).context("invalid STRK token address")?;
    let result = provider
        .call(
            FunctionCall {
                contract_address: token,
                entry_point_selector: get_selector_from_name("balance_of")
                    .context("balance_of selector")?,
                calldata: vec![holder],
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| anyhow::anyhow!("balance_of call failed: {e}"))?;
    let low = result.first().context("balance_of returned no felts")?;
    let bytes = low.to_bytes_be();
    let (high, low_bytes) = bytes.split_at(16);
    if high.iter().any(|b| *b != 0) {
        bail!("balance overflows u128");
    }
    let mut buf = [0_u8; 16];
    buf.copy_from_slice(low_bytes);
    Ok(u128::from_be_bytes(buf))
}

/// Flip `account_deployed = true` for `profile`.
fn mark_deployed(profile: &str) -> Result<()> {
    let mut cfg = config::load()?;
    if let Some(p) = cfg.profiles.get_mut(profile) {
        p.account_deployed = true;
    }
    config::save(&cfg)?;
    Ok(())
}

/// List every saved wallet profile so an agent can choose one to trade from.
fn list(ctx: &AppContext) -> Result<()> {
    let cfg = config::load()?;
    if cfg.profiles.is_empty() {
        println!("No wallets yet. Create one with `deadeye onboard` (or `--profile <name>`).");
        return Ok(());
    }
    let default = cfg.default_profile.as_deref();

    if ctx.renderer.mode() == OutputMode::Json {
        let rows: Vec<serde_json::Value> = cfg
            .profiles
            .iter()
            .map(|(name, p)| {
                serde_json::json!({
                    "profile": name,
                    "default": default == Some(name.as_str()),
                    "address": p.address,
                    "chain_id": p.chain_id,
                    "rpc_url": p.rpc_url,
                    "deployed": p.account_deployed,
                    "has_key": p.private_key.is_some(),
                    "has_mnemonic": p.mnemonic.is_some(),
                    "derivation_index": p.derivation_index,
                    "derived_from": p.derived_from,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    println!("Saved wallets ( * = default, used when --profile is omitted ):");
    for (name, p) in &cfg.profiles {
        let marker = if default == Some(name.as_str()) {
            "*"
        } else {
            " "
        };
        let addr = p.address.as_deref().unwrap_or("(no address)");
        let net = chain_label(p.chain_id.as_deref());
        let deployed = if p.account_deployed {
            "deployed"
        } else {
            "not-deployed"
        };
        let lineage = match (&p.derived_from, p.derivation_index) {
            (Some(parent), Some(i)) => format!(", hd #{i} of {parent}"),
            (None, Some(i)) => format!(", hd #{i}"),
            _ if p.mnemonic.is_some() => ", seed".to_owned(),
            _ => String::new(),
        };
        println!("  {marker} {name:<12} {addr}  [{net}, {deployed}{lineage}]");
    }
    println!("\nTrade from a specific wallet by passing `--profile <name>` to any command.");
    Ok(())
}

/// Friendly network label for a chain id.
fn chain_label(chain_id: Option<&str>) -> &'static str {
    match chain_id {
        Some(config::MAINNET_CHAIN_ID) => "mainnet",
        _ => "custom",
    }
}

async fn show(ctx: &AppContext) -> Result<()> {
    let cfg = &ctx.config;
    let address = cfg.address.as_deref().and_then(|s| Felt::from_hex(s).ok());

    let (balance_base, balance_strk) = match address {
        Some(addr) => match read_strk_balance(ctx, addr).await {
            Ok(base) => (Some(base), Some((base as f64) / 1e18_f64)),
            Err(err) => {
                ctx.renderer
                    .warning(&format!("could not read STRK balance: {err:#}"));
                (None, None)
            },
        },
        None => (None, None),
    };

    let view = AccountView {
        profile: cfg.profile_name.clone(),
        address: cfg.address.clone(),
        chain_id: cfg.chain_id.clone(),
        rpc_url: cfg.rpc_url.clone(),
        indexer_url: cfg.indexer_url.clone(),
        strk_balance_base: balance_base,
        strk_balance_strk: balance_strk,
    };
    ctx.renderer.print(&view)
}

/// Issue a `balance_of(holder)` view call against the configured STRK ERC-20.
async fn read_strk_balance(ctx: &AppContext, holder: Felt) -> Result<u128> {
    let token = Felt::from_hex(&ctx.config.strk_token)
        .with_context(|| format!("invalid STRK token address: {}", ctx.config.strk_token))?;
    let client = ctx.deadeye_client()?;
    let result = client
        .provider()
        .call(
            FunctionCall {
                contract_address: token,
                entry_point_selector: get_selector_from_name("balance_of")
                    .context("balance_of selector is a constant")?,
                calldata: vec![holder],
            },
            BlockId::Tag(BlockTag::PreConfirmed),
        )
        .await
        .map_err(|e| anyhow::anyhow!("balance_of provider call failed: {e}"))?;
    if result.len() < 2 {
        anyhow::bail!("balance_of returned {} felts (expected ≥ 2)", result.len());
    }
    let bytes = result[0].to_bytes_be();
    let (high, low) = bytes.split_at(16);
    if high.iter().any(|b| *b != 0) {
        anyhow::bail!("balance overflows u128");
    }
    let mut buf = [0_u8; 16];
    buf.copy_from_slice(low);
    Ok(u128::from_be_bytes(buf))
}
