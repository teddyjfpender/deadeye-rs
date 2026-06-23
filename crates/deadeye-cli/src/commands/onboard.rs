//! `deadeye onboard` — create/recover a wallet and deploy its account.
//!
//! The flow, in order:
//!   1. Resolve the network (RPC / indexer / chain id) from `--network`.
//!   2. Generate a fresh BIP-39 phrase, or import one with `--import`.
//!   3. Derive the Starknet account and print the address.
//!   4. Save the key into the active profile (cleartext, 0600) so every later
//!      command — and any coding agent — recovers the same wallet.
//!   5. Wait for the address to be funded with STRK for gas.
//!   6. Deploy the account contract (paid in STRK, v3).
//!
//! Steps 5–6 are skipped with `--skip-deploy` (e.g. to set the wallet up
//! now and fund/deploy later by re-running `deadeye onboard --import`).

use std::io::{self, BufRead as _, Write as _};

use anyhow::{Context as _, Result, bail};
use starknet_accounts::{AccountFactory as _, OpenZeppelinAccountFactory};
use starknet_core::{
    types::{BlockId, BlockTag, Felt, FunctionCall, StarknetError},
    utils::get_selector_from_name,
};
use starknet_providers::{JsonRpcClient, Provider as _, ProviderError, jsonrpc::HttpTransport};
use starknet_signers::{LocalWallet, SigningKey};
use url::Url;

use crate::{
    cli::{NetworkArg, OnboardArgs},
    config::{self, ProfileConfig},
    context::AppContext,
    wallet::{self, DEFAULT_OZ_ACCOUNT_CLASS_HASH},
};

/// Resolved network endpoints for onboarding.
struct NetParams {
    profile: String,
    rpc_url: String,
    indexer_url: String,
    chain_id: String,
}

fn net_params(args: &OnboardArgs) -> NetParams {
    // Mainnet is the only deployed network — the irrefutable binding keeps
    // `--network` forward-compatible while documenting the single case.
    let NetworkArg::Mainnet = args.network;
    NetParams {
        profile: args.profile.clone().unwrap_or_else(|| "mainnet".to_owned()),
        rpc_url: args
            .rpc_url
            .clone()
            .unwrap_or_else(|| config::DEFAULT_MAINNET_RPC.to_owned()),
        indexer_url: args
            .indexer_url
            .clone()
            .unwrap_or_else(|| config::DEFAULT_MAINNET_INDEXER.to_owned()),
        chain_id: config::MAINNET_CHAIN_ID.to_owned(),
    }
}

pub(crate) async fn run(args: OnboardArgs, _ctx: &AppContext, confirm: bool) -> Result<()> {
    let mut net = net_params(&args);
    let class_hash_hex = args
        .account_class_hash
        .clone()
        .unwrap_or_else(|| DEFAULT_OZ_ACCOUNT_CLASS_HASH.to_owned());
    let class_hash = Felt::from_hex(&class_hash_hex)
        .with_context(|| format!("account class hash `{class_hash_hex}` is not a hex felt"))?;

    println!("Deadeye onboarding — network: {}\n", net.profile);

    // ── 0. Choose the RPC endpoint (explicit flag wins; else prompt) ──
    net.rpc_url = resolve_rpc(args.rpc_url.as_deref(), &net.rpc_url)?;
    println!("Using RPC: {}\n", net.rpc_url);

    // ── 1. Guard: never clobber an existing wallet by accident ────────
    let existing = config::load()?.profiles.get(&net.profile).cloned();
    let existing_key = existing.as_ref().and_then(|p| p.private_key.clone());
    let existing_addr = existing.as_ref().and_then(|p| p.address.clone());
    if existing_key.is_some() && !args.force && !args.import {
        bail!(
            "profile `{p}` already has a wallet ({a}). Generating a new one would \
             overwrite it and you would lose access to any funds on it.\n  \
             • resume / recover it : deadeye onboard --import\n  \
             • make a second wallet: deadeye onboard --profile <name>\n  \
             • overwrite anyway    : deadeye onboard --force",
            p = net.profile,
            a = existing_addr
                .clone()
                .unwrap_or_else(|| "address unknown".to_owned()),
        );
    }

    // ── 2. Generate or import the wallet ──────────────────────────────
    let w = if args.import {
        let phrase = prompt_line("Enter your BIP-39 recovery phrase:")?;
        wallet::import_at_index(phrase.trim(), class_hash, args.account_index)?
    } else {
        let w = wallet::generate(class_hash)?;
        println!(
            "Generated a new recovery phrase. WRITE IT DOWN — it is the only\n\
             way to recover this wallet, and it controls real funds:\n"
        );
        println!("    {}\n", w.mnemonic);
        if args.account_index > 0 {
            wallet::import_at_index(&w.mnemonic, class_hash, args.account_index)?
        } else {
            w
        }
    };
    if args.account_index > 0 {
        println!(
            "Derivation: deadeye/hd/v1 index {} — recoverable from the phrase \
             with `deadeye onboard --import --account-index {}`.\n",
            args.account_index, args.account_index
        );
    }

    // Importing over a profile that already holds a *different* wallet would
    // clobber it too — only allow it when the phrase recovers the same key.
    if args.import && existing_key.is_some() && !args.force {
        let derived = format!("{:#066x}", w.address);
        if existing_addr.as_deref() != Some(derived.as_str()) {
            bail!(
                "that recovery phrase derives {derived}, which differs from the wallet \
                 already saved on profile `{p}` ({a}). Use --force to overwrite, or \
                 --profile <name> to keep both.",
                p = net.profile,
                a = existing_addr.clone().unwrap_or_default(),
            );
        }
    }

    println!("Account address : {:#066x}", w.address);
    println!("Public key      : {:#066x}", w.public_key);
    println!("Account class   : {class_hash:#066x}\n");

    // ── 2. Persist the wallet into the active profile ─────────────────
    save_profile(&net, &w, args.account_index)?;
    println!(
        "Saved wallet to profile `{}` (set as default). The key is stored\n\
         in cleartext at your deadeye config; keep that file private.\n",
        net.profile
    );

    if args.skip_deploy {
        println!("--skip-deploy set: stopping before funding/deploy.");
        println!(
            "Fund {:#066x} with STRK, then re-run `deadeye onboard --import`.",
            w.address
        );
        return Ok(());
    }

    // ── 3. Build provider, check it answers, verify the account class ─
    let url =
        Url::parse(&net.rpc_url).with_context(|| format!("invalid rpc_url: {}", net.rpc_url))?;
    let provider = JsonRpcClient::new(HttpTransport::new(url));
    verify_rpc_reachable(&provider, &net.rpc_url).await?;
    verify_class_declared(&provider, class_hash).await?;

    // ── 4. Wait for the address to be funded ──────────────────────────
    let strk_token = Felt::from_hex(config::STRK_TOKEN_ADDRESS)
        .context("canonical STRK address constant is valid")?;
    let min_base = strk_to_base(args.min_strk);
    wait_for_funding(&provider, strk_token, w.address, min_base, args.min_strk).await?;

    // ── 5. Deploy the account contract ────────────────────────────────
    if !confirm {
        confirm_or_bail(&format!(
            "Deploy the account contract for {:#066x} now? Gas is paid from this address.",
            w.address
        ))?;
    }
    let tx_hash = deploy_account(&provider, &net.chain_id, &w).await?;
    println!("\nAccount deployed. deploy_account tx: {tx_hash:#066x}");

    mark_deployed(&net.profile)?;
    println!(
        "\nDone. Next steps:\n  \
         deadeye account show                 # confirm address + balance\n  \
         deadeye collateral claim-grant --execute   # claim your XP grant\n  \
         deadeye markets list                 # find a market to trade"
    );
    Ok(())
}

/// Write the wallet into a profile and mark it the default.
fn save_profile(net: &NetParams, w: &wallet::Wallet, account_index: u32) -> Result<()> {
    let mut cfg = config::load()?;
    let profile = cfg.profiles.entry(net.profile.clone()).or_default();
    *profile = ProfileConfig {
        rpc_url: Some(net.rpc_url.clone()),
        indexer_url: Some(net.indexer_url.clone()),
        chain_id: Some(net.chain_id.clone()),
        address: Some(format!("{:#066x}", w.address)),
        strk_token: profile.strk_token.clone(),
        signer: None,
        external_signer: None,
        private_key: Some(format!("{:#066x}", w.private_key)),
        mnemonic: Some(w.mnemonic.clone()),
        account_class_hash: Some(format!("{:#066x}", w.class_hash)),
        account_deployed: false,
        derivation_index: (account_index > 0).then_some(account_index),
        derived_from: None,
    };
    cfg.default_profile = Some(net.profile.clone());
    config::save(&cfg)?;
    Ok(())
}

/// Flip `account_deployed = true` for `profile` after a successful deploy.
fn mark_deployed(profile: &str) -> Result<()> {
    let mut cfg = config::load()?;
    if let Some(p) = cfg.profiles.get_mut(profile) {
        p.account_deployed = true;
    }
    config::save(&cfg)?;
    Ok(())
}

/// Resolve the RPC endpoint. An explicit `--rpc-url` wins; otherwise show the
/// default and let the user paste their own, with an Alchemy suggestion for a
/// more reliable endpoint. Non-interactive input falls back to the default.
fn resolve_rpc(explicit: Option<&str>, default_rpc: &str) -> Result<String> {
    if let Some(u) = explicit {
        let u = u.trim();
        if !u.is_empty() {
            return Ok(u.to_owned());
        }
    }
    println!("Starknet RPC endpoint");
    println!("  Default (ZAN public node — free, rate-limited):");
    println!("    {default_rpc}");
    println!("  For higher reliability + rate limits, get a free Alchemy Starknet RPC key:");
    println!("    https://www.alchemy.com/rpc/starknet");
    println!("  (use the latest v0_10 JSON-RPC path for whatever endpoint you paste)");
    match prompt_line("Press Enter for the default, or paste your RPC URL:") {
        Ok(line) => {
            let t = line.trim();
            Ok(if t.is_empty() {
                default_rpc.to_owned()
            } else {
                t.to_owned()
            })
        },
        // Non-interactive (piped / EOF): accept the default.
        Err(_) => Ok(default_rpc.to_owned()),
    }
}

/// Confirm the RPC answers before we ask the user to fund anything.
pub(crate) async fn verify_rpc_reachable(
    provider: &JsonRpcClient<HttpTransport>,
    rpc_url: &str,
) -> Result<()> {
    provider.chain_id().await.map_err(|e| {
        anyhow::anyhow!(
            "RPC {rpc_url} is unreachable or invalid ({e}). Re-run with `--rpc-url <url>` \
             pointing at a working Starknet JSON-RPC v0_10 endpoint — e.g. a free Alchemy key \
             from https://www.alchemy.com/rpc/starknet."
        )
    })?;
    Ok(())
}

/// Error out early if the account class isn't declared on this network.
pub(crate) async fn verify_class_declared(
    provider: &JsonRpcClient<HttpTransport>,
    class_hash: Felt,
) -> Result<()> {
    match provider
        .get_class(BlockId::Tag(BlockTag::Latest), class_hash)
        .await
    {
        Ok(_) => Ok(()),
        Err(ProviderError::StarknetError(StarknetError::ClassHashNotFound)) => bail!(
            "account class {class_hash:#x} is not declared on this network — pass \
             `--account-class-hash 0x...` with a class that is declared (e.g. an \
             OpenZeppelin / Argent / Braavos account already on-chain)"
        ),
        Err(e) => Err(anyhow::anyhow!("could not check account class: {e}")),
    }
}

/// Poll the STRK balance until it meets `min_base`, prompting between checks.
async fn wait_for_funding(
    provider: &JsonRpcClient<HttpTransport>,
    token: Felt,
    holder: Felt,
    min_base: u128,
    min_strk: f64,
) -> Result<()> {
    println!("Fund the account with at least {min_strk} STRK for gas:\n\n    {holder:#066x}\n");
    loop {
        let bal = read_strk_balance(provider, token, holder)
            .await
            .unwrap_or(0);
        let bal_strk = (bal as f64) / 1e18_f64;
        if bal >= min_base {
            println!("Balance: {bal_strk:.6} STRK — sufficient. Continuing.");
            return Ok(());
        }
        println!("Balance: {bal_strk:.6} STRK — need {min_strk} STRK.");
        match prompt_line("Press Enter to check again once you've sent STRK, or 'q' to abort:") {
            Ok(line) if line.trim().eq_ignore_ascii_case("q") => bail!("aborted by user"),
            Ok(_) => {},
            // EOF (non-interactive / piped): don't spin forever.
            Err(_) => bail!(
                "not enough STRK and no interactive terminal to wait — fund {holder:#x} \
                 and re-run `deadeye onboard --import`"
            ),
        }
    }
}

/// Deploy the OpenZeppelin account for `w`; returns the deploy tx hash.
pub(crate) async fn deploy_account(
    provider: &JsonRpcClient<HttpTransport>,
    chain_id_hex: &str,
    w: &wallet::Wallet,
) -> Result<Felt> {
    let chain_id =
        Felt::from_hex(chain_id_hex).with_context(|| format!("bad chain id: {chain_id_hex}"))?;
    let signer = LocalWallet::from_signing_key(SigningKey::from_secret_scalar(w.private_key));
    let factory = OpenZeppelinAccountFactory::new(w.class_hash, chain_id, signer, provider)
        .await
        .context("building account factory")?;
    // Salt = public key, matching `wallet::oz_account_address`.
    let deployment = factory.deploy_v3(w.public_key);
    let projected = deployment.address();
    if projected != w.address {
        bail!(
            "factory address {projected:#x} != derived address {:#x} — refusing to deploy",
            w.address
        );
    }
    let res = deployment
        .send()
        .await
        .context("submitting deploy_account transaction")?;
    Ok(res.transaction_hash)
}

/// `balance_of(holder)` against the STRK ERC-20, low u128 limb.
async fn read_strk_balance(
    provider: &JsonRpcClient<HttpTransport>,
    token: Felt,
    holder: Felt,
) -> Result<u128> {
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

/// STRK (human) → base units (18 decimals), saturating.
fn strk_to_base(strk: f64) -> u128 {
    let scaled = (strk.max(0.0) * 1e18_f64).round();
    if scaled >= 0.0 && scaled < (u128::MAX as f64) {
        scaled as u128
    } else {
        u128::MAX
    }
}

/// Print a prompt to stderr and read one line from stdin. `Err` on EOF.
fn prompt_line(prompt: &str) -> Result<String> {
    eprint!("{prompt} ");
    io::stderr().flush().ok();
    let mut line = String::new();
    let n = io::stdin().lock().read_line(&mut line)?;
    if n == 0 {
        bail!("end of input");
    }
    Ok(line)
}

/// y/N confirmation gate (reused shape from `commands::confirm_or_bail`).
fn confirm_or_bail(prompt: &str) -> Result<()> {
    let line = prompt_line(&format!("{prompt} [y/N]"))?;
    let t = line.trim().to_ascii_lowercase();
    if t == "y" || t == "yes" {
        Ok(())
    } else {
        bail!("aborted by user");
    }
}
