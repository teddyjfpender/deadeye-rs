//! Configuration file loading + profile resolution.
//!
//! Config lives at `~/.config/deadeye/config.toml` (override path with
//! the `DEADEYE_CONFIG` env var). The file lists named profiles and an
//! optional `default_profile`. Private keys are **never** persisted to
//! disk — they must come from the `DEADEYE_PRIVATE_KEY` env var.

use std::{collections::BTreeMap, fs, io::Write as _, path::PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

/// Default public mainnet RPC endpoint — ZAN's public node, JSON-RPC v0_10
/// (the latest spec, matching the webapp). The CLI uses the `pre_confirmed`
/// block tag, so the endpoint must speak spec ≥ v0_9.
pub(crate) const DEFAULT_MAINNET_RPC: &str =
    "https://api.zan.top/public/starknet-mainnet/rpc/v0_10";

/// Default mainnet indexer URL (Hetzner, via sslip.io).
pub(crate) const DEFAULT_MAINNET_INDEXER: &str = "https://178-105-210-177.sslip.io";

/// Canonical mainnet chain id (`SN_MAIN`).
pub(crate) const MAINNET_CHAIN_ID: &str = "0x534e5f4d41494e";

/// Canonical STRK ERC-20 address.
pub(crate) const STRK_TOKEN_ADDRESS: &str =
    "0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d";

/// Top-level config document.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ConfigFile {
    /// Profile name to use when `--profile` and `DEADEYE_PROFILE` are unset.
    pub(crate) default_profile: Option<String>,
    /// Named profiles.
    pub(crate) profiles: BTreeMap<String, ProfileConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_signer_profile_resolves() {
        let cfg: ConfigFile = toml::from_str(
            r#"
            default_profile = "strkd"

            [profiles.strkd]
            rpc_url = "http://localhost:5050"
            chain_id = "0x534e5f4d41494e"
            address = "0x123"
            signer = "external"

            [profiles.strkd.external_signer]
            kind = "strkd"
            endpoint = "http://127.0.0.1:37643"
            token_path = "~/.config/deadeye/strkd.json"
            allowance_strategy = "skip-estimate"
            "#,
        )
        .expect("config parses");
        let profile = cfg.profiles.get("strkd").expect("profile");
        assert_eq!(profile.signer.as_deref(), Some("external"));
        let external = profile.external_signer.clone().expect("external signer");
        assert_eq!(external.kind.as_deref(), Some("strkd"));
        assert_eq!(
            external.allowance_strategy.as_deref(),
            Some("skip-estimate")
        );
        let resolved = ResolvedConfig::resolve(&cfg, ResolutionInputs::default()).expect("resolve");
        assert!(resolved.uses_external_signer());
        assert_eq!(resolved.address.as_deref(), Some("0x123"));
        assert_eq!(
            resolved
                .external_signer
                .as_ref()
                .and_then(|config| config.endpoint.as_deref()),
            Some("http://127.0.0.1:37643")
        );
    }
}

/// One named profile. Maps to a single RPC + indexer + address triple.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ProfileConfig {
    /// Starknet JSON-RPC URL.
    pub(crate) rpc_url: Option<String>,
    /// Indexer base URL.
    pub(crate) indexer_url: Option<String>,
    /// Chain id (hex felt).
    pub(crate) chain_id: Option<String>,
    /// Default trader / account address (hex felt).
    pub(crate) address: Option<String>,
    /// ERC-20 collateral token (defaults to canonical STRK).
    pub(crate) strk_token: Option<String>,
    /// Signing backend for write paths. Defaults to local private-key signing.
    ///
    /// Set to `"external"` to route submissions through
    /// [`external_signer`](Self::external_signer) while keeping the CLI
    /// address-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) signer: Option<String>,
    /// Companion wallet / external signer configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) external_signer: Option<ExternalSignerConfig>,
    /// Account private key (hex felt).
    ///
    /// Written by `deadeye onboard` so an agent can recover the wallet on
    /// every run without an interactive unlock. This is a **spendable
    /// secret stored in cleartext** — the file is created `0600`. Only the
    /// gas STRK on this address is at risk; XP collateral is
    /// non-transferable. Prefer `DEADEYE_PRIVATE_KEY` (env) where you can.
    pub(crate) private_key: Option<String>,
    /// BIP-39 recovery phrase for `private_key` (backup convenience).
    pub(crate) mnemonic: Option<String>,
    /// Account-contract class hash this address was derived against.
    pub(crate) account_class_hash: Option<String>,
    /// `true` once the account contract has been deployed on-chain.
    #[serde(default)]
    pub(crate) account_deployed: bool,
    /// HD derivation index under the parent's mnemonic (`deadeye/hd/v1`,
    /// issue #37). `None` for directly-onboarded (index-0) wallets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) derivation_index: Option<u32>,
    /// Name of the profile whose mnemonic this account derives from. The
    /// parent's phrase recovers this account; no key material is duplicated
    /// here beyond the derived private key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) derived_from: Option<String>,
}

/// External signer profile settings.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ExternalSignerConfig {
    /// Adapter kind. Currently `strkd` is built in.
    pub(crate) kind: Option<String>,
    /// Discovery mode: `endpoint` or `port_lock`.
    pub(crate) discovery: Option<String>,
    /// Explicit wallet JSON-RPC endpoint, e.g. `http://127.0.0.1:37643`.
    pub(crate) endpoint: Option<String>,
    /// Optional path to a strkd `port.lock` file.
    pub(crate) port_lock_path: Option<String>,
    /// JSON file containing `token` and `client_id` / `clientId`.
    pub(crate) token_path: Option<String>,
    /// Bearer token. Prefer `token_path` or `DEADEYE_STRKD_TOKEN` over storing
    /// this inline.
    pub(crate) token: Option<String>,
    /// Companion client id for the `X-Companion-Client` header.
    pub(crate) client_id: Option<String>,
    /// Whether the wallet should broadcast. Defaults to true for execute paths.
    pub(crate) submit: Option<bool>,
    /// How the adapter handles bundled approve+execute fee estimation.
    /// Defaults to `skip-estimate` for strkd.
    pub(crate) allowance_strategy: Option<String>,
}

/// Path resolver — returns `~/.config/deadeye/config.toml` unless
/// `DEADEYE_CONFIG` overrides it.
pub(crate) fn config_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("DEADEYE_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let mut dir = dirs::config_dir()
        .context("could not locate user config dir; set DEADEYE_CONFIG to override")?;
    dir.push("deadeye");
    dir.push("config.toml");
    Ok(dir)
}

/// Load the config file. Returns an empty `ConfigFile` if it doesn't exist.
pub(crate) fn load() -> Result<ConfigFile> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(ConfigFile::default());
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("reading config file at {}", path.display()))?;
    let parsed: ConfigFile =
        toml::from_str(&raw).with_context(|| format!("parsing TOML in {}", path.display()))?;
    Ok(parsed)
}

/// Persist a config file. Creates parent directories as needed.
pub(crate) fn save(cfg: &ConfigFile) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating config directory {}", parent.display()))?;
    }
    let body = toml::to_string_pretty(cfg).context("serializing config to TOML")?;
    let mut header = String::from(
        "# Deadeye CLI configuration.\n\
         #\n\
         # `deadeye onboard` may store a wallet `private_key` (and its\n\
         # `mnemonic`) here in CLEARTEXT so an agent can recover the wallet\n\
         # on every run. This file is written 0600 (owner read/write only).\n\
         # Anyone who reads it can spend the gas STRK on the address; XP\n\
         # collateral is non-transferable and cannot be drained.\n\
         #\n\
         # To keep secrets out of this file, set DEADEYE_PRIVATE_KEY in the\n\
         # environment instead — it takes precedence over the stored key.\n\
         #\n\
         # See `deadeye config --help` for management commands.\n\n",
    );
    header.push_str(&body);
    let mut file =
        fs::File::create(&path).with_context(|| format!("opening {} for write", path.display()))?;
    file.write_all(header.as_bytes())
        .with_context(|| format!("writing {}", path.display()))?;
    // Best-effort 0600 — the file may hold a spendable private key.
    restrict_permissions(&file);
    Ok(path)
}

/// Tighten file permissions to owner-only (0600) on Unix. No-op elsewhere.
#[cfg(unix)]
fn restrict_permissions(file: &fs::File) {
    use std::os::unix::fs::PermissionsExt as _;
    if let Ok(meta) = file.metadata() {
        let mut perms = meta.permissions();
        perms.set_mode(0o600);
        let _ = file.set_permissions(perms);
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_file: &fs::File) {}

/// Resolved configuration after merging CLI flags + env + file defaults.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedConfig {
    pub(crate) profile_name: String,
    pub(crate) rpc_url: String,
    pub(crate) indexer_url: String,
    pub(crate) chain_id: String,
    pub(crate) address: Option<String>,
    pub(crate) strk_token: String,
    /// Resolved private key (hex felt), from `DEADEYE_PRIVATE_KEY` (wins) or
    /// the active profile's stored `private_key`. `None` if neither is set.
    pub(crate) private_key: Option<String>,
    /// `true` iff a private key is available (env or stored profile key).
    pub(crate) has_private_key: bool,
    /// Selected write-path signer (`local` by default, `external` for wallet
    /// companion submission).
    pub(crate) signer: String,
    /// External signer details, when configured.
    pub(crate) external_signer: Option<ExternalSignerConfig>,
}

/// Inputs from the CLI that influence resolution.
#[derive(Debug, Default, Clone)]
pub(crate) struct ResolutionInputs {
    pub(crate) rpc_url: Option<String>,
    pub(crate) indexer_url: Option<String>,
    pub(crate) address: Option<String>,
    pub(crate) profile: Option<String>,
}

impl ResolvedConfig {
    pub(crate) fn uses_external_signer(&self) -> bool {
        self.signer.eq_ignore_ascii_case("external")
    }

    /// Resolve config in priority order:
    ///   1. CLI flag (passed in via `inputs`)
    ///   2. Env var (DEADEYE_RPC_URL, DEADEYE_INDEXER_URL, …)
    ///   3. Profile in `cfg`
    ///   4. Built-in defaults (mainnet public RPC + indexer).
    pub(crate) fn resolve(cfg: &ConfigFile, inputs: ResolutionInputs) -> Result<Self> {
        // Profile name: CLI > env (DEADEYE_PROFILE) > cfg.default_profile > "mainnet".
        let profile_name = inputs
            .profile
            .or_else(|| std::env::var("DEADEYE_PROFILE").ok())
            .or_else(|| cfg.default_profile.clone())
            .unwrap_or_else(|| "mainnet".to_owned());

        let profile = cfg.profiles.get(&profile_name).cloned().unwrap_or_default();

        let rpc_url = inputs
            .rpc_url
            .or_else(|| std::env::var("DEADEYE_RPC_URL").ok())
            .or(profile.rpc_url)
            .unwrap_or_else(|| DEFAULT_MAINNET_RPC.to_owned());

        let indexer_url = inputs
            .indexer_url
            .or_else(|| std::env::var("DEADEYE_INDEXER_URL").ok())
            .or(profile.indexer_url)
            .unwrap_or_else(|| DEFAULT_MAINNET_INDEXER.to_owned());

        let chain_id = std::env::var("DEADEYE_CHAIN_ID")
            .ok()
            .or(profile.chain_id)
            .unwrap_or_else(|| MAINNET_CHAIN_ID.to_owned());

        let address = inputs
            .address
            .or_else(|| std::env::var("DEADEYE_ADDRESS").ok())
            .or(profile.address);

        let strk_token = profile
            .strk_token
            .unwrap_or_else(|| STRK_TOKEN_ADDRESS.to_owned());
        let signer = std::env::var("DEADEYE_SIGNER")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or(profile.signer)
            .unwrap_or_else(|| "local".to_owned());
        let external_signer = profile.external_signer;

        // Env wins over the stored profile key so callers can override a
        // saved wallet without rewriting config.
        let private_key = std::env::var("DEADEYE_PRIVATE_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or(profile.private_key);
        let has_private_key = private_key.is_some();

        Ok(Self {
            profile_name,
            rpc_url,
            indexer_url,
            chain_id,
            address,
            strk_token,
            private_key,
            has_private_key,
            signer,
            external_signer,
        })
    }
}
