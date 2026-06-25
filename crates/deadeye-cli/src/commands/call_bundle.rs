//! Shared call-bundle handoff for keyless write paths.
//!
//! Write commands build the same account multicall they would submit locally,
//! run a skip-validation simulation, and then either emit the calls as JSON or
//! hand them to a configured external wallet companion.

use std::{
    io::{self, Write as _},
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, anyhow, bail};
use deadeye_starknet::{Account, Call, GasParams};
use serde::{Deserialize, Serialize};
use serde_json::json;
use starknet_core::types::{
    BlockId, BlockTag, Felt, MaybePreConfirmedBlockWithTxHashes, ResourcePrice,
};
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};

use crate::{
    commands::{render_helpers::SubmissionResult, runtime_resolver::build_simulation_account},
    config::ExternalSignerConfig,
    context::AppContext,
    output::{Render, Renderer},
};

/// Execution intent shared by write commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteMode {
    /// Sign and submit after validation.
    Submit,
    /// Simulate the multicall gas-free and stop.
    DryRun,
    /// Emit validated calldata for an external signer and stop.
    EmitCalldata,
}

impl WriteMode {
    pub(crate) const fn from_flags(dry_run: bool, emit_calldata: bool) -> Self {
        if emit_calldata {
            Self::EmitCalldata
        } else if dry_run {
            Self::DryRun
        } else {
            Self::Submit
        }
    }

    pub(crate) const fn is_no_submit(self) -> bool {
        !matches!(self, Self::Submit)
    }
}

/// One Starknet account-call emitted for an external signer / wallet API.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmittedCall {
    /// Contract address the account should call.
    pub(crate) contract_address: String,
    /// Back-compat alias used by earlier `trade execute --emit-calldata`.
    pub(crate) contract: String,
    /// Human-readable entrypoint name inferred from the bundle shape.
    pub(crate) entry_point: String,
    /// Back-compat alias used by earlier `trade execute --emit-calldata`.
    pub(crate) entrypoint: String,
    /// Entrypoint selector as a felt hex string.
    pub(crate) entry_point_selector: String,
    /// Back-compat alias used by earlier `trade execute --emit-calldata`.
    pub(crate) selector: String,
    /// Cairo calldata felts as hex strings.
    pub(crate) calldata: Vec<String>,
}

/// Keyless preflight result paired with emitted calldata.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CalldataPreflight {
    pub(crate) is_valid: bool,
    pub(crate) computed_collateral: Option<f64>,
    pub(crate) rejection: Option<String>,
    pub(crate) simulation_note: Option<String>,
}

/// Renderable result for `--emit-calldata`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CalldataResult {
    pub(crate) action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) family: Option<&'static str>,
    /// Market/token target. Kept as `market` for compatibility with the first
    /// trade-only emitter.
    pub(crate) market: String,
    pub(crate) account_address: String,
    /// Back-compat alias used by earlier `trade execute --emit-calldata`.
    pub(crate) account: String,
    pub(crate) call_count: usize,
    /// Back-compat boolean alias for `preflight.is_valid`.
    pub(crate) validated: bool,
    pub(crate) preflight: CalldataPreflight,
    pub(crate) calls: Vec<EmittedCall>,
    pub(crate) note: String,
}

pub(crate) struct CalldataInput<'a> {
    pub(crate) action: &'static str,
    pub(crate) family: Option<&'static str>,
    pub(crate) target: Felt,
    pub(crate) account: Felt,
    pub(crate) calls: &'a [Call],
    pub(crate) labels: &'a [String],
    pub(crate) simulation: &'a SubmissionResult,
    pub(crate) computed_collateral: Option<f64>,
}

impl Render for CalldataResult {
    fn render_pretty(&self, r: &Renderer) {
        if self.validated {
            r.success("calldata validated");
        } else {
            r.error("calldata simulation rejected");
        }
        if let Some(family) = self.family {
            r.kv("family", family);
        }
        r.kv("target", &self.market);
        r.kv("account", &self.account_address);
        r.kv("call_count", &self.call_count.to_string());
        if let Some(note) = &self.preflight.simulation_note {
            r.kv("simulation", note);
        }
        for (i, call) in self.calls.iter().enumerate() {
            r.kv(
                &format!("call_{i}"),
                &format!(
                    "{} {} calldata_felts={}",
                    call.contract_address,
                    call.entry_point,
                    call.calldata.len()
                ),
            );
        }
        r.kv("note", &self.note);
    }

    fn render_plain(&self, w: &mut dyn io::Write) -> io::Result<()> {
        writeln!(w, "action: {}", self.action)?;
        if let Some(family) = self.family {
            writeln!(w, "family: {family}")?;
        }
        writeln!(w, "market: {}", self.market)?;
        writeln!(w, "account_address: {}", self.account_address)?;
        writeln!(w, "call_count: {}", self.call_count)?;
        writeln!(w, "validated: {}", self.validated)?;
        writeln!(w, "preflight_is_valid: {}", self.preflight.is_valid)?;
        if let Some(collateral) = self.preflight.computed_collateral {
            writeln!(w, "preflight_computed_collateral: {collateral}")?;
        }
        if let Some(rejection) = &self.preflight.rejection {
            writeln!(w, "preflight_rejection: {rejection}")?;
        }
        if let Some(note) = &self.preflight.simulation_note {
            writeln!(w, "simulation_note: {note}")?;
        }
        for (i, call) in self.calls.iter().enumerate() {
            writeln!(w, "call_{i}_contract_address: {}", call.contract_address)?;
            writeln!(w, "call_{i}_entry_point: {}", call.entry_point)?;
            writeln!(
                w,
                "call_{i}_entry_point_selector: {}",
                call.entry_point_selector
            )?;
            writeln!(w, "call_{i}_calldata: {}", call.calldata.join(","))?;
        }
        writeln!(w, "note: {}", self.note)
    }
}

pub(crate) fn emitted_calls(calls: &[Call], labels: &[String]) -> Vec<EmittedCall> {
    calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            let entrypoint = labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| "unknown".to_owned());
            let contract = format!("{:#x}", call.to);
            let selector = format!("{:#x}", call.selector);
            EmittedCall {
                contract_address: contract.clone(),
                contract,
                entry_point: entrypoint.clone(),
                entrypoint,
                entry_point_selector: selector.clone(),
                selector,
                calldata: call
                    .calldata
                    .iter()
                    .map(|felt| format!("{felt:#x}"))
                    .collect(),
            }
        })
        .collect()
}

pub(crate) fn calldata_result(input: CalldataInput<'_>) -> CalldataResult {
    CalldataResult {
        action: input.action,
        family: input.family,
        market: format!("{:#x}", input.target),
        account_address: format!("{:#x}", input.account),
        account: format!("{:#x}", input.account),
        call_count: input.calls.len(),
        validated: input.simulation.accepted,
        preflight: CalldataPreflight {
            is_valid: input.simulation.accepted,
            computed_collateral: input.computed_collateral,
            rejection: input
                .simulation
                .rejection
                .as_ref()
                .map(|rejection| rejection.variant.clone())
                .or_else(|| {
                    (!input.simulation.accepted)
                        .then(|| input.simulation.note.clone())
                        .flatten()
                }),
            simulation_note: input.simulation.note.clone(),
        },
        calls: emitted_calls(input.calls, input.labels),
        note: "No transaction submitted. Send these calls through an external signer as one account multicall."
            .into(),
    }
}

pub(crate) fn print_calldata_json(result: &CalldataResult) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, result)?;
    handle.write_all(b"\n")?;
    Ok(())
}

/// Convert a fee in FRI (10^-18 STRK) to a human STRK amount for display.
fn fri_to_strk(fri: u128) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "fee is for display only")]
    let strk = fri as f64 / 1e18_f64;
    strk
}

/// Run a gas-free account simulation. This never submits.
pub(crate) async fn simulate_calls(
    action: &'static str,
    target: Felt,
    account: &impl Account,
    calls: &[Call],
) -> SubmissionResult {
    let target_s = format!("{target:#x}");
    let base = |accepted: bool, note: String| SubmissionResult {
        action,
        market: target_s.clone(),
        tx_hash: None,
        call_count: Some(calls.len()),
        accepted,
        rejection: None,
        note: Some(note),
    };
    match account.simulate(calls).await {
        Ok(Some(sim)) => match sim.revert_reason {
            Some(reason) => base(
                false,
                format!(
                    "DRY RUN — multicall WOULD REVERT on-chain: {reason}. \
                     No transaction submitted, no gas spent."
                ),
            ),
            None => base(
                true,
                format!(
                    "DRY RUN — simulation OK (≈{:.6} STRK est. fee). \
                     Re-run without --dry-run to submit.",
                    fri_to_strk(sim.estimated_fee)
                ),
            ),
        },
        Ok(None) => base(
            false,
            "DRY RUN — this account type cannot simulate (no provider-backed signer).".into(),
        ),
        Err(e) => base(false, format!("DRY RUN — simulation call failed: {e}")),
    }
}

pub(crate) fn build_write_account(
    ctx: &AppContext,
    mode: WriteMode,
) -> Result<deadeye_starknet::OwnedAccount> {
    if mode.is_no_submit() || ctx.config.uses_external_signer() {
        build_simulation_account(ctx)
    } else {
        super::runtime_resolver::build_owned_account(ctx)
    }
}

pub(crate) async fn submit_external_calls(
    ctx: &AppContext,
    action: &'static str,
    target: Felt,
    account: Felt,
    calls: &[Call],
    labels: &[String],
) -> Result<SubmissionResult> {
    let Some(config) = ctx.config.external_signer.as_ref() else {
        bail!(
            "profile signer is external but no [profiles.{}.external_signer] config is present",
            ctx.config.profile_name,
        );
    };
    let kind = config.kind.as_deref().unwrap_or("strkd");
    if !kind.eq_ignore_ascii_case("strkd") {
        bail!("unsupported external signer kind {kind:?}; built-in adapter: strkd");
    }
    let client = StrkdClient::from_config(config)?;
    let result = client
        .add_invoke_transaction(
            &ctx.config.rpc_url,
            &ctx.config.chain_id,
            account,
            calls,
            labels,
        )
        .await?;
    Ok(SubmissionResult {
        action,
        market: format!("{target:#x}"),
        tx_hash: result.transaction_hash,
        call_count: Some(calls.len()),
        accepted: true,
        rejection: None,
        note: Some(result.note),
    })
}

#[derive(Debug, Clone)]
struct StrkdClient {
    endpoint: reqwest::Url,
    token: String,
    client_id: String,
    submit: bool,
    allowance_strategy: String,
    http: reqwest::Client,
}

#[derive(Debug)]
struct StrkdResult {
    transaction_hash: Option<String>,
    note: String,
}

impl StrkdClient {
    fn from_config(config: &ExternalSignerConfig) -> Result<Self> {
        let endpoint = resolve_endpoint(config)?;
        let (file_token, file_client_id) = config
            .token_path
            .as_deref()
            .map(read_token_file)
            .transpose()?
            .unwrap_or_default();
        let token = std::env::var("DEADEYE_STRKD_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| config.token.clone())
            .or(file_token)
            .ok_or_else(|| {
                anyhow!(
                    "strkd external signer requires a bearer token; set token_path, \
                     token, or DEADEYE_STRKD_TOKEN"
                )
            })?;
        let client_id = std::env::var("DEADEYE_STRKD_CLIENT_ID")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| config.client_id.clone())
            .or(file_client_id)
            .ok_or_else(|| {
                anyhow!(
                    "strkd external signer requires a companion client id; set token_path, \
                     client_id, or DEADEYE_STRKD_CLIENT_ID"
                )
            })?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("building strkd HTTP client")?;
        Ok(Self {
            endpoint,
            token,
            client_id,
            submit: config.submit.unwrap_or(true),
            allowance_strategy: config
                .allowance_strategy
                .clone()
                .unwrap_or_else(|| "skip-estimate".to_owned()),
            http,
        })
    }

    async fn add_invoke_transaction(
        &self,
        rpc_url: &str,
        chain_id: &str,
        account: Felt,
        calls: &[Call],
        labels: &[String],
    ) -> Result<StrkdResult> {
        let mut params = json!({
            "account_address": format!("{account:#x}"),
            "calls": wallet_calls(calls, labels),
            "submit": self.submit,
            "chainId": chain_id,
        });
        if self
            .allowance_strategy
            .eq_ignore_ascii_case("skip-estimate")
        {
            let gas = current_gas_params(rpc_url).await?;
            params["resource_bounds"] = resource_bounds_json(gas);
        }
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1_u64,
            "method": "wallet_addInvokeTransaction",
            "params": params,
        });
        let req = self
            .http
            .post(self.endpoint.clone())
            .bearer_auth(&self.token)
            .header("X-Companion-Client", &self.client_id)
            .json(&body);
        let status_and_text = req
            .send()
            .await
            .context("POST wallet_addInvokeTransaction to strkd")?
            .error_for_status()
            .context("strkd wallet_addInvokeTransaction returned HTTP error")?
            .text()
            .await
            .context("reading strkd response body")?;
        let raw: serde_json::Value = serde_json::from_str(&status_and_text)
            .with_context(|| format!("decoding strkd response: {status_and_text}"))?;
        if let Some(error) = raw.get("error") {
            bail!("strkd wallet_addInvokeTransaction error: {error}");
        }
        let result = raw.get("result").unwrap_or(&raw);
        let tx_hash = find_hash(result);
        let note = if let Some(hash) = &tx_hash {
            format!("external signer submitted via strkd; transaction_hash={hash}")
        } else if result.get("signed_transaction").is_some()
            || result.get("signedTransaction").is_some()
        {
            "external signer returned a signed transaction but did not report a broadcast hash"
                .to_owned()
        } else {
            format!("external signer accepted wallet_addInvokeTransaction: {result}")
        };
        Ok(StrkdResult {
            transaction_hash: tx_hash,
            note,
        })
    }
}

const RESOURCE_PRICE_SAFETY_MULTIPLIER: u128 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResourcePrices {
    l1_gas_price: u128,
    l2_gas_price: u128,
    l1_data_gas_price: u128,
}

async fn current_gas_params(rpc_url: &str) -> Result<GasParams> {
    let url = url::Url::parse(rpc_url)
        .with_context(|| format!("rpc_url is not a valid URL: {rpc_url}"))?;
    let provider = JsonRpcClient::new(HttpTransport::new(url));
    let prices = current_resource_prices(&provider).await?;
    Ok(gas_params_with_resource_prices(prices))
}

async fn current_resource_prices<P>(provider: &P) -> Result<ResourcePrices>
where
    P: starknet_providers::Provider + Sync,
{
    match provider
        .get_block_with_tx_hashes(BlockId::Tag(BlockTag::PreConfirmed))
        .await
    {
        Ok(block) => resource_prices_from_block(block),
        Err(preconfirmed_err) => {
            let block = provider
                .get_block_with_tx_hashes(BlockId::Tag(BlockTag::Latest))
                .await
                .with_context(|| {
                    format!(
                        "fetching Starknet gas prices from pre_confirmed failed \
                         ({preconfirmed_err}); latest fallback also failed"
                    )
                })?;
            resource_prices_from_block(block)
        },
    }
}

fn resource_prices_from_block(block: MaybePreConfirmedBlockWithTxHashes) -> Result<ResourcePrices> {
    match block {
        MaybePreConfirmedBlockWithTxHashes::Block(block) => resource_prices_from_starknet_fields(
            &block.l1_gas_price,
            &block.l2_gas_price,
            &block.l1_data_gas_price,
        ),
        MaybePreConfirmedBlockWithTxHashes::PreConfirmedBlock(block) => {
            resource_prices_from_starknet_fields(
                &block.l1_gas_price,
                &block.l2_gas_price,
                &block.l1_data_gas_price,
            )
        },
    }
}

fn resource_prices_from_starknet_fields(
    l1_gas: &ResourcePrice,
    l2_gas: &ResourcePrice,
    l1_data_gas: &ResourcePrice,
) -> Result<ResourcePrices> {
    Ok(ResourcePrices {
        l1_gas_price: resource_price_fri("l1_gas", l1_gas)?,
        l2_gas_price: resource_price_fri("l2_gas", l2_gas)?,
        l1_data_gas_price: resource_price_fri("l1_data_gas", l1_data_gas)?,
    })
}

fn resource_price_fri(name: &str, price: &ResourcePrice) -> Result<u128> {
    u128::try_from(price.price_in_fri)
        .with_context(|| format!("{name} price_in_fri does not fit in u128"))
}

fn gas_params_with_resource_prices(prices: ResourcePrices) -> GasParams {
    let mut gas = GasParams::generous_defaults();
    gas.l1_gas_price = gas
        .l1_gas_price
        .max(padded_resource_price(prices.l1_gas_price));
    gas.l2_gas_price = gas
        .l2_gas_price
        .max(padded_resource_price(prices.l2_gas_price));
    gas.l1_data_gas_price = gas
        .l1_data_gas_price
        .max(padded_resource_price(prices.l1_data_gas_price));
    gas
}

const fn padded_resource_price(price: u128) -> u128 {
    price.saturating_mul(RESOURCE_PRICE_SAFETY_MULTIPLIER)
}

fn wallet_calls(calls: &[Call], labels: &[String]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            json!({
                "contract_address": format!("{:#x}", call.to),
                "entry_point": labels.get(i).map(String::as_str).unwrap_or("unknown"),
                "entry_point_selector": format!("{:#x}", call.selector),
                "calldata": call.calldata.iter().map(|felt| format!("{felt:#x}")).collect::<Vec<_>>(),
            })
        })
        .collect()
}

fn resource_bounds_json(gas: GasParams) -> serde_json::Value {
    json!({
        "l1_gas": {
            "max_amount": format!("{:#x}", gas.l1_gas),
            "max_price_per_unit": format!("{:#x}", gas.l1_gas_price),
        },
        "l2_gas": {
            "max_amount": format!("{:#x}", gas.l2_gas),
            "max_price_per_unit": format!("{:#x}", gas.l2_gas_price),
        },
        "l1_data_gas": {
            "max_amount": format!("{:#x}", gas.l1_data_gas),
            "max_price_per_unit": format!("{:#x}", gas.l1_data_gas_price),
        },
    })
}

fn find_hash(value: &serde_json::Value) -> Option<String> {
    for key in [
        "transaction_hash",
        "transactionHash",
        "tx_hash",
        "txHash",
        "hash",
    ] {
        if let Some(hash) = value.get(key).and_then(serde_json::Value::as_str) {
            return Some(hash.to_owned());
        }
    }
    None
}

fn resolve_endpoint(config: &ExternalSignerConfig) -> Result<reqwest::Url> {
    if let Some(endpoint) = std::env::var("DEADEYE_EXTERNAL_SIGNER_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| config.endpoint.clone())
    {
        return endpoint.parse().context("parsing external signer endpoint");
    }
    let discovery = config.discovery.as_deref().unwrap_or("port_lock");
    if !discovery.eq_ignore_ascii_case("port_lock") {
        bail!("external signer discovery {discovery:?} requires an explicit endpoint");
    }
    let port_lock = resolve_port_lock(config)?;
    let port = read_port_lock(&port_lock)?;
    format!("http://127.0.0.1:{port}")
        .parse()
        .context("building strkd endpoint from port_lock")
}

fn resolve_port_lock(config: &ExternalSignerConfig) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("DEADEYE_STRKD_PORT_LOCK") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = &config.port_lock_path {
        return Ok(expand_tilde(path));
    }
    let mut candidates = Vec::new();
    if let Some(dir) = dirs::data_local_dir() {
        candidates.push(dir.join("strkd").join("port.lock"));
    }
    if let Some(dir) = dirs::data_dir() {
        candidates.push(dir.join("strkd").join("port.lock"));
    }
    if let Some(dir) = dirs::config_dir() {
        candidates.push(dir.join("strkd").join("port.lock"));
    }
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| {
            anyhow!(
                "could not find strkd port.lock; set endpoint, port_lock_path, or \
                 DEADEYE_EXTERNAL_SIGNER_ENDPOINT"
            )
        })
}

fn read_port_lock(path: &Path) -> Result<u16> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading strkd port lock {}", path.display()))?;
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw)
        && let Some(port) = json.get("port").and_then(serde_json::Value::as_u64)
    {
        return u16::try_from(port).context("strkd port out of u16 range");
    }
    let digits: String = raw.chars().filter(char::is_ascii_digit).collect();
    let port = digits
        .parse::<u16>()
        .with_context(|| format!("parsing port from {}", path.display()))?;
    Ok(port)
}

#[derive(Debug, Default, Deserialize)]
struct TokenFile {
    token: Option<String>,
    bearer_token: Option<String>,
    client_id: Option<String>,
    #[serde(rename = "clientId")]
    client_id_camel: Option<String>,
}

fn read_token_file(path: &str) -> Result<(Option<String>, Option<String>)> {
    let path = expand_tilde(path);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading strkd token file {}", path.display()))?;
    let token_file: TokenFile = serde_json::from_str(&raw)
        .with_context(|| format!("parsing strkd token file {}", path.display()))?;
    Ok((
        token_file.token.or(token_file.bearer_token),
        token_file.client_id.or(token_file.client_id_camel),
    ))
}

fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

pub(crate) fn labels<I, S>(items: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    items.into_iter().map(Into::into).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_lock_accepts_json_and_plain_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let json_path = dir.path().join("port-json.lock");
        std::fs::write(&json_path, r#"{"port":37643}"#).expect("write");
        assert_eq!(read_port_lock(&json_path).expect("json"), 37_643);

        let plain_path = dir.path().join("port.lock");
        std::fs::write(&plain_path, "37644\n").expect("write");
        assert_eq!(read_port_lock(&plain_path).expect("plain"), 37_644);
    }

    #[test]
    fn emitted_call_keeps_new_and_legacy_field_names() {
        let call = Call {
            to: Felt::from(0x123_u64),
            selector: Felt::from(0x456_u64),
            calldata: vec![Felt::from(0x789_u64)],
        };
        let emitted = emitted_calls(&[call], &labels(["approve"]));
        let value = serde_json::to_value(&emitted[0]).expect("json");
        assert_eq!(value["contract_address"], "0x123");
        assert_eq!(value["contract"], "0x123");
        assert_eq!(value["entry_point"], "approve");
        assert_eq!(value["entrypoint"], "approve");
        assert_eq!(value["entry_point_selector"], "0x456");
        assert_eq!(value["selector"], "0x456");
    }

    #[test]
    fn gas_params_use_live_resource_prices_with_default_floor() {
        let gas = gas_params_with_resource_prices(ResourcePrices {
            l1_gas_price: 109_210_567_966_765,
            l2_gas_price: 1,
            l1_data_gas_price: 77_000_000_000_000,
        });

        assert_eq!(gas.l1_gas, 10_000);
        assert_eq!(gas.l1_gas_price, 218_421_135_933_530);
        assert_eq!(gas.l2_gas, 100_000_000);
        assert_eq!(gas.l2_gas_price, 100_000_000_000);
        assert_eq!(gas.l1_data_gas, 10_000);
        assert_eq!(gas.l1_data_gas_price, 154_000_000_000_000);
    }

    #[test]
    fn resource_bounds_json_serializes_supplied_prices() {
        let gas = gas_params_with_resource_prices(ResourcePrices {
            l1_gas_price: 109_210_567_966_765,
            l2_gas_price: 60_000_000_000,
            l1_data_gas_price: 77_000_000_000_000,
        });
        let value = resource_bounds_json(gas);

        assert_eq!(
            value["l1_gas"]["max_price_per_unit"],
            format!("{:#x}", 218_421_135_933_530_u128)
        );
        assert_eq!(
            value["l2_gas"]["max_price_per_unit"],
            format!("{:#x}", 120_000_000_000_u128)
        );
        assert_eq!(
            value["l1_data_gas"]["max_price_per_unit"],
            format!("{:#x}", 154_000_000_000_000_u128)
        );
    }

    #[test]
    fn resource_price_fri_rejects_values_larger_than_u128() {
        let price = ResourcePrice {
            price_in_fri: Felt::from_hex("0x100000000000000000000000000000000")
                .expect("valid felt"),
            price_in_wei: Felt::ZERO,
        };

        let err = resource_price_fri("l1_gas", &price).expect_err("overflow rejected");
        assert!(err.to_string().contains("does not fit in u128"));
    }
}
