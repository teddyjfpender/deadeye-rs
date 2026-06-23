//! `deadeye collateral …` — wraps the restricted-collateral-token (XP)
//! contract.
//!
//! The XP token exposes a one-shot `claim_initial_grant()` entrypoint
//! that mints a fixed amount of XP to the caller. This is how a fresh
//! wallet (e.g. a bot operator bootstrapping the cpi-arb-bot) obtains
//! the collateral it needs to trade against any deployed market.
//!
//! Both subcommands are read-then-write: every claim is preflighted with
//! `has_claimed_initial_grant(caller)` so re-running on an already-funded
//! wallet is a clean no-op rather than a guaranteed revert.

use anyhow::{Context, Result, bail};
use deadeye_starknet::{
    Account, CollateralTokenReader, CollateralTokenWriter, MAINNET_XP_TOKEN_ADDRESS,
};
use serde::Serialize;
use starknet_core::types::{Felt, U256};

use crate::{
    cli::{CollateralBalanceArgs, CollateralClaimGrantArgs, CollateralCmd},
    commands::{
        call_bundle::{
            CalldataInput, WriteMode, build_write_account, calldata_result, labels,
            print_calldata_json, simulate_calls, submit_external_calls,
        },
        render_helpers::{submission_from_receipt, submission_from_trade_error},
        runtime_resolver::{build_provider, parse_felt},
    },
    context::AppContext,
    output::{OutputMode, Render, Renderer},
};

pub(crate) async fn run(action: CollateralCmd, ctx: &AppContext) -> Result<()> {
    match action {
        CollateralCmd::ClaimGrant(args) => claim_grant(args, ctx).await,
        CollateralCmd::Balance(args) => balance(args, ctx).await,
    }
}

/// Resolve the token address from `--token` or fall back to the bundled
/// mainnet constant.
fn resolve_token(raw: Option<&str>) -> Result<Felt> {
    match raw {
        Some(s) => parse_felt("token address", s),
        None => Ok(MAINNET_XP_TOKEN_ADDRESS),
    }
}

/// Plain-data summary of `claim-grant` — uniform shape across dry-run,
/// skip, and execute paths so JSON consumers see one schema.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ClaimGrantPlan {
    /// Caller wallet (the recipient of the grant).
    pub(crate) account: String,
    /// XP token contract.
    pub(crate) token: String,
    /// Grant size in raw 18-decimal units (hex string — u256 can exceed u64).
    pub(crate) grant_raw_hex: String,
    /// Grant size as f64, in human XP units. Display-only.
    pub(crate) grant_xp: f64,
    /// Whether the caller has already claimed (`true` → submit is skipped).
    pub(crate) already_claimed: bool,
    /// Wallet's pre-claim XP balance, in human XP units.
    pub(crate) balance_before_xp: f64,
    /// `"dry-run" | "execute" | "skipped (already claimed)"`.
    pub(crate) mode: String,
    /// Submitted-tx hash (set after a successful execute).
    pub(crate) tx_hash: Option<String>,
}

impl Render for ClaimGrantPlan {
    fn render_pretty(&self, r: &Renderer) {
        r.kv("account", &self.account);
        r.kv("token", &self.token);
        if self.grant_xp.is_finite() {
            r.kv(
                "grant",
                &format!("{:.4} XP  (raw {})", self.grant_xp, self.grant_raw_hex),
            );
        } else {
            r.kv("grant", &self.grant_raw_hex);
        }
        r.kv(
            "balance_before",
            &format!("{:.4} XP", self.balance_before_xp),
        );
        r.kv("mode", &self.mode);
        if let Some(tx) = &self.tx_hash {
            r.kv("tx_hash", tx);
        } else if self.mode == "dry-run" {
            println!("(pass --execute to submit)");
        }
    }

    fn render_plain(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(w, "account: {}", self.account)?;
        writeln!(w, "token: {}", self.token)?;
        writeln!(w, "grant_raw_hex: {}", self.grant_raw_hex)?;
        writeln!(w, "grant_xp: {}", self.grant_xp)?;
        writeln!(w, "already_claimed: {}", self.already_claimed)?;
        writeln!(w, "balance_before_xp: {}", self.balance_before_xp)?;
        writeln!(w, "mode: {}", self.mode)?;
        if let Some(tx) = &self.tx_hash {
            writeln!(w, "tx_hash: {tx}")?;
        }
        Ok(())
    }
}

async fn claim_grant(args: CollateralClaimGrantArgs, ctx: &AppContext) -> Result<()> {
    let account_felt = ctx.resolved_address_felt()?;
    let token = resolve_token(args.token.as_deref())?;

    // Pre-flight reads: `has_claimed_initial_grant`, `initial_grant`,
    // and current balance. All cheap view calls; we batch them
    // sequentially because the chain reader doesn't expose a multicall
    // primitive and the latency is negligible.
    let provider = build_provider(ctx)?;
    let reader = CollateralTokenReader::new(&provider, token);

    let already_claimed = reader
        .has_claimed_initial_grant(account_felt)
        .await
        .context("reading has_claimed_initial_grant failed")?;
    let grant_raw = reader
        .initial_grant()
        .await
        .context("reading initial_grant failed")?;
    let balance_raw = reader
        .balance_of(account_felt)
        .await
        .context("reading balance_of failed")?;

    let mode_label = if already_claimed {
        "skipped (already claimed)"
    } else if args.emit_calldata {
        "emit-calldata"
    } else if args.execute {
        "execute"
    } else {
        "dry-run"
    };

    let mut plan = ClaimGrantPlan {
        account: format!("{account_felt:#x}"),
        token: format!("{token:#x}"),
        grant_raw_hex: format!("{grant_raw:#x}"),
        grant_xp: u256_to_human_18(grant_raw),
        already_claimed,
        balance_before_xp: u256_to_human_18(balance_raw),
        mode: mode_label.to_owned(),
        tx_hash: None,
    };

    if already_claimed {
        ctx.renderer.print(&plan)?;
        if ctx.renderer.mode() != OutputMode::Json {
            println!();
            println!("wallet has already claimed its initial grant; no action needed");
        }
        return Ok(());
    }

    if !args.execute && !args.emit_calldata {
        ctx.renderer.print(&plan)?;
        return Ok(());
    }

    // ── Submit / emit ─────────────────────────────────────────────────
    let mode = if args.emit_calldata {
        WriteMode::EmitCalldata
    } else {
        WriteMode::Submit
    };
    let signer = build_write_account(ctx, mode)?;
    let signer_addr = Account::address(&signer);
    if signer_addr != account_felt {
        bail!(
            "signer ({signer_addr:#x}) does not match --address ({account_felt:#x}); \
             claim_initial_grant is keyed on `caller`, so the wallet that signs IS the recipient",
        );
    }
    let calls = vec![deadeye_starknet::build_claim_initial_grant_call(token)];
    let call_labels = labels(["claim_initial_grant"]);
    if mode == WriteMode::EmitCalldata {
        let sim = simulate_calls("claim_initial_grant(preflight)", token, &signer, &calls).await;
        let result = calldata_result(CalldataInput {
            action: "claim_initial_grant(emit-calldata)",
            family: None,
            target: token,
            account: signer_addr,
            calls: &calls,
            labels: &call_labels,
            simulation: &sim,
            computed_collateral: None,
        });
        return print_calldata_json(&result);
    }
    if ctx.config.uses_external_signer() {
        let sim = simulate_calls("claim_initial_grant(preflight)", token, &signer, &calls).await;
        if !sim.accepted {
            ctx.renderer.print(&plan)?;
            return ctx.renderer.print(&sim);
        }
        let submission = submit_external_calls(
            ctx,
            "claim_initial_grant",
            token,
            signer_addr,
            &calls,
            &call_labels,
        )
        .await?;
        plan.tx_hash = submission.tx_hash.clone();
        ctx.renderer.print(&plan)?;
        return ctx.renderer.print(&submission);
    }
    let writer = CollateralTokenWriter::new(reader, signer);
    let market_hex = format!("{token:#x}");
    match writer.claim_initial_grant().await {
        Ok(receipt) => {
            plan.tx_hash = Some(format!("{:#x}", receipt.transaction_hash));
            ctx.renderer.print(&plan)?;
            let submission = submission_from_receipt("claim_initial_grant", market_hex, receipt);
            ctx.renderer.print(&submission)
        },
        Err(e) => {
            ctx.renderer.print(&plan)?;
            let detail = format!("{e:?}");
            let trade_err = deadeye_starknet::TradeError::from_contract(e);
            let submission =
                submission_from_trade_error("claim_initial_grant", market_hex, &trade_err);
            ctx.renderer.print(&submission)?;
            // The #1 cause for a fresh wallet: the account contract isn't
            // deployed yet, so it can't send any transaction.
            if detail.contains("ContractNotFound") {
                ctx.renderer.warning(
                    "this account isn't deployed yet — fund it with a little STRK for gas, \
                     run `deadeye account deploy`, then retry the claim",
                );
            }
            bail!("claim_initial_grant submission failed")
        },
    }
}

/// Plain-data summary of `balance`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct BalanceView {
    pub(crate) account: String,
    pub(crate) token: String,
    pub(crate) balance_raw_hex: String,
    pub(crate) balance_xp: f64,
    pub(crate) already_claimed_initial_grant: bool,
    pub(crate) grant_raw_hex: String,
    pub(crate) grant_xp: f64,
}

impl Render for BalanceView {
    fn render_pretty(&self, r: &Renderer) {
        r.kv("account", &self.account);
        r.kv("token", &self.token);
        r.kv(
            "balance",
            &format!("{:.4} XP  (raw {})", self.balance_xp, self.balance_raw_hex),
        );
        r.kv(
            "initial_grant",
            &format!("{:.4} XP  (raw {})", self.grant_xp, self.grant_raw_hex),
        );
        r.kv(
            "grant_claimed",
            if self.already_claimed_initial_grant {
                "yes"
            } else {
                "no — run `deadeye collateral claim-grant --execute`"
            },
        );
    }

    fn render_plain(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(w, "account: {}", self.account)?;
        writeln!(w, "token: {}", self.token)?;
        writeln!(w, "balance_raw_hex: {}", self.balance_raw_hex)?;
        writeln!(w, "balance_xp: {}", self.balance_xp)?;
        writeln!(
            w,
            "already_claimed_initial_grant: {}",
            self.already_claimed_initial_grant
        )?;
        writeln!(w, "grant_raw_hex: {}", self.grant_raw_hex)?;
        writeln!(w, "grant_xp: {}", self.grant_xp)
    }
}

async fn balance(args: CollateralBalanceArgs, ctx: &AppContext) -> Result<()> {
    let account_felt = match args.account.as_deref() {
        Some(s) => parse_felt("account address", s)?,
        None => ctx.resolved_address_felt()?,
    };
    let token = resolve_token(args.token.as_deref())?;

    let provider = build_provider(ctx)?;
    let reader = CollateralTokenReader::new(&provider, token);
    let balance_raw = reader.balance_of(account_felt).await?;
    let grant_raw = reader.initial_grant().await?;
    let already = reader.has_claimed_initial_grant(account_felt).await?;

    let view = BalanceView {
        account: format!("{account_felt:#x}"),
        token: format!("{token:#x}"),
        balance_raw_hex: format!("{balance_raw:#x}"),
        balance_xp: u256_to_human_18(balance_raw),
        already_claimed_initial_grant: already,
        grant_raw_hex: format!("{grant_raw:#x}"),
        grant_xp: u256_to_human_18(grant_raw),
    };
    ctx.renderer.print(&view)
}

/// Raw 18-decimal `u256` → human f64 (for display only — gates should
/// compare in raw u256 space). Saturates on overflow.
pub(crate) fn u256_to_human_18(raw: U256) -> f64 {
    const WAD: f64 = 1.0e18;
    // 2^128 in f64 — exact since 2^128 is a power of two well within
    // f64's exponent range. The literal goes through f64::from_bits
    // (via the `as` cast on a u128 sentinel) to avoid a lossy decimal
    // literal that clippy flags.
    #[allow(clippy::cast_precision_loss, reason = "2^128 is exact in f64")]
    let two_pow_128: f64 = (u128::MAX as f64) + 1.0;
    let low = raw.low() as f64;
    let high = raw.high() as f64;
    high.mul_add(two_pow_128, low) / WAD
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use super::*;

    #[test]
    fn one_xp_round_trips_through_18dec_divisor() {
        let raw = U256::from(1_000_000_000_000_000_000_u128);
        let human = u256_to_human_18(raw);
        assert!((human - 1.0).abs() < 1e-9, "got {human}");
    }

    #[test]
    fn zero_xp_is_zero() {
        // f64 == 0.0 is well-defined: zero literal is exact.
        assert!(u256_to_human_18(U256::from(0_u32)).abs() < f64::EPSILON);
    }

    #[test]
    fn resolve_token_default_matches_mainnet_xp() {
        let got = resolve_token(None).unwrap();
        assert_eq!(got, MAINNET_XP_TOKEN_ADDRESS);
    }

    #[test]
    fn resolve_token_override_parses_hex() {
        let raw = "0x4583395a85181cd495b8563d277bdadfbca15fc65d2945db6c5c11f7d0e786e";
        let got = resolve_token(Some(raw)).unwrap();
        assert_eq!(got, Felt::from_hex(raw).unwrap());
    }
}
