#![allow(
    clippy::print_stderr,
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::shadow_unrelated,
    clippy::panic,
    reason = "integration tests panic on setup failure and print debug to stderr"
)]

//! Driver B's integration tests against starknet-devnet.
//!
//! Bootstrapped via the testkit's `bootstrap_devnet`, then drives the
//! `deadeye` binary through the four write paths and `watch`.
//!
//! Enable with `DEADEYE_RUN_INTEGRATION=1` and a running
//! `starknet-devnet --seed 0 --accounts 10 --port 5050`.

use std::{process::Command, time::Duration};

use deadeye_core::Sq128;
use deadeye_testkit::fixture::{
    bootstrap_devnet,
    env::BootstrapConfig,
    erc20::approve,
    lifecycle::{
        build_initial_normal_inputs, deploy_normal_market_with_event, fetch_normal_hints,
        initialize_market, upsert_normal_profile_for_test,
    },
};
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};

fn integration_enabled() -> bool {
    std::env::var("DEADEYE_RUN_INTEGRATION").is_ok()
}

fn cli_binary() -> std::path::PathBuf {
    // assert_cmd looks up the bin from CARGO_BIN_EXE_<name>. We avoid the
    // assert_cmd dependency on `Command` directly for finer-grained control
    // over env vars / wait timing.
    assert_cmd::cargo::cargo_bin("deadeye")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadeye_trade_quote_and_execute_devnet() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1");
        return;
    }
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .expect("bootstrap succeeds");

    let admin_handle = env.account_handle(&env.admin);

    upsert_normal_profile_for_test(admin_handle.clone(), env.factory, env.collateral, 1)
        .await
        .expect("upsert normal profile");

    let (initial_dist, _placeholder) = build_initial_normal_inputs(42.0, 64.0, 1000.0);
    let hint_rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));
    let initial_hints = fetch_normal_hints(&hint_rpc, env.normal_runtime, initial_dist)
        .await
        .expect("fetch hints");
    let market = deploy_normal_market_with_event(
        &admin_handle,
        env.factory,
        1,
        starknet_core::types::Felt::from(1_u64),
        starknet_core::types::Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .expect("deploy normal market");
    initialize_market(
        &admin_handle,
        market,
        env.collateral,
        10_000_000_000_000_000_000_000_u128,
    )
    .await
    .expect("initialize market");

    let trader = env.participants.first().expect("a participant");
    let trader_handle = env.account_handle(trader);
    approve(
        trader_handle.clone(),
        env.collateral,
        market,
        1_000_000_000_000_000_000_000_u128,
    )
    .await
    .expect("approve");

    let market_hex = format!("{market:#x}");
    let chain_id_hex = format!("{:#x}", env.chain_id);
    let trader_addr = format!("{:#x}", trader.address);
    let trader_pk = format!("{:#x}", trader.private_key);
    let runtime_hex = format!("{:#x}", env.normal_runtime);

    // ── trade quote ────────────────────────────────────────────
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .env("DEADEYE_NORMAL_RUNTIME_ADDR", &runtime_hex)
        .arg("trade")
        .arg("quote")
        .arg(&market_hex)
        .arg("--family")
        .arg("normal")
        .arg("--mean")
        .arg("43.0")
        .arg("--variance")
        .arg("81.0")
        .arg("--pad")
        .arg("5.0")
        .output()
        .expect("spawn deadeye trade quote");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("trade quote stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        output.status.success(),
        "trade quote returned non-zero exit"
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let on_chain = parsed
        .get("on_chain_will_accept")
        .and_then(serde_json::Value::as_bool)
        .expect("on_chain_will_accept field present");
    assert!(on_chain, "expected the candidate to be on-chain-acceptable");

    // ── trade execute ──────────────────────────────────────────
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .arg("--confirm")
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .env("DEADEYE_PRIVATE_KEY", &trader_pk)
        .env("DEADEYE_NORMAL_RUNTIME_ADDR", &runtime_hex)
        .arg("trade")
        .arg("execute")
        .arg(&market_hex)
        .arg("--family")
        .arg("normal")
        .arg("--mean")
        .arg("43.0")
        .arg("--variance")
        .arg("81.0")
        .arg("--max-collateral")
        .arg("100.0")
        .output()
        .expect("spawn deadeye trade execute");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("trade execute stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        output.status.success(),
        "trade execute returned non-zero exit"
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert!(
        parsed
            .get("tx_hash")
            .and_then(serde_json::Value::as_str)
            .is_some()
    );

    // ── claim (graceful no-op on un-settled market) ────────────
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .arg("--confirm")
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .env("DEADEYE_PRIVATE_KEY", &trader_pk)
        .arg("claim")
        .arg(&market_hex)
        .arg("--family")
        .arg("normal")
        .output()
        .expect("spawn deadeye claim");
    eprintln!(
        "claim stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Either accepts (silent claim) or rejects with MarketNotSettled — both
    // exit 0 because we render the rejection rather than failing.
    assert!(output.status.success(), "claim returned non-zero exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadeye_watch_emits_json_updates() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1");
        return;
    }
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .expect("bootstrap succeeds");
    let admin_handle = env.account_handle(&env.admin);
    upsert_normal_profile_for_test(admin_handle.clone(), env.factory, env.collateral, 1)
        .await
        .expect("upsert profile");
    let (initial_dist, _) = build_initial_normal_inputs(42.0, 64.0, 1000.0);
    let rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));
    let hints = fetch_normal_hints(&rpc, env.normal_runtime, initial_dist)
        .await
        .expect("fetch hints");
    let market = deploy_normal_market_with_event(
        &admin_handle,
        env.factory,
        1,
        starknet_core::types::Felt::from(2_u64),
        starknet_core::types::Felt::ZERO,
        initial_dist,
        hints,
    )
    .await
    .expect("deploy");
    initialize_market(
        &admin_handle,
        market,
        env.collateral,
        10_000_000_000_000_000_000_000_u128,
    )
    .await
    .expect("init");

    // Spawn the watch process; let it run for ~3 s, kill, count JSON lines.
    let market_hex = format!("{market:#x}");
    let chain_id_hex = format!("{:#x}", env.chain_id);
    let mut child = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .arg("watch")
        .arg(&market_hex)
        .arg("--family")
        .arg("normal")
        .arg("--poll-interval-ms")
        .arg("250")
        .arg("--max-updates")
        .arg("3")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn deadeye watch");

    // Devnet doesn't auto-mine — kick the chain forward by sending a few
    // approvals so the block height advances and the stream actually
    // yields more than the initial state.
    let trader = env.participants.first().expect("a participant");
    let trader_handle = env.account_handle(trader);
    for _ in 0..3 {
        let _ = approve(
            trader_handle.clone(),
            env.collateral,
            market,
            1_000_000_000_000_000_000_000_u128,
        )
        .await;
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    let _ = child.kill();
    let output = child.wait_with_output().expect("collect output");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("watch stdout:\n{stdout}\nstderr:\n{stderr}");
    let json_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .collect();
    assert!(
        json_lines.len() >= 2,
        "expected ≥ 2 JSON update lines; got {}",
        json_lines.len()
    );
    for line in &json_lines {
        let _: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSON `{line}`: {e}"));
    }
}

// Tiny `Sq128` import to keep the type-check live in case future
// iterations need numeric helpers in-test.
const _: fn() = || {
    let _ = Sq128::from_f64;
};

// ─── trade loop (issue #40) ─────────────────────────────────────────────

/// Observe-only, gated, and execute runs of `deadeye trade loop` against
/// devnet: gates evaluate, the journal records every tick (skips included),
/// and `--execute` submits exactly one trade before `--max-trades 1` stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadeye_trade_loop_devnet() {
    if !integration_enabled() {
        eprintln!("skip: set DEADEYE_RUN_INTEGRATION=1");
        return;
    }
    let env = bootstrap_devnet(BootstrapConfig::default())
        .await
        .expect("bootstrap succeeds");
    let admin_handle = env.account_handle(&env.admin);
    upsert_normal_profile_for_test(admin_handle.clone(), env.factory, env.collateral, 1)
        .await
        .expect("upsert normal profile");
    let (initial_dist, _placeholder) = build_initial_normal_inputs(42.0, 64.0, 1000.0);
    let hint_rpc = JsonRpcClient::new(HttpTransport::new(env.url.clone()));
    let initial_hints = fetch_normal_hints(&hint_rpc, env.normal_runtime, initial_dist)
        .await
        .expect("fetch hints");
    let market = deploy_normal_market_with_event(
        &admin_handle,
        env.factory,
        1,
        starknet_core::types::Felt::from(7_u64),
        starknet_core::types::Felt::ZERO,
        initial_dist,
        initial_hints,
    )
    .await
    .expect("deploy normal market");
    initialize_market(
        &admin_handle,
        market,
        env.collateral,
        10_000_000_000_000_000_000_000_u128,
    )
    .await
    .expect("initialize market");
    let trader = env.participants.first().expect("a participant");
    let trader_handle = env.account_handle(trader);
    approve(
        trader_handle.clone(),
        env.collateral,
        market,
        1_000_000_000_000_000_000_000_u128,
    )
    .await
    .expect("approve");

    let market_hex = format!("{market:#x}");
    let chain_id_hex = format!("{:#x}", env.chain_id);
    let trader_addr = format!("{:#x}", trader.address);
    let trader_pk = format!("{:#x}", trader.private_key);
    let runtime_hex = format!("{:#x}", env.normal_runtime);
    let tmp = tempfile::tempdir().expect("tempdir");
    let read_rows = |path: &std::path::Path| -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .expect("journal written")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid journal row"))
            .collect()
    };

    // ── observe-only run: gates pass but nothing is submitted ──
    let journal = tmp.path().join("observe.jsonl");
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .args([
            "trade",
            "loop",
            &market_hex,
            "--family",
            "normal",
            "--belief",
            "43",
            "--belief-sigma",
            "8",
            "--budget",
            "50",
            "--interval",
            "1s",
            "--max-ticks",
            "1",
            "--max-trades",
            "1",
            "--min-ev",
            "0.000001",
            "--max-collateral",
            "100",
            "--journal",
            journal.to_str().unwrap(),
        ])
        .output()
        .expect("spawn deadeye trade loop (observe)");
    eprintln!(
        "trade loop (observe) stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "observe loop returned non-zero");
    let rows = read_rows(&journal);
    assert!(rows.len() >= 2, "tick + stop rows expected: {rows:?}");
    assert_eq!(rows[0]["action"], "skip", "{rows:?}");
    assert_eq!(rows[0]["reason"], "observe_only", "{rows:?}");
    assert!(rows[0].get("tx_hash").is_none(), "{rows:?}");
    assert!(
        rows[0]["gates"].as_array().is_some_and(|g| !g.is_empty()),
        "gates recorded: {rows:?}"
    );
    assert_eq!(rows.last().unwrap()["action"], "stop", "{rows:?}");

    // ── EV gate: an absurd --min-ev skips with ev_below_threshold ──
    let journal_gate = tmp.path().join("gate.jsonl");
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .args([
            "trade",
            "loop",
            &market_hex,
            "--family",
            "normal",
            "--belief",
            "43",
            "--belief-sigma",
            "8",
            "--budget",
            "50",
            "--interval",
            "1s",
            "--max-ticks",
            "1",
            "--max-trades",
            "1",
            "--min-ev",
            "1000000000",
            "--max-collateral",
            "100",
            "--journal",
            journal_gate.to_str().unwrap(),
        ])
        .output()
        .expect("spawn deadeye trade loop (gate)");
    assert!(output.status.success(), "gate loop returned non-zero");
    let rows = read_rows(&journal_gate);
    assert_eq!(rows[0]["action"], "skip", "{rows:?}");
    assert_eq!(rows[0]["reason"], "ev_below_threshold", "{rows:?}");

    // ── execute run: exactly one submitted trade, then max-trades stop ──
    let journal_exec = tmp.path().join("execute.jsonl");
    let output = Command::new(cli_binary())
        .arg("--output")
        .arg("json")
        .arg("--rpc-url")
        .arg(env.url.as_str())
        .env("DEADEYE_CHAIN_ID", &chain_id_hex)
        .env("DEADEYE_ADDRESS", &trader_addr)
        .env("DEADEYE_PRIVATE_KEY", &trader_pk)
        .env("DEADEYE_NORMAL_RUNTIME_ADDR", &runtime_hex)
        .args([
            "trade",
            "loop",
            &market_hex,
            "--family",
            "normal",
            "--belief",
            "43",
            "--belief-sigma",
            "8",
            "--budget",
            "50",
            "--interval",
            "1s",
            "--max-ticks",
            "3",
            "--max-trades",
            "1",
            "--min-ev",
            "0.000001",
            "--max-collateral",
            "100",
            "--execute",
            "--dry-run-first",
            "--journal",
            journal_exec.to_str().unwrap(),
        ])
        .output()
        .expect("spawn deadeye trade loop (execute)");
    eprintln!(
        "trade loop (execute) stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "execute loop returned non-zero");
    let rows = read_rows(&journal_exec);
    let submits: Vec<&serde_json::Value> =
        rows.iter().filter(|r| r["action"] == "submit").collect();
    assert_eq!(submits.len(), 1, "exactly one submitted trade: {rows:?}");
    assert!(
        submits[0]["tx_hash"]
            .as_str()
            .is_some_and(|h| h.starts_with("0x")),
        "tx hash recorded: {rows:?}"
    );
    assert!(
        submits[0]["dry_run"]["accepted"].as_bool().unwrap_or(false),
        "dry-run-first verdict recorded: {rows:?}"
    );
    assert_eq!(
        rows.last().unwrap()["reason"],
        "max_trades_reached",
        "loop stops on max-trades: {rows:?}"
    );
}
