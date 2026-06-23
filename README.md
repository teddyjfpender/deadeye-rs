# deadeye-rs

A Rust SDK for the Deadeye prediction-market protocol on Starknet, built for
**market makers**: low latency, small dependency surface, no hidden async.

## Crates

| Crate                | Purpose                                                                         |
| -------------------- | ------------------------------------------------------------------------------- |
| `deadeye-core`       | Signed Q128.128 fixed-point, distribution types, error hierarchy. `no_std`-friendly. |
| `deadeye-artifacts`  | Compile-time-embedded contract ABIs and release manifest.                       |
| `deadeye-collateral` | Off-chain collateral solver (L2 norm, lambda, Newton-Raphson minimiser).        |
| `deadeye-optimizer`  | EV-maximizing trade picker and LP P&L math for normal markets.                  |
| `deadeye-starknet`   | Calldata encoders, entry-point selectors, view-call wrappers over `starknet-rs`.|
| `deadeye-sdk`        | High-level façade: client, quote, per-market handles.                           |
| `deadeye-deployer`   | Typed deployment manifests (+ future declare/deploy helpers).                   |
| `deadeye-cli`        | `deadeye` — market-maker-grade CLI over the SDK (read, quote, trade, LP, loops).|
| `deadeye-indexer`    | Typed HTTP client for the production indexer (`178-105-210-177.sslip.io`).       |
| `deadeye-testkit`    | Integration-test helpers (devnet, hosted public RPC, harness). Unpublished.      |
| `deadeye-e2e`        | End-to-end tests against a live RPC. Unpublished.                               |
| `xtask`              | Workspace task runner (`cargo xtask ci`, `cargo xtask devnet-up`). Unpublished. |

Each layer is independently usable. A latency-critical MM can drive
`deadeye-starknet` directly; the SDK is convenience, not a wall.

## Design goals

1. **Pedantic from day one.** The workspace runs `clippy::all + pedantic + nursery`
   plus a curated set of `clippy::restriction` lints. `unsafe_code` is
   `forbid`. `#[expect]` everywhere — never bare `#[allow]`.
2. **Small dependency surface.** No `reqwest` in the hot path. No `serde` in
   `deadeye-core`. The `starknet-providers` crate is feature-gated so
   custom transports (multi-RPC racers, mocks) cost nothing.
3. **Numerics that match the chain bit-for-bit.** `Sq128Raw` round-trips
   bit-identical with the Cairo `SQ128x128Raw` struct, verified by proptest.
4. **`no_std`-capable core.** `deadeye-core` and `deadeye-artifacts` can
   compile without `std` so the same primitives feed bots, indexers, and
   on-chain verifier tooling.

## Quick start

```rust
use deadeye_sdk::{
    DeadeyeClient,
    collateral::MinimizationPolicy,
    core::{Distribution, NormalDistribution, Sq128},
    starknet::JsonRpcProvider,
};
use starknet_providers::{JsonRpcClient, jsonrpc::HttpTransport};
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rpc = JsonRpcClient::new(HttpTransport::new(
        Url::parse("https://api.zan.top/public/starknet-mainnet/rpc/v0_10")?,
    ));
    let client = DeadeyeClient::new(JsonRpcProvider::new(rpc));

    let market = client.normal_market("0xMARKET_ADDRESS".parse()?);

    let current = market.distribution().await?;
    println!("current mean = {}", current.mean().to_f64());

    let candidate = NormalDistribution::from_variance(
        Sq128::from_f64(105.0)?,
        Sq128::from_f64(4.0)?,
    )?;

    let quote = market
        .prepare_quote(candidate, MinimizationPolicy::standard())
        .await?;
    println!("collateral = {}", quote.collateral);

    Ok(())
}
```

## CLI quick start

The `deadeye-cli` crate ships the `deadeye` binary — the same SDK surface as a
market-maker-grade command line. Output auto-detects: a TTY renders tabular
output, a pipe renders `key: value` lines, `--output json` is machine-readable.

```bash
# Wallet onboarding (one-time): import or derive a signer, claim the XP grant.
deadeye account import          # or: deadeye account derive (HD fleet, deadeye/hd/v1)
deadeye collateral claim-grant --execute
deadeye doctor                  # readiness preflight (wallet, RPC, indexer)

# Read path — family is auto-detected; nothing here costs collateral:
deadeye markets list --limit 5
deadeye markets show 0xMARKET
deadeye markets snapshot 0xMARKET --output json > state.json   # fetch state once
deadeye trade quote 0xMARKET --from-state state.json --belief 4.18 --budget 100

# Automation — the EV-gated loop. Observe-only until you pass --execute:
deadeye forecast snapshot 0xMARKET --mean 4.05 --sd 0.10
deadeye trade loop 0xMARKET --from-forecast --interval 10m \
    --min-ev 10 --max-trades 6 --budget 250 --max-collateral 250
# add --execute to actually submit; `forecast loop` is the same with the
# committed forecast pre-wired as the belief.

# External signer handoff — no private key needed in the CLI:
deadeye trade execute 0xMARKET --family normal --mean 4.18 --variance 0.04 \
    --max-collateral 100 --emit-calldata --output json
```

**Family auto-detection.** Normal and lognormal AMMs are wire-identical on
every shared view call, so the CLI never probes a reader to guess the family —
it resolves it semantically (indexer `marketType` → class hash → factory
registration) and errors asking for `--family` when inconclusive. Pass
`--family normal|lognormal` to force it.

**Offline by default.** On mainnet the normal AMM uses library dispatch with no
separate math-runtime contract, so quotes take the offline preflight path
(σ + hints bit-exact with the chain; see below). `markets snapshot` +
`trade quote --from-state` then explore unlimited candidates at zero RPC cost.

## Normal-market: chain vs. offline preflight

The `NormalMarket` handle ships **two** preflight entry points, chosen by
whether a math-runtime contract instance is deployed on your target
network:

| Method | When to use | Chain round-trips | Guarantees |
| --- | --- | --- | --- |
| `optimize_quote(runtime, μ_b, σ_b, budget)` | A math-runtime instance is deployed (devnet or self-hosted). | 4 view calls (`distribution`, `params`, `compute_hints_view × 2`, `check_trade_view`) | Chain-validated: `on_chain_will_accept` reflects `check_trade_view`'s verdict. |
| `optimize_quote_offline(μ_b, σ_b, budget)` | **Mainnet today** — the normal AMM uses library dispatch (class hash) with no separate runtime contract. | 3 view calls (`distribution`, `params`, `lp_info`) — no math-runtime hops. | σ + hints are **bit-exact** with what the chain would derive (`Sq128::sqrt` matches `sqrt_verified` 20/20 on devnet; see [`docs/SQ128_SQRT.md`](docs/SQ128_SQRT.md)). |

The offline path eliminates `INVALID_DISTRIBUTION` and `INVALID_HINTS`
rejections by construction. The chain still re-verifies the trade on
submit (balance, nonce, policy envelope) — but the σ/hint precision
footgun is gone.

```rust
let market = client.normal_market(market_addr);
let quote = market
    .optimize_quote_offline(belief_mean, belief_sigma, budget_xp)
    .await?;
if quote.on_chain_will_accept {
    // hand to a signed handle for execute_quote()
}
```

Parity test: `deadeye-e2e/tests/offline_optimize_quote_parity.rs`
(gated on `DEADEYE_RUN_INTEGRATION=1`) — runs both paths against a
deployed runtime and asserts limb-for-limb agreement on
`(μ_g, σ_g, σ_g²)` and `(l2_norm_denom, backing_denom)`.

## Development

```bash
# Run the full local CI pipeline (fmt + clippy + tests)
cargo xtask ci

# Check workspace MSRV
cargo check --workspace --all-features

# Run unit tests
cargo test --workspace --all-features --lib --bins

# Run integration tests against a local devnet
DEADEYE_RUN_INTEGRATION=1 cargo test -p deadeye-e2e -- --nocapture

# Run integration tests against a hosted public RPC (mainnet)
DEADEYE_RUN_INTEGRATION=1 DEADEYE_TEST_TARGET=hosted \
  cargo test -p deadeye-e2e -- --nocapture

# Smoke-test the live mainnet indexer (178-105-210-177.sslip.io)
DEADEYE_RUN_INTEGRATION=1 cargo test -p deadeye-e2e --test indexer_smoke -- --nocapture
```

## MSRV

Rust 1.92 (the workspace uses `resolver = "3"` and `edition = "2024"`).

## License

Dual-licensed under MIT or Apache-2.0.
