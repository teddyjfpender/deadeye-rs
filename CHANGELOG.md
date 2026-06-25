# Changelog

All notable changes to deadeye-rs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing unreleased — the latest tagged release is below._

## [0.1.25] - 2026-06-25

### Fixed

- External signer `skip-estimate` submissions now derive explicit resource-bound
  prices from the current Starknet block, with padding, instead of using stale
  hardcoded L1/L1-data gas price ceilings. This fixes strkd validation failures
  when mainnet gas prices exceed the old defaults (#64).

## [0.1.24] - 2026-06-23

### Fixed

- Refreshed the `bitcoin_hashes` lockfile entry away from a yanked transitive
  version pulled by `bip39`, restoring cargo-deny release hygiene.
- Applied the nightly rustfmt shape expected by CI for the external-signer
  profile derivation path and address-only simulation account docs.

## [0.1.23] - 2026-06-23

### Added

- Address-only external signer profiles for write paths: profiles can set
  `signer = "external"` and configure `[profiles.<name>.external_signer]` with
  the built-in `strkd` adapter. The adapter sends validated account multicalls
  through `wallet_addInvokeTransaction`, includes `Authorization: Bearer ...`,
  `X-Companion-Client`, per-request `chainId`, and explicit resource bounds to
  avoid stale-allowance fee-estimation failures (#43).
- Universal calldata handoff for keyless custody workflows. `--emit-calldata`
  is now available on `trade execute`, `lp add`, `lp remove`, `claim`, and
  `collateral claim-grant`; it builds the exact multicall, simulates it
  keylessly, and emits JSON with `account_address`, `calls`, and `preflight`
  fields (#43).
- External signer documentation in `docs/EXTERNAL_SIGNERS.md`, including strkd
  auth headers, port-lock/endpoint discovery, sign-vs-submit behavior,
  per-request `chainId`, and the approve/estimation gotcha (#43).

### Changed

- `trade execute --emit-calldata` keeps its legacy JSON field names while also
  exposing the design-level `account_address`, `entry_point`, and `preflight`
  fields expected by wallet companions (#43).
- `trade execute --dry-run` and `--emit-calldata` continue to use address-only
  simulation accounts; `trade loop` now benefits from the same external-signer
  submit path when the active profile selects `signer = "external"` (#43).

## [0.1.22] - 2026-06-23

### Added

- `deadeye trade quote --min-ev` and `deadeye trade execute --min-ev` for
  opt-in normal-market micro-edge search. The search looks near the current
  curve, keeps chain/offline acceptability checks intact, honors budget,
  optional `--min-collateral`, `--max-cvar`, and Kelly sizing gates, and reports
  an explicit `no_trade_reason` when no candidate clears the policy (#42).
- Explicit normal candidates can now be evaluated under a supplied belief:
  `trade quote --mean <x> --variance <v> --belief <x> --belief-sigma <s>`
  populates EV, stress/CVaR fields, and sizing hints where applicable (#42).
- `is_account_deployed()` is now available from `deadeye-starknet` and the SDK
  JSON-RPC feature, giving callers a direct helper for first-trade account UX
  instead of surfacing opaque `ContractNotFound` errors (#2).

### Changed

- Normal state snapshots now carry `min_trade_collateral`, and offline
  fixed-candidate quoting rejects below-minimum trades consistently with the
  chain preflight (#42).
- Runtime prerequisite errors now point directly at
  `deadeye admin deploy-math-runtime --family <family> --confirm`, and the
  CLI/superforecaster skills document the runtime, account, gas, XP, RPC, and
  active-market prerequisites up front (#11).

### Fixed

- Strengthened the chain-runtime optimizer parity test with anchored
  sigma-tightening scenarios that must remain chain-acceptable and must use the
  stationary point where it differs from the candidate mean (#1).
- Added CLI coverage for the existing `deadeye collateral show` alias so the
  papercut stays pinned (#11).

## [0.1.21] - 2026-06-23

### Added

- `deadeye trade execute --emit-calldata` — runs the normal quote,
  chain-probe, fresh-wallet bootstrap, and gas-free simulation path, then emits
  the validated account multicall as JSON for an external wallet/signer. The
  payload labels `claim_initial_grant` when bundled, `approve`, and
  `execute_trade`, includes calldata felts, and reports the simulation verdict.
- Address-only simulation accounts for keyless no-submit flows. `trade execute
  --dry-run` and `--emit-calldata` now work with `DEADEYE_ADDRESS` and no
  `DEADEYE_PRIVATE_KEY`.

### Fixed

- `trade execute --dry-run` no longer hard-requires a local private key before
  reaching the skip-validation simulation path (#43).
- Real `trade execute` submissions still fail closed without a private key; the
  keyless path is limited to dry-run/calldata emission.

## [0.1.20] - 2026-06-13

### Added

- `deadeye trade loop` — an EV-gated belief-arbitrage loop. Each tick reads a
  fresh market snapshot, loads the belief (`--from-forecast` re-reads the
  committed forecast snapshot every tick, or explicit `--belief`/`--belief-sigma`),
  runs the same optimizer as `trade quote --belief --budget`, evaluates every
  gate, and submits at most one trade. **Observe-only unless `--execute`**;
  appends one JSONL row per tick (skips included, with the gating reason).
  Gates: `--min-ev`, `--min-edge-bps`, `--max-cvar` (normal family),
  `--max-collateral`, `--cooldown`, `--session-budget`, `--daily-budget`,
  `--max-drift-from-belief`, `--stop-if-forecast-stale`; hard bounds:
  `--max-trades`, `--max-runtime` (at least one required); `--dry-run-first`
  simulates before each submit. Never prompts, never retries a failed submit.
- `deadeye forecast loop` — the sibling that pre-wires the committed forecast
  snapshot as the belief (`trade loop --from-forecast`).
- `LognormalMarketReader::market_status()` — initialised/paused/settled reader,
  matching the normal family.

### Changed

- **⚠ Breaking — state snapshots are now family-stamped.** `markets snapshot`
  gained `--family` and the emitted JSON carries a `family` field;
  `trade quote --from-state` enforces it and **refuses legacy un-stamped
  snapshot files** unless you assert `--family normal`. Re-take old snapshots
  with `deadeye markets snapshot` — pre-fix snapshots of lognormal markets are
  indistinguishable from normal ones and quote garbage.

### Fixed

- **Family auto-detection (#38).** `trade`/`markets`/`position`/`claim`/`watch`/
  `lp`/`doctor` previously probed each family's reader to guess the market
  family — but normal and lognormal AMMs are wire-identical on every shared
  view, so the probe always concluded "normal" and silently ran normal-family
  math on lognormal markets (wrong `x*`, collateral ~13× off, wrong EV, yet
  `on_chain_will_accept: true`). Detection now uses a semantic ladder — indexer
  `marketType` → market class hash vs the bundled deployment manifest → factory
  `market_type_for_market` — and **errors asking for `--family` when
  inconclusive** rather than guessing.

## [0.1.19] - 2026-06-11

### Changed

- `install.sh` self-references the canonical `deadeye.wtf` URL (docs-only).

## [0.1.16 – 0.1.18] - 2026-06-11

### Added

- **HD-derived account fleets** — one mnemonic deterministically derives many
  Deadeye accounts (`deadeye/hd/v1` derivation path), so an operator can run a
  fleet of agent wallets from a single seed.
- `deadeye lp add` / `lp remove` — liquidity provisioning with the collateral
  ERC-20 `approve` **bundled into the multicall** (no separate approve step;
  fixes the zero-allowance revert).
- New `forecast bayes` routine flags and market-curve aggregation helpers.

### Changed

- Trade sizing: `--risk` presets (conservative/balanced/aggressive), fractional
  Kelly (`--bankroll` + `--kelly`), and a CVaR cap (`--max-cvar`).

### Fixed

- **⚠ Correctness — Sq128 σ encoding.** Lognormal candidate σ is now encoded
  from the distribution (Sq128-exact) so the runtime's hint-consistency check
  accepts it; this changed the numerical output of affected lognormal quotes.
- Lognormal quote/execute parity: probe-first execution, an optimizer-driven
  execute path, and accurate (typed) rejection reasons.

## [0.1.15] - 2026-06-11

### Added

- Lognormal **EV-max trade optimizer** (`trade quote --belief --budget` on
  lognormal markets, fully offline) plus documented per-trade movement caps
  (σ-ratio ≤ 4×, |Δμ| ≤ 4σ).
- New mainnet XP collateral token (20 000-XP initial grant).

## [0.1.13] - 2026-06-10

### Added

- Documentation links throughout `--help` footers and a `deadeye docs` command
  that prints the in-CLI documentation map.

## [0.1.12] - 2026-06-10

### Added

- `forecast score` (CRPS / z-score vs the committed snapshot), `forecast
  calibration` dashboard, and a shrink-to-market Bayesian helper (#25/#23/#21).
- `position show` mark-to-market P&L, profit/breakeven intervals, quantiles,
  market-impact, and settlement-lifecycle / claimable flags (#20/#21).
- Trade risk tooling: `--risk` presets, Kelly sizing, downside/CVaR lines,
  calibration-stress EV, and a pre-trade lint (#15/#24).
- Fetch-once RPC state snapshots + rate-limit-aware exponential backoff (#14).
- Agent skills: RPC etiquette, decide/size step, market-moves-as-evidence,
  component decomposition, efficient-market edge gate, settlement lifecycle
  (#16/#17/#21/#22/#23).

## [0.1.5 – 0.1.11] - 2026-06-08 → 06-10

### Added

- `deadeye doctor` readiness preflight, always-on tracing, `collateral show`
  (#6/#7/#11); `deadeye account deploy` (deploy a funded account); client-side
  normal `trade quote` (no math-runtime needed) with the backing σ-floor
  (#4/#5/#8/#15); `forecast` snapshot workspace → `forecast quote`/`forecast
  trade` (#9); multi-leg (trade-lot) position tracking + settlement valuation
  (#16); `config set`.
- v0.13 ABI refresh with lognormal/bivariate/multinoulli multi-leg support.

### Fixed

- Chain-certified `x*` + simulate-first execution and the ERC-20 approve
  bundled into the trade multicall, so trades land out of the box (#13);
  optimizer maximizes EV s.t. collateral ≤ budget (#12); default to the Hetzner
  mainnet indexer (removed Sepolia/Cartridge).

## Earlier — crate, SDK & collateral foundations (≤ deadeye-cli 0.1.4)

### Added — deadeye-starknet v0.1.1

- `CollateralTokenReader` / `CollateralTokenWriter` — typed view + write
  client pair for the deployed `restricted_collateral_token` (XP on
  Deadeye mainnet). Reader exposes `balance_of`,
  `allowance`, `total_supply`, `initial_grant`,
  `has_claimed_initial_grant`, `is_market_registered`,
  `is_market_enabled`; writer pairs it with an `Account` and exposes
  `claim_initial_grant`, `approve`, plus `build_*_call` builders for
  multicall composition. Mirrors the existing `NormalMarketReader` /
  `NormalMarketWriter` shape so callers don't need to learn a second
  pattern.
- `MAINNET_XP_TOKEN_ADDRESS` — `Felt` constant pinned to
  `0x01d77ce77f1d86035c5e27444da7d2fc77de1d384326074f60f973fa0dd80aff`
  (read off `deployment-mainnet.json`).
- `U256Value` — `CairoSerde`-implementing newtype around
  `starknet_core::types::U256` so `core::integer::u256` ABI returns
  decode through the same trait pipeline as every other view call.
- Constructor shorthand: `CollateralTokenReader::mainnet_xp(provider)`
  binds to the mainnet XP token without the operator typing the hex
  felt.
- 5 unit tests pin the mainnet address constant, the `u256` round-trip,
  the `approve(spender, u256)` calldata layout (`spender, low, high`),
  and selector stability / distinctness.

### Added — deadeye-sdk v0.1.5

- Transitively re-exports the new `CollateralTokenReader`,
  `CollateralTokenWriter`, `MAINNET_XP_TOKEN_ADDRESS`, and `U256Value`
  through `deadeye_sdk::starknet::*` (no SDK-side wrapper needed — the
  crate already does `pub use deadeye_starknet as starknet`). Pairs the
  Deadeye AMMs with their underlying collateral surface in one import.
- Bumps the `deadeye-starknet` dep to `0.1.1`.

### Added — deadeye-cli v0.1.4

- `deadeye collateral claim-grant` — calls `claim_initial_grant()` on
  the XP token, minting the fixed grant to the configured wallet.
  Idempotent: reads `has_claimed_initial_grant` up front and skips the
  submit on an already-funded wallet. Dry-run by default; `--execute`
  to submit. Honors `--token` for non-mainnet deploys and falls back
  to `MAINNET_XP_TOKEN_ADDRESS`.
- `deadeye collateral balance` — prints the wallet's XP balance, the
  `initial_grant` amount, and whether the grant has been claimed.
  Read-only.
- Fails loud when the signer's address doesn't match the resolved
  `--address` / `DEADEYE_ADDRESS` (since `claim_initial_grant` mints
  to the caller, a mismatch would burn gas claiming to the wrong
  wallet).

### Fixed — deadeye-sdk v0.1.4

- `NormalMarket::optimize_quote` (chain-runtime variant) now derives
  `x_star` from the audited `normal_collateral` solver and supplies the
  chain-scaled `λ_f · f(x*) − λ_g · g(x*)` collateral — matching the
  FU2 fix to `optimize_quote_offline`. Previously the runtime variant
  passed `x_star = cand_mean`, which silently tripped
  `stationary_valid` on the deployed math runtime (visible only on
  devnet / when `DEADEYE_NORMAL_RUNTIME_ADDR` is set). The two inner
  paths now run byte-identical math up to the chain hand-off: same
  `optimize_normal_trade` call, same `Sq128::from_f64` conversions,
  same `normal_collateral(..., MinimizationPolicy::standard())` solver,
  same λ-scaled collateral formula, same `(cand_mean, 0.0)` no-trade
  fallback. The two **outputs** agree on `x_star` byte-for-byte; the
  returned `required_collateral` differs only by the chain's Sq128
  re-computation (`check.verification.computed_collateral`) versus the
  offline f64 round-trip — bit-equal to within 1 ULP per
  `offline_optimize_quote_parity.rs`. Hints differ by construction
  (chain bytes via `compute_hints_view` vs Sq128 mirror via
  `compute_normal_hints_offline`).

### Added — deadeye-sdk v0.1.3

- `NormalMarket::optimize_quote_with_override(runtime, belief_mean,
  belief_sigma, budget, effective_k_override)` — chain-faithful
  preflight variant that accepts a caller-supplied `effective_k`
  instead of re-reading it from `params.k`. Use this when the caller
  already knows `effective_k` (backtest replay, simulation sweep, bot
  offline mode, unit tests). Non-positive / non-finite values are
  rejected with `CoreError::InvalidInput` before any chain I/O.
- `NormalMarket::optimize_quote_offline_with_override(belief_mean,
  belief_sigma, budget_xp, effective_k_override)` — offline twin of
  the above. Eliminates the `params` + `lp_info` reads (~150ms saved
  per quote on indexer cache miss). All chain-bit-exact behavior
  (Sq128-derived σ, λ-scaled collateral, `sqrt(mul_down(...))` hints)
  is preserved — only the `k`-derivation step is short-circuited.
- Internal `optimize_quote_offline_inner` extracted as a free function
  so the unit-test path can exercise the math without standing up a
  `Provider` mock. 8 new normal-module tests covering override
  validation, determinism, and `k`-responsiveness.
- `BacktestEngine::from_journal(path)` — real implementation
  (previously a stub returning `io::Error::other`). Reads
  newline-delimited `JournalEntry` records from disk, converts each
  to a `MarketEvent`, and seeds `initial_state` from the first
  Normal trade (falling back to N(0, 1)). Per the journal's
  permissive contract, corrupted lines are emitted as
  `tracing::warn` and skipped; pre-submission "skipped (...)" rows
  are filtered out so the replay sees only events that reached the
  chain. The cpi-bot's in-crate `entries_to_events` workaround in
  `analytics::cmd_replay` becomes redundant and can delegate (P2).
  6 new tests cover Trade/Sell/Claim entries, corrupted-line
  recovery, empty file, missing path, and skipped-row filtering.

### Fixed — deadeye-sdk v0.1.3

- `live_effective_k` doc-comment in `normal.rs` had `pool_backing`
  and `initial_backing` swapped (claimed `pool := params.backing`,
  `initial := lp_info.total_backing_deposited`). The function body
  was always correct; only the comment lied. Per `REVIEW_ITEM5`
  (Cairo storage + on-chain math runtime + TS indexer all agree):
  `pool_backing` is the live `lp_info.total_backing_deposited`,
  `initial_backing` is the immutable `params.backing`. Two new
  convention-pin tests assert (`base_k=50, pool=20_000,
  initial=10_000`) → `effective_k = 100`, and the mainnet CPI YoY
  ratio rises above `base_k` — a swapped mapping would silently
  floor at `base_k`.

### Notes — deadeye-sdk v0.1.3 (additive, non-breaking)

- The original `optimize_quote` / `optimize_quote_offline`
  signatures are unchanged. Old callers continue to work without
  modification.
- `BacktestEngine::from_journal` previously returned
  `Err(io::Error::other("not implemented"))`; the new behaviour
  succeeds on well-formed journals and returns `Ok(_)` with an
  empty event list on an empty file. Callers that relied on the
  stub erroring should switch to checking the resulting
  `engine.events.len()` instead.

### Added — deadeye-sdk v0.1.1

- `NormalMarket::optimize_quote_offline(belief_mean, belief_sigma, budget_xp)`
  — chain-bit-exact off-chain EV optimizer for the **no-math-runtime**
  preflight path. Reads live `(distribution, params, lp_info)`, derives
  σ via [`Sq128::sqrt`] (matches `sqrt_verified` 20/20 on devnet), and
  emits hints via the same `sqrt(mul_down(...))` chain the on-chain
  `compute_hints_view` runs. The output `NormalTradeQuote` survives the
  on-chain `INVALID_DISTRIBUTION` / `INVALID_HINTS` checks by construction.
  See [`docs/OFFLINE_PREFLIGHT.md`](docs/OFFLINE_PREFLIGHT.md).
- Integration test `deadeye-e2e/tests/offline_optimize_quote_parity.rs`
  — runs both `optimize_quote` and `optimize_quote_offline` against a
  freshly-bootstrapped devnet market and asserts limb-for-limb agreement
  on the candidate distribution and hints (10/10).

### Changed — deadeye-sdk v0.1.1

- `NormalMarket::optimize_quote` now constructs the candidate via
  `NormalDistribution::from_variance` (instead of `from_sigma`), so the
  candidate σ is **Sq128-derived** instead of f64-derived. Internal
  behaviour change only — the public API is unchanged. This brings the
  chain-preflight path into bit-parity with the new offline path.

### Added



- Initial workspace scaffold:
  - `deadeye-core` — signed Q128.128 fixed-point, distribution traits
    (`Distribution`, `NormalDistribution`, `LognormalDistribution`),
    typed `CoreError`.
  - `deadeye-artifacts` — compile-time-embedded contract ABIs and
    release manifest, with optional `serde_json`-backed typed view.
  - `deadeye-collateral` — `l2_norm`, `lambda`, and a damped
    Newton-Raphson collateral solver with `MinimizationPolicy`.
  - `deadeye-starknet` — `CairoSerde` trait + concrete impls,
    `Provider` abstraction with a `starknet-providers`-backed adapter,
    pre-computed entry-point selectors, `NormalMarketReader` view client.
  - `deadeye-sdk` — `DeadeyeClient`, per-market handles, `PreparedQuote`.
  - `deadeye-indexer` — typed HTTP client for the production indexer
    (`https://178-105-210-177.sslip.io`), with `health()`, `markets()`,
    and per-market detail accessors.
  - `deadeye-testkit` — devnet lifecycle helpers, hosted public RPC
    discovery, integration `Harness`.
  - `deadeye-e2e` — read-only end-to-end tests, opt-in via
    `DEADEYE_RUN_INTEGRATION=1`.
  - `xtask` — workspace task runner.
- Pedantic lint posture: `clippy::all + pedantic + nursery` plus curated
  `clippy::restriction` lints; `unsafe_code = forbid`.
- GitHub Actions CI for fmt, clippy, test, docs, MSRV, and
  `cargo-deny` checks; weekly integration workflow against a hosted
  public mainnet RPC and starknet-devnet-rs.
