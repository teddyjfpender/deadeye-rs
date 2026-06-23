//! Top-level `clap` command definitions.

use clap::{Parser, Subcommand, ValueEnum};

use crate::output::OutputMode;

/// Deadeye — market-maker-grade CLI for the Deadeye Rust SDK.
///
/// Read-path commands wrap the SDK's read surface (markets, positions,
/// LP info, indexer metadata). Output mode auto-detects: a TTY renders
/// colored / tabular output; a pipe renders plain `key: value` lines.
/// Use `--output json` for machine-readable output.
///
/// # Examples
///
/// ```text
/// # One-shot read against the default mainnet profile:
/// deadeye markets list --limit 5
///
/// # Inspect one market (family auto-detected):
/// deadeye markets show 0x53e5…0fcf4
///
/// # Pipe-friendly for jq / awk:
/// deadeye markets list --output json | jq '.[] | .address'
/// ```
/// Documentation map printed under `--help` and by `deadeye docs`.
pub(crate) const DOCS_AFTER_HELP: &str = "\
Documentation (https://deadeye.wtf/docs):
  /docs                      what distribution markets are — start here
  /docs/getting-started      10-minute intro: wallet → first trade
  /docs/trading              trading distribution markets properly
  /docs/why-probability      why trade whole distributions, not yes/no
  /docs/for-the-forecasters  the forecasting guide (pairs with `deadeye forecast`)
  /docs/lp-strategy          LP underwriting strategy (pairs with `deadeye lp`)
  /docs/glossary             μ, σ, k, λ, backing, XP — every term defined
  /docs/faq                  FAQ

Automation:
  account derive             HD account fleets from one seed (deadeye/hd/v1)
  trade loop / forecast loop EV-gated arbitrage loop — observe-only until --execute
  --family auto-detected     normal/lognormal are wire-identical; pass --family to force

RPC etiquette: fetch state once (`markets snapshot`), explore candidates with
`trade quote --from-state` (zero RPC), one dry-run, one execute. The CLI backs
off automatically on rate limits — never wrap commands in retry loops. For
multi-tick automation use `trade loop` (the sanctioned self-throttling path: it
honours `--interval` and the backoff between ticks) rather than re-invoking the
one-shot commands in a shell loop.";

#[derive(Debug, Parser)]
#[command(
    name = "deadeye",
    version,
    about = "Market-maker-grade CLI for the Deadeye Rust SDK",
    long_about = None,
    after_help = DOCS_AFTER_HELP,
    propagate_version = true,
    arg_required_else_help = true,
)]
pub(crate) struct Cli {
    /// Override the Starknet JSON-RPC URL.
    ///
    /// Falls back to `DEADEYE_RPC_URL`, then to the active profile's
    /// `rpc_url`, then to a public mainnet endpoint.
    #[arg(long, global = true, value_name = "URL", env = "DEADEYE_RPC_URL")]
    pub(crate) rpc_url: Option<String>,

    /// Override the indexer base URL.
    #[arg(long, global = true, value_name = "URL", env = "DEADEYE_INDEXER_URL")]
    pub(crate) indexer_url: Option<String>,

    /// Trader / account address (hex felt) used as the default for
    /// account-bound subcommands.
    #[arg(long, global = true, value_name = "0x...", env = "DEADEYE_ADDRESS")]
    pub(crate) address: Option<String>,

    /// Use a named profile from `~/.config/deadeye/config.toml`.
    #[arg(long, global = true, value_name = "NAME", env = "DEADEYE_PROFILE")]
    pub(crate) profile: Option<String>,

    /// Output mode. Defaults to `pretty` on a TTY and `plain` in a pipe.
    #[arg(long, global = true, value_name = "MODE")]
    pub(crate) output: Option<OutputModeArg>,

    /// Disable ANSI colors.
    ///
    /// Also honored when the `NO_COLOR` environment variable is set or
    /// stdout is not a terminal.
    #[arg(long, global = true)]
    pub(crate) no_color: bool,

    /// Enable verbose tracing (writes to stderr; does not contaminate JSON).
    #[arg(short, long, global = true)]
    pub(crate) verbose: bool,

    /// Skip interactive confirmation prompts on destructive commands.
    #[arg(long, global = true)]
    pub(crate) confirm: bool,

    #[command(subcommand)]
    pub(crate) command: Command,
}

impl Cli {
    /// Whether destructive commands should skip the y/N prompt.
    pub(crate) const fn confirm(&self) -> bool {
        self.confirm
    }
}

/// `--output` shadow so clap can derive `ValueEnum`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OutputModeArg {
    /// Colored, tabular, human-friendly output (default on a TTY).
    Pretty,
    /// `key: value` lines, no colors, no boxes (default in a pipe).
    Plain,
    /// Pretty-printed JSON dumped to stdout.
    Json,
}

impl OutputModeArg {
    pub(crate) fn into_mode(self) -> OutputMode {
        match self {
            Self::Pretty => OutputMode::Pretty,
            Self::Plain => OutputMode::Plain,
            Self::Json => OutputMode::Json,
        }
    }
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Create or recover a local wallet and deploy its account (start here).
    ///
    /// Interactive wizard: generate or import a BIP-39 phrase, derive the
    /// Starknet account, print the address to fund with STRK, poll the
    /// balance, then deploy the account contract. The key is saved to the
    /// active profile so every later command (and any agent) recovers the
    /// same wallet.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye onboard --network mainnet
    /// deadeye onboard --import            # recover from an existing phrase
    /// ```
    Onboard(OnboardArgs),
    /// Superforecasting workspace: gather evidence, blend base rates, and
    /// curate a calibrated (mean, σ) forecast that feeds `trade quote`.
    #[command(after_help = "Guide: https://deadeye.wtf/docs/for-the-forecasters")]
    Forecast {
        #[command(subcommand)]
        action: ForecastCmd,
    },
    /// Readiness preflight: check account, gas, XP, RPC, indexer, and (with
    /// `--market`) market state — so you learn up front whether you can trade.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye doctor
    /// deadeye doctor --market 0xba9a4b3… --output json
    /// ```
    Doctor(DoctorArgs),
    /// Submit a feature request / bug report as a structured GitHub issue.
    ///
    /// Builds a well-tagged issue (type, component, environment) and posts it
    /// to the repo via the `gh` CLI. Run with `--dry-run` to preview first.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye feedback --title "forecast: add CRPS scoring" \
    ///   --kind feature --component forecast \
    ///   --body "After a market resolves I want `deadeye forecast score` to ..."
    /// ```
    Feedback(FeedbackArgs),
    /// Check for a newer release and update the CLI (and skills) in place.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye update --check   # just report whether a newer version exists
    /// deadeye update           # check, then re-run the installer to update
    /// ```
    Update(UpdateArgs),
    /// Inspect the active account / profile.
    Account {
        #[command(subcommand)]
        action: AccountCmd,
    },
    /// Browse markets from the indexer and on-chain.
    #[command(
        after_help = "What these numbers mean: https://deadeye.wtf/docs/glossary · the model: https://deadeye.wtf/docs"
    )]
    Markets {
        #[command(subcommand)]
        action: MarketsCmd,
    },
    /// Read trader positions and LP shares.
    #[command(after_help = "P&L / breakeven semantics: https://deadeye.wtf/docs/trading")]
    Position {
        #[command(subcommand)]
        action: PositionCmd,
    },
    /// Manage the on-disk configuration file.
    Config {
        #[command(subcommand)]
        action: ConfigCmd,
    },
    /// Print the documentation map (deadeye.wtf/docs) — what distribution
    /// markets are, how to trade them, the glossary, and the guides.
    Docs,

    // ─── Driver B subcommands ─────────────────────────────────────────
    /// Trade preflight / execute / journal (Driver B).
    #[command(
        after_help = "How to trade distribution markets: https://deadeye.wtf/docs/trading · terms: https://deadeye.wtf/docs/glossary"
    )]
    Trade {
        #[command(subcommand)]
        action: TradeCmd,
    },
    /// LP add / remove (Driver B write paths).
    #[command(after_help = "LP underwriting strategy: https://deadeye.wtf/docs/lp-strategy")]
    Lp {
        #[command(subcommand)]
        action: LpCmd,
    },
    /// Claim a (post-settlement) position.
    Claim(ClaimArgs),
    /// Admin (factory-owner) operations: settle, pause, unpause, collect-fees.
    Admin {
        #[command(subcommand)]
        action: AdminCmd,
    },
    /// Block-driven live stream for one market.
    Watch(WatchArgs),
    /// Restricted-collateral-token (XP) operations.
    ///
    /// Wraps the deployed `restricted_collateral_token` contract — the
    /// ERC-20 that the Deadeye AMMs accept as `transfer_from` source. The
    /// `claim-grant` subcommand calls `claim_initial_grant()` on the
    /// token, which mints a fixed amount to a fresh wallet so it can
    /// start trading.
    Collateral {
        #[command(subcommand)]
        action: CollateralCmd,
    },
}

/// Target network for `deadeye onboard`. Mainnet is the only deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum NetworkArg {
    /// Starknet mainnet (real STRK for gas).
    Mainnet,
}

/// `deadeye onboard …`
#[derive(Debug, clap::Args)]
pub(crate) struct OnboardArgs {
    /// Network to onboard against. Sets RPC, indexer, and chain id.
    #[arg(long, value_name = "NETWORK", default_value = "mainnet")]
    pub(crate) network: NetworkArg,
    /// Profile name to write the wallet into (defaults to the network name).
    #[arg(long, value_name = "NAME")]
    pub(crate) profile: Option<String>,
    /// Recover from an existing recovery phrase instead of generating one.
    #[arg(long)]
    pub(crate) import: bool,
    /// HD derivation index under the phrase (`deadeye/hd/v1`, issue #37).
    /// 0 = the default single account; `--import --account-index 3` recovers
    /// the third derived account of an existing seed.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub(crate) account_index: u32,
    /// Account-contract class hash to deploy. Defaults to the bundled
    /// OpenZeppelin class; must be declared on the target network.
    #[arg(long, value_name = "0x...")]
    pub(crate) account_class_hash: Option<String>,
    /// Minimum STRK the address must hold before deploying the account.
    #[arg(long, default_value_t = 0.001)]
    pub(crate) min_strk: f64,
    /// Skip the balance/fund wait and deploy step (wallet is saved only).
    #[arg(long)]
    pub(crate) skip_deploy: bool,
    /// Override the RPC URL (otherwise derived from `--network`).
    #[arg(long, value_name = "URL")]
    pub(crate) rpc_url: Option<String>,
    /// Override the indexer URL (otherwise derived from `--network`).
    #[arg(long, value_name = "URL")]
    pub(crate) indexer_url: Option<String>,
    /// Overwrite an existing saved wallet on this profile. Without it,
    /// onboarding refuses to clobber a key you already have.
    #[arg(long)]
    pub(crate) force: bool,
}

// ─── Driver B argument types ─────────────────────────────────────────

/// Shared "trader override" flag.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct TraderOpt {
    /// Trader / account address; defaults to the active profile's address.
    #[arg(long, value_name = "0x...")]
    pub(crate) trader: Option<String>,
}

/// `deadeye trade …`
#[derive(Debug, Subcommand)]
pub(crate) enum TradeCmd {
    /// Off-chain + chain preflight for a candidate distribution.
    ///
    /// If `--belief` and `--budget` are both supplied, the optimizer
    /// picks the EV-maximizing candidate. Otherwise the caller supplies
    /// `--mean` + `--variance` (+ `--rho` / `--mu2` for bivariate).
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye trade quote 0xMARKET --mean 43.0 --variance 81.0
    /// ```
    Quote(TradeQuoteArgs),
    /// Submit a trade after a fresh preflight.
    Execute(TradeExecuteArgs),
    /// EV-gated belief arbitrage loop: each tick re-reads the market,
    /// re-loads the belief, runs the optimizer, and only trades when every
    /// EV / risk / budget gate passes. Observe-only unless `--execute`.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye forecast snapshot 0xMARKET --mean 4.05 --sd 0.10
    /// deadeye trade loop 0xMARKET --from-forecast --interval 10m \
    ///     --min-ev 10 --max-trades 6 --budget 250 --max-collateral 250 --execute
    /// ```
    #[command(
        after_help = "Observe-only until --execute. The sanctioned multi-tick path: \
it self-throttles on --interval + backoff, so never wrap one-shot commands in a \
shell loop. Automation guide: https://deadeye.wtf/docs/trading"
    )]
    Loop(TradeLoopArgs),
    /// Show / open / replay a trade journal.
    Journal(TradeJournalArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct TradeQuoteArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,
    /// Risk preset mapping to a Kelly multiplier:
    /// conservative=0.25, balanced=0.5, aggressive=1.0.
    #[arg(long, value_name = "PRESET")]
    pub(crate) risk: Option<String>,
    /// Bankroll (XP) for Kelly sizing — pairs with --kelly or --risk.
    #[arg(long, value_name = "XP")]
    pub(crate) bankroll: Option<f64>,
    /// Explicit Kelly multiplier (e.g. 0.5 = half-Kelly). Overrides --risk.
    /// With --bankroll this CAPS the staked collateral at the fractional-
    /// Kelly stake (issue #33) — the candidate is re-optimised at the capped
    /// budget; the belief itself never changes.
    #[arg(long, value_name = "FRAC")]
    pub(crate) kelly: Option<f64>,
    /// Cap the stake so the 5% CVaR loss stays within this many XP — the
    /// budget is walked down until `cvar_5pct >= -max_cvar` (issue #33).
    /// Normal-family offline quotes only.
    #[arg(long, value_name = "XP")]
    pub(crate) max_cvar: Option<f64>,
    /// Quote from a saved state snapshot (zero RPC). Produce one with
    /// `deadeye markets snapshot <ADDRESS> --output json > state.json`.
    /// Normal-family only; the snapshot's embedded `family` field is
    /// enforced (issue #38).
    #[arg(long, value_name = "PATH")]
    pub(crate) from_state: Option<std::path::PathBuf>,
    /// Candidate mean μ.
    #[arg(long)]
    pub(crate) mean: Option<f64>,
    /// Candidate variance σ².
    #[arg(long)]
    pub(crate) variance: Option<f64>,
    /// Bivariate ρ (correlation).
    #[arg(long)]
    pub(crate) rho: Option<f64>,
    /// Bivariate μ₂ (second mean).
    #[arg(long)]
    pub(crate) mu2: Option<f64>,
    /// Optimizer: trader's directional belief about the true mean.
    #[arg(long)]
    pub(crate) belief: Option<f64>,
    /// Optimizer: budget (max collateral the trader will risk).
    #[arg(long)]
    pub(crate) budget: Option<f64>,
    /// Opt into micro-edge search: return a chain-acceptable candidate whose
    /// EV is at least this many XP, even when the default coarse EV-max grid
    /// would snap back to no-trade. Normal-family offline quotes only.
    #[arg(long, value_name = "XP")]
    pub(crate) min_ev: Option<f64>,
    /// User floor on required collateral for micro-edge search. The chain's
    /// own min-trade-collateral still applies even when this is lower.
    #[arg(long, value_name = "XP")]
    pub(crate) min_collateral: Option<f64>,
    /// Optimizer: belief sigma (defaults to current market sigma).
    #[arg(long)]
    pub(crate) belief_sigma: Option<f64>,
    /// Math-runtime contract address. Defaults to env
    /// `DEADEYE_NORMAL_RUNTIME_ADDR` (etc., per family).
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Collateral pad (in STRK) applied to the chain-computed amount.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) pad: f64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TradeExecuteArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,
    /// Candidate mean μ.
    #[arg(long)]
    pub(crate) mean: Option<f64>,
    /// Candidate variance σ².
    #[arg(long)]
    pub(crate) variance: Option<f64>,
    /// Bivariate ρ.
    #[arg(long)]
    pub(crate) rho: Option<f64>,
    /// Bivariate μ₂.
    #[arg(long)]
    pub(crate) mu2: Option<f64>,
    /// Optimizer belief mean — runs the same EV-max optimizer as
    /// `trade quote --belief/--budget` and submits its candidate (log-space
    /// μ for lognormal markets, matching the on-chain distribution).
    #[arg(long)]
    pub(crate) belief: Option<f64>,
    /// Optimizer budget (max collateral the optimizer may risk).
    #[arg(long)]
    pub(crate) budget: Option<f64>,
    /// Optimizer belief sigma (defaults to the current market sigma;
    /// log-space for lognormal markets).
    #[arg(long)]
    pub(crate) belief_sigma: Option<f64>,
    /// Risk preset (conservative=0.25, balanced=0.5, aggressive=1.0) — caps
    /// the optimizer budget at the fractional-Kelly stake (issue #33).
    #[arg(long, value_name = "PRESET")]
    pub(crate) risk: Option<String>,
    /// Bankroll (XP) for Kelly sizing — pairs with --kelly or --risk.
    #[arg(long, value_name = "XP")]
    pub(crate) bankroll: Option<f64>,
    /// Kelly multiplier cap on the stake (e.g. 0.5 = half-Kelly); needs
    /// --bankroll. Same semantics as `trade quote` (issue #33).
    #[arg(long, value_name = "FRAC")]
    pub(crate) kelly: Option<f64>,
    /// Cap the stake so 5% CVaR stays within this many XP (normal family).
    #[arg(long, value_name = "XP")]
    pub(crate) max_cvar: Option<f64>,
    /// Opt into micro-edge search before execution: submit a chain-acceptable
    /// candidate whose EV is at least this many XP. Normal-family offline
    /// optimizer path only.
    #[arg(long, value_name = "XP")]
    pub(crate) min_ev: Option<f64>,
    /// User floor on required collateral for micro-edge execution. The chain's
    /// own min-trade-collateral still applies even when this is lower.
    #[arg(long, value_name = "XP")]
    pub(crate) min_collateral: Option<f64>,
    /// Maximum collateral the caller is willing to supply (STRK).
    #[arg(long)]
    pub(crate) max_collateral: f64,
    /// Math runtime address.
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Journal path — appends a `Trade` entry on success.
    #[arg(long)]
    pub(crate) journal: Option<std::path::PathBuf>,
    /// Simulate the `[approve, trade]` multicall on-chain and print the verdict
    /// **without submitting** (no gas spent, no valid signature required). Use
    /// it to confirm a trade would land before paying a fee.
    #[arg(long)]
    pub(crate) dry_run: bool,
    /// Build, chain-probe, simulate, and print the validated multicall JSON
    /// without signing. Intended for wallet companions / external signers.
    #[arg(long, conflicts_with = "dry_run")]
    pub(crate) emit_calldata: bool,
    /// ADVANCED/DIAGNOSTIC: override the off-chain-computed `x*` (collateral
    /// point) with an explicit value. Used to probe the on-chain verifier's
    /// stationary check. Offline path only.
    #[arg(long, hide = true)]
    pub(crate) x_star: Option<f64>,
}

/// `deadeye trade loop` / `deadeye forecast loop` — the EV-gated belief
/// arbitrage loop (issue #40). One tick = fresh market read → belief load →
/// optimizer → gate evaluation → at most one (optionally dry-run-first)
/// submission → a journal row (including no-trade ticks with the reason).
#[derive(Debug, Clone, clap::Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "clap flag surface — each bool is an independent CLI switch"
)]
pub(crate) struct TradeLoopArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,
    /// Re-load the committed forecast snapshot as the belief on EVERY tick —
    /// re-committing the snapshot mid-loop takes effect live.
    #[arg(long, conflicts_with = "belief")]
    pub(crate) from_forecast: bool,
    /// Explicit belief mean (log-space μ on lognormal markets). Pass this or
    /// --from-forecast.
    #[arg(long)]
    pub(crate) belief: Option<f64>,
    /// Belief sigma (defaults to the forecast snapshot's σ with
    /// --from-forecast, else the live market sigma).
    #[arg(long)]
    pub(crate) belief_sigma: Option<f64>,
    /// Optimizer budget per tick — the max collateral the optimizer may
    /// target for a single trade (XP).
    #[arg(long, value_name = "XP")]
    pub(crate) budget: f64,
    /// Risk preset (conservative=0.25, balanced=0.5, aggressive=1.0) — caps
    /// the stake at the fractional-Kelly amount each tick.
    #[arg(long, value_name = "PRESET")]
    pub(crate) risk: Option<String>,
    /// Bankroll (XP) for Kelly sizing — pairs with --kelly or --risk.
    #[arg(long, value_name = "XP")]
    pub(crate) bankroll: Option<f64>,
    /// Kelly multiplier cap on the stake (e.g. 0.5 = half-Kelly); needs
    /// --bankroll.
    #[arg(long, value_name = "FRAC")]
    pub(crate) kelly: Option<f64>,
    /// Gate: skip the tick unless 5% CVaR loss stays within this many XP
    /// (normal family only).
    #[arg(long, value_name = "XP")]
    pub(crate) max_cvar: Option<f64>,
    /// Hard ceiling (XP) on the gross collateral any single trade supplies.
    /// The chain-certified requirement is checked against it pre-submit.
    #[arg(long, value_name = "XP")]
    pub(crate) max_collateral: f64,
    /// Tick interval (e.g. `30s`, `10m`, `1h`).
    #[arg(long, value_parser = humantime::parse_duration, default_value = "10m")]
    pub(crate) interval: std::time::Duration,
    /// Stop after this many SUBMITTED trades.
    #[arg(long, value_name = "N")]
    pub(crate) max_trades: Option<u32>,
    /// Stop after this much wall-clock runtime (e.g. `6h`).
    #[arg(long, value_parser = humantime::parse_duration)]
    pub(crate) max_runtime: Option<std::time::Duration>,
    /// Stop once this much cumulative gross collateral (XP) has been
    /// supplied across the loop's submitted trades.
    #[arg(long, value_name = "XP")]
    pub(crate) session_budget: Option<f64>,
    /// Per-UTC-day cap on cumulative gross collateral (XP).
    #[arg(long, value_name = "XP")]
    pub(crate) daily_budget: Option<f64>,
    /// Minimum time between submitted trades (e.g. `20m`).
    #[arg(long, value_parser = humantime::parse_duration)]
    pub(crate) cooldown: Option<std::time::Duration>,
    /// Gate: only trade when the optimizer expected value (XP) is at least
    /// this. Required unless --allow-zero-ev-gate.
    #[arg(long, value_name = "XP")]
    pub(crate) min_ev: Option<f64>,
    /// Gate: only trade when EV per locked XP clears this many basis points.
    #[arg(long, value_name = "BPS")]
    pub(crate) min_edge_bps: Option<f64>,
    /// Sanity stop: end the loop when the optimizer candidate's mean sits
    /// more than this many BELIEF sigmas from the belief mean — a sign of
    /// stale or inconsistent input (e.g. wrong units / log-space mix-up).
    #[arg(long, value_name = "SIGMAS")]
    pub(crate) max_drift_from_belief: Option<f64>,
    /// Refuse to trade when the forecast snapshot is older than this
    /// (e.g. `12h`); the tick is journaled as `forecast_stale`.
    #[arg(long, value_parser = humantime::parse_duration)]
    pub(crate) stop_if_forecast_stale: Option<std::time::Duration>,
    /// Simulate the multicall gas-free immediately before EVERY submission;
    /// a failed simulation skips the tick instead of submitting.
    #[arg(long)]
    pub(crate) dry_run_first: bool,
    /// Actually submit trades. WITHOUT this flag the loop is observe-only:
    /// it evaluates every gate and journals what it would have done.
    #[arg(long)]
    pub(crate) execute: bool,
    /// Run without a --min-ev gate (any positive-EV candidate may trade).
    #[arg(long)]
    pub(crate) allow_zero_ev_gate: bool,
    /// Loop journal (JSONL, one row per tick incl. skips). Defaults to
    /// `~/.local/share/deadeye/loops/<market>.jsonl`. Submitted trades are
    /// ALSO appended to the canonical trade journal.
    #[arg(long)]
    pub(crate) journal: Option<std::path::PathBuf>,
    /// Math runtime address (lognormal probe fallback; optional).
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Test gate: stop after N ticks regardless of other bounds.
    #[arg(long, hide = true)]
    pub(crate) max_ticks: Option<u32>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TradeJournalArgs {
    /// Path to a journal file. If absent, falls back to
    /// `~/.local/share/deadeye/journal.jsonl`.
    #[arg(long)]
    pub(crate) path: Option<std::path::PathBuf>,
    /// Show the most recent N entries.
    #[arg(long, default_value_t = 20)]
    pub(crate) tail: usize,
}

/// `deadeye position sell …` — extend Driver A's PositionCmd.
///
/// We keep PositionCmd's read variants intact and add `Sell` here.
#[derive(Debug, clap::Args)]
pub(crate) struct PositionSellArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force family.
    #[arg(long)]
    pub(crate) family: Option<FamilyArg>,
    /// Minimum token-out (slippage floor, u128 base units).
    #[arg(long, default_value_t = 0)]
    pub(crate) min_out: u128,
    /// Math runtime address (normal only — others ignore).
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Journal path.
    #[arg(long)]
    pub(crate) journal: Option<std::path::PathBuf>,
}

/// `deadeye lp …`
#[derive(Debug, Subcommand)]
pub(crate) enum LpCmd {
    /// Add liquidity to a market.
    Add(LpAddArgs),
    /// Remove a fraction of LP shares from a market.
    Remove(LpRemoveArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct LpAddArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force family.
    #[arg(long)]
    pub(crate) family: Option<FamilyArg>,
    /// Amount of LP shares to add (in STRK-equivalent units).
    #[arg(long)]
    pub(crate) amount: f64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct LpRemoveArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force family.
    #[arg(long)]
    pub(crate) family: Option<FamilyArg>,
    /// Fraction of LP shares to remove (0 < f ≤ 1).
    #[arg(long)]
    pub(crate) fraction: f64,
}

/// `deadeye collateral …`
#[derive(Debug, Subcommand)]
pub(crate) enum CollateralCmd {
    /// Mint the one-shot initial grant of XP into the active wallet.
    ///
    /// The XP token's `claim_initial_grant()` mints `initial_grant()`
    /// tokens to the caller iff `has_claimed_initial_grant(caller)` is
    /// still `false`. Re-running on an already-claimed wallet is a clean
    /// no-op — the command short-circuits before submitting.
    ///
    /// # Example
    ///
    /// ```text
    /// # Dry-run (default): show what would happen.
    /// deadeye collateral claim-grant
    ///
    /// # Real submission.
    /// deadeye collateral claim-grant --execute
    ///
    /// # Custom token address (devnet / other).
    /// deadeye collateral claim-grant --token 0x4583… --execute
    /// ```
    ClaimGrant(CollateralClaimGrantArgs),
    /// Show the wallet's XP balance + grant-claim status.
    ///
    /// `show` is an alias for noun-consistency with `account show` etc.
    #[command(visible_alias = "show")]
    Balance(CollateralBalanceArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct CollateralClaimGrantArgs {
    /// Override the collateral-token address. Defaults to the bundled
    /// mainnet XP address (`MAINNET_XP_TOKEN_ADDRESS`). Required on
    /// non-mainnet chains.
    #[arg(long, value_name = "0x...")]
    pub(crate) token: Option<String>,
    /// Submit the transaction. Without this flag, the command performs
    /// the pre-flight reads and prints the plan but never signs.
    #[arg(long)]
    pub(crate) execute: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CollateralBalanceArgs {
    /// Override the collateral-token address. Defaults to the bundled
    /// mainnet XP address.
    #[arg(long, value_name = "0x...")]
    pub(crate) token: Option<String>,
    /// Account to inspect. Defaults to `--address` / `DEADEYE_ADDRESS`.
    #[arg(long, value_name = "0x...")]
    pub(crate) account: Option<String>,
}

/// `deadeye claim …`
#[derive(Debug, clap::Args)]
pub(crate) struct ClaimArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Trader to claim for (defaults to self).
    #[arg(long, value_name = "0x...")]
    pub(crate) trader: Option<String>,
    /// Force family.
    #[arg(long)]
    pub(crate) family: Option<FamilyArg>,
}

/// `deadeye admin …`
#[derive(Debug, Subcommand)]
pub(crate) enum AdminCmd {
    /// Settle a single market.
    Settle(AdminSettleArgs),
    /// Pause a market.
    Pause(AdminPauseArgs),
    /// Unpause a market.
    Unpause(AdminPauseArgs),
    /// Collect protocol fees.
    CollectFees(AdminCollectFeesArgs),
    /// Deploy an instance of a math-runtime class via the legacy UDC.
    ///
    /// Math runtime classes are pre-declared on mainnet, but no instances
    /// exist — consumers (e.g. cpi-arb) need an instance to do
    /// chain-faithful preflight. This command is idempotent: it caches
    /// successful deploys in `~/.config/deadeye/runtimes.toml` keyed by
    /// `(chain_id, family)` and re-uses them on subsequent invocations.
    ///
    /// # Examples
    ///
    /// ```text
    /// # Dry-run on mainnet — projects the deploy address without spending gas.
    /// deadeye admin deploy-math-runtime --family normal
    ///
    /// # Check that previously-cached runtimes are still alive on-chain.
    /// deadeye admin deploy-math-runtime --status
    ///
    /// # Real deploy (requires --confirm + DEADEYE_PRIVATE_KEY).
    /// deadeye admin deploy-math-runtime --family normal --confirm
    /// ```
    DeployMathRuntime(AdminDeployMathRuntimeArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DeployFamilyArg {
    /// Normal (Gaussian) market family.
    Normal,
    /// Lognormal market family.
    Lognormal,
    /// Multinoulli (categorical) market family.
    Multinoulli,
    /// Bivariate normal market family.
    Bivariate,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AdminDeployMathRuntimeArgs {
    /// Market family whose math runtime should be deployed.
    /// Required unless `--status` is set.
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<DeployFamilyArg>,
    /// Optional deterministic salt (hex felt). Defaults to a fresh random
    /// felt; pass the same salt across runs for a content-addressed deploy.
    #[arg(long, value_name = "FELT")]
    pub(crate) salt: Option<String>,
    /// Override the math-runtime class hash. Defaults to the canonical
    /// class hash for `(chain_id, family)` from the bundled deployment
    /// the pinned mainnet constants. Required on
    /// chains other than mainnet.
    #[arg(long, value_name = "0x...")]
    pub(crate) class_hash: Option<String>,
    /// Required for a real on-chain deploy. Without it, the command is a
    /// dry-run: it prints the projected address + class hash and exits 0.
    #[arg(long)]
    pub(crate) confirm: bool,
    /// Query the local cache + verify each entry against the chain via
    /// `getClassHashAt`. Implies `--family` is optional.
    #[arg(long)]
    pub(crate) status: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AdminSettleArgs {
    /// Factory contract address.
    #[arg(long, value_name = "0x...")]
    pub(crate) factory: Option<String>,
    /// Market address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Market family (normal | lognormal | multinoulli | bivariate).
    #[arg(long)]
    pub(crate) family: FamilyArg,
    /// x* value for normal / lognormal markets (f64; settled at this point).
    #[arg(long)]
    pub(crate) x_star: Option<f64>,
    /// Outcome index for multinoulli (u32).
    #[arg(long)]
    pub(crate) outcome: Option<u32>,
    /// Comma-separated `x1,x2` point for bivariate.
    #[arg(long, value_name = "X1,X2")]
    pub(crate) point: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AdminPauseArgs {
    /// Factory contract address.
    #[arg(long, value_name = "0x...")]
    pub(crate) factory: Option<String>,
    /// Market address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AdminCollectFeesArgs {
    /// Factory contract address.
    #[arg(long, value_name = "0x...")]
    pub(crate) factory: Option<String>,
    /// Market address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Recipient address.
    #[arg(long, value_name = "0x...")]
    pub(crate) recipient: String,
}

/// `deadeye watch …`
#[derive(Debug, clap::Args)]
pub(crate) struct WatchArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Force family.
    #[arg(long)]
    pub(crate) family: Option<FamilyArg>,
    /// Polling interval, milliseconds.
    #[arg(long, default_value_t = 1000)]
    pub(crate) poll_interval_ms: u64,
    /// Comma-separated `mean=...,variance=...` candidate to re-quote on
    /// each block; format depends on family.
    #[arg(long, value_name = "SPEC")]
    pub(crate) show_quote_for: Option<String>,
    /// Math runtime address (required for `--show-quote-for`).
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Stop after this many updates (used by tests; default: unlimited).
    #[arg(long, hide = true)]
    pub(crate) max_updates: Option<u32>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum AccountCmd {
    /// Print the resolved profile, address, balance, and chain id.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye account show
    /// ```
    Show,
    /// List every saved wallet profile (name, address, network, deployed),
    /// so an agent can pick which account to trade from with `--profile`.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye account list --output json
    /// ```
    List,
    /// Deploy the active profile's account contract on-chain.
    ///
    /// A freshly-created wallet must be **funded with STRK and deployed**
    /// before it can send any transaction (claim-grant, trade, …). Run this
    /// after funding if onboarding's deploy step was skipped or failed. It is
    /// idempotent — a no-op if the account is already deployed.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye account deploy            # deploy the default wallet
    /// deadeye account deploy --profile alice
    /// ```
    Deploy,
    /// Derive additional accounts from an existing profile's mnemonic —
    /// one seed, many independent on-chain accounts (`deadeye/hd/v1`).
    ///
    /// Each derived account is a real, separate Starknet account (own key,
    /// own address, own collateral and positions) — ideal for isolating
    /// budget per market domain or per strategy — while the single parent
    /// phrase recovers the whole fleet. Idempotent: re-running reports
    /// existing derivations instead of overwriting.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye account derive --count 5                      # main-1 … main-5
    /// deadeye account derive --index 7 --profile sports     # one account, named
    /// deadeye account derive --from-profile main --count 20 --prefix fleet
    /// deadeye account list                                  # see the fleet
    /// ```
    Derive(AccountDeriveArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct AccountDeriveArgs {
    /// Parent profile whose mnemonic to derive from (default: the active
    /// profile).
    #[arg(long, value_name = "NAME")]
    pub(crate) from_profile: Option<String>,
    /// Derive exactly this index (≥ 1; index 0 is the parent itself).
    #[arg(long, value_name = "N", conflicts_with = "count")]
    pub(crate) index: Option<u32>,
    /// Bulk-derive indices 1..=N in one call.
    #[arg(long, value_name = "N")]
    pub(crate) count: Option<u32>,
    /// Profile name for the derived account (single --index only; default
    /// `<parent>-<index>`).
    #[arg(long, value_name = "NAME", conflicts_with = "count")]
    pub(crate) profile: Option<String>,
    /// Name prefix for bulk derivation (default: the parent profile name,
    /// producing `<prefix>-1` … `<prefix>-N`).
    #[arg(long, value_name = "PREFIX", conflicts_with = "index")]
    pub(crate) prefix: Option<String>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum MarketsCmd {
    /// Print a one-shot quote state snapshot (3 view calls). Feed the JSON to
    /// `trade quote --from-state` to explore candidates with zero further RPC.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye markets snapshot 0x123… --output json > state.json
    /// deadeye trade quote 0x123… --from-state state.json --belief 4.18 --budget 100
    /// ```
    Snapshot {
        /// Market contract address.
        #[arg(value_name = "ADDRESS")]
        address: String,
        /// Force a specific family (skip auto-detect). Only `normal` markets
        /// can be snapshot today; the flag exists so detection failures have
        /// an escape hatch.
        #[arg(long, value_name = "FAMILY")]
        family: Option<FamilyArg>,
    },
    /// List markets from the indexer.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye markets list --family normal --limit 10
    /// ```
    List {
        /// Filter by market family.
        #[arg(long, value_name = "FAMILY")]
        family: Option<FamilyArg>,
        /// Max rows to display.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Read a single market's on-chain state.
    ///
    /// Family is auto-detected (indexer metadata → class hash → factory
    /// registration); pass --family to skip detection.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye markets show 0x53e5…0fcf4
    /// ```
    Show {
        /// Market contract address (`0x…`).
        #[arg(value_name = "ADDRESS")]
        address: String,
        /// Force a specific family (skip auto-detect).
        #[arg(long, value_name = "FAMILY")]
        family: Option<FamilyArg>,
    },
    /// Read indexer-side metadata (title, description, category).
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye markets info 0x53e5…0fcf4
    /// ```
    Info {
        /// Market contract address (`0x…`).
        #[arg(value_name = "ADDRESS")]
        address: String,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum PositionCmd {
    /// List a trader's open positions across the configured markets.
    ///
    /// Markets are sourced from the indexer; pass `--trader` to inspect
    /// someone else's book.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye position list --trader 0xabc…
    /// ```
    List {
        /// Trader address; defaults to the active profile's address.
        #[arg(long, value_name = "0x...")]
        trader: Option<String>,
        /// Restrict to a single family.
        #[arg(long, value_name = "FAMILY")]
        family: Option<FamilyArg>,
        /// Max markets to scan.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Read one trader's full position on a specific market.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye position show 0x53e5…0fcf4
    /// ```
    Show {
        /// Market contract address.
        #[arg(value_name = "ADDRESS")]
        market: String,
        /// Trader address; defaults to the active profile's address.
        #[arg(long, value_name = "0x...")]
        trader: Option<String>,
        /// Force a specific family.
        #[arg(long, value_name = "FAMILY")]
        family: Option<FamilyArg>,
    },
    /// Value a trader's multi-leg position: per-leg + total payout at a
    /// settlement outcome, the P&L if it settles there, and (with a belief)
    /// the expected P&L under a forecast. All families.
    ///
    /// The settlement/belief flags depend on the family: scalar `--at` /
    /// `--belief` for normal + lognormal; `--at-x1 --at-x2` /
    /// `--belief-mu1 --belief-mu2 --belief-sigma1 --belief-sigma2
    /// [--belief-rho]` for bivariate; `--outcome` / `--belief-probs` for
    /// multinoulli.
    ///
    /// # Examples
    ///
    /// ```text
    /// deadeye position value 0x53e5… --at 4.2                       # normal/lognormal
    /// deadeye position value 0x53e5… --belief 4.3 --belief-sigma 0.2
    /// deadeye position value 0xBIV   --at-x1 1.0 --at-x2 2.0        # bivariate
    /// deadeye position value 0xMULTI --outcome 3                    # multinoulli
    /// deadeye position value 0xMULTI --belief-probs 0.1,0.2,0.7
    /// ```
    Value(PositionValueArgs),
    /// Close a position via `sell_position` (Driver B write path).
    Sell(PositionSellArgs),
}

/// `deadeye position value …` — value a multi-leg position across families.
#[derive(Debug, clap::Args)]
pub(crate) struct PositionValueArgs {
    /// Market contract address.
    #[arg(value_name = "ADDRESS")]
    pub(crate) market: String,
    /// Trader address; defaults to the active profile's address.
    #[arg(long, value_name = "0x...")]
    pub(crate) trader: Option<String>,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,

    // ── normal / lognormal ──
    /// Scalar settlement x* (normal, lognormal). Defaults to the market mean.
    #[arg(long, value_name = "X")]
    pub(crate) at: Option<f64>,
    /// Forecast mean μ — expected P&L under N(μ, σ) (normal, lognormal).
    #[arg(long, value_name = "MU")]
    pub(crate) belief: Option<f64>,
    /// Forecast σ for `--belief` (defaults to the current market σ).
    #[arg(long, value_name = "SIGMA")]
    pub(crate) belief_sigma: Option<f64>,

    // ── bivariate ──
    /// First-axis settlement x1 (bivariate; pair with `--at-x2`).
    #[arg(long, value_name = "X1")]
    pub(crate) at_x1: Option<f64>,
    /// Second-axis settlement x2 (bivariate; pair with `--at-x1`).
    #[arg(long, value_name = "X2")]
    pub(crate) at_x2: Option<f64>,
    /// Bivariate belief μ₁.
    #[arg(long, value_name = "MU1")]
    pub(crate) belief_mu1: Option<f64>,
    /// Bivariate belief μ₂.
    #[arg(long, value_name = "MU2")]
    pub(crate) belief_mu2: Option<f64>,
    /// Bivariate belief σ₁.
    #[arg(long, value_name = "SIGMA1")]
    pub(crate) belief_sigma1: Option<f64>,
    /// Bivariate belief σ₂.
    #[arg(long, value_name = "SIGMA2")]
    pub(crate) belief_sigma2: Option<f64>,
    /// Bivariate belief correlation ρ (default 0).
    #[arg(long, value_name = "RHO", default_value_t = 0.0)]
    pub(crate) belief_rho: f64,

    // ── multinoulli ──
    /// Categorical settlement outcome index (multinoulli).
    #[arg(long, value_name = "I")]
    pub(crate) outcome: Option<u32>,
    /// Categorical belief probabilities (comma list; length == outcomes).
    #[arg(long, value_name = "P0,P1,…", value_delimiter = ',')]
    pub(crate) belief_probs: Vec<f64>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ConfigCmd {
    /// Create or update `~/.config/deadeye/config.toml`.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye config init --profile mainnet
    /// ```
    Init {
        /// Name of the profile to create / update.
        #[arg(long, default_value = "mainnet")]
        profile: String,
        /// Address to associate with the profile (hex felt).
        #[arg(long, value_name = "0x...")]
        address: Option<String>,
        /// Override the Starknet RPC URL.
        #[arg(long, value_name = "URL")]
        rpc_url: Option<String>,
        /// Override the indexer URL.
        #[arg(long, value_name = "URL")]
        indexer_url: Option<String>,
        /// Make this the default profile.
        #[arg(long)]
        set_default: bool,
    },
    /// Update fields on a profile — only the flags you pass are changed.
    ///
    /// Targets the **active** profile by default (the one `config show`
    /// resolves), so `deadeye config set --rpc-url <url>` just works. Pass
    /// `--profile <name>` to target another (created if it doesn't exist).
    ///
    /// # Examples
    ///
    /// ```text
    /// deadeye config set --rpc-url https://… --indexer-url https://…
    /// deadeye config set --profile bot2 --address 0x… --default
    /// ```
    Set(ConfigSetArgs),
    /// Print the resolved configuration (private key redacted).
    Show,
    /// List configured profiles.
    ProfileList,
    /// Set the default profile.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye config profile-use mainnet
    /// ```
    ProfileUse {
        /// Profile name to mark as default.
        #[arg(value_name = "NAME")]
        name: String,
    },
}

/// `deadeye config set …` — update one or more fields on a profile.
#[derive(Debug, clap::Args)]
pub(crate) struct ConfigSetArgs {
    /// Profile to update. Defaults to the active profile (`--profile`,
    /// `DEADEYE_PROFILE`, or the saved default). Created if it doesn't exist.
    #[arg(long, value_name = "NAME")]
    pub(crate) profile: Option<String>,
    /// Trader / account address (hex felt).
    #[arg(long, value_name = "0x...")]
    pub(crate) address: Option<String>,
    /// Starknet JSON-RPC URL (must speak spec ≥ v0_9 for the pre_confirmed
    /// tag).
    #[arg(long, value_name = "URL")]
    pub(crate) rpc_url: Option<String>,
    /// Indexer base URL.
    #[arg(long, value_name = "URL")]
    pub(crate) indexer_url: Option<String>,
    /// Chain id (hex felt) — e.g. `0x534e5f4d41494e` for SN_MAIN.
    #[arg(long, value_name = "HEX")]
    pub(crate) chain_id: Option<String>,
    /// ERC-20 collateral token address (defaults to canonical STRK).
    #[arg(long, value_name = "0x...")]
    pub(crate) strk_token: Option<String>,
    /// Also make this the default profile.
    #[arg(long)]
    pub(crate) default: bool,
}

// ─── Update (self-update) ────────────────────────────────────────────────

/// `deadeye update …`
#[derive(Debug, clap::Args)]
pub(crate) struct UpdateArgs {
    /// Only check whether a newer release exists; don't install anything.
    #[arg(long)]
    pub(crate) check: bool,
    /// Installer URL to run when updating. Defaults to the branded endpoint.
    #[arg(long, value_name = "URL")]
    pub(crate) url: Option<String>,
}

// ─── Doctor (readiness preflight) ────────────────────────────────────────

/// `deadeye doctor …`
#[derive(Debug, clap::Args)]
pub(crate) struct DoctorArgs {
    /// Also check a specific market is tradeable (active, initialized,
    /// not settled) and its family is on-chain readable.
    #[arg(long, value_name = "0x...")]
    pub(crate) market: Option<String>,
}

// ─── Feedback (GitHub issue submission) ──────────────────────────────────

/// Kind of feedback — drives the title prefix and the standard label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum FeedbackKind {
    /// A feature request / enhancement.
    Feature,
    /// A bug report.
    Bug,
    /// An idea / open-ended suggestion.
    Idea,
}

/// `deadeye feedback …`
#[derive(Debug, clap::Args)]
pub(crate) struct FeedbackArgs {
    /// Short issue title (a `[Feature]` / `[Bug]` prefix is added).
    #[arg(long)]
    pub(crate) title: String,
    /// The body: what you want and why. For a feature, describe the problem
    /// and a proposed solution; for a bug, steps + expected vs actual.
    #[arg(long)]
    pub(crate) body: String,
    /// Kind of feedback.
    #[arg(long, value_name = "KIND", default_value = "feature")]
    pub(crate) kind: FeedbackKind,
    /// Component this concerns (e.g. cli, forecast, wallet, trade, indexer).
    #[arg(long)]
    pub(crate) component: Option<String>,
    /// Extra GitHub label to apply (repeatable). Must already exist on the
    /// repo.
    #[arg(long = "label")]
    pub(crate) labels: Vec<String>,
    /// Target repository (`owner/name`). Defaults to the deadeye-rs repo or
    /// `DEADEYE_FEEDBACK_REPO`.
    #[arg(long, value_name = "OWNER/NAME", env = "DEADEYE_FEEDBACK_REPO")]
    pub(crate) repo: Option<String>,
    /// Print the issue title/labels/body that would be posted, without posting.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

// ─── Forecast (superforecasting workspace) ───────────────────────────────

/// `deadeye forecast …`
#[derive(Debug, Subcommand)]
pub(crate) enum ForecastCmd {
    /// Create a forecast workspace for a market.
    New(ForecastNewArgs),
    /// List markets that have a forecast workspace.
    List,
    /// Show a workspace: question, evidence, base rates, snapshot, next step.
    Show {
        /// Market contract address.
        #[arg(value_name = "MARKET")]
        market: String,
    },
    /// Append a timestamped evidence item.
    Evidence(ForecastEvidenceAddArgs),
    /// Add a reference class (base rate) to the prior.
    BaseRate(ForecastBaseRateAddArgs),
    /// Blend the recorded reference classes into a prior (prints the result).
    BlendBaseRates {
        /// Market contract address.
        #[arg(value_name = "MARKET")]
        market: String,
    },
    /// Commit the curated `(mean, σ)` snapshot for the market.
    Snapshot(ForecastSnapshotArgs),
    /// Quote a trade straight from the committed snapshot — no need to retype
    /// `--mean`/`--variance`. Reads the snapshot `(μ, σ)` and previews the
    /// trade. With `--budget`, the optimizer sizes the EV-max trade from the
    /// snapshot as the belief; otherwise it quotes the snapshot distribution.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye forecast quote 0xMARKET                # quote the snapshot curve
    /// deadeye forecast quote 0xMARKET --budget 100   # EV-max trade under budget
    /// ```
    Quote(ForecastQuoteArgs),
    /// Execute a trade straight from the committed snapshot (the write sibling
    /// of `forecast quote`). Honors `--max-collateral` and the confirm prompt.
    Trade(ForecastTradeArgs),
    /// Run the EV-gated arbitrage loop against the committed snapshot —
    /// `trade loop --from-forecast` with the forecast pre-wired. The snapshot
    /// is re-loaded every tick, so a fresh `forecast snapshot` commit takes
    /// effect live.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye forecast loop 0xMARKET --interval 10m --min-ev 10 \
    ///     --max-trades 6 --budget 250 --max-collateral 250 --execute
    /// ```
    #[command(
        after_help = "Observe-only until --execute. The sanctioned multi-tick path: \
it self-throttles on --interval + backoff, so never wrap one-shot commands in a \
shell loop. Forecasting guide: https://deadeye.wtf/docs/for-the-forecasters"
    )]
    Loop(TradeLoopArgs),
    /// Run a Bayesian / aggregation routine (JSON in, JSON + rationale out).
    Bayes(ForecastBayesArgs),
    /// Score the committed snapshot against the settled market: CRPS,
    /// z-score, signed error. Persists the record for `forecast calibration`.
    ///
    /// # Example
    ///
    /// ```text
    /// deadeye forecast score 0xMARKET
    /// ```
    Score {
        /// Market contract address (the snapshot workspace key).
        #[arg(value_name = "MARKET")]
        market: String,
    },
    /// Personal calibration dashboard across every scored market: bias,
    /// dispersion (are your σ's too tight?), interval coverage, mean CRPS —
    /// plus the bias/σ correction to apply to your next commit.
    Calibration,
}

/// `deadeye forecast quote …` — quote from the committed snapshot.
#[derive(Debug, clap::Args)]
pub(crate) struct ForecastQuoteArgs {
    /// Market contract address (the snapshot workspace key).
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Optimizer budget (max collateral, XP). When set, the snapshot is used
    /// as the belief and the optimizer picks the EV-max candidate; when
    /// omitted, the snapshot distribution is quoted directly.
    #[arg(long)]
    pub(crate) budget: Option<f64>,
    /// Override the belief σ (defaults to the snapshot's σ).
    #[arg(long)]
    pub(crate) belief_sigma: Option<f64>,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,
    /// Optional math-runtime override (not needed — quotes are client-side).
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Collateral pad (XP) applied to the chain-computed amount.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) pad: f64,
}

/// `deadeye forecast trade …` — execute the snapshot distribution.
///
/// Trades the committed snapshot `(μ, σ²)` as the candidate (the write
/// sibling of `forecast quote` with no `--budget`); preview first with
/// `forecast quote`.
#[derive(Debug, clap::Args)]
pub(crate) struct ForecastTradeArgs {
    /// Market contract address (the snapshot workspace key).
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Maximum collateral the caller will supply (XP) — a hard ceiling.
    #[arg(long)]
    pub(crate) max_collateral: f64,
    /// Force a specific family (otherwise auto-detected).
    #[arg(long, value_name = "FAMILY")]
    pub(crate) family: Option<FamilyArg>,
    /// Optional math-runtime override.
    #[arg(long)]
    pub(crate) runtime: Option<String>,
    /// Journal path — appends a `Trade` entry on success.
    #[arg(long)]
    pub(crate) journal: Option<std::path::PathBuf>,
    /// Simulate the trade gas-free and print the verdict without submitting.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// Stance of an evidence item (CLI shadow of `ledger::Stance`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum StanceArg {
    /// Pushes the outcome up / supports.
    Up,
    /// Pushes the outcome down / against.
    Down,
    /// Background context.
    Context,
    /// Mixed signal.
    Mixed,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ForecastNewArgs {
    /// Market contract address (the workspace key).
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Human question / market title. If omitted, pulled from the indexer.
    #[arg(long)]
    pub(crate) title: Option<String>,
    /// What outcome resolves the market.
    #[arg(long)]
    pub(crate) resolution: Option<String>,
    /// Lower bound of the outcome range.
    #[arg(long)]
    pub(crate) lower: Option<f64>,
    /// Upper bound of the outcome range.
    #[arg(long)]
    pub(crate) upper: Option<f64>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ForecastEvidenceAddArgs {
    /// Market contract address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// The headline claim.
    #[arg(long)]
    pub(crate) claim: String,
    /// Which way it points.
    #[arg(long, value_name = "STANCE")]
    pub(crate) stance: StanceArg,
    /// Source label (e.g. `FRED CPIAUCSL`).
    #[arg(long)]
    pub(crate) source: Option<String>,
    /// Source URL.
    #[arg(long)]
    pub(crate) url: Option<String>,
    /// Source reliability `[0, 1]`.
    #[arg(long)]
    pub(crate) reliability: Option<f64>,
    /// Relevance to the question `[0, 1]`.
    #[arg(long)]
    pub(crate) relevance: Option<f64>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ForecastBaseRateAddArgs {
    /// Market contract address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Name / description of the reference class.
    #[arg(long)]
    pub(crate) name: String,
    /// Class base rate (probability or numeric anchor).
    #[arg(long)]
    pub(crate) rate: f64,
    /// Applicability to this question `[0, 1]`.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) applicability: f64,
    /// Within-class uncertainty `>= 0`.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) uncertainty: f64,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ForecastSnapshotArgs {
    /// Market contract address.
    #[arg(value_name = "MARKET")]
    pub(crate) market: String,
    /// Forecast mean (μ).
    #[arg(long)]
    pub(crate) mean: f64,
    /// Forecast standard deviation (σ).
    #[arg(long)]
    pub(crate) sd: f64,
    /// Aggregation / pooling method label.
    #[arg(long, default_value = "manual")]
    pub(crate) method: String,
    /// Prose rationale.
    #[arg(long, default_value = "")]
    pub(crate) rationale: String,
    /// A reason the forecast could move up (repeatable).
    #[arg(long = "reason-up")]
    pub(crate) reason_up: Vec<String>,
    /// A reason the forecast could move down (repeatable).
    #[arg(long = "reason-down")]
    pub(crate) reason_down: Vec<String>,
    /// What would change your mind (repeatable).
    #[arg(long = "change-my-mind")]
    pub(crate) change_my_mind: Vec<String>,
    /// Record the market (μ, σ) this forecast was blended against, read from
    /// a `markets snapshot --output json` file (issue #34 — audit/scoring).
    #[arg(long, value_name = "SNAPSHOT_JSON")]
    pub(crate) from_state: Option<std::path::PathBuf>,
    /// Market μ blended against (overrides --from-state).
    #[arg(long)]
    pub(crate) market_mu: Option<f64>,
    /// Market σ blended against (overrides --from-state).
    #[arg(long)]
    pub(crate) market_sigma: Option<f64>,
}

/// Bayesian / aggregation routines exposed by `forecast bayes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum BayesRoutine {
    /// Aggregate component `(mu, sigma, weight, side)` beliefs → `(mean, sd,
    /// variance, quantiles)`.
    AggregateNormal,
    /// Blend reference classes `[{base_rate, applicability, uncertainty}]` →
    /// prior.
    BlendBaseRates,
    /// Pool probabilities `[{p, weight}]` via log-odds (default) or linear.
    Pool,
    /// Convert qualitative evidence → likelihood ratio.
    EvidenceWeight,
    /// Apply likelihood ratios to a prior: `{prior, lrs:[...]}`.
    LrUpdate,
    /// De-vig a binary market: `{yes, no}`.
    Devig,
    /// Effective independent count of a correlated cluster: `{n, rho}`.
    EffectiveCount,
    /// P(outcome ≤ x) under a normal: `{x, mean, sd}`.
    ProbBelow,
    /// Shrink a forecast toward the market by edge strength:
    /// `{my_mu, my_sigma, market_mu, market_sigma, edge_strength}`.
    ShrinkToMarket,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ForecastBayesArgs {
    /// Routine to run.
    #[arg(value_name = "ROUTINE")]
    pub(crate) routine: BayesRoutine,
    /// JSON input. If omitted (and no named flags follow), read from stdin.
    #[arg(long, value_name = "JSON")]
    pub(crate) input: Option<String>,
    /// Emit only JSON (no human rationale line).
    #[arg(long)]
    pub(crate) json: bool,
    /// `aggregate-normal` only: read the live market (μ, σ) from a
    /// `markets snapshot --output json` file and include it as a weighted
    /// component — zero extra RPC, no hand-copying (issue #34).
    #[arg(long, value_name = "SNAPSHOT_JSON")]
    pub(crate) from_state: Option<std::path::PathBuf>,
    /// Weight for the injected market component (default 1.0 — the market is
    /// included unless --no-market).
    #[arg(long, value_name = "WEIGHT")]
    pub(crate) market_weight: Option<f64>,
    /// Skip the market component even when --from-state is given.
    #[arg(long)]
    pub(crate) no_market: bool,
    /// Named scalar inputs as trailing flags — `--my-mu 4.32 --market-sigma
    /// 0.55` (or `--key=value`) builds the same JSON object `--input` takes;
    /// kebab-case maps to snake_case keys. Scalar routines (shrink-to-market,
    /// devig, prob-below, effective-count, evidence-weight, lr-update with a
    /// single `--lrs` value) work fully from flags; array-shaped routines
    /// (aggregate-normal, blend-base-rates, pool) still need `--input`.
    #[arg(
        trailing_var_arg = true,
        allow_hyphen_values = true,
        value_name = "NAMED_FLAGS"
    )]
    pub(crate) named: Vec<String>,
}

impl StanceArg {
    pub(crate) const fn into_ledger(self) -> crate::forecast::ledger::Stance {
        use crate::forecast::ledger::Stance;
        match self {
            Self::Up => Stance::Up,
            Self::Down => Stance::Down,
            Self::Context => Stance::Context,
            Self::Mixed => Stance::Mixed,
        }
    }
}

/// CLI-facing family enum. Mirrors [`deadeye_sdk::bulk::Family`] without
/// inheriting its derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum FamilyArg {
    Normal,
    Lognormal,
    Multinoulli,
    Bivariate,
}

impl FamilyArg {
    pub(crate) fn as_indexer_slug(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lognormal => "lognormal",
            Self::Multinoulli => "multinoulli",
            Self::Bivariate => "bivariate",
        }
    }

    pub(crate) fn as_sdk(self) -> deadeye_sdk::bulk::Family {
        match self {
            Self::Normal => deadeye_sdk::bulk::Family::Normal,
            Self::Lognormal => deadeye_sdk::bulk::Family::Lognormal,
            Self::Multinoulli => deadeye_sdk::bulk::Family::Multinoulli,
            Self::Bivariate => deadeye_sdk::bulk::Family::Bivariate,
        }
    }
}

impl DeployFamilyArg {
    /// Convert to the [`deadeye_deployer::runtime::Family`] enum.
    pub(crate) const fn as_deployer(self) -> deadeye_deployer::runtime::Family {
        match self {
            Self::Normal => deadeye_deployer::runtime::Family::Normal,
            Self::Lognormal => deadeye_deployer::runtime::Family::Lognormal,
            Self::Multinoulli => deadeye_deployer::runtime::Family::Multinoulli,
            Self::Bivariate => deadeye_deployer::runtime::Family::Bivariate,
        }
    }
}
