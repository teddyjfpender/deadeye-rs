//! `deadeye trade loop` — EV-gated belief arbitrage loop (issue #40).
//!
//! Each tick: fresh market read → belief load (committed forecast snapshot
//! re-read live, or an explicit `--belief`) → the same EV-max optimizer as
//! `trade quote --belief --budget` → every configured gate evaluated → at
//! most ONE (optionally dry-run-first) submission → one JSONL journal row,
//! including no-trade ticks with the gating reason.
//!
//! Safety posture:
//! * observe-only unless `--execute` (everything still evaluated + journaled);
//! * at least one of `--max-trades` / `--max-runtime` / `--session-budget`;
//! * `--min-ev` required unless `--allow-zero-ev-gate`;
//! * never prompts (a loop must not block on stdin) and never retries a failed
//!   submission — errors are journaled and the loop waits for the next tick
//!   (RPC etiquette: one snapshot per tick, quotes computed locally, no tight
//!   retry wrapping).

use std::{
    io::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result};
use deadeye_core::Sq128;
use deadeye_sdk::{DeadeyeClient, bulk::Family};
use deadeye_starknet::{Felt, LognormalMarketReader, NormalMarketReader, normal_amm::MarketStatus};
use serde::Serialize;
use tokio::sync::Notify;

use crate::{
    cli::TradeLoopArgs,
    commands::{
        call_bundle::WriteMode,
        render_helpers::RejectionExplanation,
        runtime_resolver::{
            build_owned_account, build_provider, family_label, parse_felt, resolve_family,
            resolve_runtime_opt,
        },
        trade::{
            COLLATERAL_BUFFER, NormalSolveOutcome, NormalSubmitOptions, SubmitOutcome,
            solve_lognormal_candidate, solve_normal_candidate, submit_lognormal_quote,
            submit_normal_quote,
        },
    },
    context::{AppContext, CliProvider},
    forecast::ledger::{Workspace, now_unix},
    output::OutputMode,
};

const SECONDS_PER_DAY: u64 = 86_400;

// ─── journal row ────────────────────────────────────────────────────────

/// Market state immediately before the tick's decision.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PreState {
    /// Market mean (log-space μ for lognormal).
    pub(crate) mu: f64,
    /// Market sigma.
    pub(crate) sigma: f64,
    /// Live effective k (normal family; lognormal reads don't surface it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effective_k: Option<f64>,
}

/// The belief used this tick and where it came from.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct BeliefInfo {
    /// `forecast` or `explicit`.
    pub(crate) source: &'static str,
    /// Belief mean (log-space μ for lognormal markets).
    pub(crate) mu: f64,
    /// Belief sigma.
    pub(crate) sigma: f64,
    /// Forecast snapshot commit time (unix s) when source == forecast.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) snapshot_created_at: Option<u64>,
}

/// The optimizer's candidate distribution.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CandidateInfo {
    /// Candidate mean.
    pub(crate) mu: f64,
    /// Candidate sigma.
    pub(crate) sigma: f64,
    /// Candidate variance.
    pub(crate) variance: f64,
}

/// One gate's verdict (all configured gates are recorded every tick, not
/// just the first failure — observe-only runs are a true preview).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct GateResult {
    /// Stable gate name (doubles as the skip reason).
    pub(crate) name: &'static str,
    /// Whether the gate passed.
    pub(crate) pass: bool,
    /// Configured threshold, when numeric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) threshold: Option<f64>,
    /// Observed value, when numeric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) actual: Option<f64>,
}

/// Dry-run simulation verdict (when `--dry-run-first`).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DryRunInfo {
    /// Whether the gas-free simulation accepted the multicall.
    pub(crate) accepted: bool,
    /// Simulator note (revert reason / fee estimate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
}

/// Market state re-read after a submitted trade.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PostState {
    /// Market mean after the trade.
    pub(crate) mu: f64,
    /// Market sigma after the trade.
    pub(crate) sigma: f64,
}

/// Loop accounting at the end of the tick.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct SessionInfo {
    /// Trades submitted (accepted submissions) so far.
    pub(crate) trades_submitted: u32,
    /// Cumulative gross collateral supplied (XP).
    pub(crate) session_spend_xp: f64,
    /// Gross collateral supplied during the current UTC day (XP).
    pub(crate) day_spend_xp: f64,
}

/// One JSONL row per tick — including skips, errors and the final stop.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct LoopTickRecord {
    /// Wall-clock timestamp (unix ms).
    pub(crate) ts_unix_ms: u64,
    /// 1-based tick counter (0 for the final stop row).
    pub(crate) tick: u32,
    /// `submit` | `skip` | `dry_run_reject` | `error` | `stop`.
    pub(crate) action: &'static str,
    /// Active profile name.
    pub(crate) profile: String,
    /// Trader address, when resolvable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) trader: Option<String>,
    /// Market address.
    pub(crate) market: String,
    /// Market family.
    pub(crate) family: &'static str,
    /// Pre-decision market state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pre: Option<PreState>,
    /// Belief used this tick.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) belief: Option<BeliefInfo>,
    /// Optimizer candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) candidate: Option<CandidateInfo>,
    /// Optimizer expected value (XP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) expected_value: Option<f64>,
    /// EV per locked XP.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) edge_per_xp: Option<f64>,
    /// `edge_per_xp` in basis points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) edge_bps: Option<f64>,
    /// Off-chain required collateral estimate (XP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) required_collateral: Option<f64>,
    /// Gross collateral actually supplied on submit (XP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) supplied_gross: Option<f64>,
    /// 5% CVaR under the belief (XP, negative = loss; normal family).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cvar_5pct: Option<f64>,
    /// Which constraint sized the stake.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sizing_basis: Option<String>,
    /// Every gate evaluated this tick.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) gates: Vec<GateResult>,
    /// Skip/stop/error reason (e.g. `ev_below_threshold`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    /// Free-form detail accompanying the reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
    /// Dry-run verdict (when `--dry-run-first`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dry_run: Option<DryRunInfo>,
    /// Transaction hash on submit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tx_hash: Option<String>,
    /// Typed rejection when the chain refused the trade.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rejection: Option<RejectionExplanation>,
    /// Market state after a submitted trade.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) post: Option<PostState>,
    /// Loop accounting after this tick.
    pub(crate) session: SessionInfo,
}

/// Wall-clock unix milliseconds.
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

impl LoopTickRecord {
    fn new(ctx: &AppContext, tick: u32, market: Felt, family: Family) -> Self {
        Self {
            ts_unix_ms: now_unix_ms(),
            tick,
            action: "skip",
            profile: ctx.config.profile_name.clone(),
            trader: ctx.config.address.clone(),
            market: format!("{market:#x}"),
            family: family_label(family),
            pre: None,
            belief: None,
            candidate: None,
            expected_value: None,
            edge_per_xp: None,
            edge_bps: None,
            required_collateral: None,
            supplied_gross: None,
            cvar_5pct: None,
            sizing_basis: None,
            gates: Vec::new(),
            reason: None,
            note: None,
            dry_run: None,
            tx_hash: None,
            rejection: None,
            post: None,
            session: SessionInfo::default(),
        }
    }
}

// ─── loop journal ───────────────────────────────────────────────────────

/// Append-only JSONL journal for loop ticks. Same durability mechanics as
/// the SDK trade journal (write + flush + fsync per row) but with the loop's
/// own flat row schema — the canonical `EntryKind::Trade` journal cannot
/// represent skip rows.
struct LoopJournal {
    file: std::fs::File,
    path: PathBuf,
}

impl LoopJournal {
    fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating journal dir {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening loop journal {}", path.display()))?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    fn append(&mut self, record: &LoopTickRecord) -> Result<()> {
        let line = serde_json::to_string(record).context("serializing loop journal row")?;
        writeln!(self.file, "{line}")
            .and_then(|()| self.file.flush())
            .and_then(|()| self.file.sync_data())
            .with_context(|| format!("appending to loop journal {}", self.path.display()))
    }
}

/// Default loop journal path: `~/.local/share/deadeye/loops/<market>.jsonl`.
fn default_loop_journal_path(market: &str) -> Result<PathBuf> {
    let mut dir =
        dirs::data_dir().context("could not locate user data dir; pass --journal explicitly")?;
    dir.push("deadeye");
    dir.push("loops");
    let key = market.trim().to_ascii_lowercase();
    Ok(dir.join(format!("{key}.jsonl")))
}

// ─── configuration ──────────────────────────────────────────────────────

/// Where the per-tick belief comes from.
enum BeliefSource {
    /// Re-load the committed forecast snapshot every tick.
    Forecast(Workspace),
    /// Fixed `--belief` (σ from `--belief-sigma` or live market σ).
    Explicit { mu: f64, sigma: Option<f64> },
}

/// Validate the operator-limit invariants that must hold before the first
/// tick (anything family-dependent is checked after detection).
fn validate_bounds(args: &TradeLoopArgs) -> Result<()> {
    anyhow::ensure!(
        args.max_trades.is_some() || args.max_runtime.is_some() || args.session_budget.is_some(),
        "trade loop needs at least one hard stop bound: --max-trades, --max-runtime \
         or --session-budget"
    );
    anyhow::ensure!(
        args.min_ev.is_some() || args.allow_zero_ev_gate,
        "trade loop refuses to run without --min-ev — pass one, or opt out explicitly \
         with --allow-zero-ev-gate"
    );
    anyhow::ensure!(
        args.from_forecast || args.belief.is_some(),
        "pass a belief source: --from-forecast (committed snapshot) or --belief <μ>"
    );
    anyhow::ensure!(args.budget > 0.0, "--budget must be > 0");
    anyhow::ensure!(args.max_collateral > 0.0, "--max-collateral must be > 0");
    anyhow::ensure!(
        args.interval >= std::time::Duration::from_secs(1),
        "--interval must be at least 1s"
    );
    if let Some(ev) = args.min_ev {
        anyhow::ensure!(ev >= 0.0, "--min-ev must be >= 0");
    }
    if let Some(b) = args.session_budget {
        anyhow::ensure!(b > 0.0, "--session-budget must be > 0");
    }
    if let Some(b) = args.daily_budget {
        anyhow::ensure!(b > 0.0, "--daily-budget must be > 0");
    }
    if (args.kelly.is_some() || args.risk.is_some()) && args.bankroll.is_none() {
        anyhow::bail!("--kelly / --risk need --bankroll (the stake cap is a bankroll fraction)");
    }
    Ok(())
}

// ─── per-tick state + gates ─────────────────────────────────────────────

/// Consecutive `tick_failed` errors before the loop stops — without this a
/// persistent failure (deleted forecast workspace, dead RPC) would spin
/// forever when only --max-trades/--session-budget bounds are set, since
/// those can never fire on an error-only loop.
const MAX_CONSECUTIVE_ERRORS: u32 = 5;

#[derive(Debug, Default)]
struct LoopState {
    tick: u32,
    trades_submitted: u32,
    session_spend_xp: f64,
    day_spend_xp: f64,
    /// `now_unix() / 86_400` of the UTC day `day_spend_xp` accumulates in.
    day_index: u64,
    last_trade_unix: Option<u64>,
    consecutive_errors: u32,
}

impl LoopState {
    /// Roll `day_spend_xp` over at UTC midnight.
    fn roll_day(&mut self, now: u64) {
        let day = now / SECONDS_PER_DAY;
        if day != self.day_index {
            self.day_index = day;
            self.day_spend_xp = 0.0;
        }
    }

    fn session(&self) -> SessionInfo {
        SessionInfo {
            trades_submitted: self.trades_submitted,
            session_spend_xp: self.session_spend_xp,
            day_spend_xp: self.day_spend_xp,
        }
    }
}

/// Numeric inputs the pure gate evaluation runs on.
struct GateInputs {
    expected_value: f64,
    edge_bps: f64,
    /// Off-chain required collateral estimate (XP).
    required: f64,
    /// 5% CVaR (XP, negative = loss); None when not computable (lognormal).
    cvar: Option<f64>,
    now_unix: u64,
    last_trade_unix: Option<u64>,
    session_spend_xp: f64,
    day_spend_xp: f64,
    /// Trader XP balance, when readable. None ⇒ gate not evaluated.
    balance_xp: Option<f64>,
}

/// Evaluate every configured numeric gate. Returns all verdicts plus the
/// first failing gate's name (the skip reason). Pure — unit-tested directly.
fn evaluate_gates(
    args: &TradeLoopArgs,
    inputs: &GateInputs,
) -> (Vec<GateResult>, Option<&'static str>) {
    let mut gates = Vec::new();
    let mut push = |name: &'static str, pass: bool, threshold: Option<f64>, actual: Option<f64>| {
        gates.push(GateResult {
            name,
            pass,
            threshold,
            actual,
        });
    };

    if let Some(min_ev) = args.min_ev {
        push(
            "ev_below_threshold",
            inputs.expected_value >= min_ev,
            Some(min_ev),
            Some(inputs.expected_value),
        );
    }
    if let Some(min_edge) = args.min_edge_bps {
        push(
            "edge_below_threshold",
            inputs.edge_bps >= min_edge,
            Some(min_edge),
            Some(inputs.edge_bps),
        );
    }
    if let Some(max_cvar) = args.max_cvar {
        // CVaR is a (negative) P&L; the cap bounds the loss magnitude.
        let pass = inputs.cvar.is_some_and(|c| c.is_finite() && c >= -max_cvar);
        push("cvar_exceeded", pass, Some(-max_cvar), inputs.cvar);
    }
    {
        // Advisory pre-check; the chain-probe ceiling inside the submit
        // pipeline is authoritative (skip reason collateral_above_max_chain).
        let est_gross = inputs.required * COLLATERAL_BUFFER;
        push(
            "collateral_above_max",
            est_gross <= args.max_collateral,
            Some(args.max_collateral),
            Some(est_gross),
        );
    }
    if let Some(cooldown) = args.cooldown {
        let pass = inputs
            .last_trade_unix
            .is_none_or(|last| inputs.now_unix.saturating_sub(last) >= cooldown.as_secs());
        push(
            "cooldown_active",
            pass,
            Some(cooldown.as_secs_f64()),
            inputs
                .last_trade_unix
                .map(|last| inputs.now_unix.saturating_sub(last) as f64),
        );
    }
    if let Some(session_budget) = args.session_budget {
        let projected = inputs
            .required
            .mul_add(COLLATERAL_BUFFER, inputs.session_spend_xp);
        push(
            "session_budget_exhausted",
            projected <= session_budget,
            Some(session_budget),
            Some(projected),
        );
    }
    if let Some(daily_budget) = args.daily_budget {
        let projected = inputs
            .required
            .mul_add(COLLATERAL_BUFFER, inputs.day_spend_xp);
        push(
            "daily_budget_exhausted",
            projected <= daily_budget,
            Some(daily_budget),
            Some(projected),
        );
    }
    if let Some(balance) = inputs.balance_xp {
        push(
            "insufficient_balance",
            balance >= inputs.required * COLLATERAL_BUFFER,
            Some(inputs.required * COLLATERAL_BUFFER),
            Some(balance),
        );
    }

    let first_failure = gates.iter().find(|g| !g.pass).map(|g| g.name);
    (gates, first_failure)
}

// ─── driver ─────────────────────────────────────────────────────────────

pub(crate) async fn run(ctx: &AppContext, args: TradeLoopArgs, _confirm: bool) -> Result<()> {
    validate_bounds(&args)?;

    let market = parse_felt("market address", &args.market)?;
    let provider = build_provider(ctx)?;
    let client = DeadeyeClient::new(provider);
    let family = resolve_family(ctx, &client, market, args.family).await?;
    match family {
        Family::Normal | Family::Lognormal => {},
        Family::Multinoulli | Family::Bivariate => anyhow::bail!(
            "trade loop supports normal + lognormal markets only (multinoulli / bivariate \
             trade drivers are not wired yet)"
        ),
    }
    anyhow::ensure!(
        args.max_cvar.is_none() || family == Family::Normal,
        "--max-cvar is normal-family only for now (lognormal risk extras are not yet wired)"
    );
    // Normal-family submits always take the offline + chain-probe path: the
    // runtime preflight path supplies the FULL --max-collateral as collateral
    // (validated, but terrible for the loop's budget accounting). The runtime
    // stays available to lognormal as the probe-miss fallback.
    let runtime = if family == Family::Normal {
        None
    } else {
        resolve_runtime_opt(args.runtime.as_deref(), family)?
    };
    // Fail fast on a missing key instead of erroring on the first submit.
    if args.execute {
        let _ = build_owned_account(ctx)?;
    }

    let belief_source = if args.from_forecast {
        BeliefSource::Forecast(Workspace::resolve(&args.market)?)
    } else {
        BeliefSource::Explicit {
            mu: args
                .belief
                .context("--belief or --from-forecast required")?,
            sigma: args.belief_sigma,
        }
    };

    let journal_path = match &args.journal {
        Some(p) => p.clone(),
        None => default_loop_journal_path(&args.market)?,
    };
    let mut journal = LoopJournal::open(&journal_path)?;

    // Cache the per-tick balance-read inputs once (RPC etiquette).
    let trader = ctx.resolved_address_felt().ok();
    let token_info = if trader.is_some() {
        let cfg = match family {
            Family::Normal => NormalMarketReader::new(client.provider(), market)
                .config()
                .await
                .ok(),
            _ => LognormalMarketReader::new(client.provider(), market)
                .config()
                .await
                .ok(),
        };
        cfg.map(|c| (c.collateral_token, c.token_decimals))
    } else {
        None
    };

    if ctx.renderer.mode() != OutputMode::Json {
        ctx.renderer.header(&format!(
            "trade loop — {} market {} ({})",
            family_label(family),
            args.market,
            if args.execute {
                "EXECUTE mode"
            } else {
                "observe-only; pass --execute to trade"
            },
        ));
        ctx.renderer
            .kv("journal", &journal_path.display().to_string());
    }

    // SIGINT → graceful stop after the in-flight tick completes; a second
    // Ctrl-C force-exits (tokio's handler replaces the default disposition,
    // so without this a hung tick would swallow every subsequent signal).
    let stop = Arc::new(Notify::new());
    let stop_for_signal = Arc::<Notify>::clone(&stop);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        stop_for_signal.notify_one();
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("second interrupt — exiting immediately");
        #[expect(
            clippy::exit,
            reason = "second Ctrl-C is an explicit force-quit; a hung RPC call \
                      would otherwise swallow the signal forever"
        )]
        std::process::exit(130);
    });

    let started = Instant::now();
    let mut state = LoopState {
        day_index: now_unix() / SECONDS_PER_DAY,
        ..LoopState::default()
    };
    let max_ticks = args.max_ticks.unwrap_or(u32::MAX);
    let mut ticker = tokio::time::interval(args.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut stop_reason: &'static str = "max_ticks_reached";
    let mut stopped_in_tick = false;
    loop {
        // `biased` so a stored Ctrl-C permit always wins over a ticker that
        // became ready while the previous tick ran — otherwise the loop can
        // execute (and submit in) one extra tick after the operator's stop.
        tokio::select! {
            biased;
            () = stop.notified() => {
                stop_reason = "interrupted";
                break;
            }
            _ = ticker.tick() => {}
        }

        // Stop bounds — checked at tick start so a bound crossed while
        // sleeping ends the loop without another market read.
        if let Some(max_rt) = args.max_runtime
            && started.elapsed() >= max_rt
        {
            stop_reason = "max_runtime_reached";
            break;
        }
        if let Some(max_trades) = args.max_trades
            && state.trades_submitted >= max_trades
        {
            stop_reason = "max_trades_reached";
            break;
        }
        if let Some(session_budget) = args.session_budget
            && state.session_spend_xp >= session_budget
        {
            stop_reason = "session_budget_exhausted";
            break;
        }

        state.tick += 1;
        state.roll_day(now_unix());
        let mut record = LoopTickRecord::new(ctx, state.tick, market, family);
        if let Err(e) = run_tick(
            ctx,
            &client,
            market,
            family,
            runtime,
            &args,
            &belief_source,
            token_info,
            trader,
            &mut state,
            &mut record,
        )
        .await
        {
            // Per-tick chain/RPC failures are journaled and the loop waits
            // for the next tick — never a tight retry, never fatal.
            record.action = "error";
            record.reason = Some("tick_failed".to_owned());
            record.note = Some(format!("{e:#}"));
        }
        record.session = state.session();
        journal.append(&record)?;
        emit(ctx, &record);

        if record.action == "error" {
            state.consecutive_errors = state.consecutive_errors.saturating_add(1);
            if state.consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                stop_reason = "persistent_errors";
                break;
            }
        } else {
            state.consecutive_errors = 0;
        }
        if record.action == "stop" {
            stopped_in_tick = true;
            break;
        }
        if state.tick >= max_ticks {
            stop_reason = "max_ticks_reached";
            break;
        }
        // Re-check trade-count/budget bounds immediately after a submit so
        // the loop ends now rather than sleeping one extra interval.
        if let Some(max_trades) = args.max_trades
            && state.trades_submitted >= max_trades
        {
            stop_reason = "max_trades_reached";
            break;
        }
        if let Some(session_budget) = args.session_budget
            && state.session_spend_xp >= session_budget
        {
            stop_reason = "session_budget_exhausted";
            break;
        }
    }

    if !stopped_in_tick {
        let mut stop_row = LoopTickRecord::new(ctx, state.tick, market, family);
        stop_row.action = "stop";
        stop_row.reason = Some(stop_reason.to_owned());
        stop_row.session = state.session();
        journal.append(&stop_row)?;
        emit(ctx, &stop_row);
    }
    if ctx.renderer.mode() != OutputMode::Json {
        ctx.renderer.header(&format!(
            "loop done — {} tick(s), {} trade(s), {:.4} XP gross supplied",
            state.tick, state.trades_submitted, state.session_spend_xp,
        ));
    }
    Ok(())
}

// ─── one tick ───────────────────────────────────────────────────────────

#[expect(
    clippy::too_many_arguments,
    reason = "tick pipeline threads loop-startup context through"
)]
async fn run_tick(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    family: Family,
    runtime: Option<Felt>,
    args: &TradeLoopArgs,
    belief_source: &BeliefSource,
    token_info: Option<(Felt, u8)>,
    trader: Option<Felt>,
    state: &mut LoopState,
    record: &mut LoopTickRecord,
) -> Result<()> {
    // 1. Market status gate — paused/settled markets skip before any math.
    let status: MarketStatus = match family {
        Family::Normal => NormalMarketReader::new(client.provider(), market)
            .market_status()
            .await
            .map_err(|e| anyhow::anyhow!("get_market_status: {e}"))?,
        _ => LognormalMarketReader::new(client.provider(), market)
            .market_status()
            .await
            .map_err(|e| anyhow::anyhow!("get_market_status: {e}"))?,
    };
    if !status.is_initialised {
        record.reason = Some("market_not_initialised".to_owned());
        return Ok(());
    }
    if status.is_paused {
        record.reason = Some("market_paused".to_owned());
        return Ok(());
    }
    if status.is_settled {
        record.reason = Some("market_settled".to_owned());
        return Ok(());
    }

    // 2. Belief — the forecast snapshot is re-read EVERY tick so a fresh `forecast
    //    snapshot` commit takes effect live.
    let now = now_unix();
    let (belief_mu, belief_sigma_opt, belief_meta) = match belief_source {
        BeliefSource::Forecast(ws) => {
            let snap = ws.load_snapshot()?.with_context(|| {
                format!(
                    "no committed forecast snapshot for {} — commit one with \
                     `deadeye forecast snapshot`",
                    record.market
                )
            })?;
            if let Some(stale_after) = args.stop_if_forecast_stale {
                let age = now.saturating_sub(snap.created_at);
                if age > stale_after.as_secs() {
                    record.belief = Some(BeliefInfo {
                        source: "forecast",
                        mu: snap.mean,
                        sigma: snap.sd,
                        snapshot_created_at: Some(snap.created_at),
                    });
                    record.reason = Some("forecast_stale".to_owned());
                    record.note = Some(format!(
                        "snapshot is {age}s old (limit {}s) — re-commit `deadeye forecast \
                         snapshot` to resume trading",
                        stale_after.as_secs(),
                    ));
                    return Ok(());
                }
            }
            let sigma = args.belief_sigma.unwrap_or(snap.sd);
            (snap.mean, Some(sigma), Some(snap.created_at))
        },
        BeliefSource::Explicit { mu, sigma } => (*mu, *sigma, None),
    };

    // 3. Snapshot + optimize (family-specific), populating the record.
    let solved = match family {
        Family::Normal => {
            solve_tick_normal(
                client,
                market,
                args,
                belief_mu,
                belief_sigma_opt,
                belief_meta,
                record,
            )
            .await?
        },
        _ => {
            solve_tick_lognormal(
                client,
                market,
                args,
                belief_mu,
                belief_sigma_opt,
                belief_meta,
                record,
            )
            .await?
        },
    };
    let Some(tick_quote) = solved else {
        // The solver may have already recorded a more specific reason
        // (e.g. cvar_unreachable).
        if record.reason.is_none() {
            record.reason = Some("optimizer_no_trade".to_owned());
        }
        return Ok(());
    };

    // 4. Drift sanity stop — the candidate sits implausibly far from the belief
    //    (σ-normalized), indicating stale or inconsistent input.
    if let (Some(max_drift), Some(belief)) = (args.max_drift_from_belief, record.belief.clone()) {
        let drift = (tick_quote.candidate_mu - belief.mu).abs() / belief.sigma.max(f64::EPSILON);
        if drift > max_drift {
            record.action = "stop";
            record.reason = Some("drift_exceeded".to_owned());
            record.note = Some(format!(
                "candidate μ {:.6} sits {drift:.2}σ from belief μ {:.6} (limit {max_drift}σ) — \
                 check the belief's units/space before re-running",
                tick_quote.candidate_mu, belief.mu,
            ));
            return Ok(());
        }
    }

    // 5. Balance read (execute mode, cached token info) + numeric gates.
    let balance_xp = match (args.execute, token_info, trader) {
        (true, Some((token, decimals)), Some(who)) => {
            read_balance_xp(client, token, decimals, who).await
        },
        _ => None,
    };
    let inputs = GateInputs {
        expected_value: tick_quote.expected_value,
        edge_bps: tick_quote.edge_bps,
        required: tick_quote.required_collateral,
        cvar: record.cvar_5pct,
        now_unix: now,
        last_trade_unix: state.last_trade_unix,
        session_spend_xp: state.session_spend_xp,
        day_spend_xp: state.day_spend_xp,
        balance_xp,
    };
    let (gates, first_failure) = evaluate_gates(args, &inputs);
    record.gates = gates;
    if let Some(reason) = first_failure {
        record.reason = Some(reason.to_owned());
        return Ok(());
    }

    // 6. Observe-only: every gate passed — journal what would have happened.
    if !args.execute {
        record.reason = Some("observe_only".to_owned());
        record.note = Some("all gates passed — pass --execute to let the loop submit".to_owned());
        return Ok(());
    }

    // 7. Pre-submit freshness guard: the gates were evaluated against the
    //    tick-start snapshot, and a dry-run + chain probe can take many seconds —
    //    re-read the distribution and skip if the market moved materially (the EV
    //    that passed the gates no longer holds).
    if let (Some(fresh), Some(pre)) = (
        read_post_state(client, market, family).await,
        record.pre.clone(),
    ) {
        let moved = (fresh.mu - pre.mu).abs() / pre.sigma.max(f64::EPSILON);
        if moved > 0.1 {
            record.reason = Some("market_moved".to_owned());
            record.note = Some(format!(
                "market μ moved {moved:.3}σ since the tick's snapshot ({:.6} → {:.6}) — \
                 gates were evaluated on stale state; re-solving next tick",
                pre.mu, fresh.mu,
            ));
            return Ok(());
        }
    }

    // 8. Submit (at most once per tick), optionally dry-run-first. The pipeline
    //    ceiling is the AUTHORITATIVE budget enforcement: cap it at the remaining
    //    session/daily allowance so the chain-certified gross (which can exceed the
    //    off-chain estimate the gates used) can never overshoot an operator budget.
    let mut effective_ceiling = args.max_collateral;
    if let Some(session_budget) = args.session_budget {
        effective_ceiling = effective_ceiling.min(session_budget - state.session_spend_xp);
    }
    if let Some(daily_budget) = args.daily_budget {
        effective_ceiling = effective_ceiling.min(daily_budget - state.day_spend_xp);
    }
    if effective_ceiling <= 0.0 {
        record.reason = Some("session_budget_exhausted".to_owned());
        return Ok(());
    }
    let opts = NormalSubmitOptions {
        max_collateral: effective_ceiling,
        mode: WriteMode::Submit,
        runtime,
        x_star_override: None,
        journal: Some(crate::commands::trade::default_journal_path()?),
        interactive: false,
        confirm: true,
        label: family_label(family),
    };
    if args.dry_run_first {
        let dry_opts = NormalSubmitOptions {
            mode: WriteMode::DryRun,
            journal: None,
            ..opts.clone()
        };
        let outcome = submit_tick(ctx, client, market, family, &tick_quote, &dry_opts).await?;
        match outcome {
            SubmitOutcome::DryRun(result) => {
                let accepted = result.accepted;
                record.dry_run = Some(DryRunInfo {
                    accepted,
                    note: result.note.clone(),
                });
                if !accepted {
                    record.action = "dry_run_reject";
                    record.reason = Some("dry_run_reject".to_owned());
                    record.rejection = result.rejection;
                    return Ok(());
                }
            },
            SubmitOutcome::CeilingExceeded { message } => {
                record.reason = Some("collateral_above_max_chain".to_owned());
                record.note = Some(message);
                return Ok(());
            },
            SubmitOutcome::PreflightRejected(result) => {
                record.reason = Some("preflight_rejected".to_owned());
                record.rejection = result.rejection;
                record.note = result.note;
                return Ok(());
            },
            SubmitOutcome::Submitted { .. } => {
                unreachable!("dry_run=true never submits")
            },
            SubmitOutcome::EmitCalldata(_) => {
                unreachable!("trade loop never emits calldata")
            },
        }
    }

    match submit_tick(ctx, client, market, family, &tick_quote, &opts).await? {
        SubmitOutcome::CeilingExceeded { message } => {
            record.reason = Some("collateral_above_max_chain".to_owned());
            record.note = Some(message);
        },
        SubmitOutcome::PreflightRejected(result) => {
            record.reason = Some("preflight_rejected".to_owned());
            record.rejection = result.rejection;
            record.note = result.note;
        },
        SubmitOutcome::DryRun(_) => unreachable!("dry_run=false never simulates only"),
        SubmitOutcome::EmitCalldata(_) => unreachable!("trade loop never emits calldata"),
        SubmitOutcome::Submitted {
            result,
            supplied_gross_xp,
            required_net_xp,
        } => {
            record.required_collateral = Some(required_net_xp);
            record.supplied_gross = Some(supplied_gross_xp);
            record.tx_hash = result.tx_hash.clone();
            if result.accepted {
                record.action = "submit";
                state.trades_submitted = state.trades_submitted.saturating_add(1);
                state.session_spend_xp += supplied_gross_xp;
                state.day_spend_xp += supplied_gross_xp;
                state.last_trade_unix = Some(now_unix());
                // Post-trade market state (one read; skip-ticks never pay it).
                record.post = read_post_state(client, market, family).await;
            } else if result.rejection.is_some() {
                // Typed rejection: the chain (or the pre-send simulation)
                // definitively refused — nothing landed. Journal + wait for
                // the next tick, never retry (issue #40's explicit rule).
                record.action = "error";
                record.reason = Some("submit_rejected".to_owned());
                record.rejection = result.rejection;
                record.note = result.note;
            } else {
                // UNTYPED failure (transport error, lost HTTP response): the
                // transaction may have landed on-chain while the loop saw an
                // error — continuing would un-account real locked collateral
                // and could double-trade past --max-trades. Stop and make
                // the operator reconcile.
                record.action = "stop";
                record.reason = Some("submit_failure_ambiguous".to_owned());
                record.note = Some(format!(
                    "submission failed without a typed rejection — the transaction MAY have \
                     landed on-chain. Check `deadeye position show {}` (and the explorer) \
                     before restarting the loop. Original error: {}",
                    record.market,
                    result.note.as_deref().unwrap_or("(none)"),
                ));
            }
        },
    }
    Ok(())
}

/// What a tick's solve produced — enough for gates + submission.
struct TickQuote {
    candidate_mu: f64,
    candidate_variance: f64,
    expected_value: f64,
    edge_bps: f64,
    required_collateral: f64,
}

/// Normal-family tick solve: ONE state snapshot, then pure optimizer math.
async fn solve_tick_normal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    args: &TradeLoopArgs,
    belief_mu: f64,
    belief_sigma_opt: Option<f64>,
    snapshot_created_at: Option<u64>,
    record: &mut LoopTickRecord,
) -> Result<Option<TickQuote>> {
    let snapshot = client
        .normal_market(market)
        .state_snapshot()
        .await
        .map_err(|e| anyhow::anyhow!("state_snapshot: {e}"))?;
    record.pre = Some(PreState {
        mu: snapshot.mean,
        sigma: snapshot.sigma,
        effective_k: Some(snapshot.effective_k),
    });
    let belief_sigma = belief_sigma_opt.unwrap_or(snapshot.sigma);
    record.belief = Some(BeliefInfo {
        source: if snapshot_created_at.is_some() {
            "forecast"
        } else {
            "explicit"
        },
        mu: belief_mu,
        sigma: belief_sigma,
        snapshot_created_at,
    });

    let kelly_mult = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
    let solve = match solve_normal_candidate(
        &snapshot,
        belief_mu,
        belief_sigma,
        args.budget,
        args.bankroll,
        kelly_mult,
        args.max_cvar,
    )? {
        NormalSolveOutcome::Solved(s) => s,
        NormalSolveOutcome::CvarUnreachable { cvar, .. } => {
            record.cvar_5pct = Some(cvar);
            record.reason = Some("cvar_unreachable".to_owned());
            return Ok(None);
        },
    };
    let q = &solve.quote;
    let required = Sq128::from_raw(q.required_collateral).to_f64();
    if !q.on_chain_will_accept || required <= 0.0 {
        return Ok(None);
    }
    let cand_mu = Sq128::from_raw(q.candidate.mean).to_f64();
    let cand_sigma = Sq128::from_raw(q.candidate.sigma).to_f64();
    let cand_var = Sq128::from_raw(q.candidate.variance).to_f64();
    record.candidate = Some(CandidateInfo {
        mu: cand_mu,
        sigma: cand_sigma,
        variance: cand_var,
    });
    record.expected_value = Some(solve.expected_value);
    record.required_collateral = Some(required);
    record.sizing_basis = Some(solve.sizing_basis.clone());
    let edge = solve.expected_value / required;
    record.edge_per_xp = Some(edge);
    record.edge_bps = Some(edge * 10_000.0);
    record.cvar_5pct = Some(super::risk::cvar_under_belief(
        snapshot.mean,
        snapshot.sigma,
        cand_mu,
        cand_sigma,
        snapshot.effective_k,
        belief_mu,
        belief_sigma,
        0.05,
    ));
    Ok(Some(TickQuote {
        candidate_mu: cand_mu,
        candidate_variance: cand_var,
        expected_value: solve.expected_value,
        edge_bps: edge * 10_000.0,
        required_collateral: required,
    }))
}

/// Lognormal tick solve (belief in LOG space). Reads 3 views per optimize
/// call — the lognormal path has no cached snapshot type yet.
async fn solve_tick_lognormal(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    args: &TradeLoopArgs,
    belief_mu: f64,
    belief_sigma_opt: Option<f64>,
    snapshot_created_at: Option<u64>,
    record: &mut LoopTickRecord,
) -> Result<Option<TickQuote>> {
    let handle = client.lognormal_market(market);
    let current = handle
        .distribution()
        .await
        .map_err(|e| anyhow::anyhow!("reading lognormal distribution: {e}"))?;
    let market_mu = current.mu().to_f64();
    let market_sigma = deadeye_core::Distribution::sigma(&current).to_f64();
    record.pre = Some(PreState {
        mu: market_mu,
        sigma: market_sigma,
        effective_k: None,
    });
    let belief_sigma = belief_sigma_opt.unwrap_or(market_sigma);
    record.belief = Some(BeliefInfo {
        source: if snapshot_created_at.is_some() {
            "forecast"
        } else {
            "explicit"
        },
        mu: belief_mu,
        sigma: belief_sigma,
        snapshot_created_at,
    });

    let kelly_mult = args
        .kelly
        .or_else(|| args.risk.as_deref().and_then(super::risk::preset_fraction));
    let (result, basis, _kelly_cap) = solve_lognormal_candidate(
        &handle,
        belief_mu,
        belief_sigma,
        args.budget,
        args.bankroll,
        kelly_mult,
    )
    .await?;
    if result.collateral_required <= 0.0 {
        return Ok(None);
    }
    record.candidate = Some(CandidateInfo {
        mu: result.optimized_mu,
        sigma: result.optimized_sigma,
        variance: result.optimized_variance,
    });
    record.expected_value = Some(result.expected_value);
    record.required_collateral = Some(result.collateral_required);
    record.sizing_basis = Some(basis);
    record.edge_per_xp = Some(result.roi);
    record.edge_bps = Some(result.roi * 10_000.0);
    Ok(Some(TickQuote {
        candidate_mu: result.optimized_mu,
        candidate_variance: result.optimized_variance,
        expected_value: result.expected_value,
        edge_bps: result.roi * 10_000.0,
        required_collateral: result.collateral_required,
    }))
}

/// Family-dispatched submit using the shared `trade execute` pipelines.
async fn submit_tick(
    ctx: &AppContext,
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    family: Family,
    quote: &TickQuote,
    opts: &NormalSubmitOptions,
) -> Result<SubmitOutcome> {
    match family {
        Family::Normal => {
            submit_normal_quote(
                ctx,
                client,
                market,
                quote.candidate_mu,
                quote.candidate_variance,
                opts,
            )
            .await
        },
        _ => {
            submit_lognormal_quote(
                ctx,
                client,
                market,
                quote.candidate_mu,
                quote.candidate_variance,
                opts,
            )
            .await
        },
    }
}

/// Trader XP balance in human units; `None` on any read failure (the
/// pre-submit simulation remains the safety net).
async fn read_balance_xp(
    client: &DeadeyeClient<CliProvider>,
    token: Felt,
    decimals: u8,
    trader: Felt,
) -> Option<f64> {
    let reader = deadeye_starknet::CollateralTokenReader::new(client.provider(), token);
    let balance = reader.balance_of(trader).await.ok()?;
    if balance.high() > 0 {
        return Some(f64::MAX);
    }
    #[expect(clippy::cast_precision_loss, reason = "balance compare is approximate")]
    Some(balance.low() as f64 / 10f64.powi(i32::from(decimals)))
}

/// One post-trade distribution read (best-effort).
async fn read_post_state(
    client: &DeadeyeClient<CliProvider>,
    market: Felt,
    family: Family,
) -> Option<PostState> {
    use deadeye_core::Distribution as _;
    if family == Family::Normal {
        let d = NormalMarketReader::new(client.provider(), market)
            .distribution()
            .await
            .ok()?;
        Some(PostState {
            mu: d.mean().to_f64(),
            sigma: d.sigma().to_f64(),
        })
    } else {
        let d = LognormalMarketReader::new(client.provider(), market)
            .distribution()
            .await
            .ok()?;
        Some(PostState {
            mu: d.mu().to_f64(),
            sigma: deadeye_core::Distribution::sigma(&d).to_f64(),
        })
    }
}

// ─── rendering ──────────────────────────────────────────────────────────

/// One line per tick: NDJSON in JSON mode, a compact status line otherwise.
fn emit(ctx: &AppContext, record: &LoopTickRecord) {
    if ctx.renderer.mode() == OutputMode::Json {
        if let Ok(line) = serde_json::to_string(record) {
            println!("{line}");
        }
        return;
    }
    let detail = if record.action == "submit" {
        format!(
            "tx {} (EV {:+.4} XP, supplied {:.4} XP gross)",
            record.tx_hash.as_deref().unwrap_or("(none)"),
            record.expected_value.unwrap_or(0.0),
            record.supplied_gross.unwrap_or(0.0),
        )
    } else {
        let reason = record.reason.as_deref().unwrap_or("-");
        match (record.expected_value, record.required_collateral) {
            (Some(ev), Some(req)) => {
                format!("{reason} (EV {ev:+.4} XP, collateral ~{req:.4} XP)")
            },
            _ => reason.to_owned(),
        }
    };
    let line = format!("[tick {}] {} — {}", record.tick, record.action, detail);
    match record.action {
        "submit" => ctx.renderer.success(&line),
        "error" | "dry_run_reject" => ctx.renderer.error(&line),
        "stop" => ctx.renderer.warning(&line),
        _ => println!("{}", ctx.renderer.dim(&line)),
    }
}

// ─── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests panic on construction failure")]
mod tests {
    use std::time::Duration;

    use super::*;

    fn base_args() -> TradeLoopArgs {
        TradeLoopArgs {
            market: "0x1".to_owned(),
            family: None,
            from_forecast: false,
            belief: Some(42.0),
            belief_sigma: None,
            budget: 100.0,
            risk: None,
            bankroll: None,
            kelly: None,
            max_cvar: None,
            max_collateral: 100.0,
            interval: Duration::from_secs(60),
            max_trades: Some(1),
            max_runtime: None,
            session_budget: None,
            daily_budget: None,
            cooldown: None,
            min_ev: Some(1.0),
            min_edge_bps: None,
            max_drift_from_belief: None,
            stop_if_forecast_stale: None,
            dry_run_first: false,
            execute: false,
            allow_zero_ev_gate: false,
            journal: None,
            runtime: None,
            max_ticks: None,
        }
    }

    fn base_inputs() -> GateInputs {
        GateInputs {
            expected_value: 5.0,
            edge_bps: 800.0,
            required: 10.0,
            cvar: Some(-3.0),
            now_unix: 1_000_000,
            last_trade_unix: None,
            session_spend_xp: 0.0,
            day_spend_xp: 0.0,
            balance_xp: Some(1_000.0),
        }
    }

    // ── startup validation ──

    #[test]
    fn validate_requires_a_stop_bound() {
        let mut args = base_args();
        args.max_trades = None;
        let err = validate_bounds(&args).unwrap_err().to_string();
        assert!(err.contains("stop bound"), "{err}");
        args.max_runtime = Some(Duration::from_secs(3600));
        validate_bounds(&args).unwrap();
    }

    #[test]
    fn validate_requires_min_ev_unless_opted_out() {
        let mut args = base_args();
        args.min_ev = None;
        let err = validate_bounds(&args).unwrap_err().to_string();
        assert!(err.contains("--min-ev"), "{err}");
        args.allow_zero_ev_gate = true;
        validate_bounds(&args).unwrap();
    }

    #[test]
    fn validate_requires_a_belief_source() {
        let mut args = base_args();
        args.belief = None;
        let err = validate_bounds(&args).unwrap_err().to_string();
        assert!(err.contains("belief source"), "{err}");
        args.from_forecast = true;
        validate_bounds(&args).unwrap();
    }

    #[test]
    fn validate_kelly_needs_bankroll() {
        let mut args = base_args();
        args.kelly = Some(0.5);
        let err = validate_bounds(&args).unwrap_err().to_string();
        assert!(err.contains("--bankroll"), "{err}");
        args.bankroll = Some(10_000.0);
        validate_bounds(&args).unwrap();
    }

    // ── gate evaluation ──

    #[test]
    fn gates_all_pass_on_clean_inputs() {
        let (gates, failure) = evaluate_gates(&base_args(), &base_inputs());
        assert!(failure.is_none(), "{gates:?}");
        assert!(gates.iter().all(|g| g.pass));
    }

    #[test]
    fn ev_gate_blocks_low_ev() {
        let mut args = base_args();
        args.min_ev = Some(10.0);
        let inputs = GateInputs {
            expected_value: 4.2,
            ..base_inputs()
        };
        let (gates, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("ev_below_threshold"));
        let g = gates
            .iter()
            .find(|g| g.name == "ev_below_threshold")
            .unwrap();
        assert_eq!(g.threshold, Some(10.0));
        assert_eq!(g.actual, Some(4.2));
    }

    #[test]
    fn edge_gate_blocks_thin_edge() {
        let mut args = base_args();
        args.min_edge_bps = Some(500.0);
        let inputs = GateInputs {
            edge_bps: 120.0,
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("edge_below_threshold"));
    }

    #[test]
    fn cvar_gate_blocks_deep_tail_loss() {
        let mut args = base_args();
        args.max_cvar = Some(2.0);
        // CVaR −3 XP is a deeper loss than the 2 XP cap allows.
        let (_, failure) = evaluate_gates(&args, &base_inputs());
        assert_eq!(failure, Some("cvar_exceeded"));
        args.max_cvar = Some(5.0);
        let (_, relaxed_failure) = evaluate_gates(&args, &base_inputs());
        assert!(relaxed_failure.is_none());
    }

    #[test]
    fn cvar_gate_fails_closed_when_unavailable() {
        let mut args = base_args();
        args.max_cvar = Some(5.0);
        let inputs = GateInputs {
            cvar: None,
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("cvar_exceeded"));
    }

    #[test]
    fn collateral_pre_gate_blocks_oversized_requirement() {
        let mut args = base_args();
        args.max_collateral = 10.0;
        let inputs = GateInputs {
            required: 9.9, // ×1.05 buffer pushes past the 10 XP ceiling
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("collateral_above_max"));
    }

    #[test]
    fn cooldown_gate_blocks_until_elapsed() {
        let mut args = base_args();
        args.cooldown = Some(Duration::from_secs(1200));
        let inputs = GateInputs {
            last_trade_unix: Some(1_000_000 - 600),
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("cooldown_active"));
        let elapsed_inputs = GateInputs {
            last_trade_unix: Some(1_000_000 - 1800),
            ..base_inputs()
        };
        let (_, elapsed_failure) = evaluate_gates(&args, &elapsed_inputs);
        assert!(elapsed_failure.is_none());
    }

    #[test]
    fn session_budget_gate_counts_projected_spend() {
        let mut args = base_args();
        args.session_budget = Some(100.0);
        let inputs = GateInputs {
            session_spend_xp: 95.0,
            required: 10.0,
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("session_budget_exhausted"));
    }

    #[test]
    fn daily_budget_gate_counts_projected_spend() {
        let mut args = base_args();
        args.daily_budget = Some(50.0);
        let inputs = GateInputs {
            day_spend_xp: 45.0,
            required: 10.0,
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&args, &inputs);
        assert_eq!(failure, Some("daily_budget_exhausted"));
    }

    #[test]
    fn balance_gate_blocks_underfunded_wallet() {
        let inputs = GateInputs {
            balance_xp: Some(5.0),
            required: 10.0,
            ..base_inputs()
        };
        let (_, failure) = evaluate_gates(&base_args(), &inputs);
        assert_eq!(failure, Some("insufficient_balance"));
    }

    #[test]
    fn balance_gate_skipped_when_unreadable() {
        let inputs = GateInputs {
            balance_xp: None,
            ..base_inputs()
        };
        let (gates, failure) = evaluate_gates(&base_args(), &inputs);
        assert!(failure.is_none());
        assert!(gates.iter().all(|g| g.name != "insufficient_balance"));
    }

    // ── day rollover ──

    #[test]
    fn day_spend_rolls_over_at_utc_midnight() {
        let mut state = LoopState {
            day_index: 100,
            day_spend_xp: 42.0,
            ..LoopState::default()
        };
        // Same day: spend survives.
        state.roll_day(100 * SECONDS_PER_DAY + 5);
        assert!((state.day_spend_xp - 42.0).abs() < f64::EPSILON);
        // Next UTC day: spend resets.
        state.roll_day(101 * SECONDS_PER_DAY + 5);
        assert!(state.day_spend_xp.abs() < f64::EPSILON);
        assert_eq!(state.day_index, 101);
    }

    // ── journal row shape ──

    #[test]
    fn skip_row_serializes_flat_action_and_reason() {
        let record = LoopTickRecord {
            ts_unix_ms: 1,
            tick: 3,
            action: "skip",
            profile: "test".to_owned(),
            trader: None,
            market: "0x1".to_owned(),
            family: "normal",
            pre: None,
            belief: None,
            candidate: None,
            expected_value: Some(4.2),
            edge_per_xp: None,
            edge_bps: None,
            required_collateral: None,
            supplied_gross: None,
            cvar_5pct: None,
            sizing_basis: None,
            gates: vec![GateResult {
                name: "ev_below_threshold",
                pass: false,
                threshold: Some(10.0),
                actual: Some(4.2),
            }],
            reason: Some("ev_below_threshold".to_owned()),
            note: None,
            dry_run: None,
            tx_hash: None,
            rejection: None,
            post: None,
            session: SessionInfo::default(),
        };
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["action"], "skip");
        assert_eq!(json["reason"], "ev_below_threshold");
        assert_eq!(json["expected_value"], 4.2);
        assert_eq!(json["gates"][0]["name"], "ev_below_threshold");
        // Unset optionals must be absent, not null — keeps rows jq-friendly.
        assert!(json.get("tx_hash").is_none());
        assert!(json.get("candidate").is_none());
    }

    #[test]
    fn loop_journal_appends_one_line_per_row() {
        let tmp = std::env::temp_dir().join(format!(
            "deadeye-loop-journal-test-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let mut journal = LoopJournal::open(&tmp).unwrap();
        let mut record = LoopTickRecord {
            ts_unix_ms: 1,
            tick: 1,
            action: "skip",
            profile: "t".to_owned(),
            trader: None,
            market: "0x1".to_owned(),
            family: "normal",
            pre: None,
            belief: None,
            candidate: None,
            expected_value: None,
            edge_per_xp: None,
            edge_bps: None,
            required_collateral: None,
            supplied_gross: None,
            cvar_5pct: None,
            sizing_basis: None,
            gates: Vec::new(),
            reason: Some("observe_only".to_owned()),
            note: None,
            dry_run: None,
            tx_hash: None,
            rejection: None,
            post: None,
            session: SessionInfo::default(),
        };
        journal.append(&record).unwrap();
        record.tick = 2;
        journal.append(&record).unwrap();
        let contents = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(row["reason"], "observe_only");
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
