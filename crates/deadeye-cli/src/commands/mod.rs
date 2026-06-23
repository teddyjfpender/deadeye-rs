//! Subcommand dispatch.
//!
//! `dispatch(cli)` is the single async entry point called by
//! [`crate::main`]. It builds the per-invocation [`crate::context::AppContext`]
//! once and forwards to per-subcommand handlers.
//!
//! Driver A (read paths) and Driver B (write paths + stream) coexist
//! here — each subcommand is its own submodule. Stubs for Driver A's
//! read paths return a friendly "not implemented yet" message so the
//! binary still compiles end-to-end with only one driver landed.

use anyhow::Result;

use crate::{
    cli::{Cli, Command},
    context::AppContext,
};

pub(crate) mod account;
pub(crate) mod admin;
pub(crate) mod call_bundle;
pub(crate) mod claim;
pub(crate) mod collateral;
pub(crate) mod config_cmd;
pub(crate) mod doctor;
pub(crate) mod feedback;
pub(crate) mod forecast;
pub(crate) mod lp;
pub(crate) mod markets;
pub(crate) mod onboard;
pub(crate) mod position;
pub(crate) mod render_helpers;
pub(crate) mod risk;
pub(crate) mod runtime_resolver;
pub(crate) mod trade;
pub(crate) mod trade_loop;
pub(crate) mod update;
pub(crate) mod watch;

/// Interactive y/N prompt — returns Ok if accepted, an error on rejection.
///
/// Reads from stdin; callers are expected to gate this on a TTY check
/// before invoking (so scripted runs never block).
pub(crate) fn confirm_or_bail(prompt: &str) -> Result<()> {
    use std::io::{self, BufRead as _, Write as _};
    eprint!("{prompt} [y/N] ");
    io::stderr().flush().ok();
    let mut line = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    handle.read_line(&mut line)?;
    let trimmed = line.trim().to_ascii_lowercase();
    if trimmed == "y" || trimmed == "yes" {
        Ok(())
    } else {
        anyhow::bail!("aborted by user");
    }
}

/// Top-level dispatch: parse → build context → forward to handler.
pub(crate) async fn dispatch(cli: Cli) -> Result<()> {
    let ctx = AppContext::from_cli(&cli)?;
    let confirm = cli.confirm();
    match cli.command {
        // Onboard runs before the context's profile is required — it
        // creates that profile — so it takes the raw args, not `ctx`.
        Command::Onboard(args) => onboard::run(args, &ctx, confirm).await,
        Command::Forecast { action } => forecast::run(action, &ctx, confirm).await,
        Command::Doctor(args) => doctor::run(args, &ctx).await,
        Command::Feedback(args) => feedback::run(args, confirm).await,
        Command::Update(args) => update::run(args, confirm).await,
        Command::Account { action } => account::run(action, &ctx, confirm).await,
        Command::Markets { action } => markets::run(action, &ctx).await,
        Command::Docs => {
            println!("{}", crate::cli::DOCS_AFTER_HELP);
            Ok(())
        },
        Command::Position { action } => position::run(action, &ctx, confirm).await,
        Command::Config { action } => config_cmd::run(action, &ctx).await,
        Command::Trade { action } => trade::run(action, &ctx, confirm).await,
        Command::Lp { action } => lp::run(action, &ctx, confirm).await,
        Command::Claim(args) => claim::run(args, &ctx).await,
        Command::Admin { action } => admin::run(action, &ctx, confirm).await,
        Command::Watch(args) => watch::run(args, &ctx).await,
        Command::Collateral { action } => collateral::run(action, &ctx).await,
    }
}
