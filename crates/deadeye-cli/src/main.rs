//! `deadeye` — market-maker-grade CLI for the Deadeye Rust SDK.
//!
//! This binary wraps the SDK's read paths (and Driver B will layer the
//! write paths on top). It is designed to feel polished at a TTY and
//! script-friendly in a pipe — every command supports
//! `--output {pretty,plain,json}` and auto-detects when stdout is not a
//! terminal.

#![doc(html_no_source)]
// A CLI binary legitimately writes to stdout/stderr, and `pub(crate)`
// inside `bin/`-only modules is idiomatic Rust — relax the workspace
// pedantic gates that fight that shape.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::redundant_pub_crate,
    clippy::trivially_copy_pass_by_ref,
    clippy::needless_pass_by_value,
    clippy::ref_option,
    clippy::wildcard_enum_match_arm,
    clippy::string_slice,
    clippy::big_endian_bytes,
    clippy::map_unwrap_or,
    clippy::pathbuf_init_then_push,
    clippy::default_trait_access,
    clippy::assigning_clones,
    clippy::format_push_string,
    clippy::unnecessary_wraps,
    clippy::struct_field_names,
    clippy::doc_markdown,
    clippy::significant_drop_tightening,
    clippy::redundant_clone,
    clippy::unused_async,
    clippy::items_after_statements,
    reason = "CLI binary: stdout/stderr printing is intentional; pedantic API \
              shape lints don't apply to a single-bin internal crate"
)]

use std::process::ExitCode;

use clap::Parser;

mod cli;
mod commands;
mod config;
mod context;
mod forecast;
mod output;
mod render;
mod wallet;

use crate::cli::Cli;

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Install the tracing subscriber **unconditionally** so `RUST_LOG` is
    // honored even without `-v` (previously it only installed under `-v`, so
    // `RUST_LOG=debug deadeye …` produced nothing). Routed to stderr so it
    // never contaminates `--output json` on stdout. `-v` raises the default to
    // `debug` for the deadeye crates; without it (and without RUST_LOG) we stay
    // quiet at `warn`. Span close-events surface each instrumented RPC/read
    // (contract address, timing) so a failing call is visible from the trace.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let default = if cli.verbose {
            "info,deadeye=debug"
        } else {
            "warn"
        };
        tracing_subscriber::EnvFilter::new(default)
    });
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .try_init();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("error: failed to start async runtime: {err}");
            return ExitCode::from(2);
        },
    };

    match runtime.block_on(commands::dispatch(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(1)
        },
    }
}
