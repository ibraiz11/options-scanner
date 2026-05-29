//! Axum server. Proxies /api/scan and /api/backtest to Python (Polars) scripts
//! by spawning them via `uv run`. Static frontend served from ./static.
//!
//! Phase-1 additions: /api/execute, /api/state, /api/killswitch, /api/risk-caps.

mod broker;
mod executor;
mod risk;
mod scheduler;
mod sim;
mod state;

use axum::{extract::{Query, State as AxState}, response::Json, routing::{get, post}, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;
use tower_http::services::ServeDir;

use crate::broker::default_broker;
use crate::executor::{Executor, Signal, TradingMode};
use crate::scheduler::SchedulerConfig;
use crate::state::State;
use tokio::sync::RwLock;

#[derive(Clone)]
struct AppState {
    executor: Arc<Executor>,
    mode: Arc<RwLock<TradingMode>>,
    scheduler_cfg: Arc<RwLock<SchedulerConfig>>,
}

#[derive(Deserialize)]
struct ScanParams {
    min_score: Option<f64>,
    min_relvol: Option<f64>,
    direction: Option<String>,
    tickers: Option<String>,
}

#[derive(Deserialize)]
struct BacktestParams {
    min_score: Option<f64>,
    min_relvol: Option<f64>,
    hold_bars: Option<u32>,
    period: Option<String>,
    tickers: Option<String>,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

async fn run_python(script: &str, args: Vec<String>) -> Result<Value, String> {
    let py_dir = project_root().join("python");
    let output = Command::new("uv")
        .current_dir(&py_dir)
        .arg("run").arg("--quiet").arg("python").arg(script).args(&args)
        .output().await
        .map_err(|e| format!("failed to spawn uv: {e}. Install: https://docs.astral.sh/uv/"))?;
    if !output.status.success() {
        return Err(format!("python exited {}: {}", output.status, String::from_utf8_lossy(&output.stderr)));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| {
        format!("json parse failed: {e}; stdout: {}",
            String::from_utf8_lossy(&output.stdout).chars().take(500).collect::<String>())
    })
}

async fn api_scan(Query(p): Query<ScanParams>) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    let mut args = vec![];
    if let Some(s) = p.min_score { args.push(format!("--min-score={s}")); }
    if let Some(r) = p.min_relvol { args.push(format!("--min-relvol={r}")); }
    if let Some(d) = p.direction { args.push(format!("--direction={d}")); }
    if let Some(t) = p.tickers { args.push(format!("--tickers={t}")); }
    match run_python("scanner.py", args).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(e) => {
            tracing::error!(error=%e, "scan failed");
            // Return 502 so the frontend's `res.ok` check fires — empty results
            // and an underlying subprocess failure should NOT look identical.
            (StatusCode::BAD_GATEWAY, Json(json!({
                "error": e,
                "hint": "Visit /api/health for what's wrong.",
            })))
        }
    }
}

async fn api_backtest(Query(p): Query<BacktestParams>) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    let mut args = vec![];
    if let Some(s) = p.min_score { args.push(format!("--min-score={s}")); }
    if let Some(r) = p.min_relvol { args.push(format!("--min-relvol={r}")); }
    if let Some(h) = p.hold_bars { args.push(format!("--hold-bars={h}")); }
    if let Some(per) = p.period { args.push(format!("--period={per}")); }
    if let Some(t) = p.tickers { args.push(format!("--tickers={t}")); }
    match run_python("backtest.py", args).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(e) => {
            tracing::error!(error=%e, "backtest failed");
            (StatusCode::BAD_GATEWAY, Json(json!({
                "error": e,
                "hint": "Visit /api/health for what's wrong.",
            })))
        }
    }
}

async fn api_universe() -> Json<Value> {
    match run_python("universe.py", vec![]).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e})),
    }
}

/// Reports every common failure mode the UI should warn about so the user knows
/// exactly what to fix before clicking buttons that depend on these.
async fn api_health(AxState(st): AxState<AppState>) -> Json<Value> {
    let mut checks: Vec<serde_json::Value> = vec![];
    let mut all_ok = true;

    // 1. `uv` binary present?
    let uv_status = tokio::process::Command::new("uv")
        .arg("--version").output().await;
    match uv_status {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            checks.push(json!({
                "name": "uv installed",
                "ok": true,
                "detail": v,
            }));
        }
        _ => {
            all_ok = false;
            checks.push(json!({
                "name": "uv installed",
                "ok": false,
                "detail": "uv binary not found on PATH",
                "fix": "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh — then restart the server.",
            }));
        }
    }

    // 2. Python env synced (only meaningful if uv exists)
    let py_dir = project_root().join("python");
    let sync_check = tokio::process::Command::new("uv")
        .current_dir(&py_dir)
        .args(["run", "--quiet", "python", "-c", "import polars, yfinance; print('ok')"])
        .output().await;
    match sync_check {
        Ok(o) if o.status.success() => {
            checks.push(json!({"name": "python deps (polars + yfinance)", "ok": true}));
        }
        Ok(o) => {
            all_ok = false;
            checks.push(json!({
                "name": "python deps (polars + yfinance)",
                "ok": false,
                "detail": String::from_utf8_lossy(&o.stderr).chars().take(400).collect::<String>(),
                "fix": "cd python && uv sync",
            }));
        }
        Err(e) => {
            all_ok = false;
            checks.push(json!({
                "name": "python deps (polars + yfinance)",
                "ok": false,
                "detail": format!("could not run python: {e}"),
                "fix": "Install uv first, then `cd python && uv sync`.",
            }));
        }
    }

    // 3. Broker — paper creds present? (only matters for live fills, not for the scanner)
    let broker_name = st.executor.broker.name();
    if broker_name == "dry-run" {
        checks.push(json!({
            "name": "broker credentials",
            "ok": false,
            "severity": "warning",
            "detail": "Using DryRunBroker — orders are logged but never sent. The scanner still works.",
            "fix": "Set ALPACA_API_KEY and ALPACA_API_SECRET (get free paper keys at alpaca.markets) and restart the server.",
        }));
    } else {
        match st.executor.broker.account_equity().await {
            Ok(eq) => checks.push(json!({
                "name": "broker reachable",
                "ok": true,
                "detail": format!("{} · equity ${:.2}", broker_name, eq),
            })),
            Err(e) => {
                all_ok = false;
                checks.push(json!({
                    "name": "broker reachable",
                    "ok": false,
                    "detail": format!("{broker_name}: {e}"),
                    "fix": "Check ALPACA_API_KEY/ALPACA_API_SECRET are correct and the account is enabled.",
                }));
            }
        }
    }

    // 4. State file writable
    let state_path = project_root().join("data/state.json");
    let state_ok = tokio::fs::metadata(&state_path).await.is_ok()
        || tokio::fs::create_dir_all(state_path.parent().unwrap()).await.is_ok();
    checks.push(json!({
        "name": "state directory writable",
        "ok": state_ok,
        "detail": state_path.display().to_string(),
    }));
    if !state_ok { all_ok = false; }

    Json(json!({"ok": all_ok, "checks": checks}))
}

async fn api_execute(AxState(st): AxState<AppState>, Json(sig): Json<Signal>) -> Json<Value> {
    match st.executor.execute_signal(sig).await {
        Ok(r) => Json(serde_json::to_value(r).unwrap_or(json!({"error": "serialize"}))),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// MCP-safe execution. Adds a hard refusal if anything other than a paper
/// broker is connected, so an LLM cannot accidentally trigger live orders
/// through this surface even if the human switched the underlying broker.
/// The dashboard's manual Send button stays on /api/execute and is unaffected.
async fn api_execute_mcp(AxState(st): AxState<AppState>, Json(sig): Json<Signal>) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    if !st.executor.broker.is_paper() {
        return (StatusCode::FORBIDDEN, Json(json!({
            "error": "refused: MCP execution is paper-only",
            "broker": st.executor.broker.name(),
            "hint": "The execute_paper_trade tool refuses to route orders to a live broker. Connect a paper broker (Alpaca paper) or use the dashboard's manual Send button on the scan tab.",
        })));
    }
    if st.scheduler_cfg.read().await.on {
        return (StatusCode::CONFLICT, Json(json!({
            "error": "refused: scheduler is currently auto-trading",
            "hint": "Stop the unattended scheduler from the dashboard before placing manual MCP trades, so orders don't race.",
        })));
    }
    match st.executor.execute_signal(sig).await {
        Ok(r) => (StatusCode::OK, Json(serde_json::to_value(r).unwrap_or(json!({"error": "serialize"})))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    }
}

async fn api_state(AxState(st): AxState<AppState>) -> Json<Value> {
    let snap = st.executor.state.snapshot().await;
    let equity = st.executor.broker.account_equity().await.unwrap_or(0.0);
    let mode = *st.mode.read().await;
    let scheduler = st.scheduler_cfg.read().await.clone();
    let market_open = st.executor.broker.is_market_open().await.unwrap_or(true);
    Json(json!({
        "broker": st.executor.broker.name(),
        "is_paper": st.executor.broker.is_paper(),
        "market_open": market_open,
        "account_equity": equity,
        "mode": mode,
        "caps": mode.caps(),
        "scheduler": scheduler,
        "counters": snap.counters,
        "recent_trades": snap.recent_trades,
    }))
}

#[derive(Deserialize)]
struct ModeBody { mode: TradingMode }
async fn api_mode(AxState(st): AxState<AppState>, Json(body): Json<ModeBody>) -> Json<Value> {
    *st.mode.write().await = body.mode;
    tracing::info!(?body.mode, "trading mode switched");
    Json(json!({"mode": body.mode, "caps": body.mode.caps()}))
}

#[derive(Deserialize)]
struct SchedulerBody {
    on: Option<bool>,
    scan_interval_seconds: Option<u64>,
    min_score_to_execute: Option<f64>,
}
async fn api_scheduler(AxState(st): AxState<AppState>, Json(body): Json<SchedulerBody>) -> Json<Value> {
    let mut cfg = st.scheduler_cfg.write().await;
    if let Some(on) = body.on { cfg.on = on; }
    if let Some(s) = body.scan_interval_seconds { cfg.scan_interval_seconds = s.max(30); }
    if let Some(m) = body.min_score_to_execute { cfg.min_score_to_execute = m; }
    let snapshot = cfg.clone();
    tracing::info!(?snapshot, "scheduler updated");
    Json(serde_json::to_value(snapshot).unwrap_or(json!({})))
}

#[derive(Deserialize)]
struct SimulateParams {
    min_score: Option<f64>,
    min_relvol: Option<f64>,
    hold_bars: Option<u32>,
    period: Option<String>,
    tickers: Option<String>,
    starting_equity: Option<f64>,
    slippage_bps: Option<f64>,
    oos: Option<bool>,
    split_fraction: Option<f64>,
}

async fn api_simulate(AxState(st): AxState<AppState>, Query(p): Query<SimulateParams>) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    // 1. Fetch back-test trades via Python (same as /api/backtest).
    let mut args = vec![];
    if let Some(s) = p.min_score { args.push(format!("--min-score={s}")); }
    if let Some(r) = p.min_relvol { args.push(format!("--min-relvol={r}")); }
    if let Some(h) = p.hold_bars { args.push(format!("--hold-bars={h}")); }
    if let Some(per) = p.period { args.push(format!("--period={per}")); }
    if let Some(t) = p.tickers { args.push(format!("--tickers={t}")); }
    let bt = match run_python("backtest.py", args).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({
            "error": format!("backtest fetch failed: {e}"),
            "hint": "Visit /api/health for what's wrong.",
        }))),
    };

    let trades_json = bt.get("trades").cloned().unwrap_or(json!([]));
    let trades: Vec<crate::sim::BacktestTrade> = match serde_json::from_value(trades_json) {
        Ok(t) => t,
        Err(e) => return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({
            "error": format!("trade parse failed: {e}"),
        }))),
    };

    if trades.is_empty() {
        return (StatusCode::OK, Json(json!({
            "mode": "single",
            "report": null,
            "warning": "Back-test produced 0 trades. Lower the score filter, widen the universe, or check that the scanner is finding any signals at all.",
        })));
    }

    let active_mode = *st.mode.read().await;
    let simulator = crate::sim::Simulator {
        caps: active_mode.caps(),
        starting_equity: p.starting_equity.unwrap_or(100_000.0),
        target_pct_per_trade: active_mode.target_pct_per_trade(),
        slippage_bps: p.slippage_bps.unwrap_or(10.0),
    };
    if p.oos.unwrap_or(false) {
        match simulator.run_oos(trades, p.split_fraction.unwrap_or(0.5)).await {
            Ok((train, test)) => (StatusCode::OK, Json(json!({"mode": "oos", "train": train, "test": test}))),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
        }
    } else {
        match simulator.run(trades, "single").await {
            Ok(r) => (StatusCode::OK, Json(json!({"mode": "single", "report": r}))),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
        }
    }
}

#[derive(Deserialize)]
struct KillswitchBody { on: bool }
async fn api_killswitch(AxState(st): AxState<AppState>, Json(body): Json<KillswitchBody>) -> Json<Value> {
    st.executor.state.set_killswitch(body.on).await;
    Json(json!({"killswitch": body.on}))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info".into())).init();

    let state = State::load(project_root().join("data/state.json")).await
        .expect("load state");

    // Default to Tiny mode if TRADING_MODE=tiny is set (handy for $100 accounts);
    // Standard otherwise. Either way, the running server can flip via /api/mode.
    let initial_mode = match std::env::var("TRADING_MODE").as_deref() {
        Ok("tiny") => TradingMode::Tiny,
        _ => TradingMode::Standard,
    };
    let mode = Arc::new(RwLock::new(initial_mode));
    tracing::info!(?initial_mode, "starting in mode");

    let executor = Arc::new(Executor {
        broker: default_broker(),
        state,
        mode: mode.clone(),
    });
    let scheduler_cfg = Arc::new(RwLock::new(SchedulerConfig::default()));

    // Background scheduler — off by default, gated on the config flag.
    tokio::spawn(scheduler::run(
        executor.clone(),
        scheduler_cfg.clone(),
        project_root().join("python"),
    ));

    let app_state = AppState { executor, mode, scheduler_cfg };

    let static_dir = project_root().join("static");
    let app = Router::new()
        .route("/api/scan", get(api_scan))
        .route("/api/backtest", get(api_backtest))
        .route("/api/universe", get(api_universe))
        .route("/api/execute", post(api_execute))
        .route("/api/execute-mcp", post(api_execute_mcp))
        .route("/api/state", get(api_state))
        .route("/api/health", get(api_health))
        .route("/api/killswitch", post(api_killswitch))
        .route("/api/simulate", get(api_simulate))
        .route("/api/mode", post(api_mode))
        .route("/api/scheduler", post(api_scheduler))
        .with_state(app_state)
        .fallback_service(ServeDir::new(static_dir));

    let addr = "0.0.0.0:8000";
    tracing::info!("options-scanner listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
