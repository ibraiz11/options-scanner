//! Unattended scheduler.
//!
//! When enabled, periodically:
//!   1. Asks the broker if the market is open. If not, sleeps a short tick and re-checks.
//!   2. Spawns the Python scanner with the configured threshold.
//!   3. Routes every result above min_score_to_execute through the Executor.
//!
//! Crucially does NOT bypass the risk layer — every order it generates passes
//! through `Executor::execute_signal`, which runs `risk::check` like any human-clicked order.
//!
//! Off by default. Flip via POST /api/scheduler {"on": true}.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::sync::RwLock;

use crate::executor::{Executor, Signal};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub on: bool,
    pub scan_interval_seconds: u64,
    pub min_score_to_execute: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { on: false, scan_interval_seconds: 300, min_score_to_execute: 4.0 }
    }
}

pub async fn run(
    executor: Arc<Executor>,
    cfg: Arc<RwLock<SchedulerConfig>>,
    python_dir: PathBuf,
) {
    // Tick frequently when disabled to pick up "on" quickly; sleep the configured
    // interval when actually scanning.
    loop {
        let snapshot = cfg.read().await.clone();
        if !snapshot.on {
            tokio::time::sleep(Duration::from_secs(15)).await;
            continue;
        }

        match executor.broker.is_market_open().await {
            Ok(true) => {}
            Ok(false) => {
                tracing::debug!("scheduler: market closed");
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
            Err(e) => {
                tracing::warn!("scheduler: market-open check failed: {e}");
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
        }

        if executor.state.killswitch_active().await {
            tracing::warn!("scheduler: killswitch active, holding off scans");
            tokio::time::sleep(Duration::from_secs(snapshot.scan_interval_seconds)).await;
            continue;
        }

        match run_scan(&python_dir, snapshot.min_score_to_execute).await {
            Ok(signals) => {
                tracing::info!("scheduler: {} candidate(s) above min_score={}",
                    signals.len(), snapshot.min_score_to_execute);
                for sig in signals {
                    match executor.execute_signal(sig.clone()).await {
                        Ok(r) => tracing::info!(symbol = %sig.symbol,
                            "scheduler exec: {:?}", r.decision),
                        Err(e) => tracing::warn!(symbol = %sig.symbol,
                            "scheduler exec failed: {e}"),
                    }
                }
            }
            Err(e) => tracing::warn!("scheduler: scan failed: {e}"),
        }

        tokio::time::sleep(Duration::from_secs(snapshot.scan_interval_seconds)).await;
    }
}

async fn run_scan(python_dir: &PathBuf, min_score: f64) -> anyhow::Result<Vec<Signal>> {
    let output = Command::new("uv")
        .current_dir(python_dir)
        .args(["run", "--quiet", "python", "scanner.py",
               &format!("--min-score={}", min_score)])
        .output().await?;
    if !output.status.success() {
        anyhow::bail!("python exit {}: {}",
            output.status, String::from_utf8_lossy(&output.stderr));
    }
    let val: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let results = val.get("results").cloned().unwrap_or(serde_json::json!([]));
    let signals: Vec<Signal> = serde_json::from_value(results).unwrap_or_default();
    Ok(signals)
}
