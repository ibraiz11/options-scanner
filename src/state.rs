//! Persistent state: open orders, executed trades, daily P&L counter, killswitch flag.
//! Backed by a JSON file under ./data/state.json — small, human-readable, easy to inspect.
//! Switch to SQLite when the file gets >1MB or we need concurrent multi-process access.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::broker::{AssetClass, OrderRequest, OrderResult, OrderSize};
use crate::risk::{OrderSide, RiskDecision};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLogEntry {
    pub id: uuid::Uuid,
    pub timestamp: DateTime<Utc>,
    pub broker: String,
    pub symbol: String,
    pub side: OrderSide,
    pub qty: u32,                 // whole-share count, 0 when fractional notional used
    pub size_repr: String,        // "5 sh" or "$50.00" — what the UI should display
    pub asset_class: AssetClass,
    pub estimated_notional: f64,
    pub broker_order_id: Option<String>,
    pub status: String,        // "approved", "rejected", "submitted", "filled", "error"
    pub risk_decision: String, // "approve" | "resize: …" | "reject: …"
    pub notes: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyCounters {
    pub utc_date: String,           // "YYYY-MM-DD"
    pub trades_opened: u32,
    pub open_notional: f64,
    pub realized_pnl: f64,
    pub starting_equity: f64,
    pub killswitch_tripped: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub counters: DailyCounters,
    pub recent_trades: Vec<TradeLogEntry>, // capped at 200
}

#[derive(Clone)]
pub struct State {
    inner: Arc<RwLock<StateSnapshot>>,
    path: Option<PathBuf>,
}

impl State {
    pub async fn load(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let snap = match tokio::fs::read(&path).await {
            Ok(bytes) => serde_json::from_slice::<StateSnapshot>(&bytes).unwrap_or_default(),
            Err(_) => StateSnapshot::default(),
        };
        let st = Self { inner: Arc::new(RwLock::new(snap)), path: Some(path) };
        st.roll_day_if_needed().await;
        Ok(st)
    }

    /// In-memory only — never writes to disk. Used by the simulator so it can't
    /// pollute production state. Caller passes a starting equity to anchor P&L.
    pub fn ephemeral(starting_equity: f64) -> Self {
        let snap = StateSnapshot {
            counters: DailyCounters {
                utc_date: Utc::now().date_naive().to_string(),
                starting_equity,
                ..Default::default()
            },
            recent_trades: vec![],
        };
        Self { inner: Arc::new(RwLock::new(snap)), path: None }
    }

    /// Decrement open exposure and credit realized P&L when a position closes.
    /// Trips the killswitch if cumulative daily P&L breaches the cap.
    pub async fn close_position(&self, notional: f64, realized_pnl: f64, killswitch_pct: f64) {
        let mut w = self.inner.write().await;
        w.counters.open_notional = (w.counters.open_notional - notional).max(0.0);
        w.counters.realized_pnl += realized_pnl;
        if w.counters.starting_equity > 0.0 {
            let pnl_pct = w.counters.realized_pnl / w.counters.starting_equity * 100.0;
            if pnl_pct <= killswitch_pct {
                w.counters.killswitch_tripped = true;
            }
        }
    }

    pub async fn force_date(&self, ymd: &str) {
        let mut w = self.inner.write().await;
        if w.counters.utc_date != ymd {
            // new sim day — reset daily counters but preserve cumulative realized_pnl
            // so the killswitch threshold reflects the *whole sim*, not just today.
            let realized = w.counters.realized_pnl;
            let starting = w.counters.starting_equity;
            w.counters = DailyCounters {
                utc_date: ymd.into(),
                starting_equity: starting,
                realized_pnl: realized,
                ..Default::default()
            };
        }
    }

    pub async fn snapshot(&self) -> StateSnapshot {
        self.inner.read().await.clone()
    }

    pub async fn roll_day_if_needed(&self) {
        let today = Utc::now().date_naive().to_string();
        let mut w = self.inner.write().await;
        if w.counters.utc_date != today {
            w.counters = DailyCounters {
                utc_date: today,
                ..Default::default()
            };
        }
    }

    pub async fn record_decision(
        &self,
        order: &OrderRequest,
        notional: f64,
        broker_name: &str,
        decision: &RiskDecision,
        result: Option<&OrderResult>,
    ) -> Result<TradeLogEntry> {
        let (status, note) = match (decision, result) {
            (RiskDecision::Approve, Some(r))
            | (RiskDecision::Resize { .. }, Some(r)) => (r.status.clone(), String::new()),
            (RiskDecision::Approve, None) => ("error".into(), "no broker result".into()),
            (RiskDecision::Resize { reason, .. }, None) => ("error".into(), reason.clone()),
            (RiskDecision::Reject { reason }, _) => ("rejected".into(), reason.clone()),
        };
        let risk_str = match decision {
            RiskDecision::Approve => "approve".into(),
            RiskDecision::Resize { reason, .. } => format!("resize: {reason}"),
            RiskDecision::Reject { reason } => format!("reject: {reason}"),
        };
        let (qty_u32, size_repr) = match &order.size {
            OrderSize::Shares { qty } => (*qty, format!("{qty} sh")),
            OrderSize::Notional { dollars } => (0, format!("${dollars:.2}")),
        };
        let entry = TradeLogEntry {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            broker: broker_name.into(),
            symbol: order.symbol.clone(),
            side: order.side,
            qty: qty_u32,
            size_repr,
            asset_class: order.asset_class,
            estimated_notional: notional,
            broker_order_id: result.map(|r| r.broker_order_id.clone()),
            status,
            risk_decision: risk_str,
            notes: note,
        };

        let mut w = self.inner.write().await;
        if matches!(decision, RiskDecision::Approve | RiskDecision::Resize { .. })
            && entry.broker_order_id.is_some()
        {
            w.counters.trades_opened += 1;
            w.counters.open_notional += notional;
        }
        w.recent_trades.insert(0, entry.clone());
        w.recent_trades.truncate(200);
        let snap = w.clone();
        drop(w);

        if let Some(p) = &self.path {
            let bytes = serde_json::to_vec_pretty(&snap)?;
            tokio::fs::write(p, bytes).await?;
        }
        Ok(entry)
    }

    pub async fn set_killswitch(&self, on: bool) {
        let mut w = self.inner.write().await;
        w.counters.killswitch_tripped = on;
    }

    pub async fn set_starting_equity_if_unset(&self, equity: f64) {
        let mut w = self.inner.write().await;
        if w.counters.starting_equity == 0.0 {
            w.counters.starting_equity = equity;
        }
    }

    pub async fn killswitch_active(&self) -> bool {
        self.inner.read().await.counters.killswitch_tripped
    }
}
