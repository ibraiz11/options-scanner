//! Order pipeline. Takes a scanner Signal and walks it through:
//!   1. instrument selection  (Phase-1: equity proxy; Phase-2: option contract picker)
//!   2. position sizing       (mode-aware: whole shares in Standard, dollar notional in Tiny)
//!   3. risk check            (risk::check — the firewall)
//!   4. broker submission     (every approved/resized order is logged regardless of fill)
//!
//! Single entry point: `execute_signal()`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::broker::{AssetClass, Broker, OrderRequest, OrderSize};
use crate::risk::{self, OrderSide, ProposedOrder, RiskCaps, RiskDecision, RiskSnapshot};
use crate::state::{State, TradeLogEntry};

/// Operational profile. Standard is for normally-sized accounts; Tiny is for $100–$1000.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TradingMode { Standard, Tiny }

impl TradingMode {
    pub fn caps(&self) -> RiskCaps {
        match self {
            TradingMode::Standard => RiskCaps::MODERATE,
            TradingMode::Tiny => RiskCaps::TINY,
        }
    }
    pub fn target_pct_per_trade(&self) -> f64 {
        match self {
            TradingMode::Standard => 5.0,
            TradingMode::Tiny => 75.0,
        }
    }
    pub fn use_fractional(&self) -> bool {
        matches!(self, TradingMode::Tiny)
    }
    pub fn allows_options(&self) -> bool {
        matches!(self, TradingMode::Standard)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Signal {
    pub symbol: String,
    pub direction: String,
    pub entry: f64,
    pub stop: f64,
    pub target_10: f64,
    pub target_20: f64,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct ExecutionReport {
    pub signal: Signal,
    pub decision: RiskDecision,
    pub trade: Option<TradeLogEntry>,
    pub error: Option<String>,
}

pub struct Executor {
    pub broker: Box<dyn Broker>,
    pub state: State,
    pub mode: Arc<RwLock<TradingMode>>,
}

impl Executor {
    pub fn current_caps(&self) -> RiskCaps {
        // Synchronous helper for snapshots; uses blocking_read sparingly.
        match self.mode.try_read() {
            Ok(m) => m.caps(),
            Err(_) => RiskCaps::MODERATE,
        }
    }

    fn pick_instrument(&self, sig: &Signal) -> (String, AssetClass, OrderSide) {
        let side = if sig.direction == "bullish" { OrderSide::Buy } else { OrderSide::Sell };
        // Phase-1: trade the underlying. Phase-2 will branch on mode.allows_options().
        (sig.symbol.clone(), AssetClass::UsEquity, side)
    }

    pub async fn execute_signal(&self, sig: Signal) -> Result<ExecutionReport> {
        self.state.roll_day_if_needed().await;

        let mode = *self.mode.read().await;
        let caps = mode.caps();
        let target_pct = mode.target_pct_per_trade();

        let equity = match self.broker.account_equity().await {
            Ok(e) => e,
            Err(e) => {
                return Ok(ExecutionReport {
                    signal: sig,
                    decision: RiskDecision::Reject { reason: format!("broker equity fetch failed: {e}") },
                    trade: None,
                    error: Some(e.to_string()),
                });
            }
        };
        self.state.set_starting_equity_if_unset(equity).await;

        let snap = self.state.snapshot().await;
        let daily_pnl_pct = if snap.counters.starting_equity > 0.0 {
            (equity - snap.counters.starting_equity) / snap.counters.starting_equity * 100.0
        } else { 0.0 };

        let target_notional = equity * target_pct / 100.0;
        let (symbol, asset_class, side) = self.pick_instrument(&sig);

        let proposed = ProposedOrder {
            symbol: symbol.clone(), side, estimated_notional: target_notional,
        };
        let risk_snap = RiskSnapshot {
            account_equity: equity,
            trades_today: snap.counters.trades_opened,
            open_notional: snap.counters.open_notional,
            daily_pnl_pct,
            killswitch_tripped: snap.counters.killswitch_tripped,
        };
        let decision = risk::check(&caps, &risk_snap, &proposed);

        let final_notional = match &decision {
            RiskDecision::Approve => target_notional,
            RiskDecision::Resize { new_notional, .. } => *new_notional,
            RiskDecision::Reject { .. } => {
                let dummy_size = if mode.use_fractional() {
                    OrderSize::Notional { dollars: target_notional }
                } else {
                    OrderSize::Shares { qty: 0 }
                };
                let entry = self.state.record_decision(
                    &OrderRequest { symbol: symbol.clone(), size: dummy_size, side, asset_class },
                    target_notional, self.broker.name(), &decision, None,
                ).await?;
                return Ok(ExecutionReport { signal: sig, decision, trade: Some(entry), error: None });
            }
        };

        // Mode-aware sizing: fractional uses dollars, standard uses whole shares.
        let size = if mode.use_fractional() {
            OrderSize::Notional { dollars: (final_notional * 100.0).round() / 100.0 }
        } else {
            let qty = (final_notional / sig.entry).max(1.0).floor() as u32;
            OrderSize::Shares { qty }
        };
        let order = OrderRequest { symbol: symbol.clone(), size, side, asset_class };

        let result = self.broker.place_order(&order).await;
        let (broker_result, error) = match result {
            Ok(r) => (Some(r), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let entry = self.state.record_decision(
            &order, final_notional, self.broker.name(), &decision, broker_result.as_ref(),
        ).await?;

        if daily_pnl_pct <= caps.daily_drawdown_killswitch_pct {
            self.state.set_killswitch(true).await;
            tracing::error!("killswitch TRIPPED — daily P&L {:.2}%", daily_pnl_pct);
        }

        Ok(ExecutionReport { signal: sig, decision, trade: Some(entry), error })
    }
}
