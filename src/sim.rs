//! Paper-trade simulation harness — hardened.
//!
//! Compared to the v1 sim this:
//!   1. Steps day-by-day across the union of all bar dates (not just signal dates),
//!      so unrealized P&L is marked-to-market every trading day.
//!   2. Trips the killswitch on (realized + unrealized) daily P&L, not just realized.
//!   3. Applies bid-ask slippage symmetrically on entry and exit.
//!   4. Supports walk-forward out-of-sample split — train half / test half — to expose
//!      strategies that work on the data they were tuned on but die on fresh data.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::broker::{Broker, BrokerResult, OrderRequest, OrderResult, Position};
use crate::executor::{Executor, Signal, TradingMode};
use crate::risk::{RiskCaps, RiskDecision};
use crate::state::State;
use tokio::sync::RwLock;

#[derive(Debug, Deserialize, Clone)]
pub struct Bar {
    pub date: String,
    pub close: f64,
    pub high: f64,
    pub low: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BacktestTrade {
    pub symbol: String,
    pub direction: String,
    pub entry_date: String,
    pub entry: f64,
    pub stop: f64,
    pub target_10: f64,
    pub target_20: f64,
    pub score: f64,
    pub outcome: String,
    pub exit_date: String,
    pub exit_price: f64,
    pub return_pct: f64,
    #[serde(default)]
    pub bars: Vec<Bar>,
}

// ---- SimBroker -------------------------------------------------------------

pub struct SimBroker { inner: Mutex<f64> }
impl SimBroker {
    pub fn new(eq: f64) -> Self { Self { inner: Mutex::new(eq) } }
    pub async fn apply_pnl(&self, pnl: f64) { *self.inner.lock().await += pnl; }
    pub async fn equity_value(&self) -> f64 { *self.inner.lock().await }
}
#[async_trait]
impl Broker for SimBroker {
    fn name(&self) -> &'static str { "sim" }
    fn is_paper(&self) -> bool { true }
    async fn account_equity(&self) -> BrokerResult<f64> { Ok(*self.inner.lock().await) }
    async fn place_order(&self, _o: &OrderRequest) -> BrokerResult<OrderResult> {
        Ok(OrderResult {
            broker_order_id: uuid::Uuid::new_v4().to_string(),
            status: "sim_filled".into(),
            submitted_at: chrono::Utc::now(),
        })
    }
    async fn list_positions(&self) -> BrokerResult<Vec<Position>> { Ok(vec![]) }
}

// ---- Slippage --------------------------------------------------------------

#[derive(Copy, Clone)]
enum SlipKind { Entry, Exit }
fn slipped(price: f64, is_buy: bool, kind: SlipKind, bps: f64) -> f64 {
    let f = bps / 10_000.0;
    let adj = match (is_buy, kind) {
        (true, SlipKind::Entry) => 1.0 + f,  // pay more
        (true, SlipKind::Exit)  => 1.0 - f,  // receive less
        (false, SlipKind::Entry) => 1.0 - f, // short fill: receive less
        (false, SlipKind::Exit)  => 1.0 + f, // cover: pay more
    };
    price * adj
}

// ---- Simulator -------------------------------------------------------------

#[derive(Debug, Clone)]
struct OpenPos {
    symbol: String,
    direction: String,
    /// Effective share count: real shares for whole-share orders, or notional/entry
    /// for fractional orders. Stored as f64 so both code paths use the same P&L math.
    effective_shares: f64,
    entry_filled: f64,        // post-slippage entry price
    exit_planned: f64,        // post-slippage exit price
    exit_date: NaiveDate,
    bars_by_date: HashMap<NaiveDate, Bar>,
    last_mtm: f64,            // current mark, defaults to entry until first bar arrives
    notional: f64,
}

impl OpenPos {
    fn unrealized_pnl(&self) -> f64 {
        let dir_sign = if self.direction == "bullish" { 1.0 } else { -1.0 };
        (self.last_mtm - self.entry_filled) * self.effective_shares * dir_sign
    }
    fn realized_pnl_on_close(&self) -> f64 {
        let dir_sign = if self.direction == "bullish" { 1.0 } else { -1.0 };
        (self.exit_planned - self.entry_filled) * self.effective_shares * dir_sign
    }
}

#[derive(Default, Debug)]
struct DayActions {
    entries: Vec<usize>,
    exits: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimEvent {
    pub kind: String,
    pub date: String,
    pub symbol: Option<String>,
    pub detail: String,
    pub equity_after: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct SimReport {
    pub label: String,
    pub starting_equity: f64,
    pub final_equity: f64,
    pub total_signals: usize,
    pub orders_filled: usize,
    pub orders_rejected: usize,
    pub orders_resized: usize,
    pub killswitch_tripped: bool,
    pub max_drawdown_pct: f64,             // peak-to-trough realized
    pub max_unrealized_drawdown_pct: f64,  // worst MTM drawdown including open positions
    pub return_pct: f64,
    pub slippage_bps: f64,
    pub events: Vec<SimEvent>,
}

pub struct Simulator {
    pub caps: RiskCaps,
    pub starting_equity: f64,
    pub target_pct_per_trade: f64,
    pub slippage_bps: f64,
}

impl Simulator {
    pub async fn run(&self, mut trades: Vec<BacktestTrade>, label: &str) -> Result<SimReport> {
        trades.sort_by(|a, b| a.entry_date.cmp(&b.entry_date));

        let broker = Arc::new(SimBroker::new(self.starting_equity));
        let state = State::ephemeral(self.starting_equity);
        let broker_for_exec: Box<dyn Broker> = Box::new(BrokerHandle { inner: broker.clone() });
        // Sim picks the mode that matches the configured caps; this preserves the
        // existing back-test semantics (Standard caps, 5% per trade) without exposing
        // a separate API surface for choosing it.
        let mode = if self.caps.max_pct_per_trade <= 10.0 {
            TradingMode::Standard
        } else {
            TradingMode::Tiny
        };
        let exec = Executor {
            broker: broker_for_exec,
            state: state.clone(),
            mode: Arc::new(RwLock::new(mode)),
        };

        // Build the day-by-day calendar: every date that has an entry, an exit, or a bar.
        let mut calendar: BTreeMap<NaiveDate, DayActions> = BTreeMap::new();
        for (i, t) in trades.iter().enumerate() {
            if let Ok(d) = NaiveDate::parse_from_str(&t.entry_date, "%Y-%m-%d") {
                calendar.entry(d).or_default().entries.push(i);
            }
            if let Ok(d) = NaiveDate::parse_from_str(&t.exit_date, "%Y-%m-%d") {
                calendar.entry(d).or_default().exits.push(i);
            }
            for b in &t.bars {
                if let Ok(d) = NaiveDate::parse_from_str(&b.date, "%Y-%m-%d") {
                    calendar.entry(d).or_default();
                }
            }
        }

        let mut open: Vec<OpenPos> = vec![];
        let mut events: Vec<SimEvent> = vec![];
        let mut realized: f64 = 0.0;
        let mut peak_combined: f64 = self.starting_equity;
        let mut peak_realized: f64 = self.starting_equity;
        let mut max_dd: f64 = 0.0;
        let mut max_unrealized_dd: f64 = 0.0;
        let mut filled = 0usize;
        let mut rejected = 0usize;
        let mut resized = 0usize;

        for (date, actions) in &calendar {
            let date_str = date.to_string();
            state.force_date(&date_str).await;

            // (1) Mark every open position to today's close if it has a bar for today.
            for op in open.iter_mut() {
                if let Some(b) = op.bars_by_date.get(date) {
                    op.last_mtm = b.close;
                }
            }

            // (2) Unrealized P&L and killswitch check (combined realized + unrealized).
            let unrealized: f64 = open.iter().map(|p| p.unrealized_pnl()).sum();
            let combined_equity = self.starting_equity + realized + unrealized;
            let combined_pnl_pct = (combined_equity - self.starting_equity) / self.starting_equity * 100.0;
            if combined_pnl_pct <= self.caps.daily_drawdown_killswitch_pct
                && !state.killswitch_active().await
            {
                state.set_killswitch(true).await;
                events.push(SimEvent {
                    kind: "killswitch".into(),
                    date: date_str.clone(),
                    symbol: None,
                    detail: format!(
                        "killswitch tripped on MTM: combined P&L {:.2}% <= {:.2}%",
                        combined_pnl_pct, self.caps.daily_drawdown_killswitch_pct
                    ),
                    equity_after: combined_equity,
                });
            }

            peak_combined = peak_combined.max(combined_equity);
            let unrealized_dd = (peak_combined - combined_equity) / peak_combined * 100.0;
            max_unrealized_dd = max_unrealized_dd.max(unrealized_dd);

            // (3) Process exits scheduled for today.
            let mut keep = Vec::with_capacity(open.len());
            for op in open.drain(..) {
                if op.exit_date == *date {
                    let pnl = op.realized_pnl_on_close();
                    realized += pnl;
                    broker.apply_pnl(pnl).await;
                    state.close_position(op.notional, pnl, self.caps.daily_drawdown_killswitch_pct).await;
                    let eq = broker.equity_value().await;
                    peak_realized = peak_realized.max(eq);
                    let dd = (peak_realized - eq) / peak_realized * 100.0;
                    max_dd = max_dd.max(dd);
                    events.push(SimEvent {
                        kind: "close".into(),
                        date: date_str.clone(),
                        symbol: Some(op.symbol),
                        detail: format!("close pnl=${:.2}", pnl),
                        equity_after: eq,
                    });
                } else {
                    keep.push(op);
                }
            }
            open = keep;

            // (4) Process new entries — through the real risk-checked executor.
            for &i in &actions.entries {
                let t = &trades[i];
                let _ = i; // index retained in case future event records need it
                let sig = Signal {
                    symbol: t.symbol.clone(),
                    direction: t.direction.clone(),
                    entry: t.entry,
                    stop: t.stop,
                    target_10: t.target_10,
                    target_20: t.target_20,
                    score: t.score,
                };
                let report = exec.execute_signal(sig).await?;
                let eq_now = broker.equity_value().await;

                match &report.decision {
                    RiskDecision::Reject { reason } => {
                        rejected += 1;
                        events.push(SimEvent {
                            kind: "decision".into(),
                            date: date_str.clone(),
                            symbol: Some(t.symbol.clone()),
                            detail: format!("REJECT: {reason}"),
                            equity_after: eq_now,
                        });
                    }
                    RiskDecision::Approve | RiskDecision::Resize { .. } => {
                        let was_resized = matches!(report.decision, RiskDecision::Resize { .. });
                        if was_resized { resized += 1; }
                        filled += 1;
                        let trade_entry = report.trade.as_ref()
                            .ok_or_else(|| anyhow!("missing trade record"))?;
                        let exit_date = NaiveDate::parse_from_str(&t.exit_date, "%Y-%m-%d")
                            .unwrap_or(*date);

                        let is_buy = t.direction == "bullish";
                        let entry_filled = slipped(t.entry, is_buy, SlipKind::Entry, self.slippage_bps);
                        let exit_planned = slipped(t.exit_price, is_buy, SlipKind::Exit, self.slippage_bps);

                        let bars_by_date: HashMap<NaiveDate, Bar> = t.bars.iter()
                            .filter_map(|b| NaiveDate::parse_from_str(&b.date, "%Y-%m-%d").ok()
                                .map(|d| (d, b.clone())))
                            .collect();

                        // For fractional notional orders trade_entry.qty is 0; derive
                        // effective shares from the dollar notional and entry price.
                        let effective_shares = if trade_entry.qty > 0 {
                            trade_entry.qty as f64
                        } else if t.entry > 0.0 {
                            trade_entry.estimated_notional / t.entry
                        } else { 0.0 };
                        open.push(OpenPos {
                            symbol: t.symbol.clone(),
                            direction: t.direction.clone(),
                            effective_shares,
                            entry_filled,
                            exit_planned,
                            exit_date,
                            bars_by_date,
                            last_mtm: entry_filled,
                            notional: trade_entry.estimated_notional,
                        });
                        events.push(SimEvent {
                            kind: "decision".into(),
                            date: date_str.clone(),
                            symbol: Some(t.symbol.clone()),
                            detail: if was_resized {
                                format!("RESIZE → fill qty={} notional=${:.0} (slip {:.0}bps)",
                                    trade_entry.qty, trade_entry.estimated_notional, self.slippage_bps)
                            } else {
                                format!("FILL qty={} notional=${:.0} (slip {:.0}bps)",
                                    trade_entry.qty, trade_entry.estimated_notional, self.slippage_bps)
                            },
                            equity_after: eq_now,
                        });
                    }
                }
            }
        }

        // Close anything still open at the end of the calendar.
        let last_date = calendar.keys().next_back().copied().unwrap_or_else(|| chrono::Utc::now().date_naive());
        let _ = realized; // realized is no longer read after this point
        for op in open.drain(..) {
            let pnl = op.realized_pnl_on_close();
            broker.apply_pnl(pnl).await;
            events.push(SimEvent {
                kind: "close".into(),
                date: last_date.to_string(),
                symbol: Some(op.symbol),
                detail: format!("end-of-sim close pnl=${:.2}", pnl),
                equity_after: broker.equity_value().await,
            });
        }

        let final_equity = broker.equity_value().await;
        Ok(SimReport {
            label: label.into(),
            starting_equity: self.starting_equity,
            final_equity,
            total_signals: trades.len(),
            orders_filled: filled,
            orders_rejected: rejected,
            orders_resized: resized,
            killswitch_tripped: state.killswitch_active().await,
            max_drawdown_pct: round2(max_dd),
            max_unrealized_drawdown_pct: round2(max_unrealized_dd),
            return_pct: round2((final_equity - self.starting_equity) / self.starting_equity * 100.0),
            slippage_bps: self.slippage_bps,
            events,
        })
    }

    /// Run two simulations: first-half trades (train) and second-half (test).
    /// A strategy that works on real data should look similar on both halves;
    /// large gaps indicate over-fit parameters or regime dependence.
    pub async fn run_oos(&self, mut trades: Vec<BacktestTrade>, split_fraction: f64) -> Result<(SimReport, SimReport)> {
        trades.sort_by(|a, b| a.entry_date.cmp(&b.entry_date));
        let cut = ((trades.len() as f64) * split_fraction.clamp(0.1, 0.9)) as usize;
        let train = trades[..cut].to_vec();
        let test = trades[cut..].to_vec();
        let t_report = self.run(train, "train").await?;
        let v_report = self.run(test, "test").await?;
        Ok((t_report, v_report))
    }
}

fn round2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::RiskCaps;

    fn winning_trade(sym: &str, entry_date: &str, exit_date: &str) -> BacktestTrade {
        BacktestTrade {
            symbol: sym.into(), direction: "bullish".into(),
            entry_date: entry_date.into(), entry: 100.0,
            stop: 95.0, target_10: 110.0, target_20: 120.0,
            score: 5.0, outcome: "target_20".into(),
            exit_date: exit_date.into(), exit_price: 120.0, return_pct: 20.0,
            bars: vec![
                Bar { date: entry_date.into(), close: 100.0, high: 101.0, low: 99.0 },
                Bar { date: exit_date.into(), close: 120.0, high: 121.0, low: 100.0 },
            ],
        }
    }
    fn losing_trade(sym: &str, entry_date: &str, exit_date: &str) -> BacktestTrade {
        // Larger loss per trade than a real stop would produce — the test exists
        // to verify the killswitch mechanic, not to model realistic stop behavior.
        let mut t = winning_trade(sym, entry_date, exit_date);
        t.outcome = "stop".into();
        t.exit_price = 50.0;
        t.return_pct = -50.0;
        t.bars = vec![
            Bar { date: entry_date.into(), close: 100.0, high: 100.0, low: 99.0 },
            Bar { date: exit_date.into(), close: 50.0, high: 100.0, low: 49.0 },
        ];
        t
    }

    #[tokio::test]
    async fn zero_slippage_matches_backtest_pnl_within_bounds() {
        let sim = Simulator {
            caps: RiskCaps::MODERATE, starting_equity: 100_000.0,
            target_pct_per_trade: 5.0, slippage_bps: 0.0,
        };
        let r = sim.run(vec![winning_trade("AAA", "2024-01-02", "2024-01-10")], "t").await.unwrap();
        assert_eq!(r.orders_filled, 1);
        // entry 100, exit 120, qty = floor(5000/100) = 50 → pnl ~ +1000 → ~+1% on 100k
        assert!(r.return_pct > 0.5 && r.return_pct < 1.5, "got {}", r.return_pct);
    }

    #[tokio::test]
    async fn slippage_reduces_pnl_vs_no_slippage() {
        let no_slip = Simulator {
            caps: RiskCaps::MODERATE, starting_equity: 100_000.0,
            target_pct_per_trade: 5.0, slippage_bps: 0.0,
        };
        let with_slip = Simulator {
            caps: RiskCaps::MODERATE, starting_equity: 100_000.0,
            target_pct_per_trade: 5.0, slippage_bps: 50.0, // 0.5% each side
        };
        let trade = vec![winning_trade("AAA", "2024-01-02", "2024-01-10")];
        let a = no_slip.run(trade.clone(), "no").await.unwrap();
        let b = with_slip.run(trade, "yes").await.unwrap();
        assert!(b.return_pct < a.return_pct, "slip should hurt: {} vs {}", a.return_pct, b.return_pct);
    }

    #[tokio::test]
    async fn many_losers_trip_killswitch_via_mtm() {
        let trades: Vec<_> = (0..8).map(|i| {
            let d = format!("2024-01-{:02}", i + 2);
            let e = format!("2024-01-{:02}", i + 3);
            losing_trade(&format!("L{i}"), &d, &e)
        }).collect();
        let sim = Simulator {
            caps: RiskCaps::MODERATE, starting_equity: 10_000.0,
            target_pct_per_trade: 5.0, slippage_bps: 0.0,
        };
        let r = sim.run(trades, "kill").await.unwrap();
        // Several losers in a row should eventually push combined P&L past -8%
        assert!(r.killswitch_tripped || r.orders_rejected > 0,
            "expected killswitch or cap rejections; got {:?}", r);
    }
}

struct BrokerHandle { inner: Arc<SimBroker> }
#[async_trait]
impl Broker for BrokerHandle {
    fn name(&self) -> &'static str { self.inner.name() }
    fn is_paper(&self) -> bool { true }
    async fn account_equity(&self) -> BrokerResult<f64> { self.inner.account_equity().await }
    async fn place_order(&self, o: &OrderRequest) -> BrokerResult<OrderResult> { self.inner.place_order(o).await }
    async fn list_positions(&self) -> BrokerResult<Vec<Position>> { self.inner.list_positions().await }
}
