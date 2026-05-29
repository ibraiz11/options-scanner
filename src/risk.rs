//! Risk caps enforcement. This file is the firewall between the strategy and the broker.
//! Every order must pass `check()` before it can be sent. If you change these caps,
//! the change should be reviewed like a security patch — bugs here cost real money.
//!
//! Moderate-tier caps (user-selected 2026-05):
//!   - max 5 trades opened per UTC day
//!   - max 5% of account equity per individual trade
//!   - max 25% of account equity total open notional
//!   - kill-switch: halt all new orders if daily realized+unrealized P&L <= -8%

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RiskCaps {
    pub max_trades_per_day: u32,
    pub max_pct_per_trade: f64,
    pub max_pct_total_exposure: f64,
    pub daily_drawdown_killswitch_pct: f64,
}

impl RiskCaps {
    pub const MODERATE: Self = Self {
        max_trades_per_day: 5,
        max_pct_per_trade: 5.0,
        max_pct_total_exposure: 25.0,
        daily_drawdown_killswitch_pct: -8.0,
    };

    /// For accounts under ~$1,000 where diversification is mathematically impossible.
    /// Trade one position at a time using most of the available cash; tighter killswitch
    /// because a single bad trade can wipe a large fraction of the account.
    pub const TINY: Self = Self {
        max_trades_per_day: 3,
        max_pct_per_trade: 80.0,
        max_pct_total_exposure: 80.0,
        daily_drawdown_killswitch_pct: -25.0,
    };
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskSnapshot {
    pub account_equity: f64,
    pub trades_today: u32,
    pub open_notional: f64,
    pub daily_pnl_pct: f64,
    pub killswitch_tripped: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProposedOrder {
    pub symbol: String,
    pub side: OrderSide,
    pub estimated_notional: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide { Buy, Sell }

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "decision", rename_all = "lowercase")]
pub enum RiskDecision {
    Approve,
    Reject { reason: String },
    Resize { new_notional: f64, reason: String },
}

/// Pure function — easy to unit-test, no I/O. The whole risk model lives here.
pub fn check(caps: &RiskCaps, snap: &RiskSnapshot, order: &ProposedOrder) -> RiskDecision {
    if snap.killswitch_tripped {
        return RiskDecision::Reject {
            reason: format!(
                "killswitch active (daily P&L {:.2}% <= {:.2}%)",
                snap.daily_pnl_pct, caps.daily_drawdown_killswitch_pct
            ),
        };
    }
    if snap.daily_pnl_pct <= caps.daily_drawdown_killswitch_pct {
        return RiskDecision::Reject {
            reason: format!(
                "killswitch tripping now: daily P&L {:.2}% <= {:.2}%",
                snap.daily_pnl_pct, caps.daily_drawdown_killswitch_pct
            ),
        };
    }
    if snap.trades_today >= caps.max_trades_per_day {
        return RiskDecision::Reject {
            reason: format!(
                "daily trade count cap reached ({} >= {})",
                snap.trades_today, caps.max_trades_per_day
            ),
        };
    }
    if snap.account_equity <= 0.0 {
        return RiskDecision::Reject {
            reason: "non-positive account equity".into(),
        };
    }
    if order.estimated_notional <= 0.0 {
        return RiskDecision::Reject {
            reason: "non-positive order notional".into(),
        };
    }

    let max_per_trade_dollars = snap.account_equity * caps.max_pct_per_trade / 100.0;
    let max_total_dollars = snap.account_equity * caps.max_pct_total_exposure / 100.0;
    let remaining_exposure = (max_total_dollars - snap.open_notional).max(0.0);

    // Effective per-order ceiling is the min of (per-trade cap, remaining total cap).
    let ceiling = max_per_trade_dollars.min(remaining_exposure);

    if ceiling <= 0.0 {
        return RiskDecision::Reject {
            reason: format!(
                "no exposure room left (open {:.2} / cap {:.2})",
                snap.open_notional, max_total_dollars
            ),
        };
    }

    if order.estimated_notional <= ceiling {
        return RiskDecision::Approve;
    }

    // Don't silently truncate by a lot — if the requested size is way over the cap,
    // reject so a human notices. Only resize when within 4x the ceiling.
    if order.estimated_notional > ceiling * 4.0 {
        return RiskDecision::Reject {
            reason: format!(
                "requested notional ${:.0} is >4x the per-order ceiling ${:.0} — refusing to resize",
                order.estimated_notional, ceiling
            ),
        };
    }

    RiskDecision::Resize {
        new_notional: ceiling,
        reason: format!(
            "scaled from ${:.0} to ${:.0} to respect caps",
            order.estimated_notional, ceiling
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(equity: f64, trades: u32, open: f64, pnl: f64, kill: bool) -> RiskSnapshot {
        RiskSnapshot {
            account_equity: equity,
            trades_today: trades,
            open_notional: open,
            daily_pnl_pct: pnl,
            killswitch_tripped: kill,
        }
    }
    fn order(notional: f64) -> ProposedOrder {
        ProposedOrder {
            symbol: "AAPL".into(),
            side: OrderSide::Buy,
            estimated_notional: notional,
        }
    }

    #[test]
    fn approves_when_under_caps() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 0.0, 0.0, false), &order(4_000.0));
        assert!(matches!(r, RiskDecision::Approve));
    }

    #[test]
    fn resizes_when_over_per_trade() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 0.0, 0.0, false), &order(8_000.0));
        match r {
            RiskDecision::Resize { new_notional, .. } => {
                assert!((new_notional - 5_000.0).abs() < 0.01)
            }
            other => panic!("expected resize, got {other:?}"),
        }
    }

    #[test]
    fn rejects_when_killswitch_active() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 0.0, 0.0, true), &order(1_000.0));
        assert!(matches!(r, RiskDecision::Reject { .. }));
    }

    #[test]
    fn rejects_when_drawdown_exceeded() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 0.0, -9.0, false), &order(1_000.0));
        assert!(matches!(r, RiskDecision::Reject { .. }));
    }

    #[test]
    fn rejects_when_daily_count_reached() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 5, 0.0, 0.0, false), &order(1_000.0));
        assert!(matches!(r, RiskDecision::Reject { .. }));
    }

    #[test]
    fn rejects_when_total_exposure_full() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 25_000.0, 0.0, false), &order(1.0));
        assert!(matches!(r, RiskDecision::Reject { .. }));
    }

    #[test]
    fn rejects_when_request_absurdly_large() {
        let r = check(&RiskCaps::MODERATE, &snap(100_000.0, 0, 0.0, 0.0, false), &order(50_000.0));
        assert!(matches!(r, RiskDecision::Reject { .. }));
    }
}
