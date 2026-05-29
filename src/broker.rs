//! Broker abstraction. The Broker trait is what the executor talks to;
//! concrete implementations (Alpaca paper / IBKR / Schwab / dry-run) live below.
//!
//! Phase-1 ships:
//!   - DryRunBroker  — never touches network, just logs. Default if no creds set.
//!   - AlpacaPaper   — REST against paper-api.alpaca.markets when ALPACA_API_KEY set.
//!
//! Phase-2 adds: IBKR (via local ib-gateway bridge) and Schwab (OAuth).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::risk::OrderSide;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OrderSize {
    /// Whole shares — universal across brokers and asset classes.
    Shares { qty: u32 },
    /// Dollar notional — fractional shares. Alpaca paper supports this for equities only,
    /// market orders only, RTH only. Other brokers may reject.
    Notional { dollars: f64 },
}

impl OrderSize {
    pub fn approx_shares(&self, price: f64) -> u32 {
        match self {
            OrderSize::Shares { qty } => *qty,
            OrderSize::Notional { dollars } if price > 0.0 => (dollars / price).max(0.0) as u32,
            _ => 0,
        }
    }
    pub fn notional(&self, price: f64) -> f64 {
        match self {
            OrderSize::Shares { qty } => *qty as f64 * price,
            OrderSize::Notional { dollars } => *dollars,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub symbol: String,           // equity ticker OR OCC option symbol
    pub size: OrderSize,
    pub side: OrderSide,
    pub asset_class: AssetClass,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetClass { UsEquity, UsOption }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResult {
    pub broker_order_id: String,
    pub status: String,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub qty: i64,
    pub market_value: f64,
    pub unrealized_pl: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum BrokerError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("auth: {0}")]
    Auth(String),
    #[error("broker rejected: {0}")]
    Rejected(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type BrokerResult<T> = Result<T, BrokerError>;

#[async_trait]
pub trait Broker: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_paper(&self) -> bool;
    async fn account_equity(&self) -> BrokerResult<f64>;
    async fn place_order(&self, order: &OrderRequest) -> BrokerResult<OrderResult>;
    async fn list_positions(&self) -> BrokerResult<Vec<Position>>;
    /// Default impl returns true so non-trading brokers (sim/dry-run) always work.
    /// Live brokers must override with the venue's real clock to avoid out-of-hours rejects.
    async fn is_market_open(&self) -> BrokerResult<bool> { Ok(true) }
}

// ---- DryRun ----------------------------------------------------------------

pub struct DryRunBroker {
    pub mock_equity: f64,
}

#[async_trait]
impl Broker for DryRunBroker {
    fn name(&self) -> &'static str { "dry-run" }
    fn is_paper(&self) -> bool { true }
    async fn account_equity(&self) -> BrokerResult<f64> { Ok(self.mock_equity) }
    async fn place_order(&self, order: &OrderRequest) -> BrokerResult<OrderResult> {
        tracing::info!(?order, "dry-run order");
        Ok(OrderResult {
            broker_order_id: uuid::Uuid::new_v4().to_string(),
            status: "accepted_dry_run".into(),
            submitted_at: chrono::Utc::now(),
        })
    }
    async fn list_positions(&self) -> BrokerResult<Vec<Position>> { Ok(vec![]) }
}

// ---- Alpaca paper ----------------------------------------------------------

pub struct AlpacaPaper {
    client: reqwest::Client,
    base: String,
    key: String,
    secret: String,
}

impl AlpacaPaper {
    pub fn from_env() -> Option<Self> {
        let key = std::env::var("ALPACA_API_KEY").ok()?;
        let secret = std::env::var("ALPACA_API_SECRET").ok()?;
        Some(Self {
            client: reqwest::Client::new(),
            base: "https://paper-api.alpaca.markets".into(),
            key,
            secret,
        })
    }

    fn auth_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("APCA-API-KEY-ID", &self.key)
           .header("APCA-API-SECRET-KEY", &self.secret)
    }
}

#[async_trait]
impl Broker for AlpacaPaper {
    fn name(&self) -> &'static str { "alpaca-paper" }
    fn is_paper(&self) -> bool { true }

    async fn account_equity(&self) -> BrokerResult<f64> {
        #[derive(Deserialize)]
        struct Acct { equity: String }
        let r = self.auth_headers(self.client.get(format!("{}/v2/account", self.base)))
            .send().await?.error_for_status()?.json::<Acct>().await?;
        r.equity.parse::<f64>().map_err(|e| BrokerError::Auth(e.to_string()))
    }

    async fn place_order(&self, order: &OrderRequest) -> BrokerResult<OrderResult> {
        // Alpaca fractional shares use `notional` (dollars); whole shares use `qty`.
        // Only one of the two may be present in the request body.
        let side_str = match order.side { OrderSide::Buy => "buy", OrderSide::Sell => "sell" };
        let mut body = serde_json::json!({
            "symbol": order.symbol,
            "side": side_str,
            "type": "market",
            "time_in_force": "day",
        });
        match &order.size {
            OrderSize::Shares { qty } => { body["qty"] = serde_json::json!(qty.to_string()); }
            OrderSize::Notional { dollars } => {
                if matches!(order.asset_class, AssetClass::UsOption) {
                    return Err(BrokerError::Unsupported(
                        "Alpaca does not support fractional options".into(),
                    ));
                }
                body["notional"] = serde_json::json!(format!("{:.2}", dollars));
            }
        }
        #[derive(Deserialize)]
        struct OrderResp { id: String, status: String }
        let resp = self.auth_headers(self.client.post(format!("{}/v2/orders", self.base)))
            .json(&body).send().await?;
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(BrokerError::Rejected(txt));
        }
        let parsed: OrderResp = resp.json().await?;
        Ok(OrderResult {
            broker_order_id: parsed.id,
            status: parsed.status,
            submitted_at: chrono::Utc::now(),
        })
    }

    async fn list_positions(&self) -> BrokerResult<Vec<Position>> {
        #[derive(Deserialize)]
        struct Pos { symbol: String, qty: String, market_value: String, unrealized_pl: String }
        let v: Vec<Pos> = self.auth_headers(self.client.get(format!("{}/v2/positions", self.base)))
            .send().await?.error_for_status()?.json().await?;
        Ok(v.into_iter().map(|p| Position {
            symbol: p.symbol,
            qty: p.qty.parse().unwrap_or(0),
            market_value: p.market_value.parse().unwrap_or(0.0),
            unrealized_pl: p.unrealized_pl.parse().unwrap_or(0.0),
        }).collect())
    }

    async fn is_market_open(&self) -> BrokerResult<bool> {
        #[derive(Deserialize)] struct Clock { is_open: bool }
        let c: Clock = self.auth_headers(self.client.get(format!("{}/v2/clock", self.base)))
            .send().await?.error_for_status()?.json().await?;
        Ok(c.is_open)
    }
}

// ---- factory ---------------------------------------------------------------

pub fn default_broker() -> Box<dyn Broker> {
    if let Some(a) = AlpacaPaper::from_env() {
        tracing::info!("using Alpaca paper broker");
        Box::new(a)
    } else {
        tracing::warn!("no broker creds — using dry-run. Set ALPACA_API_KEY/ALPACA_API_SECRET to enable paper.");
        Box::new(DryRunBroker { mock_equity: 100_000.0 })
    }
}
