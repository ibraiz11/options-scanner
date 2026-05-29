//! Stdio-based MCP server exposing this project's scanner, back-test, simulator,
//! and risk state to LLM clients (Claude Code, ChatGPT, Cursor, Codex).
//!
//! Why: Robinhood's Agentic Trading (launched 2026-05-27) is an MCP server consumed
//! by an LLM agent — not a REST API consumed by a deterministic bot. To plug our
//! analytics into that flow, the user attaches BOTH MCP servers to their LLM
//! client: ours provides signals + risk context, Robinhood's provides execution.
//!
//! This binary is a thin shim over the project's HTTP server. It does not
//! re-implement any logic; it converts MCP tool calls into HTTP requests against
//! `SCANNER_HTTP_BASE` (default http://localhost:8000). Run the main HTTP server
//! separately (`./run.sh`) so this MCP server has something to call.
//!
//! Wire protocol: newline-delimited JSON-RPC 2.0 over stdio, per the MCP spec.

use std::io::Write;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

const PROTOCOL_VERSION: &str = "2025-03-26";
const SERVER_NAME: &str = "options-scanner";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[serde(default)]
    jsonrpc: String,
    #[serde(default)]
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

struct Ctx {
    http: reqwest::Client,
    base: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    // Trace to stderr only — stdout is reserved for JSON-RPC frames and MUST NOT
    // contain anything else, or the MCP client will fail to parse the stream.
    eprintln!("options-scanner-mcp v{SERVER_VERSION} starting; speaking MCP {PROTOCOL_VERSION}");

    let ctx = Ctx {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest client"),
        base: std::env::var("SCANNER_HTTP_BASE").unwrap_or_else(|_| "http://localhost:8000".into()),
    };
    eprintln!("proxying tools to HTTP server at {}", ctx.base);

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() { continue; }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                emit_error(&Value::Null, -32700, &format!("parse error: {e}"));
                continue;
            }
        };
        if req.jsonrpc != "2.0" {
            emit_error(&req.id, -32600, "invalid Request — jsonrpc must be \"2.0\"");
            continue;
        }

        // Notifications (no id) get no reply. We handle them silently.
        let is_notification = matches!(req.id, Value::Null);
        let result = dispatch(&ctx, &req.method, &req.params).await;
        if is_notification { continue; }

        let response = match result {
            Ok(v) => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: Some(v),
                error: None,
            },
            Err((code, message)) => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: None,
                error: Some(JsonRpcError { code, message }),
            },
        };
        emit(&response);
    }

    Ok(())
}

fn emit<T: Serialize>(v: &T) {
    let bytes = serde_json::to_vec(v).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(&bytes);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn emit_error(id: &Value, code: i32, message: &str) {
    emit(&JsonRpcResponse {
        jsonrpc: "2.0",
        id: id.clone(),
        result: None,
        error: Some(JsonRpcError { code, message: message.into() }),
    });
}

async fn dispatch(ctx: &Ctx, method: &str, params: &Value) -> Result<Value, (i32, String)> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
            "capabilities": { "tools": {}, "prompts": {} },
        })),
        "initialized" | "notifications/initialized" => Ok(Value::Null),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools_list() })),
        "tools/call" => call_tool(ctx, params).await,
        "prompts/list" => Ok(json!({ "prompts": prompts_list() })),
        "prompts/get" => get_prompt(params),
        other => Err((-32601, format!("method not found: {other}"))),
    }
}

fn tools_list() -> Vec<Value> {
    vec![
        json!({
            "name": "scan_market",
            "description": "Run the technical scanner across the configured liquid optionable universe. \
                Combines rolling-VWAP relationship, MACD momentum, relative volume gate, and liquidity-sweep \
                detection to produce ranked bullish/bearish candidates with entry, stop, +10% target, and \
                +20% target. Use this when an agent needs fresh signals to inform a trading decision.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_score": { "type": "number", "default": 3.0, "description": "Minimum composite score (0-7) to include" },
                    "min_relvol": { "type": "number", "default": 1.3, "description": "Minimum relative volume multiple vs 20-day avg" },
                    "direction": { "type": "string", "enum": ["both", "bullish", "bearish"], "default": "both" },
                    "tickers": { "type": "string", "description": "Optional comma-separated symbol list to override the default universe" }
                }
            }
        }),
        json!({
            "name": "run_backtest",
            "description": "Walk-forward back-test of the scanner's strategy on historical bars. For every \
                day in the lookback window, the same signal logic is re-evaluated on the data available at \
                that point; outcomes (target_20 hit, stop hit, timeout) are tracked. Returns per-symbol and \
                aggregate stats including win rate, expectancy, and outcome distribution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_score": { "type": "number", "default": 3.0 },
                    "min_relvol": { "type": "number", "default": 1.3 },
                    "hold_bars": { "type": "integer", "default": 20, "description": "Max bars to hold a position before timeout exit" },
                    "period": { "type": "string", "enum": ["1y", "2y", "5y"], "default": "2y" }
                }
            }
        }),
        json!({
            "name": "simulate_strategy",
            "description": "Replay historical signals through the LIVE executor + risk caps + slippage + \
                mark-to-market killswitch — the same code path that handles real orders. Returns equity \
                curve, max drawdown, killswitch trip status, and full event log. Supports out-of-sample \
                split: set oos=true to run train/test halves and compare. Use this before recommending \
                any parameter change so the agent sees how caps interact with the strategy.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "starting_equity": { "type": "number", "default": 100000 },
                    "min_score": { "type": "number", "default": 3.0 },
                    "slippage_bps": { "type": "number", "default": 10, "description": "Round-trip slippage in basis points (each side)" },
                    "period": { "type": "string", "enum": ["1y", "2y", "5y"], "default": "2y" },
                    "oos": { "type": "boolean", "default": false },
                    "split_fraction": { "type": "number", "default": 0.5, "description": "When oos=true, fraction of trades used for training" }
                }
            }
        }),
        json!({
            "name": "get_risk_state",
            "description": "Current account + risk snapshot: which broker is connected, account equity, \
                trading mode (standard/tiny), active risk caps, today's trade counters, recent trade log, \
                and killswitch status. The agent should call this BEFORE proposing trades, to know whether \
                the human caps have already been breached and how much exposure budget remains.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "check_health",
            "description": "Diagnoses every common operational failure: uv installed, Python deps present, \
                broker reachable, state directory writable. Call this first if any other tool returns \
                surprising errors — the response includes concrete `fix` strings for each failed check.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "render_chart_for_vision",
            "description": "Render the symbol's candlestick chart (with volume and the top geometric \
                detections annotated as trigger/target/stop lines) and return it as an IMAGE you can \
                look at directly. This is the LLM-vision second-opinion layer: after calling \
                detect_chart_patterns for the deterministic geometry, call this to SEE the chart and \
                form an independent visual read. Look for things the rule engine can't catch — sloppy \
                or forming patterns, trendline breaks, channels, divergences — and explicitly note where \
                your visual read AGREES or DISAGREES with the geometric detections (blue=trigger, \
                green=target, red=stop lines on the chart). Requires a vision-capable client.",
            "inputSchema": {
                "type": "object",
                "required": ["symbol"],
                "properties": {
                    "symbol": { "type": "string" },
                    "period": { "type": "string", "enum": ["3mo", "6mo", "1y", "2y"], "default": "6mo" },
                    "annotate": { "type": "boolean", "default": true, "description": "Overlay geometric detection levels" }
                }
            }
        }),
        json!({
            "name": "detect_chart_patterns",
            "description": "Run rule-based geometric chart-pattern detection on a symbol's daily bars. \
                Detects reversal patterns (head & shoulders + inverse, double/triple top & bottom), \
                continuation patterns (flags, pennants, triangles, wedges), candlestick signals \
                (engulfing, hammer, shooting star, doji, morning/evening star), and context levels \
                (support/resistance, Fibonacci retracements, VWAP bands). Each detection returns the \
                direction, a confidence score (0-1), the confirmation trigger price, a measured-move \
                target, and an invalidation stop. \
                \n\nThese are DETERMINISTIC geometric facts, not predictions — a detected double_top is \
                a description of the price shape, not a guarantee it resolves bearishly. Cross-reference \
                with scan_market (which combines VWAP/MACD/RVOL/sweep) and run_backtest before acting. \
                Use this when the user asks 'what patterns are forming on X' or to confirm a scanner signal.",
            "inputSchema": {
                "type": "object",
                "required": ["symbol"],
                "properties": {
                    "symbol": { "type": "string", "description": "Ticker symbol, e.g. AAPL" },
                    "period": { "type": "string", "enum": ["3mo", "6mo", "1y", "2y"], "default": "6mo" },
                    "pivot_order": { "type": "integer", "default": 3, "description": "Swing-pivot sensitivity; lower finds more (noisier) pivots" }
                }
            }
        }),
        json!({
            "name": "execute_paper_trade",
            "description": "Place a real order at the configured broker — STRICTLY PAPER ONLY. The server \
                refuses with HTTP 403 if any live broker is connected, and refuses with HTTP 409 if the \
                unattended scheduler is currently auto-trading (to prevent races). The order still passes \
                through the full risk-cap pipeline (per-trade size limit, daily count, total exposure, \
                killswitch); the response includes the risk decision (Approve / Resize / Reject) and the \
                broker order id when accepted. \
                \n\nUse this tool ONLY after: (1) calling scan_market for fresh signals, (2) calling \
                get_risk_state to confirm caps have room, and (3) explicit human confirmation. The order \
                fields (entry, stop, target_10, target_20) must come from a scan_market result — do not \
                fabricate prices.",
            "inputSchema": {
                "type": "object",
                "required": ["symbol", "direction", "entry", "stop", "target_10", "target_20", "score"],
                "properties": {
                    "symbol": { "type": "string", "description": "Ticker symbol from a scan_market result" },
                    "direction": { "type": "string", "enum": ["bullish", "bearish"] },
                    "entry": { "type": "number" },
                    "stop": { "type": "number" },
                    "target_10": { "type": "number" },
                    "target_20": { "type": "number" },
                    "score": { "type": "number" }
                }
            }
        }),
    ]
}

fn prompts_list() -> Vec<Value> {
    vec![
        json!({
            "name": "morning_briefing",
            "description": "Generate a structured morning brief: current risk state, top scanner candidates, \
                what's worth watching today. No execution.",
            "arguments": []
        }),
        json!({
            "name": "risk_audit",
            "description": "Audit whether the strategy still has edge by running an out-of-sample simulation \
                and comparing train vs test halves. Flags over-fit or regime-change risk.",
            "arguments": [
                { "name": "period", "description": "Lookback window (1y, 2y, or 5y)", "required": false }
            ]
        }),
        json!({
            "name": "propose_trade",
            "description": "Find the best current setup, verify the risk caps allow another position, and \
                output a structured proposal. Does NOT execute — the user must confirm and re-prompt with \
                execute_paper_trade.",
            "arguments": [
                { "name": "min_score", "description": "Minimum composite score (default 4)", "required": false },
                { "name": "direction", "description": "bullish | bearish | both (default both)", "required": false }
            ]
        }),
        json!({
            "name": "analyze_chart",
            "description": "Full dual-read chart analysis for one symbol: deterministic geometric pattern \
                detection PLUS an LLM-vision second opinion, reconciled into one view.",
            "arguments": [
                { "name": "symbol", "description": "Ticker to analyze", "required": true },
                { "name": "period", "description": "3mo | 6mo | 1y | 2y (default 6mo)", "required": false }
            ]
        }),
        json!({
            "name": "bridge_to_robinhood",
            "description": "Dual-MCP workflow: use this scanner's signals + risk discipline to drive a trade \
                in your Robinhood Agentic Trading sub-account. Requires both MCPs registered in the same \
                LLM session.",
            "arguments": [
                { "name": "min_score", "description": "Minimum composite score (default 4)", "required": false },
                { "name": "size_pct_of_rh_equity", "description": "Fraction of Robinhood agentic-account equity to risk (default 5)", "required": false }
            ]
        })
    ]
}

fn get_prompt(params: &Value) -> Result<Value, (i32, String)> {
    let name = params.get("name").and_then(|v| v.as_str())
        .ok_or((-32602, "missing prompt name".into()))?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));
    let arg = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();

    let (description, text) = match name {
        "morning_briefing" => (
            "Morning briefing workflow",
            "I need a concise pre-market briefing. Please:\n\
            1. Call `get_risk_state` to show current account equity, trading mode, today's trade counters, and killswitch status.\n\
            2. If the killswitch is tripped, STOP and report — do not scan for trades.\n\
            3. Otherwise call `scan_market` with min_score=3.5 and direction=both.\n\
            4. Group the results by direction (bullish / bearish) and rank by score within each group.\n\
            5. For the top 3 candidates overall, summarize: symbol, direction, score, entry, stop, +10% target, +20% target, and the signal notes (which indicators fired).\n\
            6. Conclude with one sentence on which (if any) setup looks worth a closer look, and a reminder that nothing is executed without my explicit go-ahead.".to_string()
        ),
        "risk_audit" => {
            let period = if arg("period").is_empty() { "2y".to_string() } else { arg("period") };
            (
                "Strategy edge audit",
                format!(
                    "Run a strategy health audit using an out-of-sample simulation over a {period} lookback.\n\
                    \n\
                    1. Call `simulate_strategy` with: starting_equity=100000, period={period}, oos=true, split_fraction=0.5, slippage_bps=10.\n\
                    2. Report the train-period and test-period returns side by side, plus max drawdown for each.\n\
                    3. Highlight the gap between train and test returns:\n\
                       - If test ≥ 0 and within 30% of train: strategy looks robust.\n\
                       - If test < 0 or gap > 30%: strategy may be over-fit or regime has shifted. Recommend NOT trading.\n\
                    4. Report whether the killswitch tripped in either half — that's the most important data point.\n\
                    5. Be specific about what to do next: continue paper, retune parameters, or stop trading entirely."
                )
            )
        }
        "propose_trade" => {
            let min_score = if arg("min_score").is_empty() { "4".to_string() } else { arg("min_score") };
            let direction = if arg("direction").is_empty() { "both".to_string() } else { arg("direction") };
            (
                "Trade proposal workflow",
                format!(
                    "Propose (but do NOT execute) the best current trade. Workflow:\n\
                    \n\
                    1. Call `check_health` — if anything is broken, report and stop.\n\
                    2. Call `get_risk_state`. If `killswitch_tripped` is true, or `trades_opened` has reached the daily cap, or `open_notional / account_equity` is already at the exposure cap, STOP and explain — there's no room for a new trade.\n\
                    3. Call `scan_market` with min_score={min_score} and direction={direction}.\n\
                    4. Pick the top-scoring candidate. If no candidate meets the threshold, say so.\n\
                    5. Output a structured proposal in this exact format:\n\
                       ```\n\
                       PROPOSAL\n\
                       Symbol:    <sym>\n\
                       Direction: <bullish|bearish>\n\
                       Score:     <n>\n\
                       Entry:     $<entry>\n\
                       Stop:      $<stop>  (risk: <abs_dollars_at_5%_of_equity>)\n\
                       Target +10%: $<t10>\n\
                       Target +20%: $<t20>\n\
                       Signals: <notes from scanner>\n\
                       ```\n\
                    6. Conclude with: 'To execute this paper trade, confirm and I will call execute_paper_trade with these exact values.' Do NOT call execute_paper_trade in this turn."
                )
            )
        }
        "analyze_chart" => {
            let symbol = arg("symbol");
            if symbol.is_empty() {
                return Err((-32602, "analyze_chart requires a `symbol` argument".into()));
            }
            let period = if arg("period").is_empty() { "6mo".to_string() } else { arg("period") };
            (
                "Dual-read chart analysis (geometric + vision)",
                format!(
                    "Analyze {symbol} ({period}) using BOTH detection layers, then reconcile:\n\
                    \n\
                    1. Call `detect_chart_patterns` with symbol={symbol}, period={period}. This is the \
                    deterministic geometric read — exact patterns with trigger/target/stop levels.\n\
                    2. Call `render_chart_for_vision` with symbol={symbol}, period={period}. Actually LOOK \
                    at the returned chart image. Form your own visual read.\n\
                    3. Reconcile the two into a single table:\n\
                       | Pattern | Geometric? | Visual confirm? | Verdict |\n\
                       - CONFIRMED: both the rules and your eyes agree.\n\
                       - GEOMETRIC-ONLY: rules flagged it but it looks weak/invalid to you — say why.\n\
                       - VISUAL-ONLY: you see something (channel, trendline break, divergence) the rules missed.\n\
                    4. For each CONFIRMED pattern, restate its trigger / target / stop.\n\
                    5. Cross-check against `scan_market` (does the momentum scanner agree on direction?) and \
                    note alignment or conflict.\n\
                    6. Conclude with the single highest-conviction setup (or 'no clean setup') and the \
                    specific level that would confirm or invalidate it. Do NOT place any order — analysis only."
                )
            )
        }
        "bridge_to_robinhood" => {
            let min_score = if arg("min_score").is_empty() { "4".to_string() } else { arg("min_score") };
            let size_pct = if arg("size_pct_of_rh_equity").is_empty() { "5".to_string() } else { arg("size_pct_of_rh_equity") };
            (
                "Bridge scanner analytics → Robinhood Agentic execution",
                format!(
                    "Drive a trade in my Robinhood Agentic sub-account using this scanner's analytics. \
                    This workflow assumes BOTH MCPs are connected:\n\
                    - `options-scanner` (this server) provides signals, back-test evidence, and risk context.\n\
                    - `robinhood-agentic` provides quote, account, and order tools against my sandboxed sub-account.\n\
                    \n\
                    Step-by-step:\n\
                    \n\
                    1. From `options-scanner`, call `check_health` — if anything is broken, stop.\n\
                    2. From `options-scanner`, call `get_risk_state`. Note today's trade count, killswitch state, mode.\n\
                    3. From `robinhood-agentic`, fetch the Agentic sub-account info: equity, buying power, current positions. (Tool name will be something like `get_account` or `account_info` — use whatever the connected Robinhood MCP exposes.)\n\
                    4. From `options-scanner`, call `scan_market` with min_score={min_score}.\n\
                    5. Filter the scan results to symbols Robinhood actually offers equities for (drop ETFs the agentic sandbox doesn't support if any).\n\
                    6. Take the top candidate. Cross-check it isn't already in the Robinhood positions list — if it is, skip to the next candidate to avoid double-sizing.\n\
                    7. Compute the position size as {size_pct}% of the Robinhood agentic-account equity (NOT this scanner's mock equity — Robinhood is the real account being traded).\n\
                    8. Get a fresh quote from `robinhood-agentic` for the chosen symbol. If the live price has moved >2% from the scanner's `entry`, abort and report — the signal has gone stale.\n\
                    9. Output a structured proposal in this EXACT format:\n\
                       ```\n\
                       PROPOSAL FOR ROBINHOOD AGENTIC\n\
                       Symbol:     <sym>\n\
                       Direction:  <bullish=BUY | bearish=skip equity, agentic shorts not supported>\n\
                       Score:      <n>/7\n\
                       Robinhood live quote: $<live>\n\
                       Scanner entry:        $<entry>  (drift: <pct>%)\n\
                       Stop loss:            $<stop>\n\
                       +10% target:          $<t10>\n\
                       +20% target:          $<t20>\n\
                       Size:                 <qty> sh (~$<notional>, {size_pct}% of agentic equity)\n\
                       Signals fired:        <notes>\n\
                       ```\n\
                    10. End with: 'Confirm to place this order via Robinhood Agentic. I will use the order tool from `robinhood-agentic` only after explicit confirmation.'\n\
                    \n\
                    Critical rules:\n\
                    - Bearish signals require shorts. Robinhood's agentic account may not support shorts at launch — if not, REPORT and skip rather than placing a long against a bearish signal.\n\
                    - Never call `execute_paper_trade` (this scanner's tool) AND a Robinhood order tool in the same turn. They're separate accounts — pick one.\n\
                    - If `get_risk_state` shows our scanner's killswitch tripped, that's a signal about market regime, not just our paper account. Be skeptical about placing the Robinhood trade and say so."
                )
            )
        }
        other => return Err((-32602, format!("unknown prompt: {other}"))),
    };

    Ok(json!({
        "description": description,
        "messages": [
            { "role": "user", "content": { "type": "text", "text": text } }
        ]
    }))
}

async fn call_tool(ctx: &Ctx, params: &Value) -> Result<Value, (i32, String)> {
    let name = params.get("name").and_then(|v| v.as_str())
        .ok_or((-32602, "missing tool name".into()))?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    // execute_paper_trade uses POST with a JSON body; everything else is a GET with query params.
    if name == "execute_paper_trade" {
        return execute_paper_trade(ctx, &args).await;
    }
    // render_chart_for_vision returns MCP image content, not text.
    if name == "render_chart_for_vision" {
        return render_chart_for_vision(ctx, &args).await;
    }

    let endpoint = match name {
        "scan_market" => path_with_query("/api/scan", &args, &["min_score", "min_relvol", "direction", "tickers"]),
        "run_backtest" => path_with_query("/api/backtest", &args, &["min_score", "min_relvol", "hold_bars", "period"]),
        "simulate_strategy" => path_with_query("/api/simulate", &args, &["starting_equity", "min_score", "slippage_bps", "period", "oos", "split_fraction"]),
        "get_risk_state" => "/api/state".to_string(),
        "check_health" => "/api/health".to_string(),
        "detect_chart_patterns" => path_with_query("/api/patterns", &args, &["symbol", "period", "pivot_order"]),
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };

    let url = format!("{}{}", ctx.base, endpoint);
    let body = match ctx.http.get(&url).send().await {
        Ok(r) => {
            let status = r.status();
            let text = r.text().await.unwrap_or_default();
            if !status.is_success() {
                // Return as tool result so the LLM sees and can act on the error
                // rather than silently failing with a JSON-RPC error.
                let err = serde_json::from_str::<Value>(&text)
                    .unwrap_or_else(|_| json!({ "raw": text }));
                return Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Tool call returned HTTP {status}.\n\n{}", serde_json::to_string_pretty(&err).unwrap_or_default())
                    }],
                    "isError": true,
                }));
            }
            serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({ "raw": text }))
        }
        Err(e) => return Ok(json!({
            "content": [{
                "type": "text",
                "text": format!(
                    "HTTP request to {url} failed: {e}\n\n\
                    The MCP server proxies tools to the main scanner HTTP server. \
                    Make sure it's running: `cd /Users/ibraizqazi/RustWorks/options-scanner && ./run.sh`"
                )
            }],
            "isError": true,
        })),
    };

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&body).unwrap_or_default()
        }],
    }))
}

async fn execute_paper_trade(ctx: &Ctx, args: &Value) -> Result<Value, (i32, String)> {
    let required = ["symbol", "direction", "entry", "stop", "target_10", "target_20", "score"];
    for k in required {
        if args.get(k).is_none() {
            return Ok(json!({
                "content": [{ "type": "text", "text":
                    format!("Refusing — missing required field `{k}`. The execute_paper_trade tool requires \
                            the full signal object (symbol, direction, entry, stop, target_10, target_20, score) \
                            as returned by scan_market. Do not fabricate prices; call scan_market first.")
                }],
                "isError": true,
            }));
        }
    }
    let url = format!("{}/api/execute-mcp", ctx.base);
    let resp = match ctx.http.post(&url).json(args).send().await {
        Ok(r) => r,
        Err(e) => return Ok(json!({
            "content": [{ "type": "text", "text":
                format!("HTTP POST to {url} failed: {e}.\n\nIs the scanner HTTP server running? Start it with `./run.sh`.")
            }],
            "isError": true,
        })),
    };
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({ "raw": text }));
    let is_err = !status.is_success();
    let summary = if is_err {
        format!(
            "Order refused (HTTP {status}).\n\n{}\n\nThis is a SAFETY refusal — the MCP execute tool is hard-gated to paper-only. \
            If you intended a live trade, use Robinhood's agentic-trading MCP tools instead. The dashboard's manual Send \
            button bypasses this gate for human-approved orders.",
            serde_json::to_string_pretty(&parsed).unwrap_or_default()
        )
    } else {
        format!(
            "Order placed (paper).\n\n{}",
            serde_json::to_string_pretty(&parsed).unwrap_or_default()
        )
    };
    Ok(json!({
        "content": [{ "type": "text", "text": summary }],
        "isError": is_err,
    }))
}

async fn render_chart_for_vision(ctx: &Ctx, args: &Value) -> Result<Value, (i32, String)> {
    if args.get("symbol").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
        return Ok(json!({
            "content": [{ "type": "text", "text": "Refusing — `symbol` is required." }],
            "isError": true,
        }));
    }
    let endpoint = path_with_query("/api/chart", args, &["symbol", "period", "annotate"]);
    let url = format!("{}{}", ctx.base, endpoint);
    let resp = match ctx.http.get(&url).send().await {
        Ok(r) => r,
        Err(e) => return Ok(json!({
            "content": [{ "type": "text", "text":
                format!("Chart request to {url} failed: {e}. Is the scanner HTTP server running (./run.sh)?") }],
            "isError": true,
        })),
    };
    let body: Value = serde_json::from_slice(&resp.bytes().await.unwrap_or_default())
        .unwrap_or_else(|_| json!({}));

    if let Some(err) = body.get("error").and_then(|v| v.as_str()) {
        return Ok(json!({
            "content": [{ "type": "text", "text": format!("Chart render error: {err}") }],
            "isError": true,
        }));
    }

    let png = body.get("png_base64").and_then(|v| v.as_str()).unwrap_or("");
    if png.is_empty() {
        return Ok(json!({
            "content": [{ "type": "text", "text": "No image returned from chart renderer." }],
            "isError": true,
        }));
    }

    // Pair the image with the geometric detections so the model can compare its
    // visual read against the deterministic facts.
    let detections = body.get("annotated_detections").cloned().unwrap_or(json!([]));
    let symbol = body.get("symbol").and_then(|v| v.as_str()).unwrap_or("?");
    let guidance = format!(
        "Chart for {symbol}. Annotation lines: blue=trigger, green=target, red=stop.\n\n\
        Geometric detector found these (compare your visual read against them):\n{}\n\n\
        Now look at the chart image and give an independent second opinion: which detections do you \
        visually confirm, which look wrong, and what does the rule engine miss?",
        serde_json::to_string_pretty(&detections).unwrap_or_default()
    );

    Ok(json!({
        "content": [
            { "type": "image", "data": png, "mimeType": "image/png" },
            { "type": "text", "text": guidance }
        ]
    }))
}

fn path_with_query(path: &str, args: &Value, keys: &[&str]) -> String {
    let mut q: Vec<String> = vec![];
    if let Some(obj) = args.as_object() {
        for k in keys {
            if let Some(v) = obj.get(*k) {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => b.to_string(),
                    Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                q.push(format!("{k}={}", urlencode(&s)));
            }
        }
    }
    if q.is_empty() { path.into() } else { format!("{path}?{}", q.join("&")) }
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}
