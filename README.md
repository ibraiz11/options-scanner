# options-scanner

A Rust + Polars technical scanner for US equities, with a back-test, a paper-trade simulator, a risk-capped order executor against Alpaca paper, and an MCP server that lets an LLM agent (Claude Code, ChatGPT, Cursor, Codex) consume the analytics and drive execution — including Robinhood's Agentic Trading sandbox.

**Status:** alpha. Not financial advice. The strategy has not been validated with live capital. Paper-trade for weeks before risking anything real, and read the [caveats](#caveats) before turning the scheduler on.

---

## What's in the box

```
                                           ┌─────────────────────────────────┐
                                           │   LLM client (Claude Code, etc) │
                                           └────────────┬────────────────────┘
                                                        │ MCP stdio
                              ┌─────────────────────────┴─────────────────────┐
                              │                                                │
                  ┌───────────▼──────────────┐                  ┌──────────────▼───────────┐
                  │ options-scanner-mcp      │                  │ robinhood-agentic MCP    │
                  │ (this repo)              │                  │ (Robinhood, beta)        │
                  │ 6 tools, 4 prompts       │                  │ trading tools, sandboxed │
                  └───────────┬──────────────┘                  └──────────────────────────┘
                              │ HTTP
                  ┌───────────▼──────────────────────────────────────────────┐
                  │ options-scanner HTTP server (Axum)                       │
                  │ /api/scan  /api/backtest  /api/simulate  /api/execute    │
                  │ /api/state /api/health    /api/scheduler /api/killswitch │
                  └───────────┬──────────────────────────────────────────────┘
                              │ subprocess
                  ┌───────────▼──────────────┐         ┌──────────────────────┐
                  │ python/scanner.py        │         │ broker (Alpaca paper │
                  │ python/backtest.py       │         │ or DryRun)           │
                  │ Polars on yfinance data  │         └──────────────────────┘
                  └──────────────────────────┘
```

Two execution paths share one analytics layer:
- **Deterministic auto-trader** → Alpaca paper, gated by hard risk caps, optionally unattended via a market-hours-aware scheduler.
- **LLM-mediated agentic trading** → Claude (or another MCP client) reads signals + risk state from this server's MCP, places trades through Robinhood Agentic Trading's MCP in a sandboxed sub-account.

---

## The strategy

A composite-score momentum scanner tuned for swing trades targeting **±10% to ±20% moves** on liquid US optionable names over a 30–45-bar hold horizon.

### Universe

The default universe is ~65 liquid optionable tickers across mega-cap tech, banks, energy, retail, healthcare, industrials, and high-volatility growth names. Defined in `python/universe.py` — edit freely, or pass a custom comma-separated list via the `tickers` parameter.

### Indicators (computed in Polars on daily bars)

| Indicator | Window | What it measures |
|---|---|---|
| **Rolling VWAP** | 20-day | Mean trade price weighted by volume. `close > VWAP` is bullish bias; `close < VWAP` is bearish. |
| **ATR** | 14-day, Wilder smoothing | True-range volatility. Used for stop sizing (`2 × ATR`) and to filter dead tickers (`ATR/close < 1%` → skip). |
| **MACD** | 12/26/9 | Trend + momentum. Bullish when `line > signal` AND `hist` is rising; bearish when both inverted. |
| **Relative volume** | 20-day | Today's volume / 20-day average. Required to be ≥ 1.3× for a signal to count (default; configurable). |
| **Liquidity sweep** | 20-day | Bullish: today's low pierces the prior 20-day low but closes back above it (stop-run reclaim). Bearish: the mirror with the prior high. |

### Scoring

Each indicator contributes to either a bull or bear tally:

| Signal | Bull / Bear points |
|---|---|
| Price > / < VWAP | +1 |
| MACD bull / bear cross with hist momentum | +1.5 |
| Relative volume ≥ 1.3× and today's candle aligns | +1 |
| Liquidity sweep in the corresponding direction | +2 |

A candidate is emitted if the larger tally is **≥ 3**. ATR filter (≥ 1% of price) is a hard gate — too-quiet tickers can't produce 10%–20% moves regardless of score.

### Decision

For each candidate the scanner reports:

```
Entry:       last close
Stop:        entry ± 2 × ATR    (long: entry - 2·ATR; short: entry + 2·ATR)
Target +10%: entry × 1.10       (long) or × 0.90 (short)
Target +20%: entry × 1.20       (long) or × 0.80 (short)
```

The 2×ATR stop is a heuristic; the ±10/±20% targets are fixed percentage bands. These are the easiest part of the strategy to tune.

### Risk caps

Every order — whether from the dashboard's Send button, the unattended scheduler, or the MCP `execute_paper_trade` tool — passes through `risk::check()` (`src/risk.rs`). Two presets:

| Cap | **Standard mode** (default, accounts ≥ $10k) | **Tiny mode** (accounts $100 – $1k) |
|---|---|---|
| Max trades opened per UTC day | 5 | 3 |
| Max % of equity per trade | 5% | 80% |
| Max % of equity in total open exposure | 25% | 80% |
| Daily P&L kill-switch | −8% | −25% |
| Order type | Whole shares | Dollar-notional (fractional via Alpaca) |
| Options allowed | yes (Phase 2) | no |

The risk layer has 7 unit tests covering each rejection path; resizing the request rather than refusing it happens when the requested size is within 4× the cap (further out and it refuses, since that's a bug-shape, not a tradable proposal).

The kill-switch trips on **realized + unrealized** P&L combined, not just realized. Once tripped it blocks every new order until manually reset from the Approvals tab.

---

## The analysis pipeline

Two layers of validation between "the scanner says X" and "should we trade it":

### 1. Walk-forward back-test (`python/backtest.py`)

For every historical day in the lookback window:
1. Re-evaluate the scanner on data available **up to that point** (no peeking).
2. If a signal fires, walk forward day-by-day checking which of `stop`, `target_10`, `target_20` hits first against the daily OHLC.
3. Conservative: if a stop and a target both fall inside the same daily bar, assume the stop hit first.

Outputs win rate, expectancy per trade, outcome distribution (hit-target / stopped / timed out), and per-symbol breakdown. Run it from the dashboard's Back-test tab or via `GET /api/backtest`.

### 2. Paper-trade simulator (`src/sim.rs`)

Replays the back-test's signals through the **live executor + risk layer** day-by-day, so we see:
- How many signals get rejected by the caps (vs the back-test which ignores them).
- Whether the kill-switch ever trips during the simulated period.
- What the equity curve actually does after slippage and risk constraints.

Three upgrades over a naïve back-test:

- **Slippage:** symmetric basis-point haircut on entry and exit. Default 10 bps; bump to 25–50 for thinner names. If the strategy is profitable at 0 bps but unprofitable at 25, it doesn't have edge.
- **Mark-to-market exposure:** each open position is marked daily against historical closes. Combined realized + unrealized P&L is checked every simulated day — the kill-switch sees through the discretization.
- **Walk-forward OOS split:** set `oos=true` (default in the UI) to run the first N% as training and the rest as test, both with the same parameters. A large gap between train and test return is the loudest signal of over-fit you'll get.

Three unit tests pin these properties: zero-slippage matches the back-test, increasing slippage decreases P&L, and concentrated losers correctly trip the kill-switch via MTM.

### How to read the output

| Field | What it tells you |
|---|---|
| `expectancy_pct_per_trade > 0` | Strategy has positive edge at these settings. Negative is fatal. |
| `win_rate` alone | Misleading. A 40% win rate at +18% / −10% is profitable; a 60% rate at +4% / −8% is not. |
| Train / Test gap | < 30% → robust evidence. > 30% → almost certainly overfit. |
| `max_unrealized_drawdown_pct` | Worst peak-to-trough including open positions. The number that should match your stomach. |
| `killswitch_tripped: YES` | Strategy has at least one bad week the caps can't survive. Tighten before live. |

---

## Execution

### Option 1 — Dashboard (manual, every order is a human click)

```bash
./run.sh      # builds the binary, sources .env.local, runs the server
```

Open http://localhost:8000. Three tabs: **Live scan**, **Back-test**, **Approvals & Risk**. The Live scan tab has a "Send" button on every result row — clicking it routes the signal through the same executor pipeline.

### Option 2 — Unattended scheduler (auto-trader)

From the Approvals tab, set the scan interval and `min_score` threshold, then click **Start auto-trading**. The scheduler:

- Polls every 15s when off, every `scan_interval_seconds` when on
- Asks the broker `is_market_open()` before scanning (Alpaca's clock endpoint)
- Skips when the kill-switch is tripped
- Routes every signal ≥ `min_score` through `Executor::execute_signal` — the same risk pipeline as a human click

Off by default. Server restart resets it to off. You opt in every time.

### Option 3 — Agentic via MCP (Claude Code + Robinhood Agentic Trading)

This is the path designed for **Robinhood's Agentic Trading** product (launched 2026-05-27), where an LLM consumes MCP tools to trade in a sandboxed sub-account.

Register this server's MCP in Claude Code:

```bash
claude mcp add options-scanner \
  /Users/ibraizqazi/RustWorks/options-scanner/target/release/options-scanner-mcp \
  -e SCANNER_HTTP_BASE=http://localhost:8000
```

Eight tools become available:

| Tool | What it does |
|---|---|
| `scan_market` | Return ranked candidates with entry/stop/targets |
| `run_backtest` | Walk-forward back-test stats |
| `simulate_strategy` | Replay through caps + slippage + MTM, optional OOS split |
| `get_risk_state` | Account, mode, caps, today's counters, kill-switch status |
| `check_health` | Diagnose `uv` / Python / broker / state-dir issues, with concrete fixes |
| `detect_chart_patterns` | Rule-based geometric pattern detection (see below) |
| `render_chart_for_vision` | Render the candlestick chart as an image for an LLM-vision second opinion |
| `execute_paper_trade` | **Paper-only gated** order placement against Alpaca paper |

Five prompts (invokable as slash commands in Claude Code):

| Prompt | What it does |
|---|---|
| `/options-scanner:morning_briefing` | Risk state → scan → ranked summary, no execution |
| `/options-scanner:risk_audit` | OOS simulation, train-vs-test comparison, edge call |
| `/options-scanner:propose_trade` | Health → state → scan → structured proposal, no execution |
| `/options-scanner:analyze_chart` | Dual-read: geometric detection + vision second opinion, reconciled |
| `/options-scanner:bridge_to_robinhood` | Orchestrate analytics from this MCP + execution via Robinhood Agentic's MCP, in one workflow |

### Chart-pattern detection

`python/patterns.py` is a deterministic, rule-based geometric pattern detector. It finds swing pivots (a fractal zigzag) and matches classical patterns, returning for each a direction, confidence (0–1), confirmation trigger, measured-move target, and invalidation stop.

| Family | Patterns |
|---|---|
| **Reversal** | head & shoulders (+ inverse), double top/bottom, triple top/bottom |
| **Continuation** | bull/bear flags, ascending/descending/symmetrical triangles, rising/falling wedges |
| **Candlestick** | engulfing, hammer, shooting star, doji, morning/evening star |
| **Context** | support/resistance zones, Fibonacci retracements, VWAP bands |

These are *facts about the price shape*, not predictions — a detected `double_top` describes geometry, it does not guarantee a bearish resolution. Cross-reference with `scan_market` and `run_backtest` before acting.

The **LLM-vision second opinion** (`render_chart_for_vision` / `/analyze_chart`) renders the actual candlestick chart with the detections annotated (blue=trigger, green=target, red=stop) and hands it to a vision-capable model. The model forms an independent visual read and reports where it agrees or disagrees with the rule engine — catching forming/sloppy patterns, trendline breaks, and channels the geometry misses. `python/test_patterns.py` has 8 unit tests pinning the geometric detectors against hand-built synthetic patterns.

When your Robinhood Agentic access activates, register their MCP alongside this one, then invoke `/options-scanner:bridge_to_robinhood`. Claude follows a fixed recipe: our `check_health` → our `get_risk_state` → Robinhood account info → our `scan_market` → Robinhood quote → drift check → structured proposal → wait for confirm → Robinhood order tool.

---

## Setup

### Prerequisites

- **Rust** (1.75+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **uv** (Python package manager, Rust-built): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Alpaca paper account** (free): https://app.alpaca.markets/ → toggle to **Paper Trading** → "Your API Keys" panel → Generate. Keys start with **`PK`**. Trading-API keys, not Broker API (`CK…`) keys.

### First-time setup

```bash
git clone https://github.com/ibraiz11/options-scanner.git
cd options-scanner

# Python env
cd python && uv sync && cd ..

# Secrets
cp .env.example .env.local
chmod 600 .env.local
$EDITOR .env.local      # paste your ALPACA_API_KEY and ALPACA_API_SECRET

# Build
cargo build --release

# Run
./run.sh
```

Open http://localhost:8000. The health banner at the top will tell you about every misconfiguration (missing keys, broken Python env, etc.) with the concrete fix for each.

### Quick verification

```bash
# Health check
curl -sS http://localhost:8000/api/health | jq

# Scan one ticker
curl -sS 'http://localhost:8000/api/scan?tickers=AAPL'

# Smoke-test the MCP server
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","clientInfo":{"name":"t","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | ./target/release/options-scanner-mcp
```

---

## Project layout

```
options-scanner/
├── Cargo.toml                    Two binaries: HTTP server + MCP server
├── run.sh                        Sources .env.local, builds, runs
├── .env.example                  Template for ALPACA_API_KEY / ALPACA_API_SECRET
├── .gitignore                    .env*, /target, /data, editor cruft
├── src/
│   ├── main.rs                   Axum HTTP server, 9 endpoints
│   ├── broker.rs                 Broker trait, DryRunBroker, AlpacaPaper
│   ├── risk.rs                   Hard caps + risk::check() (7 unit tests)
│   ├── executor.rs               Signal → instrument → size → risk → broker
│   ├── state.rs                  JSON-backed counters, killswitch, trade log
│   ├── sim.rs                    Day-by-day MTM sim + slippage + OOS (3 unit tests)
│   ├── scheduler.rs              Unattended auto-trader, market-hours gated
│   └── bin/
│       └── mcp_server.rs         Stdio MCP server: 6 tools, 4 prompts
├── python/
│   ├── pyproject.toml            uv-managed deps (polars, yfinance)
│   ├── universe.py               Default ticker list
│   ├── scanner.py                Polars indicators + scoring
│   └── backtest.py               Walk-forward back-test
└── static/
    └── index.html                Single-page dashboard
```

---

## Tests

```bash
cargo test
```

10 unit tests:

```
risk::tests::approves_when_under_caps              ok
risk::tests::resizes_when_over_per_trade           ok
risk::tests::rejects_when_killswitch_active        ok
risk::tests::rejects_when_drawdown_exceeded        ok
risk::tests::rejects_when_daily_count_reached      ok
risk::tests::rejects_when_total_exposure_full      ok
risk::tests::rejects_when_request_absurdly_large   ok
sim::tests::zero_slippage_matches_backtest_pnl     ok
sim::tests::slippage_reduces_pnl_vs_no_slippage    ok
sim::tests::many_losers_trip_killswitch_via_mtm    ok
```

---

## Caveats

These are real and you should read them before running any of this with real money.

1. **The strategy has not been validated with live capital.** The back-test and simulator give positive expectancy on the default universe over 2y in my testing, but I haven't watched it run forward in live paper for long enough to claim it has actual edge. Run the OOS simulator yourself. If train and test diverge, don't trade it.

2. **Daily-bar discretization overstates fills.** The back-test and simulator check `target` and `stop` against daily highs/lows, which assumes you'd get filled at the exact level. Live trading has slippage, gaps, and intraday whipsaws not modeled. Add 5–15 bps of mental haircut on top of whatever the simulator says.

3. **No correlation modeling.** Eight tech-stock longs that all moved together in 2024 are treated as eight independent risks. Real correlated drawdowns will be worse than the sim shows. The 25% total exposure cap helps but isn't a substitute for actual correlation-aware sizing.

4. **The risk caps' specific numbers are heuristics.** 5% per trade and −8% killswitch in Standard mode are reasonable defaults but not derived from your specific risk tolerance. Tune them.

5. **`execute_paper_trade` is paper-only by design.** The MCP tool refuses to route orders to a live broker. Live execution should happen through Robinhood Agentic's sandboxed sub-account (or, for non-MCP workflows, through the dashboard's manual Send button which intentionally bypasses this gate so you can authorize live orders consciously).

6. **The bridge_to_robinhood prompt refers to Robinhood MCP tools generically** (e.g. "use whatever order tool the connected Robinhood MCP exposes"). Their exact tool schema isn't published yet. When you connect their MCP the LLM will see the real names and adapt; if anything reads oddly, tune the prompt text in `src/bin/mcp_server.rs`.

7. **The scheduler does not survive a graceful shutdown.** SIGTERM kills it mid-tick. Open positions persist in `data/state.json`; any in-flight HTTP requests don't. Don't ctrl-C during a trade window unless you're prepared to reconcile.

8. **This is not financial advice.** I'm an LLM agent that wrote some Rust. The math is reproducible; the judgment about whether to trade is yours.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

- Strategy primitives from standard technical-analysis literature (VWAP, ATR-Wilder, MACD, RVOL, ICT-style liquidity sweeps).
- Indicator math implemented in [Polars](https://pola.rs/) for fast Rust-backed Python.
- Built with [Axum](https://github.com/tokio-rs/axum) and the [Model Context Protocol](https://modelcontextprotocol.io/) reference implementation.
- Initial scaffolding co-authored with [Claude](https://claude.com/claude-code).
