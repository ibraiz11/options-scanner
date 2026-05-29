"""
Walk-forward back-test for the scanner. Pure Polars on the analytics path.
For every historical day, evaluate() runs on the slice of data available at that point;
forward bars are then walked to see whether stop / target_10 / target_20 hit first.
"""
from __future__ import annotations

import argparse
import json
import sys

import polars as pl

from scanner import evaluate, fetch
from universe import DEFAULT_UNIVERSE


def _trade_record(cand: dict, df: pl.DataFrame, entry_idx: int, bars_held: int,
                  exit_price: float, outcome: str, bars: list[dict] | None = None) -> dict:
    bull = cand["direction"] == "bullish"
    ret = (exit_price - cand["entry"]) / cand["entry"] * 100 * (1 if bull else -1)
    exit_idx = min(entry_idx + bars_held, df.height - 1)
    return {
        "symbol": cand["symbol"],
        "direction": cand["direction"],
        "entry_date": str(df["date"][entry_idx]),
        "entry": cand["entry"],
        "stop": cand["stop"],
        "target_10": cand["target_10"],
        "target_20": cand["target_20"],
        "score": cand["score"],
        "outcome": outcome,
        "exit_date": str(df["date"][exit_idx]),
        "exit_price": float(exit_price),
        "bars_held": bars_held,
        "return_pct": round(float(ret), 3),
        # Per-day OHLC for the holding period — consumed by the Rust simulator
        # for mark-to-market exposure tracking and unrealized-drawdown killswitch.
        "bars": bars or [],
    }


def simulate_forward(df: pl.DataFrame, idx: int, cand: dict, hold_bars: int) -> dict | None:
    n = df.height
    if idx + 1 >= n:
        return None
    end = min(idx + 1 + hold_bars, n)
    forward = df.slice(idx + 1, end - idx - 1)
    bull = cand["direction"] == "bullish"

    bars: list[dict] = []
    for j, row in enumerate(forward.iter_rows(named=True), start=1):
        bars.append({
            "date": str(row["date"]),
            "close": float(row["close"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
        })
        hi, lo = row["high"], row["low"]
        if bull:
            stop_hit = lo <= cand["stop"]
            t20_hit = hi >= cand["target_20"]
        else:
            stop_hit = hi >= cand["stop"]
            t20_hit = lo <= cand["target_20"]
        # Conservative: if stop and target both hit in same bar, assume stop first.
        if stop_hit:
            return _trade_record(cand, df, idx, j, cand["stop"], "stop", bars)
        if t20_hit:
            return _trade_record(cand, df, idx, j, cand["target_20"], "target_20", bars)

    last = forward.row(-1, named=True)
    return _trade_record(cand, df, idx, forward.height, float(last["close"]), "timeout", bars)


def backtest_symbol(symbol: str, df: pl.DataFrame, min_score: float, min_relvol: float,
                    hold_bars: int, cooldown_bars: int = 5) -> list[dict]:
    if df is None or df.height < 80:
        return []
    trades: list[dict] = []
    last_signal_idx = -10**9
    for i in range(60, df.height - 1):
        if i - last_signal_idx < cooldown_bars:
            continue
        window = df.slice(0, i + 1)
        cand = evaluate(symbol, window, min_relvol)
        if not cand or cand["score"] < min_score:
            continue
        trade = simulate_forward(df, i, cand, hold_bars)
        if trade:
            trades.append(trade)
            last_signal_idx = i
    return trades


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    df = pl.DataFrame(trades)
    n = df.height
    wins = df.filter(pl.col("return_pct") > 0)
    losses = df.filter(pl.col("return_pct") <= 0)
    avg_win = float(wins["return_pct"].mean()) if wins.height else 0.0
    avg_loss = float(losses["return_pct"].mean()) if losses.height else 0.0
    win_rate = wins.height / n
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    by_outcome = (
        df.group_by("outcome").agg(pl.len().alias("count")).to_dict(as_series=False)
    )
    outcomes = dict(zip(by_outcome["outcome"], by_outcome["count"], strict=False))

    by_dir: dict[str, dict] = {}
    for d, g in df.group_by("direction"):
        key = d[0] if isinstance(d, tuple) else d
        by_dir[str(key)] = {
            "trades": g.height,
            "win_rate_pct": round(float((g["return_pct"] > 0).mean()) * 100, 2),
            "avg_return_pct": round(float(g["return_pct"].mean()), 3),
        }

    return {
        "trades": n,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_return_pct": round(float(df["return_pct"].mean()), 3),
        "expectancy_pct_per_trade": round(expectancy, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "median_bars_held": int(df["bars_held"].median()),
        "outcomes": {
            "target_20": int(outcomes.get("target_20", 0)),
            "stop": int(outcomes.get("stop", 0)),
            "timeout": int(outcomes.get("timeout", 0)),
        },
        "by_direction": by_dir,
    }


def run_backtest(universe: list[str], period: str, min_score: float,
                 min_relvol: float, hold_bars: int) -> dict:
    data = fetch(universe, period=period)
    all_trades: list[dict] = []
    per_symbol: dict[str, dict] = {}
    for sym, df in data.items():
        trades = backtest_symbol(sym, df, min_score, min_relvol, hold_bars)
        if trades:
            all_trades.extend(trades)
            per_symbol[sym] = summarize(trades)

    return {
        "params": {
            "period": period,
            "min_score": min_score,
            "min_relvol": min_relvol,
            "hold_bars": hold_bars,
            "universe_size": len(universe),
        },
        "overall": summarize(all_trades),
        "per_symbol": per_symbol,
        "trades": all_trades[-200:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=3.0)
    ap.add_argument("--min-relvol", type=float, default=1.3)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--period", default="2y", choices=["1y", "2y", "5y", "10y"])
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    universe = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers else DEFAULT_UNIVERSE
    )
    result = run_backtest(
        universe=universe,
        period=args.period,
        min_score=args.min_score,
        min_relvol=args.min_relvol,
        hold_bars=args.hold_bars,
    )
    json.dump(result, sys.stdout, default=str)


if __name__ == "__main__":
    main()
