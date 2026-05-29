"""
Back-test the geometric chart patterns to find which (if any) actually have edge.

Method (forward test, no lookahead in the outcome):
  1. Detect all patterns on a symbol's full history.
  2. Each actionable detection (has trigger/target/stop) "completes" at end_idx.
  3. After end_idx, find the first bar that confirms the pattern (price crosses the
     trigger in the pattern's direction). Entry = trigger.
  4. From the confirmation bar forward, check whether target or stop is hit first
     within `horizon` bars. Conservative: if both fall in the same bar, stop wins.
  5. Aggregate by pattern type: confirmation rate, win rate, expectancy.

This tells us, e.g., "double_bottom confirmed 61% of the time and when confirmed hit
target before stop 54% of the time for +2.1% expectancy" — the number that decides
whether a pattern is worth trading. Context-only detections (support, fib, vwap,
doji) are skipped since they have no trigger/target/stop triple.

NOT financial advice. Daily-bar discretization overstates fills; treat results as
relative ranking between patterns, not absolute tradeable returns.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import yfinance as yf

from patterns import detect_all
from universe import DEFAULT_UNIVERSE


def _log(msg: str) -> None:
    print(f"[pattern_backtest] {msg}", file=sys.stderr, flush=True)


def _forward_test(df, detection: dict, horizon: int) -> dict | None:
    """Return an outcome dict for one detection, or None if not actionable."""
    trigger = detection.get("trigger")
    target = detection.get("target")
    stop = detection.get("stop")
    direction = detection["direction"]
    if trigger is None or target is None or stop is None or direction == "neutral":
        return None

    end_idx = detection["end_idx"]
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    bullish = direction == "bullish"

    # 1. find confirmation: first bar after end_idx that crosses the trigger
    conf_idx = None
    search_end = min(end_idx + 1 + horizon, n)
    for i in range(end_idx + 1, search_end):
        if bullish and highs[i] >= trigger:
            conf_idx = i
            break
        if not bullish and lows[i] <= trigger:
            conf_idx = i
            break
    if conf_idx is None:
        return {"pattern": detection["pattern"], "confirmed": False,
                "outcome": "no_confirmation", "return_pct": 0.0}

    # 2. from confirmation forward, target vs stop first
    outcome = "timeout"
    exit_price = float(df["close"].iloc[min(conf_idx + horizon, n - 1)])
    for i in range(conf_idx, min(conf_idx + 1 + horizon, n)):
        hi, lo = highs[i], lows[i]
        if bullish:
            if lo <= stop:  # conservative: stop checked first
                outcome, exit_price = "stop", stop
                break
            if hi >= target:
                outcome, exit_price = "target", target
                break
        else:
            if hi >= stop:
                outcome, exit_price = "stop", stop
                break
            if lo <= target:
                outcome, exit_price = "target", target
                break

    ret = (exit_price - trigger) / trigger * 100 * (1 if bullish else -1)
    return {"pattern": detection["pattern"], "confirmed": True,
            "outcome": outcome, "return_pct": round(float(ret), 3)}


def backtest_symbol(symbol: str, period: str, horizon: int, pivot_order: int) -> list[dict]:
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        _log(f"{symbol}: download failed: {e}")
        return []
    if raw is None or raw.empty or len(raw) < 60:
        return []
    raw = raw.reset_index()
    raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in raw.columns]

    detections = detect_all(raw, pivot_order=pivot_order)
    results = []
    for d in detections:
        r = _forward_test(raw, d, horizon)
        if r is not None:
            # carry context tags through so we can segment by them
            r["trend_aligned"] = d.get("trend_aligned")
            r["volume_confirmed"] = d.get("volume_confirmed")
            results.append(r)
    return results


def summarize(results: list[dict]) -> dict:
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_pattern[r["pattern"]].append(r)

    summary = {}
    for pat, rs in by_pattern.items():
        confirmed = [r for r in rs if r["confirmed"]]
        n_total = len(rs)
        n_conf = len(confirmed)
        if n_conf == 0:
            summary[pat] = {"detections": n_total, "confirmed": 0,
                            "confirmation_rate_pct": 0.0, "note": "never confirmed"}
            continue
        wins = [r for r in confirmed if r["outcome"] == "target"]
        losses = [r for r in confirmed if r["outcome"] == "stop"]
        avg_ret = sum(r["return_pct"] for r in confirmed) / n_conf
        win_rate = len(wins) / n_conf * 100
        summary[pat] = {
            "detections": n_total,
            "confirmed": n_conf,
            "confirmation_rate_pct": round(n_conf / n_total * 100, 1),
            "win_rate_pct": round(win_rate, 1),
            "avg_return_pct": round(avg_ret, 3),
            "target_hits": len(wins),
            "stop_hits": len(losses),
            "timeouts": n_conf - len(wins) - len(losses),
        }
    # sort patterns by expectancy
    summary = dict(sorted(summary.items(),
                          key=lambda kv: kv[1].get("avg_return_pct", -999), reverse=True))
    return summary


def run(universe: list[str], period: str, horizon: int, pivot_order: int) -> dict:
    all_results: list[dict] = []
    n_ok = 0
    for sym in universe:
        rs = backtest_symbol(sym, period, horizon, pivot_order)
        if rs:
            all_results.extend(rs)
            n_ok += 1
    overall = summarize(all_results)
    total_conf = sum(1 for r in all_results if r["confirmed"])

    # Verification: does the trend-alignment filter actually improve expectancy?
    aligned = [r for r in all_results if r.get("trend_aligned") is True]
    fighting = [r for r in all_results if r.get("trend_aligned") is False]

    def _exp(rs):
        conf = [r for r in rs if r["confirmed"]]
        if not conf:
            return {"confirmed": 0}
        wins = [r for r in conf if r["outcome"] == "target"]
        return {"confirmed": len(conf),
                "win_rate_pct": round(len(wins) / len(conf) * 100, 1),
                "avg_return_pct": round(sum(r["return_pct"] for r in conf) / len(conf), 3)}

    return {
        "params": {"period": period, "horizon_bars": horizon, "pivot_order": pivot_order,
                   "symbols_tested": n_ok, "universe_size": len(universe)},
        "total_detections": len(all_results),
        "total_confirmed": total_conf,
        "trend_filter_verification": {
            "trend_aligned": _exp(aligned),
            "fighting_trend": _exp(fighting),
        },
        "by_pattern": overall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2y", choices=["1y", "2y", "5y"])
    ap.add_argument("--horizon", type=int, default=20, help="bars to hold after confirmation")
    ap.add_argument("--pivot-order", type=int, default=3)
    ap.add_argument("--tickers", default=None, help="comma list; default = full universe")
    args = ap.parse_args()
    universe = ([t.strip().upper() for t in args.tickers.split(",")]
                if args.tickers else DEFAULT_UNIVERSE)
    result = run(universe, args.period, args.horizon, args.pivot_order)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
