"""
Polars-based daily options play scanner.

Indicators: rolling VWAP, ATR(14), MACD(12,26,9), relative volume, liquidity sweep.
Targets: ±10% and ±20% from entry, ATR-based stop.

NOT FINANCIAL ADVICE. Educational scaffold. Paper-trade and back-test before risking capital.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

import pandas as pd
import polars as pl
import yfinance as yf

from universe import DEFAULT_UNIVERSE

Direction = Literal["bullish", "bearish"]


# ---- indicators ----------------------------------------------------------

def with_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical")
    )
    df = df.with_columns([
        (pl.col("typical") * pl.col("volume")).rolling_sum(window_size=20).alias("pv_sum"),
        pl.col("volume").rolling_sum(window_size=20).alias("v_sum"),
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias("rvol"),
    ])
    df = df.with_columns(
        (pl.col("pv_sum") / pl.col("v_sum")).alias("vwap")
    )
    # ATR via Wilder's smoothing
    df = df.with_columns(
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]).alias("tr")
    )
    df = df.with_columns(
        pl.col("tr").ewm_mean(alpha=1 / 14, adjust=False).alias("atr")
    )
    # MACD
    df = df.with_columns([
        pl.col("close").ewm_mean(span=12, adjust=False).alias("ema_fast"),
        pl.col("close").ewm_mean(span=26, adjust=False).alias("ema_slow"),
    ])
    df = df.with_columns(
        (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")
    )
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=9, adjust=False).alias("macd_sig")
    )
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_sig")).alias("macd_hist")
    )
    return df


def liquidity_sweep(df: pl.DataFrame, lookback: int = 20) -> tuple[bool, bool]:
    """
    Bullish sweep: today's low < prior `lookback` low, but close > prior low (took stops, reclaimed).
    Bearish sweep: today's high > prior `lookback` high, but close < prior high.
    """
    if df.height < lookback + 1:
        return False, False
    prior = df.slice(df.height - lookback - 1, lookback)
    today = df.row(-1, named=True)
    prior_low = prior["low"].min()
    prior_high = prior["high"].max()
    bull = today["low"] < prior_low and today["close"] > prior_low
    bear = today["high"] > prior_high and today["close"] < prior_high
    return bool(bull), bool(bear)


# ---- evaluation ----------------------------------------------------------

def evaluate(symbol: str, df: pl.DataFrame, min_relvol: float = 1.3) -> dict | None:
    if df is None or df.height < 60:
        return None

    df = with_indicators(df)
    last = df.row(-1, named=True)
    prev = df.row(-2, named=True)

    required = ("vwap", "atr", "rvol", "macd_hist", "macd_line", "macd_sig")
    if any(last[k] is None for k in required):
        return None

    bull_sweep, bear_sweep = liquidity_sweep(df, 20)

    notes: list[str] = []
    bull_score = 0.0
    bear_score = 0.0

    # 1. VWAP
    if last["close"] > last["vwap"]:
        bull_score += 1
        notes.append("price>VWAP")
    elif last["close"] < last["vwap"]:
        bear_score += 1
        notes.append("price<VWAP")

    # 2. MACD direction + histogram momentum
    if last["macd_line"] > last["macd_sig"] and last["macd_hist"] > prev["macd_hist"]:
        bull_score += 1.5
        notes.append("MACD bull")
    if last["macd_line"] < last["macd_sig"] and last["macd_hist"] < prev["macd_hist"]:
        bear_score += 1.5
        notes.append("MACD bear")

    # 3. Relative volume gate
    rvol = float(last["rvol"])
    if rvol >= min_relvol:
        if last["close"] > last["open"]:
            bull_score += 1
        else:
            bear_score += 1
        notes.append(f"rvol={rvol:.2f}")

    # 4. Liquidity sweep
    if bull_sweep:
        bull_score += 2
        notes.append("bull sweep")
    if bear_sweep:
        bear_score += 2
        notes.append("bear sweep")

    # 5. ATR sanity — too quiet for a 10–20% move
    atr = float(last["atr"])
    close = float(last["close"])
    if atr / close * 100 < 1.0:
        return None

    if bull_score < 3 and bear_score < 3:
        return None

    if bull_score >= bear_score:
        direction: Direction = "bullish"
        score = bull_score
        entry = close
        stop = entry - 2 * atr
        t10 = entry * 1.10
        t20 = entry * 1.20
    else:
        direction = "bearish"
        score = bear_score
        entry = close
        stop = entry + 2 * atr
        t10 = entry * 0.90
        t20 = entry * 0.80

    return {
        "symbol": symbol,
        "direction": direction,
        "score": round(float(score), 2),
        "entry": round(entry, 2),
        "stop": round(float(stop), 2),
        "target_10": round(float(t10), 2),
        "target_20": round(float(t20), 2),
        "atr": round(atr, 4),
        "rel_volume": round(rvol, 2),
        "macd_hist": round(float(last["macd_hist"]), 4),
        "vwap": round(float(last["vwap"]), 2),
        "sweep": bool(bull_sweep or bear_sweep),
        "notes": ", ".join(notes),
    }


# ---- data fetch ----------------------------------------------------------

def _log(msg: str) -> None:
    """Diagnostic output — stderr only. Stdout is reserved for the JSON the Rust
    server parses; any print() to stdout from this script will break that."""
    print(f"[scanner] {msg}", file=sys.stderr, flush=True)


def fetch(universe: list[str], period: str = "6mo") -> dict[str, pl.DataFrame]:
    """Download daily bars for every symbol in `universe`. Robust to per-ticker
    failures (delistings, transient Yahoo DNS errors, cache races) — bad tickers
    are skipped with a stderr warning rather than crashing the scan.

    threads=False is deliberate: yfinance's SQLite cache races under
    threads=True and produces 'unable to open database file' on ~20+ tickers.
    Sequential is slower but reliable."""
    try:
        raw = yf.download(
            tickers=universe,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        _log(f"yf.download failed entirely: {type(e).__name__}: {e}")
        return {}

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        _log("yf.download returned no data")
        return {}

    out: dict[str, pl.DataFrame] = {}
    skipped: list[tuple[str, str]] = []

    for sym in universe:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Batch download returns columns indexed by (ticker, field).
                if sym not in raw.columns.get_level_values(0):
                    skipped.append((sym, "no_data"))
                    continue
                df_pd = raw[sym]
            else:
                # Single-ticker download returns a flat frame.
                df_pd = raw

            df_pd = df_pd.dropna().reset_index()
            if df_pd.empty or len(df_pd) < 10:
                skipped.append((sym, "too_few_bars"))
                continue

            df_pd.columns = [str(c).lower() for c in df_pd.columns]
            out[sym] = pl.from_pandas(df_pd)
        except Exception as e:
            # Per-ticker failure (parse error, polars conversion glitch) — log and continue.
            skipped.append((sym, type(e).__name__))

    if skipped:
        head = ", ".join(f"{s}({r})" for s, r in skipped[:8])
        more = f" + {len(skipped) - 8} more" if len(skipped) > 8 else ""
        _log(f"skipped {len(skipped)} of {len(universe)} tickers: {head}{more}")
    if not out:
        _log("WARNING: zero tickers loaded successfully. Check network and that the universe contains valid symbols.")

    return out


# ---- CLI -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=3.0)
    ap.add_argument("--min-relvol", type=float, default=1.3)
    ap.add_argument("--direction", default="both", choices=["both", "bullish", "bearish"])
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    universe = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers else DEFAULT_UNIVERSE
    )
    data = fetch(universe)

    results: list[dict] = []
    for sym, df in data.items():
        cand = evaluate(sym, df, args.min_relvol)
        if cand and cand["score"] >= args.min_score:
            if args.direction == "both" or cand["direction"] == args.direction:
                results.append(cand)

    results.sort(key=lambda x: x["score"], reverse=True)
    json.dump({"count": len(results), "results": results}, sys.stdout)


if __name__ == "__main__":
    main()
