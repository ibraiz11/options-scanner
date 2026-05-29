"""
Render a candlestick chart to PNG for the LLM-vision second-opinion layer.

The geometric detector in patterns.py is precise but literal — it only finds the
shapes it has rules for. A vision-capable LLM looking at the actual chart can catch
context the rules miss (a sloppy head-and-shoulders, a trendline break, an obvious
channel) and, just as usefully, can DISAGREE with a geometric detection that looks
wrong to a human eye.

This script renders the chart (optionally annotating the geometric detections as
horizontal trigger/target/stop lines) and emits a base64 PNG plus the detection
list as JSON. The MCP layer wraps the PNG as image content so the model can see it.

stdout = JSON only. Diagnostics → stderr.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import mplfinance as mpf
import pandas as pd
import yfinance as yf

from patterns import detect_all


def _log(msg: str) -> None:
    print(f"[chart_render] {msg}", file=sys.stderr, flush=True)


def render(symbol: str, period: str, annotate: bool) -> dict:
    raw = yf.download(symbol, period=period, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return {"error": f"no data for {symbol}", "symbol": symbol}

    raw = raw.reset_index()
    raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in raw.columns]

    detections = detect_all(raw, pivot_order=3) if annotate else []

    # The datetime column may be named 'date', 'datetime', or 'index' depending on
    # yfinance/pandas version and whether the source index was named. Find it.
    date_col = next((c for c in ("date", "datetime", "index") if c in raw.columns), None)
    if date_col is None:
        return {"error": f"could not locate date column in {list(raw.columns)}", "symbol": symbol}

    # mplfinance wants a DatetimeIndex and capitalized OHLC columns.
    plot_df = raw.rename(columns={
        date_col: "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    }).set_index("Date")
    plot_df.index = pd.to_datetime(plot_df.index)

    # Collect horizontal lines from the highest-confidence actionable detections.
    hlines, colors = [], []
    for d in detections[:6]:
        for key in ("trigger", "target", "stop"):
            v = d.get(key)
            if v is not None:
                hlines.append(v)
                colors.append({"trigger": "#6ea8fe", "target": "#22c55e", "stop": "#ef4444"}[key])

    addplot_kwargs = {}
    if hlines:
        addplot_kwargs["hlines"] = dict(hlines=hlines, colors=colors, linewidths=0.8, alpha=0.6)

    buf = io.BytesIO()
    mpf.plot(
        plot_df, type="candle", volume=True, style="nightclouds",
        title=f"\n{symbol}  ({period})",
        figsize=(12, 7), savefig=dict(fname=buf, dpi=110, format="png"),
        **addplot_kwargs,
    )
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {
        "symbol": symbol,
        "period": period,
        "png_base64": png_b64,
        "annotated_detections": detections[:6],
        "legend": {"trigger": "blue", "target": "green", "stop": "red"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--period", default="6mo", choices=["3mo", "6mo", "1y", "2y"])
    ap.add_argument("--annotate", default="true", choices=["true", "false"])
    args = ap.parse_args()
    try:
        out = render(args.symbol.strip().upper(), args.period, args.annotate == "true")
    except Exception as e:
        _log(f"render failed: {e}")
        out = {"error": f"render failed: {e}", "symbol": args.symbol}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
