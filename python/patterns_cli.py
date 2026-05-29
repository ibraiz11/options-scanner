"""
CLI wrapper around patterns.detect_all for the Rust HTTP server.
Fetches daily bars via yfinance, runs detection, prints JSON to stdout.
Diagnostics go to stderr (stdout must stay pure JSON for the Rust parser).
"""
from __future__ import annotations

import argparse
import json
import sys

import yfinance as yf

from patterns import detect_all


def _log(msg: str) -> None:
    print(f"[patterns] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--period", default="6mo", choices=["3mo", "6mo", "1y", "2y"])
    ap.add_argument("--pivot-order", type=int, default=3)
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()
    try:
        raw = yf.download(symbol, period=args.period, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        _log(f"download failed: {e}")
        json.dump({"error": f"download failed for {symbol}: {e}", "symbol": symbol,
                   "count": 0, "detections": []}, sys.stdout)
        return

    if raw is None or raw.empty:
        json.dump({"error": f"no data for {symbol} (delisted or bad symbol?)",
                   "symbol": symbol, "count": 0, "detections": []}, sys.stdout)
        return

    raw = raw.reset_index()
    raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in raw.columns]

    try:
        detections = detect_all(raw, pivot_order=args.pivot_order)
    except Exception as e:
        _log(f"detection error: {e}")
        json.dump({"error": f"detection failed: {e}", "symbol": symbol,
                   "count": 0, "detections": []}, sys.stdout)
        return

    json.dump({"symbol": symbol, "count": len(detections), "detections": detections},
              sys.stdout)


if __name__ == "__main__":
    main()
