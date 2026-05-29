"""
Unit tests for pattern detection. Each test constructs a synthetic OHLC series
containing a known pattern and asserts the detector finds it with the right
direction and sane levels. Run: uv run python -m pytest test_patterns.py -q
(or: uv run python test_patterns.py for a no-pytest fallback).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patterns import (
    find_pivots, detect_head_and_shoulders, detect_double_triple,
    detect_candlesticks, detect_support_resistance, detect_all,
)


def _ohlc(prices: list[float], vol: float = 1_000_000) -> pd.DataFrame:
    """Build a simple OHLC frame from a close path. High/low bracket each close
    individually (NOT off the open) so turning-point bars don't accidentally share
    an identical high/low with their neighbour — which would defeat the strict
    single-maximum pivot test. Candlestick-specific tests build their own frames."""
    closes = np.array(prices, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = closes * 1.002
    lows = closes * 0.998
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(len(closes), vol),
    })


def _zigzag(points: list[float], step: int = 5) -> list[float]:
    """Linearly interpolate between turning points to make a smooth path with
    detectable pivots `step` bars apart."""
    path: list[float] = []
    for i in range(len(points) - 1):
        seg = np.linspace(points[i], points[i + 1], step, endpoint=False)
        path.extend(seg.tolist())
    path.append(points[-1])
    return path


def test_pivots_alternate():
    df = _ohlc(_zigzag([100, 110, 102, 112, 104]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    kinds = [p.kind for p in pivots]
    # pivots must strictly alternate
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)), kinds
    assert len(pivots) >= 3


def test_double_top_detected():
    # two equal peaks at ~120 with a valley at ~108 between
    df = _ohlc(_zigzag([100, 120, 108, 120, 105]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    dets = detect_double_triple(df, pivots, tol_pct=2.0)
    tops = [d for d in dets if d.pattern == "double_top"]
    assert tops, f"no double_top in {[d.pattern for d in dets]}"
    d = tops[0]
    assert d.direction == "bearish"
    assert d.target < d.trigger < d.stop  # bearish: target below trigger below stop(peak)


def test_double_bottom_detected():
    df = _ohlc(_zigzag([120, 100, 112, 100, 118]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    dets = detect_double_triple(df, pivots, tol_pct=2.0)
    bottoms = [d for d in dets if d.pattern == "double_bottom"]
    assert bottoms, f"no double_bottom in {[d.pattern for d in dets]}"
    d = bottoms[0]
    assert d.direction == "bullish"
    assert d.stop < d.trigger < d.target  # bullish: stop(valley) below trigger below target


def test_head_and_shoulders_bearish():
    # LS=115, head=125, RS=115 with troughs ~108
    df = _ohlc(_zigzag([100, 115, 108, 125, 108, 115, 100]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    dets = detect_head_and_shoulders(df, pivots, tol_pct=4.0)
    hs = [d for d in dets if d.pattern == "head_and_shoulders"]
    assert hs, f"no H&S in {[(p.kind, round(p.price)) for p in pivots]}"
    d = hs[0]
    assert d.direction == "bearish"
    assert d.key_levels["head"] > d.key_levels["left_shoulder"]
    assert d.target < d.trigger  # target below neckline


def test_inverse_head_and_shoulders_bullish():
    df = _ohlc(_zigzag([120, 105, 112, 95, 112, 105, 120]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    dets = detect_head_and_shoulders(df, pivots, tol_pct=4.0)
    ihs = [d for d in dets if d.pattern == "inverse_head_and_shoulders"]
    assert ihs, f"no inverse H&S in {[(p.kind, round(p.price)) for p in pivots]}"
    assert ihs[0].direction == "bullish"
    assert ihs[0].target > ihs[0].trigger


def test_bullish_engulfing():
    # prior down bar, then a bar whose body engulfs it to the upside
    df = pd.DataFrame({
        "open":  [100, 100, 98, 95],
        "high":  [101, 100.5, 99, 101],
        "low":   [99, 97, 94, 94.5],
        "close": [100, 98, 95, 100],
        "volume": [1e6] * 4,
    })
    dets = detect_candlesticks(df, lookback=2)
    assert any(d.pattern == "bullish_engulfing" and d.direction == "bullish" for d in dets), \
        [d.pattern for d in dets]


def test_support_resistance_levels():
    # repeated touches near 100 (support) and 120 (resistance)
    df = _ohlc(_zigzag([110, 120, 100, 120, 100, 120, 108]))
    pivots = find_pivots(df, order=2, dedupe_pct=1.0)
    dets = detect_support_resistance(df, pivots, cluster_pct=1.5, min_touches=2)
    assert dets, "no S/R levels found"
    assert all(0 <= d.confidence <= 1 for d in dets)


def test_detect_all_is_json_ready():
    df = _ohlc(_zigzag([100, 120, 108, 120, 105, 115, 100]))
    out = detect_all(df, pivot_order=2)
    assert isinstance(out, list)
    for d in out:
        assert "pattern" in d and "direction" in d and "confidence" in d
        assert 0 <= d["confidence"] <= 1
        # key_levels values must be JSON-serializable floats
        for v in d["key_levels"].values():
            assert isinstance(v, float)


def _run_fallback():
    """Run tests without pytest."""
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_fallback() else 1)
