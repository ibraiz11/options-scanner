"""
Rule-based geometric chart-pattern detection.

Detects classical technical-analysis patterns on OHLC data and returns, for each,
an actionable trade structure: direction, entry/trigger, measured-move target,
invalidation stop, and a confidence score. Deterministic and unit-testable — no
ML, no training data. The LLM consumes these detections as structured facts; it
does not invent the geometry.

Families:
  - Reversal:      head & shoulders (+ inverse), double/triple top & bottom
  - Continuation:  flags, pennants, triangles (asc/desc/sym), wedges
  - Candlestick:   engulfing, hammer, shooting star, doji, morning/evening star
  - Context:       support/resistance zones, Fibonacci retracements, VWAP bands

Design notes
------------
Every detector works off a shared "swing pivot" series (the zigzag of local
highs/lows). Pivots are found with a fractal window: a bar is a pivot-high if its
high is the maximum within `order` bars on each side. A percentage de-noise filter
then drops pivots that don't represent a meaningful reversal.

Each detection is a dict with a common schema so the HTTP/MCP layers can pass them
through unchanged. `target` uses the classical measured-move projection; `stop`
uses the pattern's invalidation point.

NOT financial advice. Pattern detection ≠ edge. Validate with the back-test/sim
before trading any of these.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["bullish", "bearish", "neutral"]


# ---------------------------------------------------------------------------
# Detection schema
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    pattern: str
    family: str                       # "reversal" | "continuation" | "candlestick" | "context"
    direction: Direction
    confidence: float                 # 0..1
    start_idx: int
    end_idx: int
    trigger: float | None = None      # price that confirms the pattern (breakout/neckline)
    target: float | None = None       # measured-move objective
    stop: float | None = None         # invalidation level
    key_levels: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("confidence", "trigger", "target", "stop"):
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4)
        d["key_levels"] = {k: round(float(v), 4) for k, v in d["key_levels"].items()}
        return d


# ---------------------------------------------------------------------------
# Swing pivot detection (the zigzag every pattern builds on)
# ---------------------------------------------------------------------------

@dataclass
class Pivot:
    idx: int
    price: float
    kind: Literal["high", "low"]


def find_pivots(df: pd.DataFrame, order: int = 3, dedupe_pct: float = 1.5) -> list[Pivot]:
    """
    Fractal pivot detection. A bar i is a pivot-high if df.high[i] is the strict
    maximum of the window [i-order, i+order]; mirror for pivot-low. The dedupe pass
    removes consecutive same-kind pivots and pivots whose move from the previous
    pivot is smaller than `dedupe_pct` percent (noise).
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    raw: list[Pivot] = []

    for i in range(order, n - order):
        win_hi = highs[i - order : i + order + 1]
        win_lo = lows[i - order : i + order + 1]
        if highs[i] == win_hi.max() and (win_hi == highs[i]).sum() == 1:
            raw.append(Pivot(i, float(highs[i]), "high"))
        elif lows[i] == win_lo.min() and (win_lo == lows[i]).sum() == 1:
            raw.append(Pivot(i, float(lows[i]), "low"))

    if not raw:
        return []

    # Enforce alternating high/low, keeping the more extreme of consecutive same-kind
    # pivots, and drop moves smaller than dedupe_pct.
    cleaned: list[Pivot] = [raw[0]]
    for p in raw[1:]:
        last = cleaned[-1]
        if p.kind == last.kind:
            # keep the more extreme one
            if (p.kind == "high" and p.price > last.price) or (
                p.kind == "low" and p.price < last.price
            ):
                cleaned[-1] = p
        else:
            move_pct = abs(p.price - last.price) / last.price * 100
            if move_pct >= dedupe_pct:
                cleaned.append(p)
    return cleaned


def _pct_close(a: float, b: float, tol_pct: float) -> bool:
    """True if a and b are within tol_pct percent of each other."""
    return abs(a - b) / ((a + b) / 2) * 100 <= tol_pct


# ---------------------------------------------------------------------------
# Reversal patterns
# ---------------------------------------------------------------------------

def detect_head_and_shoulders(df: pd.DataFrame, pivots: list[Pivot],
                              tol_pct: float = 3.0) -> list[Detection]:
    """
    Head & shoulders (bearish): high(LS) < high(Head) > high(RS), shoulders roughly
    equal, two intervening lows forming the neckline. Inverse H&S (bullish) is the
    mirror on lows. Confirmation = close beyond the neckline. Target = neckline -/+
    (head - neckline).
    """
    out: list[Detection] = []
    highs = [p for p in pivots]

    # Scan windows of 5 alternating pivots: shoulder-trough-head-trough-shoulder
    for i in range(len(pivots) - 4):
        seq = pivots[i : i + 5]
        kinds = [p.kind for p in seq]

        # Bearish H&S: high low high low high, middle high is the head
        if kinds == ["high", "low", "high", "low", "high"]:
            ls, t1, head, t2, rs = seq
            if head.price > ls.price and head.price > rs.price and _pct_close(ls.price, rs.price, tol_pct):
                neckline = (t1.price + t2.price) / 2
                height = head.price - neckline
                conf = _hs_confidence(ls.price, head.price, rs.price, t1.price, t2.price, tol_pct)
                out.append(Detection(
                    pattern="head_and_shoulders", family="reversal", direction="bearish",
                    confidence=conf, start_idx=ls.idx, end_idx=rs.idx,
                    trigger=neckline, target=neckline - height, stop=head.price,
                    key_levels={"left_shoulder": ls.price, "head": head.price,
                                "right_shoulder": rs.price, "neckline": neckline},
                    notes="Bearish reversal; confirm on close below neckline.",
                ))

        # Inverse H&S (bullish): low high low high low, middle low is the head
        if kinds == ["low", "high", "low", "high", "low"]:
            ls, t1, head, t2, rs = seq
            if head.price < ls.price and head.price < rs.price and _pct_close(ls.price, rs.price, tol_pct):
                neckline = (t1.price + t2.price) / 2
                height = neckline - head.price
                conf = _hs_confidence(ls.price, head.price, rs.price, t1.price, t2.price, tol_pct)
                out.append(Detection(
                    pattern="inverse_head_and_shoulders", family="reversal", direction="bullish",
                    confidence=conf, start_idx=ls.idx, end_idx=rs.idx,
                    trigger=neckline, target=neckline + height, stop=head.price,
                    key_levels={"left_shoulder": ls.price, "head": head.price,
                                "right_shoulder": rs.price, "neckline": neckline},
                    notes="Bullish reversal; confirm on close above neckline.",
                ))
    return out


def _hs_confidence(ls, head, rs, t1, t2, tol_pct) -> float:
    """Higher when shoulders are symmetric, troughs are level, and head is prominent."""
    shoulder_sym = 1 - min(abs(ls - rs) / ((ls + rs) / 2), tol_pct / 100) / (tol_pct / 100)
    trough_level = 1 - min(abs(t1 - t2) / ((t1 + t2) / 2), tol_pct / 100) / (tol_pct / 100)
    head_prom = min(abs(head - (ls + rs) / 2) / ((ls + rs) / 2) / 0.05, 1.0)
    return round(float(0.4 * shoulder_sym + 0.3 * trough_level + 0.3 * head_prom), 3)


def detect_double_triple(df: pd.DataFrame, pivots: list[Pivot],
                         tol_pct: float = 2.0) -> list[Detection]:
    """
    Double/triple top (bearish) and bottom (bullish). Two (or three) swing highs
    (lows) at approximately the same level, separated by a counter pivot. Trigger =
    break of the intervening pivot; target = measured move of the pattern height.
    """
    out: list[Detection] = []

    # Double top: high low high (two equal highs)
    for i in range(len(pivots) - 2):
        a, mid, b = pivots[i], pivots[i + 1], pivots[i + 2]
        if [a.kind, mid.kind, b.kind] == ["high", "low", "high"] and _pct_close(a.price, b.price, tol_pct):
            height = (a.price + b.price) / 2 - mid.price
            out.append(Detection(
                pattern="double_top", family="reversal", direction="bearish",
                confidence=round(1 - abs(a.price - b.price) / a.price / (tol_pct / 100), 3),
                start_idx=a.idx, end_idx=b.idx,
                trigger=mid.price, target=mid.price - height, stop=max(a.price, b.price),
                key_levels={"top1": a.price, "top2": b.price, "valley": mid.price},
                notes="Bearish; confirm on close below the valley.",
            ))
        if [a.kind, mid.kind, b.kind] == ["low", "high", "low"] and _pct_close(a.price, b.price, tol_pct):
            height = mid.price - (a.price + b.price) / 2
            out.append(Detection(
                pattern="double_bottom", family="reversal", direction="bullish",
                confidence=round(1 - abs(a.price - b.price) / a.price / (tol_pct / 100), 3),
                start_idx=a.idx, end_idx=b.idx,
                trigger=mid.price, target=mid.price + height, stop=min(a.price, b.price),
                key_levels={"bottom1": a.price, "bottom2": b.price, "peak": mid.price},
                notes="Bullish; confirm on close above the peak.",
            ))

    # Triple top/bottom: 5 pivots, three equal extremes
    for i in range(len(pivots) - 4):
        seq = pivots[i : i + 5]
        kinds = [p.kind for p in seq]
        if kinds == ["high", "low", "high", "low", "high"]:
            h1, _, h2, mid2, h3 = seq
            if _pct_close(h1.price, h2.price, tol_pct) and _pct_close(h2.price, h3.price, tol_pct):
                support = min(seq[1].price, seq[3].price)
                height = (h1.price + h2.price + h3.price) / 3 - support
                out.append(Detection(
                    pattern="triple_top", family="reversal", direction="bearish",
                    confidence=0.7, start_idx=h1.idx, end_idx=h3.idx,
                    trigger=support, target=support - height, stop=max(h1.price, h2.price, h3.price),
                    key_levels={"top1": h1.price, "top2": h2.price, "top3": h3.price, "support": support},
                    notes="Bearish; strong resistance rejected three times.",
                ))
        if kinds == ["low", "high", "low", "high", "low"]:
            l1, _, l2, _, l3 = seq
            if _pct_close(l1.price, l2.price, tol_pct) and _pct_close(l2.price, l3.price, tol_pct):
                resistance = max(seq[1].price, seq[3].price)
                height = resistance - (l1.price + l2.price + l3.price) / 3
                out.append(Detection(
                    pattern="triple_bottom", family="reversal", direction="bullish",
                    confidence=0.7, start_idx=l1.idx, end_idx=l3.idx,
                    trigger=resistance, target=resistance + height, stop=min(l1.price, l2.price, l3.price),
                    key_levels={"bottom1": l1.price, "bottom2": l2.price, "bottom3": l3.price, "resistance": resistance},
                    notes="Bullish; strong support held three times.",
                ))
    return out


# ---------------------------------------------------------------------------
# Continuation patterns
# ---------------------------------------------------------------------------

def _fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Least-squares slope, intercept."""
    if len(xs) < 2:
        return 0.0, float(ys[0]) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def detect_triangles_wedges(df: pd.DataFrame, pivots: list[Pivot],
                            min_pivots: int = 4) -> list[Detection]:
    """
    Fit trendlines to recent swing highs and swing lows, then classify by the slope
    pair:
      ascending triangle:  flat highs, rising lows   (bullish)
      descending triangle: falling highs, flat lows  (bearish)
      symmetrical triangle: highs down, lows up       (neutral → trade the break)
      rising wedge:  both up, converging              (bearish)
      falling wedge: both down, converging            (bullish)
    """
    out: list[Detection] = []
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    if len(highs) < 2 or len(lows) < 2 or len(pivots) < min_pivots:
        return out

    # Use the most recent run of pivots
    hi_x = np.array([p.idx for p in highs[-3:]], dtype=float)
    hi_y = np.array([p.price for p in highs[-3:]], dtype=float)
    lo_x = np.array([p.idx for p in lows[-3:]], dtype=float)
    lo_y = np.array([p.price for p in lows[-3:]], dtype=float)

    hi_slope, _ = _fit_line(hi_x, hi_y)
    lo_slope, _ = _fit_line(lo_x, lo_y)

    # Normalise slopes to "per-bar % of price" so thresholds are scale-free
    price = float(df["close"].iloc[-1])
    hi_s = hi_slope / price * 100
    lo_s = lo_slope / price * 100
    flat = 0.05  # within ±0.05%/bar counts as flat

    start = min(highs[-3].idx if len(highs) >= 3 else highs[0].idx,
                lows[-3].idx if len(lows) >= 3 else lows[0].idx)
    end = len(df) - 1
    apex_top = hi_y[-1]
    apex_bot = lo_y[-1]
    height = float(hi_y.max() - lo_y.min())

    def emit(pattern, direction, conf, trigger, target, stop, note):
        out.append(Detection(
            pattern=pattern, family="continuation", direction=direction, confidence=conf,
            start_idx=int(start), end_idx=int(end), trigger=trigger, target=target, stop=stop,
            key_levels={"upper": float(apex_top), "lower": float(apex_bot)}, notes=note,
        ))

    if abs(hi_s) < flat and lo_s > flat:
        emit("ascending_triangle", "bullish", 0.6, apex_top, apex_top + height, apex_bot,
             "Bullish continuation; buy break above flat resistance.")
    elif hi_s < -flat and abs(lo_s) < flat:
        emit("descending_triangle", "bearish", 0.6, apex_bot, apex_bot - height, apex_top,
             "Bearish continuation; sell break below flat support.")
    elif hi_s < -flat and lo_s > flat:
        emit("symmetrical_triangle", "neutral", 0.5, apex_top, apex_top + height, apex_bot,
             "Neutral; trade the breakout direction with volume confirmation.")
    elif hi_s > flat and lo_s > flat and hi_s < lo_s:
        emit("rising_wedge", "bearish", 0.55, apex_bot, apex_bot - height, apex_top,
             "Bearish; rising wedge typically resolves downward.")
    elif hi_s < -flat and lo_s < -flat and lo_s > hi_s:
        emit("falling_wedge", "bullish", 0.55, apex_top, apex_top + height, apex_bot,
             "Bullish; falling wedge typically resolves upward.")
    return out


def detect_flags(df: pd.DataFrame, pole_bars: int = 8, flag_bars: int = 8,
                 min_pole_pct: float = 8.0) -> list[Detection]:
    """
    Flag / pennant: a sharp directional "pole" followed by a shallow counter-trend
    consolidation. Continuation in the pole's direction. Target = breakout +/- pole
    height (measured move).
    """
    out: list[Detection] = []
    n = len(df)
    if n < pole_bars + flag_bars + 1:
        return out

    close = df["close"].to_numpy()
    flag_start = n - flag_bars
    pole_start = flag_start - pole_bars
    pole_move_pct = (close[flag_start] - close[pole_start]) / close[pole_start] * 100
    pole_height = abs(close[flag_start] - close[pole_start])

    flag_slice = df.iloc[flag_start:]
    flag_range_pct = (flag_slice["high"].max() - flag_slice["low"].min()) / close[flag_start] * 100

    # Flag consolidation should be much tighter than the pole.
    if abs(pole_move_pct) < min_pole_pct or flag_range_pct > abs(pole_move_pct) * 0.6:
        return out

    if pole_move_pct > 0:
        trigger = float(flag_slice["high"].max())
        out.append(Detection(
            pattern="bull_flag", family="continuation", direction="bullish", confidence=0.55,
            start_idx=int(pole_start), end_idx=n - 1, trigger=trigger,
            target=trigger + pole_height, stop=float(flag_slice["low"].min()),
            key_levels={"pole_start": float(close[pole_start]), "pole_top": float(close[flag_start])},
            notes="Bullish continuation; buy break above flag high.",
        ))
    else:
        trigger = float(flag_slice["low"].min())
        out.append(Detection(
            pattern="bear_flag", family="continuation", direction="bearish", confidence=0.55,
            start_idx=int(pole_start), end_idx=n - 1, trigger=trigger,
            target=trigger - pole_height, stop=float(flag_slice["high"].max()),
            key_levels={"pole_start": float(close[pole_start]), "pole_bottom": float(close[flag_start])},
            notes="Bearish continuation; sell break below flag low.",
        ))
    return out


# ---------------------------------------------------------------------------
# Candlestick patterns (last-bar / few-bar timing signals)
# ---------------------------------------------------------------------------

def detect_candlesticks(df: pd.DataFrame, lookback: int = 3) -> list[Detection]:
    """
    Single/few-bar candlestick reversal signals on the most recent bars. These are
    timing signals — best used to confirm entry within a larger structural pattern,
    not as standalone triggers.
    """
    out: list[Detection] = []
    n = len(df)
    if n < 3:
        return out

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    def body(i): return abs(c[i] - o[i])
    def rng(i): return max(h[i] - l[i], 1e-9)
    def upper_wick(i): return h[i] - max(o[i], c[i])
    def lower_wick(i): return min(o[i], c[i]) - l[i]

    for i in range(max(2, n - lookback), n):
        b, r = body(i), rng(i)

        # Doji: tiny body relative to range
        if b <= 0.1 * r:
            out.append(Detection(
                pattern="doji", family="candlestick", direction="neutral", confidence=0.4,
                start_idx=i, end_idx=i, key_levels={"close": float(c[i])},
                notes="Indecision; potential reversal when at a structural level.",
            ))
        # Hammer: small body near top, long lower wick, little upper wick (bullish)
        if lower_wick(i) >= 2 * b and upper_wick(i) <= 0.3 * b and b > 0:
            out.append(Detection(
                pattern="hammer", family="candlestick", direction="bullish", confidence=0.5,
                start_idx=i, end_idx=i, stop=float(l[i]), key_levels={"low": float(l[i])},
                notes="Bullish reversal candle; stronger at support.",
            ))
        # Shooting star: small body near bottom, long upper wick (bearish)
        if upper_wick(i) >= 2 * b and lower_wick(i) <= 0.3 * b and b > 0:
            out.append(Detection(
                pattern="shooting_star", family="candlestick", direction="bearish", confidence=0.5,
                start_idx=i, end_idx=i, stop=float(h[i]), key_levels={"high": float(h[i])},
                notes="Bearish reversal candle; stronger at resistance.",
            ))
        # Bullish engulfing: prior red, current green body engulfs prior body
        if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
            out.append(Detection(
                pattern="bullish_engulfing", family="candlestick", direction="bullish", confidence=0.55,
                start_idx=i - 1, end_idx=i, stop=float(l[i]), key_levels={"engulf_low": float(l[i])},
                notes="Bullish reversal; current candle engulfs prior down candle.",
            ))
        # Bearish engulfing
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
            out.append(Detection(
                pattern="bearish_engulfing", family="candlestick", direction="bearish", confidence=0.55,
                start_idx=i - 1, end_idx=i, stop=float(h[i]), key_levels={"engulf_high": float(h[i])},
                notes="Bearish reversal; current candle engulfs prior up candle.",
            ))
        # Morning star (bullish 3-bar): down, small body, strong up
        if i >= 2:
            if (c[i - 2] < o[i - 2] and body(i - 1) <= 0.4 * body(i - 2)
                    and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) / 2):
                out.append(Detection(
                    pattern="morning_star", family="candlestick", direction="bullish", confidence=0.6,
                    start_idx=i - 2, end_idx=i, stop=float(l[i - 1]), key_levels={"star_low": float(l[i - 1])},
                    notes="Bullish 3-bar reversal.",
                ))
            # Evening star (bearish 3-bar)
            if (c[i - 2] > o[i - 2] and body(i - 1) <= 0.4 * body(i - 2)
                    and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2):
                out.append(Detection(
                    pattern="evening_star", family="candlestick", direction="bearish", confidence=0.6,
                    start_idx=i - 2, end_idx=i, stop=float(h[i - 1]), key_levels={"star_high": float(h[i - 1])},
                    notes="Bearish 3-bar reversal.",
                ))
    return out


# ---------------------------------------------------------------------------
# Context: support/resistance, Fibonacci, VWAP bands
# ---------------------------------------------------------------------------

def detect_support_resistance(df: pd.DataFrame, pivots: list[Pivot],
                              cluster_pct: float = 1.0, min_touches: int = 2) -> list[Detection]:
    """
    Cluster swing pivots into horizontal price zones. A level touched `min_touches`+
    times is reported with confidence scaling on touch count.
    """
    out: list[Detection] = []
    if not pivots:
        return out
    prices = sorted(p.price for p in pivots)
    clusters: list[list[float]] = []
    for pr in prices:
        if clusters and abs(pr - np.mean(clusters[-1])) / pr * 100 <= cluster_pct:
            clusters[-1].append(pr)
        else:
            clusters.append([pr])

    last = float(df["close"].iloc[-1])
    for cl in clusters:
        if len(cl) >= min_touches:
            level = float(np.mean(cl))
            direction = "bullish" if level < last else "bearish"  # support below, resistance above
            out.append(Detection(
                pattern="support" if level < last else "resistance",
                family="context", direction=direction,
                confidence=round(min(len(cl) / 4, 1.0), 3),
                start_idx=0, end_idx=len(df) - 1, trigger=level,
                key_levels={"level": level, "touches": float(len(cl))},
                notes=f"{'Support' if level < last else 'Resistance'} tested {len(cl)}x.",
            ))
    return out


def fibonacci_levels(df: pd.DataFrame, pivots: list[Pivot]) -> list[Detection]:
    """
    Fibonacci retracements from the most recent significant swing (last low→high or
    high→low). Reports the standard 0.236/0.382/0.5/0.618/0.786 levels.
    """
    if len(pivots) < 2:
        return []
    a, b = pivots[-2], pivots[-1]
    lo, hi = (a.price, b.price) if a.price < b.price else (b.price, a.price)
    diff = hi - lo
    up = b.price > a.price  # most recent leg direction
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    levels = {f"fib_{int(r*1000)/10}": (hi - diff * r) if up else (lo + diff * r) for r in ratios}
    return [Detection(
        pattern="fibonacci_retracement", family="context",
        direction="bullish" if up else "bearish",
        confidence=0.5, start_idx=a.idx, end_idx=b.idx,
        key_levels={"swing_low": lo, "swing_high": hi, **levels},
        notes="Retracement levels from most recent swing; 0.618 is the key zone.",
    )]


def vwap_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> list[Detection]:
    """
    Rolling VWAP with ±n_std volume-weighted bands. Reports where price sits relative
    to the bands (reversion context).
    """
    if len(df) < window:
        return []
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).rolling(window).sum()
    vv = df["volume"].rolling(window).sum()
    vwap = (pv / vv)
    dev = (typical - vwap)
    std = dev.rolling(window).std()
    last_v = float(vwap.iloc[-1])
    last_s = float(std.iloc[-1]) if not np.isnan(std.iloc[-1]) else 0.0
    last_c = float(df["close"].iloc[-1])
    upper, lower = last_v + n_std * last_s, last_v - n_std * last_s
    if last_c > upper:
        direction, note = "bearish", "Price above upper VWAP band; stretched, mean-reversion risk."
    elif last_c < lower:
        direction, note = "bullish", "Price below lower VWAP band; stretched, mean-reversion candidate."
    else:
        direction, note = "neutral", "Price within VWAP bands."
    return [Detection(
        pattern="vwap_bands", family="context", direction=direction, confidence=0.45,
        start_idx=len(df) - window, end_idx=len(df) - 1,
        key_levels={"vwap": last_v, "upper": upper, "lower": lower, "close": last_c},
        notes=note,
    )]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_all(df: pd.DataFrame, pivot_order: int = 3, dedupe_pct: float = 1.5) -> list[dict]:
    """
    Run every detector and return a flat, JSON-ready list sorted by confidence.
    `df` must have lowercase columns: open, high, low, close, volume.
    """
    df = df.reset_index(drop=True)
    pivots = find_pivots(df, order=pivot_order, dedupe_pct=dedupe_pct)

    detections: list[Detection] = []
    detections += detect_head_and_shoulders(df, pivots)
    detections += detect_double_triple(df, pivots)
    detections += detect_triangles_wedges(df, pivots)
    detections += detect_flags(df)
    detections += detect_candlesticks(df)
    detections += detect_support_resistance(df, pivots)
    detections += fibonacci_levels(df, pivots)
    detections += vwap_bands(df)

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return [d.to_dict() for d in detections]


if __name__ == "__main__":
    import json
    import sys
    import yfinance as yf

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    raw = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    raw = raw.reset_index()
    raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in raw.columns]
    result = detect_all(raw)
    print(json.dumps({"symbol": symbol, "count": len(result), "detections": result}, indent=2))
