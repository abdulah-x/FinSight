"""Confidence engine: turns tool results into a bias and an evidence score, in plain code.

The LLM never decides the bias or the score. Each tool result becomes Evidence
(direction -1/0/+1 and strength 0-1) by a written rule; evidence is averaged per
group, groups are weighted by asset and timeframe, and the result is scaled by how
much evidence we actually have (coverage) and whether a big event is close.

Every number in CONFIG is a documented starting guess, to be tuned only from
forward-test results (Stage 7). Until then the score measures how strongly the
evidence agrees, not a probability of being right.
"""

from typing import Literal

from pydantic import BaseModel

CONFIG = {
    "neutral_band": 0.15,          # |net| below this -> Neutral
    "buckets": (35, 65),           # score < 35 Low, <= 65 Medium, > 65 High
    "event_penalty": 0.7,          # high-impact USD event inside the timeframe
    "unknown_events_penalty": 0.85,  # calendar not checked or unavailable
    "event_window_hours": {"1H": 4, "4H": 16, "1D": 24, "1W": 168},
    # group weights per (asset class, timeframe); a group with no evidence lowers coverage
    "weights": {
        ("crypto", "1H"): {"price": 0.40, "news": 0.25, "positioning": 0.20, "sentiment": 0.15, "macro": 0.00},
        ("crypto", "4H"): {"price": 0.40, "news": 0.20, "positioning": 0.20, "sentiment": 0.15, "macro": 0.05},
        ("crypto", "1D"): {"price": 0.35, "news": 0.20, "positioning": 0.15, "sentiment": 0.15, "macro": 0.15},
        ("crypto", "1W"): {"price": 0.40, "news": 0.15, "positioning": 0.10, "sentiment": 0.15, "macro": 0.20},
        ("gold", "1H"): {"price": 0.45, "news": 0.30, "macro": 0.20, "sentiment": 0.05},
        ("gold", "4H"): {"price": 0.45, "news": 0.25, "macro": 0.25, "sentiment": 0.05},
        ("gold", "1D"): {"price": 0.40, "news": 0.20, "macro": 0.35, "sentiment": 0.05},
        ("gold", "1W"): {"price": 0.40, "news": 0.15, "macro": 0.40, "sentiment": 0.05},
    },
    # a move of `scale` counts as full strength; under 10% of it counts as no move.
    # (short = 1H/4H uses 1-day changes, long = 1D/1W uses 5-day changes)
    "macro": {
        "gold": [  # (snapshot key, label, unit, inverse?, short scale, long scale)
            ("dollar_index", "dollar index", "%", True, 0.4, 1.0),   # stronger dollar -> weaker gold
            ("us10y_yield", "US 10y yield", "bp", True, 7, 20),      # higher yields -> weaker gold
        ],
        "crypto": [
            ("sp500", "S&P 500", "%", False, 1.0, 2.5),              # risk-on -> crypto up
            ("vix", "VIX", "%", True, 8, 15),                        # fear up -> crypto down
        ],
    },
    "fear_greed": {"fear": 25, "greed": 75},          # contrarian, only at extremes
    "funding_pct_8h": {"crowded_long": 0.05, "crowded_short": -0.03},  # contrarian, only at extremes
    "social_min_tagged": 5,                           # fewer tagged posts -> no evidence
}

GROUP_LABELS = {"price": "Price", "macro": "Macro", "news": "News",
                "positioning": "Positioning", "sentiment": "Sentiment"}


class Evidence(BaseModel):
    source: str
    group: str
    direction: int    # -1 bearish, 0 neutral, +1 bullish
    strength: float   # 0..1
    rationale: str


class Assessment(BaseModel):
    bias: Literal["Bullish", "Bearish", "Neutral"]
    score: int        # 0..100 evidence score
    bucket: Literal["Low", "Medium", "High"]
    net: float        # weighted direction, -1..+1
    coverage: float   # share of expected evidence weight we actually got
    event_penalty: float
    groups: dict[str, float]
    evidence: list[Evidence]
    breakdown: list[str]  # human-readable lines, shown to the trader


def asset_class(symbol: str) -> str:
    return "gold" if symbol == "GC=F" else "crypto"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sign(x: float) -> int:
    return int(x > 0) - int(x < 0)  # int() also handles numpy numbers from the tools


def _word(direction: float) -> str:
    return "bullish" if direction > 0.1 else "bearish" if direction < -0.1 else "neutral"


# ---------- tool result -> evidence (one rule per source) ----------

def price_evidence(pa: dict) -> list[Evidence]:
    last, ema20, ema50 = pa["last_price"], pa["ema20"], pa["ema50"]
    if pa["trend"] in ("up", "down"):
        direction, weight = (1 if pa["trend"] == "up" else -1), 1.0
    elif last < min(ema20, ema50) or last > max(ema20, ema50):
        # "mixed" but price is beyond both EMAs: an early turn, counted at half strength
        direction, weight = (1 if last > ema50 else -1), 0.5
    else:
        direction, weight = 0, 0.0  # price between the EMAs: no lean
    # how far price is from EMA50, in units of 3 typical bars: 3+ ATRs away = full strength
    strength = _clamp(abs(last - ema50) / (3 * pa["atr14"])) * weight
    return [Evidence(
        source="price_action", group="price", direction=direction, strength=round(strength, 2),
        rationale=f"price {last} vs EMA20 {ema20} / EMA50 {ema50} "
                  f"({pa['trend']} trend, {pa['change_pct_20_bars']:+}% over 20 bars)",
    )]


def macro_evidence(macro: dict, symbol: str, timeframe: str) -> list[Evidence]:
    short = timeframe in ("1H", "4H")
    change_key, period = ("change_1d", "1d") if short else ("change_5d", "5d")
    evidence = []
    for key, label, unit, inverse, short_scale, long_scale in CONFIG["macro"][asset_class(symbol)]:
        item = macro.get(key) or {}
        if change_key not in item:
            continue
        change, scale = item[change_key], short_scale if short else long_scale
        moved = abs(change) >= 0.1 * scale
        direction = _sign(change) * (-1 if inverse else 1) if moved else 0
        evidence.append(Evidence(
            source=f"macro.{key}", group="macro", direction=direction,
            strength=round(_clamp(abs(change) / scale), 2) if moved else 0.0,
            rationale=f"{label} {change:+}{unit} ({period})",
        ))
    return evidence


def fear_greed_evidence(fg: dict) -> list[Evidence]:
    value, cfg = fg["today"], CONFIG["fear_greed"]
    if value <= cfg["fear"]:
        direction, strength = 1, 0.5 + (cfg["fear"] - value) / 50   # crowd fear: contrarian bullish
    elif value >= cfg["greed"]:
        direction, strength = -1, 0.5 + (value - cfg["greed"]) / 50  # crowd greed: contrarian bearish
    else:
        direction, strength = 0, 0.0
    return [Evidence(source="fear_greed", group="sentiment", direction=direction,
                     strength=round(_clamp(strength), 2),
                     rationale=f"Fear & Greed {value} ({fg['classification']})")]


def funding_evidence(f: dict) -> list[Evidence]:
    rate, cfg = f["latest_pct_8h"], CONFIG["funding_pct_8h"]
    if rate >= cfg["crowded_long"]:
        direction, strength = -1, rate / (2 * cfg["crowded_long"])     # crowded longs: contrarian bearish
    elif rate <= cfg["crowded_short"]:
        direction, strength = 1, rate / (2 * cfg["crowded_short"])     # crowded shorts: contrarian bullish
    else:
        direction, strength = 0, 0.0
    return [Evidence(source="funding", group="positioning", direction=direction,
                     strength=round(_clamp(strength), 2),
                     rationale=f"funding {rate:+}% per 8h ({'normal' if direction == 0 else 'crowded'})")]


def social_evidence(s: dict) -> list[Evidence]:
    tagged = s["bullish"] + s["bearish"]
    if tagged < CONFIG["social_min_tagged"]:
        return [Evidence(source="social", group="sentiment", direction=0, strength=0.0,
                         rationale=f"StockTwits: only {tagged} tagged posts")]
    lean = (s["bullish"] - s["bearish"]) / tagged
    direction = _sign(lean) if abs(lean) > 0.3 else 0
    return [Evidence(source="social", group="sentiment", direction=direction,
                     strength=round(abs(lean) * 0.5, 2) if direction else 0.0,  # retail chatter: half strength at most
                     rationale=f"StockTwits {s['bullish']} bullish / {s['bearish']} bearish tags")]


def news_evidence(scores: list[dict]) -> list[Evidence]:
    """`scores`: one {direction, relevance} per headline, from the news-scoring LLM.
    All headlines collapse into ONE piece of evidence, so article count can't dominate."""
    relevant = [s for s in scores if s["relevance"] > 0]
    if not relevant:
        return [Evidence(source="news", group="news", direction=0, strength=0.0,
                         rationale="no relevant headlines")]
    lean = sum(s["direction"] * s["relevance"] for s in relevant) / len(relevant)
    direction = _sign(lean) if abs(lean) >= 0.15 else 0
    ups, downs = sum(s["direction"] > 0 for s in relevant), sum(s["direction"] < 0 for s in relevant)
    return [Evidence(source="news", group="news", direction=direction,
                     strength=round(_clamp(abs(lean)), 2) if direction else 0.0,
                     rationale=f"{len(relevant)} relevant headlines: {ups} bullish, {downs} bearish")]


def event_penalty(calendar: dict | None, timeframe: str) -> tuple[float, str]:
    if not calendar or "error" in calendar:
        return CONFIG["unknown_events_penalty"], "calendar not checked: upcoming events unknown"
    window = CONFIG["event_window_hours"][timeframe]
    high = [e for e in calendar["events"] if e["impact"] == "High" and e["hours_away"] <= window]
    if high:
        return CONFIG["event_penalty"], f"{high[0]['event']} in {high[0]['hours_away']:.0f}h"
    return 1.0, f"no high-impact USD events in the next {window}h"


# ---------- evidence -> bias + score ----------

def score(evidence: list[Evidence], symbol: str, timeframe: str,
          penalty: float = 1.0, penalty_note: str = "") -> Assessment:
    weights = CONFIG["weights"][(asset_class(symbol), timeframe)]
    expected = {g: w for g, w in weights.items() if w > 0}

    groups = {}
    for g in expected:
        items = [e for e in evidence if e.group == g]
        if items:  # average inside the group, so correlated inputs count once
            groups[g] = sum(e.direction * e.strength for e in items) / len(items)

    present_weight = sum(expected[g] for g in groups)
    net = sum(expected[g] * groups[g] for g in groups) / present_weight if present_weight else 0.0
    coverage = present_weight / sum(expected.values())

    band = CONFIG["neutral_band"]
    bias = "Bullish" if net >= band else "Bearish" if net <= -band else "Neutral"
    value = round(100 * abs(net) * coverage * penalty)
    low, high = CONFIG["buckets"]
    bucket = "Low" if value < low else "Medium" if value <= high else "High"

    breakdown = []
    for g, w in sorted(expected.items(), key=lambda kv: -kv[1]):
        if g in groups:
            # every item carries its own direction, so a mixed group can't be misread
            reasons = "; ".join(f"{e.rationale} [{_word(e.direction)}]" for e in evidence if e.group == g)
            breakdown.append(f"{GROUP_LABELS[g]} ({w:.0%}): {_word(groups[g])} - {reasons}")
        else:
            breakdown.append(f"{GROUP_LABELS[g]} ({w:.0%}): missing")
    breakdown.append(f"Coverage {coverage:.0%} of expected evidence")
    if penalty_note:
        breakdown.append(f"Events: {penalty_note}" + (f" (score x{penalty})" if penalty < 1 else ""))

    return Assessment(bias=bias, score=value, bucket=bucket, net=round(net, 3),
                      coverage=round(coverage, 2), event_penalty=penalty,
                      groups={g: round(v, 3) for g, v in groups.items()},
                      evidence=evidence, breakdown=breakdown)


def assess(results: dict, symbol: str, timeframe: str, news_scores: list[dict] | None = None) -> Assessment:
    """`results`: gather_evidence output ({source: tool result}). Failed sources are skipped."""
    def ok(source):
        return source in results and "error" not in results[source]

    evidence = []
    if ok("price_action"):
        evidence += price_evidence(results["price_action"])
    if ok("macro"):
        evidence += macro_evidence(results["macro"], symbol, timeframe)
    if ok("fear_greed"):
        evidence += fear_greed_evidence(results["fear_greed"])
    if ok("funding"):
        evidence += funding_evidence(results["funding"])
    if ok("social"):
        evidence += social_evidence(results["social"])
    if ok("news") and news_scores is not None:
        evidence += news_evidence(news_scores)

    penalty, note = event_penalty(results.get("calendar"), timeframe)
    return score(evidence, symbol, timeframe, penalty, note)
