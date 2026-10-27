"""The confidence engine: pure code, so every rule is checked against known numbers."""

import pytest

from confidence import (
    CONFIG,
    Evidence,
    assess,
    event_penalty,
    fear_greed_evidence,
    funding_evidence,
    macro_evidence,
    news_evidence,
    price_evidence,
    score,
    social_evidence,
)


def pa(trend, last, ema20, ema50, atr=10.0):
    return {"trend": trend, "last_price": last, "ema20": ema20, "ema50": ema50,
            "atr14": atr, "change_pct_20_bars": -1.0}


def macro(**changes):
    """macro(dollar_index=(1d, 5d), us10y_yield=(1d, 5d), ...)"""
    return {k: {"change_1d": d1, "change_5d": d5} for k, (d1, d5) in changes.items()}


def ev(group, direction, strength=1.0, source="x"):
    return Evidence(source=source, group=group, direction=direction, strength=strength, rationale=source)


# ---------- price ----------

def test_price_downtrend_far_from_ema50_is_full_strength_bearish():
    [e] = price_evidence(pa("down", last=900, ema20=950, ema50=1000, atr=10))  # 100 away = 10 ATRs
    assert (e.direction, e.strength) == (-1, 1.0)


def test_price_strength_scales_with_distance_in_atrs():
    [e] = price_evidence(pa("up", last=1015, ema20=1010, ema50=1000, atr=10))  # 1.5 ATRs of 3
    assert (e.direction, e.strength) == (1, 0.5)


def test_price_mixed_but_below_both_emas_is_half_strength_bearish():
    [e] = price_evidence(pa("mixed", last=900, ema20=1050, ema50=1000, atr=10))
    assert (e.direction, e.strength) == (-1, 0.5)


def test_price_between_the_emas_has_no_lean():
    [e] = price_evidence(pa("mixed", last=1005, ema20=1010, ema50=1000))
    assert (e.direction, e.strength) == (0, 0.0)


# ---------- macro: the rules the LLM kept getting backwards ----------

def test_falling_yields_are_bullish_for_gold():
    [_, yields] = macro_evidence(macro(dollar_index=(0, 0), us10y_yield=(0, -15)), "GC=F", "1W")
    assert yields.direction == 1 and yields.strength == 0.75   # 15bp of a 20bp full-strength move


def test_rising_dollar_is_bearish_for_gold():
    [dollar, _] = macro_evidence(macro(dollar_index=(0, 0.5), us10y_yield=(0, 0)), "GC=F", "1D")
    assert dollar.direction == -1 and dollar.strength == 0.5


def test_short_timeframes_use_the_1_day_change():
    [dollar, _] = macro_evidence(macro(dollar_index=(-0.4, 5.0), us10y_yield=(0, 0)), "GC=F", "1H")
    assert dollar.direction == 1 and dollar.strength == 1.0     # -0.4% on the 0.4% short scale
    assert "(1d)" in dollar.rationale


def test_tiny_moves_count_as_no_move():
    [_, yields] = macro_evidence(macro(dollar_index=(0, 0), us10y_yield=(-0.2, 0)), "GC=F", "4H")
    assert (yields.direction, yields.strength) == (0, 0.0)      # 0.2bp < 10% of 7bp


def test_crypto_macro_uses_stocks_and_vix_not_the_dollar():
    items = macro_evidence(macro(sp500=(1.0, 0), vix=(8, 0), dollar_index=(9, 9)), "BTC-USD", "4H")
    assert [(e.source, e.direction) for e in items] == [("macro.sp500", 1), ("macro.vix", -1)]


def test_missing_macro_item_is_skipped():
    assert macro_evidence({"dollar_index": {"error": "x"}}, "GC=F", "1D") == []


# ---------- contrarian sentiment and positioning ----------

@pytest.mark.parametrize("value, direction", [(10, 1), (25, 1), (26, 0), (48, 0), (74, 0), (75, -1), (95, -1)])
def test_fear_greed_only_counts_at_extremes_and_against_the_crowd(value, direction):
    [e] = fear_greed_evidence({"today": value, "classification": "x"})
    assert e.direction == direction


@pytest.mark.parametrize("rate, direction", [(0.01, 0), (0.0087, 0), (-0.02, 0), (0.05, -1), (0.12, -1), (-0.03, 1)])
def test_funding_only_counts_when_crowded(rate, direction):
    [e] = funding_evidence({"latest_pct_8h": rate})
    assert e.direction == direction


def test_normal_funding_is_neutral_not_bearish():
    [e] = funding_evidence({"latest_pct_8h": 0.01})  # what the LLM called "mild bearish pressure"
    assert (e.direction, e.strength) == (0, 0.0) and "normal" in e.rationale


def test_social_needs_enough_tagged_posts():
    [e] = social_evidence({"bullish": 3, "bearish": 0})
    assert e.direction == 0


def test_social_lean_is_at_most_half_strength():
    [e] = social_evidence({"bullish": 10, "bearish": 0})
    assert (e.direction, e.strength) == (1, 0.5)


# ---------- news ----------

def test_news_collapses_into_one_piece_of_evidence():
    scores = [{"direction": -1, "relevance": 1.0}] * 5
    assert len(news_evidence(scores)) == 1
    assert news_evidence(scores)[0].direction == -1


def test_irrelevant_headlines_are_ignored():
    scores = [{"direction": 1, "relevance": 0.0}] * 4 + [{"direction": -1, "relevance": 0.8}]
    [e] = news_evidence(scores)
    assert (e.direction, e.strength) == (-1, 0.8)


def test_balanced_news_is_neutral():
    [e] = news_evidence([{"direction": 1, "relevance": 1.0}, {"direction": -1, "relevance": 1.0}])
    assert e.direction == 0


# ---------- events ----------

def test_high_impact_event_inside_the_timeframe_cuts_the_score():
    calendar = {"events": [{"event": "FOMC Statement", "impact": "High", "hours_away": 35.0}]}
    assert event_penalty(calendar, "1W") == (0.7, "FOMC Statement in 35h")


def test_event_outside_the_timeframe_does_not():
    calendar = {"events": [{"event": "FOMC Statement", "impact": "High", "hours_away": 35.0}]}
    assert event_penalty(calendar, "4H")[0] == 1.0     # 4H window is 16h


def test_medium_impact_events_do_not_count():
    calendar = {"events": [{"event": "Claims", "impact": "Medium", "hours_away": 1.0}]}
    assert event_penalty(calendar, "1H")[0] == 1.0


@pytest.mark.parametrize("calendar", [None, {"error": "429"}])
def test_unknown_calendar_is_a_smaller_penalty(calendar):
    assert event_penalty(calendar, "1D")[0] == CONFIG["unknown_events_penalty"]


# ---------- scoring ----------

def test_all_evidence_agreeing_at_full_strength_scores_100():
    evidence = [ev("price", -1), ev("macro", -1), ev("news", -1), ev("sentiment", -1)]
    a = score(evidence, "GC=F", "1D")
    assert (a.bias, a.score, a.bucket, a.coverage) == ("Bearish", 100, "High", 1.0)


def test_correlated_items_in_one_group_count_once():
    # 3 macro items vs 1 price item, opposite ways: with grouping it's 40/35 by weight, not 3 vs 1
    evidence = [ev("price", 1), ev("macro", -1), ev("macro", -1), ev("macro", -1)]
    a = score(evidence, "GC=F", "1D")  # price 0.40, macro 0.35
    assert a.net == pytest.approx((0.40 - 0.35) / 0.75, abs=1e-3)


def test_conflicting_evidence_is_neutral():
    a = score([ev("price", -1), ev("macro", 1)], "GC=F", "1W")  # equal 40/40 weights
    assert (a.bias, a.score) == ("Neutral", 0)


def test_neutral_band():
    weak = score([ev("price", -1, strength=0.1)], "GC=F", "1D")
    assert weak.bias == "Neutral"                      # |net| 0.1 < 0.15


def test_missing_groups_lower_coverage_and_the_score():
    a = score([ev("price", -1)], "GC=F", "1D")         # only price: 40% of expected weight
    assert a.coverage == 0.4 and a.score == 40
    assert "News (20%): missing" in a.breakdown


def test_event_penalty_multiplies_the_score():
    a = score([ev("price", -1), ev("macro", -1), ev("news", -1), ev("sentiment", -1)],
              "GC=F", "1D", penalty=0.7, penalty_note="FOMC in 5h")
    assert a.score == 70 and a.bucket == "High"
    assert "Events: FOMC in 5h (score x0.7)" in a.breakdown


@pytest.mark.parametrize("strength, bucket", [(0.34, "Low"), (0.35, "Medium"), (0.65, "Medium"), (0.66, "High")])
def test_buckets(strength, bucket):
    evidence = [ev(g, 1, strength) for g in ("price", "macro", "news", "sentiment")]
    assert score(evidence, "GC=F", "1D").bucket == bucket


def test_evidence_in_a_zero_weight_group_is_ignored():
    a = score([ev("price", 1), ev("macro", -1)], "BTC-USD", "1H")  # crypto 1H: macro weight 0
    assert "macro" not in a.groups and a.bias == "Bullish"


def test_no_evidence_at_all_is_neutral_zero():
    a = score([], "BTC-USD", "4H")
    assert (a.bias, a.score, a.coverage) == ("Neutral", 0, 0.0)


def test_each_item_in_the_breakdown_carries_its_own_direction():
    a = assess({"macro": macro(dollar_index=(0, 0.38), us10y_yield=(0, -15))}, "GC=F", "1W")
    line = next(b for b in a.breakdown if b.startswith("Macro"))
    assert "dollar index +0.38% (5d) [bearish]" in line and "-15bp (5d) [bullish]" in line


# ---------- assess: tool results in, assessment out ----------

def test_assess_skips_failed_sources():
    results = {"price_action": pa("down", 900, 950, 1000), "macro": {"error": "yahoo down"},
               "calendar": {"error": "429"}}
    a = assess(results, "GC=F", "1D")
    assert [e.source for e in a.evidence] == ["price_action"]
    assert a.event_penalty == CONFIG["unknown_events_penalty"]


def test_assess_counts_news_only_when_scored():
    results = {"news": {"articles": [{"title": "x"}]}}
    assert assess(results, "BTC-USD", "1D").groups.get("news") is None           # scorer failed
    scored = assess(results, "BTC-USD", "1D", [{"direction": 1, "relevance": 1.0}])
    assert scored.groups["news"] == 1.0
