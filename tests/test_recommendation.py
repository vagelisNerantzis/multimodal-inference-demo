"""Adapter recommendations depend on explicit forecasts and costs only."""
import pytest
from multimodal_inference_demo.acquisition.recommend import recommend_next


def recommendation(result, *, gains=None, costs=None, **kwargs):
    return recommend_next(result, ["first", "second", "third"],
                          {"second": .2, "third": .3} if gains is None else gains,
                          {"second": 1., "third": 3.} if costs is None else costs, **kwargs)


def test_highest_estimated_utility_is_selected(initial_result):
    chosen = recommendation(initial_result)
    assert chosen.recommended_modality == "third"
    assert chosen.estimated_utility == pytest.approx(.3 - .02 * 3)
    assert chosen.reason


def test_cost_can_change_the_recommended_modality(initial_result):
    chosen = recommendation(initial_result, costs={"second": 1., "third": 100.})
    assert chosen.recommended_modality == "second"


def test_recommendation_tie_break_is_lexical(initial_result):
    chosen = recommendation(initial_result, gains={"third": .2, "second": .2},
                            costs={"third": 1., "second": 1.})
    assert chosen.recommended_modality == "second"


def test_no_remaining_modality_returns_none(initial_result):
    chosen = recommend_next(initial_result, ["first"], {}, {})
    assert chosen.recommended_modality is None
    assert chosen.reason


def test_confidence_threshold_stops_recommendation(initial_result):
    chosen = recommendation(initial_result, confidence_threshold=.5)
    assert chosen.recommended_modality is None
    assert "confidence" in chosen.reason.lower()


def test_nonpositive_utility_has_no_recommendation(initial_result):
    chosen = recommendation(initial_result, gains={"second": .01, "third": .01}, cost_weight=1.)
    assert chosen.recommended_modality is None
    assert chosen.estimated_utility <= 0


@pytest.mark.parametrize("changes", [
    {"costs": {"second": -1., "third": 3.}},
    {"costs": {"second": 1.}},
    {"gains": {"second": float("nan"), "third": .3}},
    {"gains": {"second": 1.1, "third": .3}},
])
def test_invalid_or_missing_recommendation_inputs_fail(initial_result, changes):
    with pytest.raises((ValueError, TypeError)):
        recommendation(initial_result, **changes)
