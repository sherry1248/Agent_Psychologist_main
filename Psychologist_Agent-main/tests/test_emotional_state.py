"""
Tests for the rule-based emotional state agent.
"""

from dataclasses import fields

from src.agent.intent import classify_intent
from src.agent.models import EmotionalStateVector, RAW_TEXT_FIELD_NAMES
from src.agent.state import (
    EmotionalStateAgent,
    normalize_checkin_score,
    summarize_emotional_state,
    update_emotional_state,
)


def test_previous_state_none_starts_from_default_values():
    state = update_emotional_state(previous_state=None)

    assert state.to_dict() == EmotionalStateVector().to_dict()


def test_sleep_problem_intent_lowers_sleep_value():
    previous = EmotionalStateVector()
    intent = classify_intent("요즘 잠을 못 자요")

    state = update_emotional_state(previous_state=previous, intent_result=intent)

    assert state.sleep < previous.sleep


def test_anxiety_support_intent_raises_anxiety_value():
    previous = EmotionalStateVector()
    intent = classify_intent("요즘 불안해요")

    state = EmotionalStateAgent().update(previous_state=previous, intent_result=intent)

    assert state.anxiety > previous.anxiety


def test_danger_risk_stage_lowers_safety_substantially():
    previous = EmotionalStateVector()

    state = update_emotional_state(previous_state=previous, risk_stage="위험")

    assert state.safety < 0.8


def test_low_wellness_sleep_quality_lowers_sleep():
    previous = EmotionalStateVector()

    state = update_emotional_state(
        previous_state=previous,
        wellness_checkin={"sleep_quality": 1},
    )

    assert state.sleep < previous.sleep
    assert normalize_checkin_score(1) == 0.0
    assert normalize_checkin_score(10) == 1.0


def test_all_state_values_are_clamped_to_unit_range():
    previous = EmotionalStateVector(
        mood=-10,
        anxiety=10,
        stress=10,
        sleep=-10,
        energy=10,
        safety=-10,
        rapport=10,
    )
    state = update_emotional_state(
        previous_state=previous,
        intent_result=classify_intent("죽고 싶고 잠을 못 자고 불안하고 스트레스 받아요"),
        risk_stage="위험",
        wellness_checkin={
            "mood_score": -5,
            "anxiety_score": 100,
            "sleep_quality": -5,
            "energy_score": 100,
            "stress_score": 100,
        },
    )

    assert all(0.0 <= value <= 1.0 for value in state.to_dict().values())


def test_state_summary_returns_prompt_labels():
    state = update_emotional_state(
        previous_state=EmotionalStateVector(),
        intent_result=classify_intent("잠을 못 자고 불안하고 스트레스 받아요"),
        risk_stage="위험",
    )

    summary = summarize_emotional_state(state)

    assert "low sleep" in summary
    assert "elevated anxiety" in summary
    assert "high stress" in summary
    assert "low energy" in summary
    assert "safety concern" in summary


def test_emotional_state_vector_has_no_raw_text_fields():
    names = {item.name for item in fields(EmotionalStateVector)}

    assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)

