"""
Tests for the deterministic rule-based decision agent.
"""

from dataclasses import fields

from src.agent.decision import DecisionAgent, decide_action
from src.agent.intent import classify_intent
from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    EmotionalStateVector,
    IntentAgentResult,
    IntentLabel,
    IntentSeverity,
    RAW_TEXT_FIELD_NAMES,
)


def test_danger_risk_stage_escalates_safety():
    result = decide_action(risk_stage="위험")

    assert result.primary_action == DecisionAction.ESCALATE_SAFETY
    assert result.action == DecisionAction.ESCALATE_SAFETY
    assert "crisis_priority" in result.reason_codes
    assert result.response_constraints["tone"] == "safety_first"


def test_s3_sos_intent_escalates_safety():
    intent = classify_intent("죽고 싶어요")

    result = DecisionAgent().decide(intent_result=intent)

    assert result.primary_action == DecisionAction.ESCALATE_SAFETY
    assert "crisis_priority" in result.reason_codes


def test_sleep_problem_low_risk_asks_follow_up():
    intent = IntentAgentResult(
        primary_intent=IntentLabel.SLEEP_PROBLEM,
        severity=IntentSeverity.S1_CONCERN,
    )

    result = decide_action(risk_stage="관심", intent_result=intent)

    assert result.primary_action == DecisionAction.ASK_FOLLOW_UP
    assert DecisionAction.RESPOND_SUPPORTIVELY in result.secondary_actions
    assert result.response_constraints["must_include_followup"] is True
    assert "sleep_problem_needs_clarification" in result.reason_codes


def test_wellness_hint_adds_small_action_secondary_action():
    result = decide_action(wellness_hint="sleep_hygiene")

    assert DecisionAction.SUGGEST_SMALL_ACTION in result.secondary_actions
    assert result.response_constraints["must_include_small_action"] is True
    assert "actionable_wellness_hint" in result.reason_codes


def test_high_anxiety_state_sets_calm_or_empathic_tone_constraint():
    result = decide_action(
        emotional_state=EmotionalStateVector(anxiety=0.8),
    )

    assert result.response_constraints["tone"] in {"calm", "empathic"}
    assert result.response_constraints["tone"] == "calm"
    assert "elevated_anxiety_state" in result.reason_codes


def test_crisis_does_not_choose_follow_up_or_small_action():
    intent = IntentAgentResult(
        primary_intent=IntentLabel.SLEEP_PROBLEM,
        severity=IntentSeverity.S3_SOS,
        s3_sos=True,
    )

    result = decide_action(
        risk_stage="위험",
        intent_result=intent,
        wellness_hint="breathing",
        emotional_state=EmotionalStateVector(anxiety=0.9, sleep=0.2),
    )

    assert result.primary_action == DecisionAction.ESCALATE_SAFETY
    assert DecisionAction.ASK_FOLLOW_UP not in result.secondary_actions
    assert DecisionAction.SUGGEST_SMALL_ACTION not in result.secondary_actions
    assert result.response_constraints["must_include_followup"] is False
    assert result.response_constraints["must_include_small_action"] is False


def test_decision_result_has_no_raw_text_fields():
    names = {item.name for item in fields(DecisionAgentResult)}

    assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)
    assert "raw_text" not in names
    assert "user_input" not in names
    assert "conversation" not in names
    assert "content" not in names
