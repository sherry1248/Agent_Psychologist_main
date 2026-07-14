"""
Tests for the deterministic rule-based small action planner.
"""

from dataclasses import fields

from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    IntentAgentResult,
    IntentLabel,
    IntentSeverity,
    RAW_TEXT_FIELD_NAMES,
    SmallActionPlan,
)
from src.agent.planner import SmallActionPlanner, generate_small_action_plan


def _intent(label: IntentLabel) -> IntentAgentResult:
    return IntentAgentResult(
        primary_intent=label,
        severity=IntentSeverity.S1_CONCERN,
    )


def test_sleep_problem_generates_sleep_small_action():
    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
    )

    assert "잠들기 전" in plan.action_text
    assert "조명" in plan.action_text
    assert plan.intent_label == IntentLabel.SLEEP_PROBLEM.value


def test_anxiety_support_generates_grounding_small_action():
    plan = SmallActionPlanner().generate(
        intent_result=_intent(IntentLabel.ANXIETY_SUPPORT),
    )

    assert "발바닥" in plan.action_text
    assert "30초" in plan.action_text


def test_low_mood_support_generates_low_burden_action():
    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.LOW_MOOD_SUPPORT),
    )

    assert "물 한 잔" in plan.action_text
    assert "1분" in plan.action_text


def test_danger_risk_stage_does_not_generate_general_small_action():
    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.ANXIETY_SUPPORT),
        risk_stage="위험",
    )

    assert plan.action_text == ""
    assert plan.status == "safety_blocked"
    assert plan.rationale_label == "safety_priority"


def test_safety_escalation_decision_does_not_generate_general_small_action():
    decision = DecisionAgentResult(
        primary_action=DecisionAction.ESCALATE_SAFETY,
    )

    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
        decision_result=decision,
    )

    assert plan.action_text == ""
    assert "호흡" not in plan.action_text
    assert "산책" not in plan.action_text


def test_status_default_is_suggested():
    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.OTHER_CONCERN),
    )

    assert plan.status == "suggested"


def test_check_after_turns_default_is_one():
    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.NEED_ADVICE),
    )

    assert plan.check_after_turns == 1


def test_action_text_does_not_include_raw_user_input():
    raw_user_input = "요즘 잠을 못 자서 회사에서 실수한 이야기를 그대로 저장하지 마세요"

    plan = generate_small_action_plan(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
        counseling_hint=raw_user_input,
    )

    assert raw_user_input not in plan.action_text
    assert plan.rationale_label == "counseling_hint_considered"


def test_small_action_plan_has_no_raw_text_fields():
    names = {item.name for item in fields(SmallActionPlan)}

    assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)
    assert "raw_text" not in names
    assert "user_input" not in names
    assert "conversation" not in names
    assert "content" not in names
