"""
Tests for the deterministic rule-based follow-up question generator.
"""

from dataclasses import fields, is_dataclass

from src.agent.followup import (
    GENERAL_FOLLOWUP_QUESTION,
    FollowUpQuestionGenerator,
    generate_followup_question,
)
from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    IntentAgentResult,
    IntentLabel,
    IntentSeverity,
    RAW_TEXT_FIELD_NAMES,
)


def _intent(label: IntentLabel, s3_sos: bool = False) -> IntentAgentResult:
    return IntentAgentResult(
        primary_intent=label,
        severity=IntentSeverity.S3_SOS if s3_sos else IntentSeverity.S1_CONCERN,
        s3_sos=s3_sos,
    )


def test_sleep_problem_generates_sleep_followup_question():
    question = generate_followup_question(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
    )

    assert "잠드는 데" in question
    assert "자주 깨는" in question


def test_anxiety_support_generates_anxiety_followup_question():
    question = FollowUpQuestionGenerator().generate(
        intent_result=_intent(IntentLabel.ANXIETY_SUPPORT),
    )

    assert "불안" in question
    assert "가장 크게 올라오는 순간" in question


def test_danger_risk_stage_returns_empty_question():
    question = generate_followup_question(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
        risk_stage="위험",
    )

    assert question == ""


def test_safety_escalation_decision_returns_empty_question():
    decision = DecisionAgentResult(
        primary_action=DecisionAction.ESCALATE_SAFETY,
    )

    question = generate_followup_question(
        intent_result=_intent(IntentLabel.ANXIETY_SUPPORT),
        decision_result=decision,
    )

    assert question == ""


def test_previous_followup_is_preferred_when_allowed():
    previous = "이전 질문을 그대로 이어가도 괜찮을까요?"

    question = generate_followup_question(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
        previous_followup=previous,
    )

    assert question == previous


def test_generates_at_most_one_question():
    question = generate_followup_question(
        intent_result=_intent(IntentLabel.WORK_OR_STUDY_STRESS),
    )

    assert question
    assert question.count("?") + question.count("？") <= 1


def test_avoid_topics_falls_back_to_general_question():
    question = generate_followup_question(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM),
        avoid_topics=["잠"],
    )

    assert question == GENERAL_FOLLOWUP_QUESTION


def test_s3_sos_intent_returns_empty_question():
    question = generate_followup_question(
        intent_result=_intent(IntentLabel.SLEEP_PROBLEM, s3_sos=True),
    )

    assert question == ""


def test_followup_generator_defines_no_raw_text_fields():
    assert is_dataclass(FollowUpQuestionGenerator)

    names = {item.name for item in fields(FollowUpQuestionGenerator)}

    assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)
    assert "raw_text" not in names
    assert "user_input" not in names
    assert "conversation" not in names
    assert "content" not in names
