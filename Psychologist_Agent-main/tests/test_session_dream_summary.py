"""
Tests for the deterministic session dream summary builder.
"""

from dataclasses import fields

from src.agent.models import (
    EmotionAgentResult,
    EmotionLabel,
    IntentAgentResult,
    IntentLabel,
    IntentSeverity,
    RAW_TEXT_FIELD_NAMES,
    SessionDreamSummary,
    SmallActionPlan,
)
from src.agent.summary import SessionDreamSummaryBuilder, build_session_dream_summary


RAW_SUMMARY_FIELD_NAMES = RAW_TEXT_FIELD_NAMES.union({"transcript", "message"})


def _intent(label: IntentLabel) -> IntentAgentResult:
    return IntentAgentResult(
        primary_intent=label,
        severity=IntentSeverity.S1_CONCERN,
    )


def test_intent_results_create_main_issue():
    summary = build_session_dream_summary(
        session_id="session-1",
        intent_results=[
            _intent(IntentLabel.SLEEP_PROBLEM),
            _intent(IntentLabel.ANXIETY_SUPPORT),
            _intent(IntentLabel.WORK_OR_STUDY_STRESS),
        ],
    )

    assert summary.main_issue == ["sleep", "anxiety", "work_or_study"]
    assert summary.recurring_themes == summary.main_issue


def test_emotion_results_create_emotional_trend():
    summary = SessionDreamSummaryBuilder().build(
        session_id="session-1",
        emotion_results=[
            EmotionAgentResult(
                primary_label=EmotionLabel.ANXIETY,
                intensity=0.8,
            ),
            EmotionAgentResult(
                primary_label=EmotionLabel.SADNESS,
                intensity=0.6,
            ),
        ],
    )

    assert "anxiety_high" in summary.emotional_trend
    assert "sadness_moderate" in summary.emotional_trend


def test_risk_stage_start_and_end_are_first_and_last_values():
    summary = build_session_dream_summary(
        session_id="session-1",
        risk_stages=["관심", "주의", "위험"],
    )

    assert summary.risk_stage_start == "관심"
    assert summary.risk_stage_end == "위험"


def test_last_small_action_is_included():
    plan = SmallActionPlan(
        action_id="action_12345678",
        title="Small action",
        action_text="물 한 잔을 마시고 창문 근처에서 1분만 서 있어보세요.",
    )

    summary = build_session_dream_summary(
        session_id="session-1",
        last_small_action=plan,
    )

    assert summary.last_small_action == plan.action_text


def test_next_followup_is_included():
    followup = "지금 이야기에서 가장 먼저 다뤄보고 싶은 부분은 무엇인가요?"

    summary = build_session_dream_summary(
        session_id="session-1",
        next_followup=followup,
    )

    assert summary.next_follow_up == followup


def test_important_user_directives_are_included():
    summary = build_session_dream_summary(
        session_id="session-1",
        important_user_directives=["prefer_style:listening", "avoid_topic:family"],
    )

    assert summary.important_user_directives == [
        "prefer_style:listening",
        "avoid_topic:family",
    ]
    assert summary.memory_updates == summary.important_user_directives


def test_session_dream_summary_has_no_raw_text_fields():
    names = {item.name for item in fields(SessionDreamSummary)}

    assert names.isdisjoint(RAW_SUMMARY_FIELD_NAMES)
    assert "raw_text" not in names
    assert "user_input" not in names
    assert "assistant_response" not in names
    assert "conversation" not in names
    assert "content" not in names
    assert "transcript" not in names
    assert "message" not in names


def test_raw_conversation_sentence_is_not_used_as_issue_or_emotional_trend():
    raw_sentence = "사용자가 오늘 회사에서 있었던 일을 길게 말했습니다."

    intent = _intent(IntentLabel.SLEEP_PROBLEM)
    intent.primary_intent = raw_sentence
    emotion = EmotionAgentResult(primary_label=EmotionLabel.ANXIETY, intensity=0.8)
    emotion.primary_label = raw_sentence

    summary = build_session_dream_summary(
        session_id="session-1",
        intent_results=[intent],
        emotion_results=[emotion],
    )

    assert raw_sentence not in summary.main_issue
    assert raw_sentence not in summary.emotional_trend
    assert summary.main_issue == []
    assert summary.emotional_trend == []


def test_default_summary_is_created_with_minimal_input():
    summary = build_session_dream_summary(session_id="session-1")

    assert summary.session_id == "session-1"
    assert summary.summary_id.startswith("dream_")
    assert summary.main_issue == []
    assert summary.emotional_trend == []
    assert summary.risk_stage_start == "관심"
    assert summary.risk_stage_end == "관심"
    assert summary.last_small_action == ""
    assert summary.next_follow_up == ""
    assert summary.important_user_directives == []
    assert summary.created_at
