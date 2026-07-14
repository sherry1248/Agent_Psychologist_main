"""
Tests for agent pipeline schemas.
"""

from dataclasses import fields

from src.agent.models import (
    ALL_AGENT_SCHEMAS,
    DatasetStrategyResult,
    DecisionAction,
    DecisionAgentResult,
    EmotionAgentResult,
    EmotionLabel,
    EmotionalStateVector,
    IntentAgentResult,
    IntentCandidate,
    IntentLabel,
    IntentSeverity,
    ProactiveRecallResult,
    RAW_TEXT_FIELD_NAMES,
    SafetyAgentResult,
    SessionDreamSummary,
    SmallActionPlan,
    validate_no_raw_fields,
)


def test_agent_schemas_are_importable():
    assert SafetyAgentResult()
    assert EmotionAgentResult()
    assert IntentCandidate(label=IntentLabel.SUPPORT_REQUEST)
    assert IntentAgentResult()
    assert DatasetStrategyResult()
    assert ProactiveRecallResult()
    assert EmotionalStateVector()
    assert DecisionAgentResult()
    assert SmallActionPlan(action_id="micro-breathing", title="Breathing pause")
    assert SessionDreamSummary(session_id="session-1", summary_id="summary-1")
    assert EmotionLabel.ANXIETY.value == "anxiety"
    assert IntentSeverity.CRITICAL.value == "critical"


def test_emotional_state_vector_clamps_scores_to_unit_range():
    state = EmotionalStateVector(
        mood=-0.5,
        anxiety=1.7,
        stress=2.0,
        sleep=-1,
        energy=0.75,
        safety=9,
        rapport=-3,
    )

    assert state.mood == 0.0
    assert state.anxiety == 1.0
    assert state.stress == 1.0
    assert state.sleep == 0.0
    assert state.energy == 0.75
    assert state.safety == 1.0
    assert state.rapport == 0.0


def test_emotional_state_vector_defaults_match_design():
    state = EmotionalStateVector()

    assert state.to_dict() == {
        "mood": 0.5,
        "anxiety": 0.3,
        "stress": 0.3,
        "sleep": 0.5,
        "energy": 0.5,
        "safety": 0.9,
        "rapport": 0.2,
    }


def test_decision_action_enum_values_are_expected():
    assert {item.value for item in DecisionAction} == {
        "respond_supportively",
        "ask_follow_up",
        "suggest_small_action",
        "summarize_state",
        "update_memory",
        "escalate_safety",
    }


def test_agent_schemas_do_not_define_raw_text_fields():
    assert validate_no_raw_fields(ALL_AGENT_SCHEMAS) is True

    for schema_class in ALL_AGENT_SCHEMAS:
        names = {item.name for item in fields(schema_class)}
        assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)


def test_session_dream_summary_does_not_store_raw_conversation():
    names = {item.name for item in fields(SessionDreamSummary)}

    assert "conversation" not in names
    assert "content" not in names
    assert "user_input" not in names
    assert "assistant_response" not in names

    summary = SessionDreamSummary(
        session_id="session-1",
        summary_id="summary-1",
        emotional_arc=[EmotionLabel.ANXIETY, EmotionLabel.RELIEF],
        recurring_themes=["work", "sleep"],
        memory_updates=["prefers_listening"],
        unresolved_needs=["rest"],
        safety_notes=["no_escalation"],
        next_session_focus=["sleep_hygiene"],
    )

    data = summary.to_dict()
    assert "conversation" not in data
    assert "content" not in data
    assert data["recurring_themes"] == ["work", "sleep"]

