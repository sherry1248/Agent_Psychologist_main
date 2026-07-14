"""
Tests for the deterministic proactive recall agent.
"""

from dataclasses import fields

from src.agent.models import ProactiveRecallResult, RAW_TEXT_FIELD_NAMES, SmallActionPlan
from src.agent.recall import ProactiveRecallAgent, build_proactive_recall
from src.memory.models import (
    EmotionalStateEntry,
    FactMemoryEntry,
    MemoryContext,
    RecentMemoryEntry,
    UserDirective,
)


def _fact(
    normalized_value: str,
    evidence_count: int,
    category: str = "concern",
) -> FactMemoryEntry:
    return FactMemoryEntry(
        fact_id=f"fact-{normalized_value}",
        session_id="session-1",
        category=category,
        label=normalized_value,
        normalized_value=normalized_value,
        confidence=0.9,
        evidence_count=evidence_count,
        first_seen_at="2026-01-01T00:00:00Z",
        last_seen_at="2026-01-02T00:00:00Z",
    )


def test_last_small_action_is_included():
    plan = SmallActionPlan(
        action_id="action_12345678",
        title="Small action",
        action_text="지금 자리에서 발바닥 감각을 30초만 느껴보세요.",
        status="suggested",
    )

    result = build_proactive_recall(last_small_action=plan)

    assert result.last_small_action == plan.action_text
    assert "last_small_action" in result.recalled_keys
    assert "last_small_action_status_suggested" in result.recalled_keys


def test_next_followup_is_included_when_allowed():
    followup = "잠드는 데 오래 걸리는 편인가요, 아니면 자다가 자주 깨는 편인가요?"

    result = ProactiveRecallAgent().build(next_followup=followup)

    assert result.next_follow_up == followup
    assert "next_follow_up" in result.recalled_keys


def test_preferred_response_style_is_included():
    result = build_proactive_recall(preferred_response_style=["listening", "brief"])

    assert result.preferred_response_style == ["listening", "brief"]
    assert "preferred_response_style" in result.recalled_keys


def test_avoid_topics_are_included():
    result = build_proactive_recall(avoid_topics=["family"])

    assert result.avoid_topics == ["family"]
    assert "avoid_topics" in result.recalled_keys


def test_emotional_trend_summary_is_generated_from_labels():
    context = MemoryContext(
        emotional_trend=[
            EmotionalStateEntry(
                session_id="session-1",
                label="anxiety",
                intensity=0.8,
                confidence=0.9,
                source="agent",
                risk_stage="관심",
            ),
            EmotionalStateEntry(
                session_id="session-1",
                label="sleep_low",
                intensity=0.7,
                confidence=0.9,
                source="agent",
                risk_stage="관심",
            ),
        ]
    )

    result = build_proactive_recall(memory_context=context)

    assert result.emotional_trend_summary == "recent anxiety and poor sleep observed"
    assert "emotional_trend" in result.recalled_keys


def test_repeated_concerns_are_sorted_by_evidence_count():
    context = MemoryContext(
        facts=[
            _fact("sleep", evidence_count=2),
            _fact("anxiety", evidence_count=5),
            _fact("stress", evidence_count=3),
            _fact("listening", evidence_count=10, category="support_style"),
        ],
        recent_summaries=[
            RecentMemoryEntry(
                session_id="session-1",
                summary="raw-like summary should not be used",
                key_topics=["work"],
                emotional_themes=["stress"],
                risk_stage="관심",
            )
        ],
    )

    result = build_proactive_recall(memory_context=context)

    assert result.repeated_concerns[:3] == ["anxiety", "stress", "sleep"]
    assert "work" in result.repeated_concerns
    assert "listening" not in result.repeated_concerns
    assert "repeated_concerns" in result.recalled_keys


def test_active_directives_are_used_when_explicit_values_absent():
    context = MemoryContext(
        directives=[
            UserDirective(
                directive_id="directive-1",
                session_id="session-1",
                kind="prefer_style",
                term="gentle",
                active=True,
            ),
            UserDirective(
                directive_id="directive-2",
                session_id="session-1",
                kind="avoid_topic",
                term="school",
                active=True,
            ),
            UserDirective(
                directive_id="directive-3",
                session_id="session-1",
                kind="prefer_style",
                term="inactive",
                active=False,
            ),
        ]
    )

    result = build_proactive_recall(memory_context=context)

    assert result.preferred_response_style == ["gentle"]
    assert result.avoid_topics == ["school"]
    assert "inactive" not in result.preferred_response_style


def test_next_followup_is_excluded_when_it_conflicts_with_avoid_topics():
    result = build_proactive_recall(
        next_followup="가족 이야기를 조금 더 해볼까요?",
        avoid_topics=["가족"],
    )

    assert result.next_follow_up == ""
    assert "next_follow_up" not in result.recalled_keys


def test_proactive_recall_result_has_no_raw_text_fields():
    names = {item.name for item in fields(ProactiveRecallResult)}

    assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)
    assert "raw_text" not in names
    assert "user_input" not in names
    assert "conversation" not in names
    assert "content" not in names


def test_unknown_raw_looking_dict_is_not_output_whole():
    raw_value = "사용자 원문을 통째로 넣으면 안 됩니다"
    context = MemoryContext(
        facts=[
            {
                "category": "concern",
                "label": "",
                "normalized_value": "",
                "evidence_count": 9,
                "raw_text": raw_value,
            }
        ],
        recent_summaries=[
            {
                "summary": raw_value,
                "content": raw_value,
                "key_topics": ["sleep"],
                "emotional_themes": ["anxiety"],
            }
        ],
    )

    result = build_proactive_recall(memory_context=context)
    rendered = str(result.to_dict())

    assert raw_value not in rendered
    assert "raw_text" not in rendered
    assert "content" not in rendered
    assert result.repeated_concerns == ["sleep"]
