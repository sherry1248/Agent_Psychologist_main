"""
Tests for structured memory models and rule-based extractors.
"""

import os

os.environ["LLM_TYPE"] = "MOCK"

import pytest

from src.memory.extractors import (
    build_recent_memory_entry,
    extract_emotional_states,
    extract_fact_candidates,
    extract_user_directives,
)
from src.memory.models import (
    EmotionalStateEntry,
    FactMemoryEntry,
    MemoryContext,
    RecentMemoryEntry,
    UserDirective,
)


def test_recent_memory_does_not_store_raw_masked_text():
    masked_text = "회사 일이 너무 힘들고 불안해서 잠을 못 자요"

    entry = build_recent_memory_entry(
        masked_text=masked_text,
        session_id="session-1",
        risk_stage="주의",
        created_at="2026-06-05T00:00:00Z",
    )

    assert isinstance(entry, RecentMemoryEntry)
    assert "work" in entry.key_topics
    assert "sleep" in entry.key_topics
    assert "anxiety" in entry.emotional_themes
    assert entry.risk_stage == "주의"
    assert masked_text not in entry.summary
    assert "회사 일이 너무 힘들고 불안해서 잠을 못 자요" not in str(entry.to_dict())


def test_recent_memory_rejects_raw_metadata_keys():
    with pytest.raises(ValueError):
        RecentMemoryEntry(
            session_id="session-1",
            summary="masked_turn_summary: topics=work",
            key_topics=["work"],
            emotional_themes=["anxiety"],
            risk_stage="관심",
            metadata={"raw_text": "원문을 저장하면 안 됩니다"},
        )


def test_user_directive_extraction_uses_normalized_terms_only():
    masked_text = "가족 얘기는 하기 싫어. 조언보다 그냥 들어줘."

    directives = extract_user_directives(
        masked_text=masked_text,
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )

    assert directives
    assert all(isinstance(item, UserDirective) for item in directives)
    assert any(item.kind == "avoid_topic" and item.term == "family" for item in directives)
    assert any(item.kind == "prefer_style" and item.term == "listening" for item in directives)
    assert all(masked_text not in str(item.to_dict()) for item in directives)


def test_emotion_label_extraction_returns_structured_entries():
    masked_text = "요즘 너무 불안하고 외로워요"

    states = extract_emotional_states(
        masked_text=masked_text,
        session_id="session-1",
        risk_stage="주의",
        created_at="2026-06-05T00:00:00Z",
    )

    labels = {state.label for state in states}
    assert all(isinstance(state, EmotionalStateEntry) for state in states)
    assert "anxiety" in labels
    assert "loneliness" in labels
    assert all(0 <= state.intensity <= 1 for state in states)
    assert all(0 <= state.confidence <= 1 for state in states)
    assert all(masked_text not in str(state.to_dict()) for state in states)


def test_fact_candidate_extraction_is_normalized():
    masked_text = "직장 스트레스 때문에 잠을 못 자고, 그냥 들어줬으면 좋겠어요"

    facts = extract_fact_candidates(
        masked_text=masked_text,
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )

    assert all(isinstance(item, FactMemoryEntry) for item in facts)
    assert any(item.category == "concern" and item.normalized_value == "work" for item in facts)
    assert any(item.category == "concern" and item.normalized_value == "sleep" for item in facts)
    assert any(item.category == "support_style" and item.normalized_value == "listening" for item in facts)
    assert all(masked_text not in str(item.to_dict()) for item in facts)


def test_memory_context_composes_structured_layers():
    recent = build_recent_memory_entry(
        masked_text="회사 일 때문에 불안해요",
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )
    facts = extract_fact_candidates(
        masked_text="회사 일 때문에 불안해요",
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )
    directives = extract_user_directives(
        masked_text="조언보다 들어줘",
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )
    states = extract_emotional_states(
        masked_text="회사 일 때문에 불안해요",
        session_id="session-1",
        created_at="2026-06-05T00:00:00Z",
    )

    context = MemoryContext(
        recent_summaries=[recent],
        facts=facts,
        directives=directives,
        emotional_trend=states,
    )

    assert not context.is_empty()
    data = context.to_dict()
    assert len(data["recent_summaries"]) == 1
    assert data["facts"]
    assert data["directives"]
    assert data["emotional_trend"]
