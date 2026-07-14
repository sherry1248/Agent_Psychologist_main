"""
Tests for MemoryStore structured memory layers.
"""

import asyncio
import os

os.environ["LLM_TYPE"] = "MOCK"

from src.memory.models import MemoryContext
from src.memory.store import MemoryStore


def run(coro):
    return asyncio.run(coro)


def test_existing_history_api_unchanged():
    async def scenario():
        store = MemoryStore()
        await store.add(
            session_id="session-1",
            user_input="요즘 회사 일 때문에 불안해요",
            response="많이 부담스러웠겠어요.",
        )

        history = await store.get_history("session-1")
        cloud_history, profile = await store.get_cloud_context("session-1")
        local_history = await store.get_local_context("session-1")
        memory_context = await store.get_memory_context("session-1")

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert "사용자 요약:" in history[0].content
        assert len(cloud_history) == 2
        assert len(local_history) == 2
        assert profile.user_id == "session-1"
        assert memory_context.is_empty()

    run(scenario())


def test_add_structured_memory_uses_masked_text_only():
    async def scenario():
        store = MemoryStore()
        masked_text = "회사 일이 너무 힘들고 불안해서 잠을 못 자요"

        await store.add_structured_memory(
            session_id="session-1",
            masked_text=masked_text,
            risk_stage="주의",
        )

        context = await store.get_memory_context("session-1")
        data = context.to_dict()

        assert not context.is_empty()
        assert masked_text not in str(data)
        assert data["recent_summaries"][0]["risk_stage"] == "주의"
        assert "work" in data["recent_summaries"][0]["key_topics"]
        assert "anxiety" in data["recent_summaries"][0]["emotional_themes"]

    run(scenario())


def test_get_memory_context_returns_all_structured_layers():
    async def scenario():
        store = MemoryStore()
        await store.add_structured_memory(
            session_id="session-1",
            masked_text="가족 얘기는 하기 싫어. 회사 일 때문에 불안하고 잠을 못 자요. 그냥 들어줘.",
            risk_stage="주의",
        )

        context = await store.get_memory_context("session-1")

        assert isinstance(context, MemoryContext)
        assert context.recent_summaries
        assert context.facts
        assert context.directives
        assert context.emotional_trend
        assert any(item.term == "family" for item in context.directives)
        assert any(item.label == "anxiety" for item in context.emotional_trend)

    run(scenario())


def test_fact_memory_merges_duplicates():
    async def scenario():
        store = MemoryStore()
        masked_text = "직장 스트레스 때문에 잠을 못 자요"

        await store.add_structured_memory("session-1", masked_text)
        await store.add_structured_memory("session-1", masked_text)

        facts = await store.get_fact_memory("session-1", categories=["concern"], limit=20)
        work_fact = next(
            item for item in facts
            if item.category == "concern" and item.normalized_value == "work"
        )
        sleep_fact = next(
            item for item in facts
            if item.category == "concern" and item.normalized_value == "sleep"
        )

        assert work_fact.evidence_count == 2
        assert sleep_fact.evidence_count == 2

    run(scenario())


def test_directive_memory_merges_duplicates():
    async def scenario():
        store = MemoryStore()
        masked_text = "가족 얘기는 하기 싫어. 그냥 들어줘."

        await store.add_structured_memory("session-1", masked_text)
        await store.add_structured_memory("session-1", masked_text)

        directives = await store.get_user_directives("session-1", limit=20)
        family_directive = next(
            item for item in directives
            if item.kind == "avoid_topic" and item.term == "family"
        )
        listening_directive = next(
            item for item in directives
            if item.kind == "prefer_style" and item.term == "listening"
        )

        assert family_directive.hit_count == 2
        assert listening_directive.hit_count == 2

    run(scenario())


def test_clear_structured_memory_and_clear_session():
    async def scenario():
        store = MemoryStore()

        await store.add_structured_memory(
            session_id="session-1",
            masked_text="회사 일 때문에 불안하고 잠을 못 자요",
        )
        assert not (await store.get_memory_context("session-1")).is_empty()

        await store.clear_structured_memory("session-1")
        assert (await store.get_memory_context("session-1")).is_empty()

        await store.add(
            session_id="session-1",
            user_input="요즘 회사 일 때문에 불안해요",
            response="많이 부담스러웠겠어요.",
        )
        await store.add_structured_memory(
            session_id="session-1",
            masked_text="회사 일 때문에 불안하고 잠을 못 자요",
        )
        assert not (await store.get_memory_context("session-1")).is_empty()

        await store.clear_session("session-1")
        assert await store.get_history("session-1") == []
        assert (await store.get_memory_context("session-1")).is_empty()

    run(scenario())
