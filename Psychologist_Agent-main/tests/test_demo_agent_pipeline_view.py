"""Tests for the demo Agent Pipeline View helper."""

import asyncio
from types import SimpleNamespace

from demo import app as demo_app
from demo.app import build_agent_pipeline_markdown


RAW_KEYS = (
    "user_input",
    "raw_text",
    "conversation",
    "content",
    "assistant_response",
)

INTERNAL_HINT_LABELS = (
    "상담 참고",
    "공감 참고",
    "웰니스 참고",
    "심리상담 데이터 기반 힌트",
    "공감형 대화 기반 힌트",
    "웰니스 기반 힌트",
)


def test_pipeline_markdown_includes_core_agent_sections():
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심", "risk_level": "attention"},
        {
            "pipeline_details": {
                "agents": {
                    "intent": {"primary_intent": "sleep_problem"},
                    "decision": {"primary_action": "ask_follow_up"},
                }
            }
        },
    )

    assert "Safety Agent" in markdown
    assert "Intent Agent" in markdown
    assert "Decision Agent" in markdown


def test_pipeline_markdown_does_not_include_raw_user_input():
    raw_input = "요즘 회사에서 있었던 일을 원문 그대로 보여주면 안 됩니다"
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심"},
        {
            "response": "safe response",
            "pipeline_details": {
                "agents": {
                    "intent": {
                        "primary_intent": "anxiety_support",
                        "user_input": raw_input,
                    }
                },
                "raw_text": raw_input,
            },
        },
    )

    assert raw_input not in markdown


def test_pipeline_markdown_does_not_include_raw_memory_transcript():
    raw_memory = "raw memory transcript should not be visible"
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심"},
        {
            "pipeline_details": {
                "agents": {
                    "memory_recall": {
                        "recalled_keys": ["next_follow_up"],
                        "conversation": raw_memory,
                        "content": raw_memory,
                    }
                }
            }
        },
    )

    assert raw_memory not in markdown


def test_pipeline_markdown_does_not_display_raw_looking_keys():
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심"},
        {
            "pipeline_details": {
                "agents": {
                    "memory": {
                        "user_input": "hidden",
                        "raw_text": "hidden",
                        "conversation": "hidden",
                        "content": "hidden",
                        "assistant_response": "hidden",
                    }
                }
            }
        },
    )

    for key in RAW_KEYS:
        assert key not in markdown


def test_pipeline_markdown_does_not_reveal_internal_hint_labels():
    markdown = build_agent_pipeline_markdown(
        {
            "risk_stage": "관심",
            "counseling_hint": "상담 참고: 내부 힌트",
            "empathy_style_hint": "공감 참고: 내부 힌트",
            "wellness_hint": "웰니스 참고: 내부 힌트",
        },
        {
            "pipeline_details": {
                "counseling": {
                    "intervention_hint": "심리상담 데이터 기반 힌트 원문",
                    "category": "stress",
                    "score": 0.8,
                },
                "empathy": {
                    "intervention_hint": "공감형 대화 기반 힌트 원문",
                    "category": "validation",
                    "score": 0.7,
                },
                "wellness": {
                    "support_hint": "웰니스 기반 힌트 원문",
                    "category": "sleep",
                    "score": 0.6,
                },
            }
        },
    )

    for label in INTERNAL_HINT_LABELS:
        assert label not in markdown


def test_pipeline_markdown_falls_back_without_agents_block():
    markdown = build_agent_pipeline_markdown(
        {
            "risk_stage": "주의",
            "requires_crisis_response": False,
            "risk_level": "moderate",
            "counseling_hint": "internal hint should not appear",
        },
        {
            "pipeline_details": {
                "counseling": {"category": "stress", "score": 0.75},
                "memory_context": {
                    "recent_summaries": 1,
                    "facts": 2,
                    "directives": 1,
                    "emotional_trend": 3,
                },
            }
        },
    )

    assert "Agent Pipeline View" in markdown
    assert "risk_stage: 주의" in markdown
    assert "counseling: category=stress, score=0.75" in markdown
    assert "recent_summaries=1" in markdown
    assert "internal hint should not appear" not in markdown


def test_pipeline_markdown_uses_agent_results_when_available():
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심"},
        {
            "response": "response body is not displayed in pipeline view",
            "pipeline_details": {
                "agents": {
                    "safety": {
                        "risk_stage": "관심",
                        "requires_crisis_response": False,
                        "risk_level": "attention",
                    },
                    "intent": {
                        "primary_intent": "sleep_problem",
                        "s2_suspected": False,
                        "s3_sos": False,
                        "labels": ["sleep_problem", "anxiety_support"],
                    },
                    "decision": {
                        "primary_action": "ask_follow_up",
                        "secondary_actions": ["respond_supportively"],
                        "reason_codes": ["sleep_problem_needs_clarification"],
                        "response_constraints": {
                            "must_include_followup": True,
                            "max_questions": 1,
                        },
                    },
                }
            },
        },
    )

    assert "primary_intent: sleep_problem" in markdown
    assert "sleep_problem, anxiety_support" in markdown
    assert "primary_action: ask_follow_up" in markdown
    assert "sleep_problem_needs_clarification" in markdown
    assert "response body is not displayed" not in markdown


def test_pipeline_markdown_uses_memory_recall_agent_results():
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심"},
        {
            "pipeline_details": {
                "agents": {
                    "memory_recall": {
                        "recalled_keys": ["next_follow_up", "previous_emotional_state"],
                        "has_next_follow_up": True,
                    }
                }
            }
        },
    )

    assert "recalled_keys: next_follow_up, previous_emotional_state" in markdown


def test_dataset_strategy_agent_shows_hint_keys_not_hint_text():
    raw_hint = "이 힌트 문장 전체는 표시되면 안 됩니다"

    markdown = build_agent_pipeline_markdown(
        {
            "risk_stage": "관심",
            "counseling_hint": raw_hint,
            "empathy_style_hint": raw_hint,
            "wellness_hint": raw_hint,
        },
        {
            "counseling_hint": raw_hint,
            "pipeline_details": {
                "counseling": {
                    "category": "sleep",
                    "score": 0.9,
                    "intervention_hint": raw_hint,
                }
            },
        },
    )

    assert "hint_keys: counseling_hint, empathy_style_hint, wellness_hint" in markdown
    assert "category=sleep" in markdown
    assert "score=0.9" in markdown
    assert raw_hint not in markdown


def test_pipeline_markdown_uses_final_risk_stage_for_safety_agent():
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "주의", "risk_level": "moderate"},
        {
            "risk_stage": "주의",
            "risk_level": "moderate",
            "pipeline_details": {
                "agents": {
                    "safety": {
                        "risk_stage": "관심",
                        "risk_level": "none",
                    }
                }
            },
        },
    )

    assert "risk_stage: 주의" in markdown
    assert "risk_stage: 관심" not in markdown


def test_dataset_strategy_flags_low_confidence_category_without_raw_text():
    raw_text = "원문 데이터셋 문장은 노출되면 안 됩니다"
    markdown = build_agent_pipeline_markdown(
        {"risk_stage": "관심", "counseling_hint": "hint present"},
        {
            "pipeline_details": {
                "counseling": {
                    "category": "ADDICTION",
                    "score": 0.0,
                    "content": raw_text,
                }
            },
        },
    )

    assert "category=ADDICTION" in markdown
    assert "score=0.0" in markdown
    assert "low_confidence_match=True" in markdown
    assert raw_text not in markdown


def test_get_agent_caches_initialized_instance(monkeypatch):
    class FakeSessionManager:
        async def create_session(self):
            return SimpleNamespace(session_id="session-1")

    class FakeAgent:
        created = 0
        initialized = 0

        def __init__(self):
            FakeAgent.created += 1
            self.session_manager = FakeSessionManager()

        async def initialize(self):
            FakeAgent.initialized += 1

    async def scenario():
        demo_app.agent = None
        demo_app.current_session_id = None
        monkeypatch.setattr("src.main.PsychologistAgent", FakeAgent)
        first = await demo_app.get_agent()
        second = await demo_app.get_agent()
        return first, second

    first, second = asyncio.run(scenario())

    assert first is second
    assert FakeAgent.created == 1
    assert FakeAgent.initialized == 1
    demo_app.agent = None
    demo_app.current_session_id = None
