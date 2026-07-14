"""Tests for shallow agent pipeline integration in src.main."""

import asyncio
import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

os.environ["LLM_TYPE"] = "MOCK"

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda _: None))

if "pydantic" not in sys.modules:
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self):
            return dict(self.__dict__)

    def Field(default=None, **kwargs):
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=BaseModel, Field=Field)

sys.modules.setdefault(
    "numpy",
    types.SimpleNamespace(
        ndarray=object,
        argmax=lambda values: 0,
        max=max,
    ),
)

from src.main import AgentConfig, PsychologistAgent
from src.api.models import AnalysisResult
from src.counseling.retriever import CounselingRetriever
from src.empathy.retriever import EmpathyRetriever
from src.inference.generator import GenerationResult
from src.wellness.recommender import WellnessRecommender


RAW_KEYS = (
    "raw_text",
    "user_input",
    "conversation",
    "content",
    "assistant_response",
)


async def _run_message(message: str, wellness_checkin=None):
    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=True,
    )
    agent.counseling_retriever = SimpleNamespace(
        recommend=lambda _: SimpleNamespace(
            intervention_hint="지금 당장 해결하려 하기보다, 가장 작은 한 가지를 정해보세요.",
            matched_record_id="counseling-test",
            category="sleep",
            score=1.0,
        )
    )
    agent.empathy_retriever = SimpleNamespace(
        recommend=lambda _: SimpleNamespace(
            empathy_style_hint="감정을 먼저 확인하고 차분하게 공감하세요.",
            emotion_label="불안",
            empathy_label="위로",
            matched_record_id="empathy-test",
            score=1.0,
        )
    )
    agent.wellness_recommender = SimpleNamespace(
        recommend=lambda _: SimpleNamespace(
            support_hint="지금 자리에서 발바닥 감각을 30초만 느껴보세요.",
            risk_stage="관심",
            matched_record_id="wellness-test",
            matched_topic="anxiety",
            distance=0.1,
        )
    )
    await agent.initialize()
    session = await agent.session_manager.create_session()
    try:
        return await agent.process_message(
            user_input=message,
            session_id=session.session_id,
            wellness_checkin=wellness_checkin,
        )
    finally:
        await agent.shutdown()


def test_non_crisis_input_creates_agents_pipeline_details():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    agents = result["pipeline_details"].get("agents")

    assert isinstance(agents, dict)
    assert "intent" in agents
    assert "emotional_state" in agents
    assert "decision" in agents
    assert "followup" in agents
    assert "small_action" in agents


def test_non_crisis_mock_flow_creates_agent_prompt_context():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    prompt_context = result["pipeline_details"]["agents"].get("prompt_context")

    assert isinstance(prompt_context, dict)
    assert "decision" in prompt_context
    assert "emotional_state" in prompt_context
    assert "proactive_recall" in prompt_context
    assert "followup" in prompt_context
    assert "small_action" in prompt_context
    assert "primary_action" in prompt_context["decision"]
    assert "state_summary" in prompt_context["emotional_state"]


def test_sleep_and_anxiety_input_sets_sleep_intent_label():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    intent = result["pipeline_details"]["agents"]["intent"]

    assert (
        intent["primary_intent"] == "SLEEP_PROBLEM"
        or "SLEEP_PROBLEM" in intent["labels"]
    )


def test_decision_selects_followup_or_small_action():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            wellness_checkin={
                "mood_score": 3,
                "anxiety_score": 5,
                "loneliness_score": 3,
                "sleep_quality": 1,
                "meal_status": 3,
                "energy_score": 2,
                "stress_score": 4,
            },
        )
    )

    decision = result["pipeline_details"]["agents"]["decision"]

    assert (
        decision["primary_action"] == "ASK_FOLLOW_UP"
        or "SUGGEST_SMALL_ACTION" in decision["secondary_actions"]
    )


def test_mock_response_includes_followup_question():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    followup = result["pipeline_details"]["agents"]["followup"]

    assert followup["has_question"] is True
    assert followup["question"]
    assert followup["question"] in result["response"]


def test_mock_response_includes_small_action_when_planned():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            wellness_checkin={
                "mood_score": 3,
                "anxiety_score": 5,
                "loneliness_score": 3,
                "sleep_quality": 1,
                "meal_status": 3,
                "energy_score": 2,
                "stress_score": 4,
            },
        )
    )

    small_action = result["pipeline_details"]["agents"]["small_action"]

    assert small_action["has_action"] is True
    assert small_action["status"] == "suggested"
    assert small_action["action_text"] in result["response"]
    assert "해볼까" in result["response"] or "해보자" in result["response"]


def test_sleep_anxiety_mock_response_does_not_include_low_mood_phrase():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    assert "기분이 우울" not in result["response"]
    assert "우울하시군요" not in result["response"]


def test_sleep_anxiety_followup_is_sleep_related():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    followup = result["pipeline_details"]["agents"]["followup"]["question"]

    assert "잠들기 전 걱정이 많아지는 편인가요" in followup
    assert followup in result["response"]


def test_sleep_anxiety_small_action_is_action_not_empathy_sentence():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            wellness_checkin={
                "mood_score": 3,
                "anxiety_score": 5,
                "loneliness_score": 3,
                "sleep_quality": 1,
                "meal_status": 3,
                "energy_score": 2,
                "stress_score": 4,
            },
        )
    )

    small_action = result["pipeline_details"]["agents"]["small_action"]
    action_text = small_action["action_text"]

    assert any(keyword in action_text for keyword in ("잠", "수면", "불안", "발바닥", "화면", "조명"))
    assert "감정을 먼저 확인" not in action_text
    assert "공감" not in action_text
    assert "기분이 우울" not in action_text
    assert action_text in result["response"]
    assert "해볼까" in result["response"] or "해보자" in result["response"]


def test_risk_stage_matches_safety_agent_view():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            wellness_checkin={
                "mood_score": 3,
                "anxiety_score": 5,
                "loneliness_score": 3,
                "sleep_quality": 1,
                "meal_status": 3,
                "energy_score": 2,
                "stress_score": 4,
            },
        )
    )

    safety = result["pipeline_details"]["agents"]["safety"]

    assert result["risk_stage"] == safety["risk_stage"]


def test_crisis_flow_takes_priority_without_general_followup_or_small_action():
    result = asyncio.run(_run_message("죽고 싶어요. 지금 자해하고 싶어요."))

    agents = result["pipeline_details"].get("agents", {})

    assert result["requires_crisis_response"] is True
    assert result["risk_stage"] == "위험"
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "prompt_context" not in agents
    assert "잠드는 데 오래 걸리는 편인가요" not in result["response"]
    assert "발바닥 감각" not in result["response"]


def test_non_mock_local_prompt_receives_agent_context():
    class PromptCapture:
        def __init__(self):
            self.local_kwargs = None

        def gen_cloud_prompt(self, **kwargs):
            return SimpleNamespace(system_message="system", user_message="user")

        def gen_local_prompt(self, **kwargs):
            self.local_kwargs = kwargs
            return SimpleNamespace(
                to_messages=lambda: [{"role": "user", "content": "user"}],
            )

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_risk_audit=False,
            enable_audit_logging=False,
        ),
        mock_mode=True,
    )
    agent.mock_mode = False
    agent._initialized = True
    agent.prompt_generator = PromptCapture()
    agent.pii_redactor = SimpleNamespace(
        redact=Mock(return_value=SimpleNamespace(redacted_text="redacted", entity_count=0, entities=[]))
    )
    agent.cloud_client = SimpleNamespace(analyze=AsyncMock(return_value=AnalysisResult()))
    agent.local_generator = SimpleNamespace(
        create_chat_completion=AsyncMock(
            return_value=GenerationResult(
                text="Generated response",
                tokens_generated=2,
                finish_reason="stop",
                generation_time_ms=1.0,
            )
        )
    )
    agent.memory_store = SimpleNamespace(
        get_memory_context=AsyncMock(
            return_value=SimpleNamespace(
                is_empty=lambda: True,
                recent_summaries=[],
                facts=[],
                directives=[],
                emotional_trend=[],
            )
        ),
        get_cloud_context=AsyncMock(return_value=([], None)),
        get_local_context=AsyncMock(return_value=[]),
    )
    agent.session_manager = SimpleNamespace(
        add_to_history=AsyncMock(),
        update_activity=AsyncMock(),
    )
    agent.counseling_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                intervention_hint="작은 단계를 제안하세요.",
                matched_record_id="counseling-test",
                category="support",
                score=1.0,
            )
        )
    )
    agent.empathy_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                empathy_style_hint="차분하게 공감하세요.",
                emotion_label="불안",
                empathy_label="위로",
                matched_record_id="empathy-test",
                score=1.0,
            )
        )
    )
    agent.wellness_recommender = SimpleNamespace(recommend=Mock(return_value=None))

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-1"))
    agent_context = agent.prompt_generator.local_kwargs["agent_context"]

    assert result["requires_crisis_response"] is False
    assert agent_context["decision"]["primary_action"]
    assert agent_context["emotional_state"]["state_summary"]
    assert "question" in agent_context["followup"]
    assert "small_action" in agent_context


def test_agents_pipeline_details_do_not_include_raw_looking_keys():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))
    rendered = str(result["pipeline_details"].get("agents", {}))

    for key in RAW_KEYS:
        assert key not in rendered


def test_internal_hint_labels_are_not_exposed_in_response():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    assert "상담 참고" not in result["response"]
    assert "공감 참고" not in result["response"]
    assert "웰니스 참고" not in result["response"]


def test_pipeline_details_include_debug_timing_without_raw_text():
    raw_message = "요즘 잠을 못 자고 불안해요"
    result = asyncio.run(_run_message(raw_message))

    timing = result["pipeline_details"].get("timing")

    assert isinstance(timing, dict)
    for key in (
        "safety",
        "dataset_retrieval",
        "memory_context",
        "agent_pipeline",
        "response_generation",
        "total",
    ):
        assert key in timing
        assert isinstance(timing[key], (int, float))
        assert timing[key] >= 0
    assert raw_message not in str(timing)


def test_mock_mode_does_not_call_cloud_or_local_generation():
    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=True,
    )
    agent._initialized = True
    agent.cloud_client = SimpleNamespace(analyze=AsyncMock())
    agent.local_generator = SimpleNamespace(create_chat_completion=AsyncMock())
    agent.counseling_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                intervention_hint="작은 단계를 제안하세요.",
                matched_record_id="counseling-test",
                category="sleep",
                score=1.0,
            )
        )
    )
    agent.empathy_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                empathy_style_hint="차분하게 공감하세요.",
                emotion_label="불안",
                empathy_label="위로",
                matched_record_id="empathy-test",
                score=1.0,
            )
        )
    )
    agent.wellness_recommender = SimpleNamespace(recommend=Mock(return_value=None))
    agent.memory_store = SimpleNamespace(
        get_memory_context=AsyncMock(
            return_value=SimpleNamespace(
                recent_summaries=[],
                facts=[],
                directives=[],
                emotional_trend=[],
            )
        )
    )
    agent.session_manager = SimpleNamespace(add_to_history=AsyncMock())

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-1"))

    assert result["response"]
    agent.cloud_client.analyze.assert_not_called()
    agent.local_generator.create_chat_completion.assert_not_called()


def test_ollama_mode_falls_back_when_server_is_unavailable(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_ollama_response = AsyncMock(return_value="")

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-ollama"))

    generation = result["pipeline_details"]["response_generation"]
    assert result["response"]
    assert generation["llm_type"] == "OLLAMA"
    assert generation["model"] == "qwen2.5-coder:3b"
    assert generation["timeout_seconds"] == 3.0
    assert generation["generation_strategy"] == "agent_pipeline_with_optional_ollama_naturalization"
    assert generation["used_ollama"] is False
    assert generation["fallback_used"] is True
    assert result["pipeline_details"]["agents"]["followup"]["question"] in result["response"]


def test_ollama_mode_times_out_and_uses_agent_pipeline_fallback(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "0.01")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )

    async def slow_ollama(**kwargs):
        await asyncio.sleep(0.2)
        return "too late"

    agent._generate_ollama_response = slow_ollama

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-ollama-timeout"))

    generation = result["pipeline_details"]["response_generation"]
    assert result["response"]
    assert "too late" not in result["response"]
    assert generation["timeout_seconds"] == 0.01
    assert generation["used_ollama"] is False
    assert generation["fallback_used"] is True


def test_ollama_mode_is_not_mock_and_uses_generated_text(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_ollama_response = AsyncMock(return_value="올라마가 만든 따뜻한 응답입니다.")

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-ollama-success"))

    generation = result["pipeline_details"]["response_generation"]
    assert agent.mock_mode is False
    assert agent.use_ollama_generation is True
    assert "올라마가 만든 따뜻한 응답입니다." in result["response"]
    assert generation["llm_type"] == "OLLAMA"
    assert generation["model"] == "qwen2.5-coder:3b"
    assert generation["timeout_seconds"] == 3.0
    assert generation["generation_strategy"] == "agent_pipeline_with_optional_ollama_naturalization"
    assert generation["attempted_ollama"] is True
    assert generation["used_ollama"] is True
    assert generation["fallback_used"] is False


def test_gemini_mode_falls_back_when_api_key_is_missing(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-gemini-missing-key"))

    generation = result["pipeline_details"]["response_generation"]
    assert result["response"]
    assert agent.mock_mode is False
    assert agent.use_gemini_generation is True
    assert generation["llm_type"] == "GEMINI"
    assert generation["provider"] == "GEMINI"
    assert generation["model"] == "gemini-2.5-flash"
    assert generation["timeout_seconds"] == 4.0
    assert generation["generation_strategy"] == "agent_pipeline_with_gemini_naturalization"
    assert generation["used_cloud_llm"] is False
    assert generation["fallback_used"] is True


def test_gemini_mode_uses_generated_text(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    gemini_response = "지금 많이 힘들었겠어요. 잠깐 숨을 고르면서 부담을 나눠봐도 괜찮습니다. 오늘은 가장 작은 한 가지만 정해보세요."
    agent._generate_gemini_response = AsyncMock(return_value=gemini_response)

    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-gemini-success"))

    generation = result["pipeline_details"]["response_generation"]
    assert gemini_response in result["response"]
    assert generation["llm_type"] == "GEMINI"
    assert generation["provider"] == "GEMINI"
    assert generation["used_cloud_llm"] is True
    assert generation["fallback_used"] is False


def test_incomplete_gemini_study_fragment_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_gemini_response = AsyncMock(return_value="공부 때문에 많이 힘드")

    result = asyncio.run(agent.process_message("공부하느라 너무 힘들어", "session-gemini-truncated-study"))

    generation = result["pipeline_details"]["response_generation"]
    assert "공부 때문에 많이 힘드" not in result["response"]
    assert generation["used_cloud_llm"] is False
    assert generation["fallback_used"] is True
    assert result["response"].rstrip().endswith(("요.", "다.", "세요.", "요", "다", "세요"))


def test_incomplete_gemini_exam_fragment_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_gemini_response = AsyncMock(return_value="시험 때문에 마음이 많이 무겁고 부담")

    result = asyncio.run(agent.process_message("시험 때문에 너무 부담돼", "session-gemini-truncated-exam"))

    generation = result["pipeline_details"]["response_generation"]
    assert "시험 때문에 마음이 많이 무겁고 부담" not in result["response"]
    assert generation["used_cloud_llm"] is False
    assert generation["fallback_used"] is True
    assert "공부량" in result["response"] or "성적" in result["response"] or "지쳐서" in result["response"]


def test_study_exam_fallback_response_is_complete(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_safety_check=False,
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_gemini_response = AsyncMock(return_value="")

    result = asyncio.run(agent.process_message("시험 때문에 너무 부담돼", "session-study-fallback"))
    response = result["response"]

    assert "공부" in response or "시험" in response or "과제" in response
    assert "공부량" in response
    assert "성적" in response
    assert "가장 작은" in response or "10분" in response or "짧게" in response
    assert response.rstrip().endswith((".", "요", "다", "세요", "!", "?"))


def test_crisis_input_does_not_call_gemini_generation(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_gemini_response = AsyncMock(return_value="gemini response")

    result = asyncio.run(agent.process_message("죽고 싶어요. 지금 자해하고 싶어요.", "session-gemini-crisis"))

    assert result["requires_crisis_response"] is True
    assert result["risk_stage"] == "위험"
    agent._generate_gemini_response.assert_not_called()


def test_crisis_input_does_not_call_ollama_generation(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "OLLAMA")

    agent = PsychologistAgent(
        config=AgentConfig(
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=None,
    )
    agent._generate_ollama_response = AsyncMock(return_value="ollama response")

    result = asyncio.run(agent.process_message("죽고 싶어요. 지금 자해하고 싶어요.", "session-crisis"))

    assert result["requires_crisis_response"] is True
    assert result["risk_stage"] == "위험"
    agent._generate_ollama_response.assert_not_called()


def _write_safe_hints(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_dataset": "unit",
                        "label_or_topic": "sleep anxiety",
                        "intent_hint": "수면 문제와 불안 지원 요청으로 해석한다.",
                        "emotion_hint": "불안을 먼저 인정하고 차분하게 공감한다.",
                        "cause_hint": "수면, 걱정, 생활 리듬을 원인 후보로 점검한다.",
                        "action_hint": "잠들기 전 걱정 하나를 짧게 적어보세요.",
                        "safety_hint": "명시적 위기 신호는 낮지만 필요 시 안전 질문을 우선한다.",
                        "short_summary": "수면과 불안 상담 패턴을 요약한 안전 힌트다.",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source_dataset": "unit",
                        "label_or_topic": "relationship hurt",
                        "intent_hint": "관계 스트레스 지원 요청으로 해석한다.",
                        "emotion_hint": "상처와 서운함을 먼저 반영한다.",
                        "cause_hint": "대인관계 맥락을 원인 후보로 점검한다.",
                        "action_hint": "상처였던 표현 하나를 짧게 적어보세요.",
                        "safety_hint": "명시적 위기 신호는 낮다.",
                        "short_summary": "관계 상처 상담 패턴을 요약한 안전 힌트다.",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_live_chat_uses_cached_safe_hints_instead_of_processed_retrievers(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    safe_hints_path = tmp_path / "balanced_safe_hints.jsonl"
    _write_safe_hints(safe_hints_path)
    monkeypatch.setenv("SAFE_HINTS_PATH", str(safe_hints_path))
    monkeypatch.setattr(CounselingRetriever, "recommend", Mock(side_effect=AssertionError("processed counseling loaded")))
    monkeypatch.setattr(EmpathyRetriever, "recommend", Mock(side_effect=AssertionError("processed empathy loaded")))
    monkeypatch.setattr(WellnessRecommender, "recommend", Mock(side_effect=AssertionError("processed wellness loaded")))

    agent = PsychologistAgent(
        config=AgentConfig(enable_safety_check=False, enable_rag=False, enable_audit_logging=False),
        mock_mode=None,
    )
    result = asyncio.run(
        agent.process_message(
            "요즘 잠을 못 자고 불안해요",
            "session-safe-hints",
            wellness_checkin={"sleep_quality": 1, "anxiety_score": 5},
        )
    )

    assert result["response"]
    assert result["counseling_hint"]
    assert result["empathy_style_hint"]
    assert result["wellness_hint"]
    assert result["pipeline_details"]["safe_hints"]["cache_size"] == 2
    assert result["pipeline_details"]["safe_hints"]["selected_count"] >= 1


def test_missing_balanced_safe_hints_does_not_crash_or_use_processed_retrievers(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SAFE_HINTS_PATH", str(tmp_path / "missing.jsonl"))
    monkeypatch.setattr(CounselingRetriever, "recommend", Mock(side_effect=AssertionError("processed counseling loaded")))
    monkeypatch.setattr(EmpathyRetriever, "recommend", Mock(side_effect=AssertionError("processed empathy loaded")))
    monkeypatch.setattr(WellnessRecommender, "recommend", Mock(side_effect=AssertionError("processed wellness loaded")))

    agent = PsychologistAgent(
        config=AgentConfig(enable_safety_check=False, enable_rag=False, enable_audit_logging=False),
        mock_mode=None,
    )
    result = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-missing-safe-hints"))

    assert result["response"]
    assert result["pipeline_details"]["safe_hints"]["available"] is False
    assert result["pipeline_details"]["safe_hints"]["selected_count"] == 0


def test_safe_hints_cache_is_reused_across_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "MOCK")
    safe_hints_path = tmp_path / "balanced_safe_hints.jsonl"
    _write_safe_hints(safe_hints_path)
    monkeypatch.setenv("SAFE_HINTS_PATH", str(safe_hints_path))

    agent = PsychologistAgent(
        config=AgentConfig(enable_safety_check=False, enable_rag=False, enable_audit_logging=False),
        mock_mode=None,
    )
    asyncio.run(agent.initialize())
    assert len(agent.safe_hints_cache) == 2
    agent._load_safe_hints_once = Mock(side_effect=AssertionError("safe hints reloaded"))

    first = asyncio.run(agent.process_message("요즘 잠을 못 자고 불안해요", "session-cache"))
    second = asyncio.run(agent.process_message("친구 때문에 마음이 상했어요", "session-cache"))

    assert first["pipeline_details"]["safe_hints"]["cache_size"] == 2
    assert second["pipeline_details"]["safe_hints"]["cache_size"] == 2
    agent._load_safe_hints_once.assert_not_called()


def test_crisis_input_does_not_call_processed_retrievers(monkeypatch):
    monkeypatch.setenv("LLM_TYPE", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(CounselingRetriever, "recommend", Mock(side_effect=AssertionError("processed counseling loaded")))
    monkeypatch.setattr(EmpathyRetriever, "recommend", Mock(side_effect=AssertionError("processed empathy loaded")))
    monkeypatch.setattr(WellnessRecommender, "recommend", Mock(side_effect=AssertionError("processed wellness loaded")))

    agent = PsychologistAgent(config=AgentConfig(enable_rag=False, enable_audit_logging=False), mock_mode=None)
    agent._generate_gemini_response = AsyncMock(return_value="gemini response")

    result = asyncio.run(agent.process_message("죽고 싶어요. 지금 자해하고 싶어요.", "session-crisis-no-datasets"))

    assert result["requires_crisis_response"] is True
    agent._generate_gemini_response.assert_not_called()
