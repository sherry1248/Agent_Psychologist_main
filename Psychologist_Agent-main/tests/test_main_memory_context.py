"""
Tests for MemoryContext wiring in the main agent pipeline.
"""

import os
import sys
import types
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

os.environ["LLM_TYPE"] = "MOCK"

# The test environment may not have PyYAML installed. These tests do not
# exercise YAML parsing, so a minimal stub keeps imports focused on wiring.
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda _: None))

if "pydantic" not in sys.modules:
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

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

from src.api.models import AnalysisResult, RiskLevel as ApiRiskLevel
from src.audit.risk_checker import InterventionLevel, RiskAssessment
from src.inference.generator import GenerationResult
from src.main import PsychologistAgent
from src.safety.gateway import SafetyResult
from src.safety.patterns import RiskLevel as SafetyRiskLevel


class PromptCapture:
    def __init__(self):
        self.cloud_kwargs = None
        self.local_kwargs = None

    def gen_cloud_prompt(self, **kwargs):
        self.cloud_kwargs = kwargs
        return SimpleNamespace(system_message="system", user_message="user")

    def gen_local_prompt(self, **kwargs):
        self.local_kwargs = kwargs
        return SimpleNamespace(
            system_message="system",
            user_message="user",
            full_prompt="full",
            to_messages=lambda: [{"role": "user", "content": "user"}],
        )


def normal_risk_assessment():
    return RiskAssessment(
        risk_level=ApiRiskLevel.LOW,
        risk_stage="관심",
        intervention_level=InterventionLevel.SUPPORTIVE,
        requires_crisis_response=False,
        requires_escalation=False,
        confidence=1.0,
    )


def crisis_risk_assessment():
    return RiskAssessment(
        risk_level=ApiRiskLevel.HIGH,
        risk_stage="위험",
        intervention_level=InterventionLevel.CRISIS,
        requires_crisis_response=True,
        requires_escalation=True,
        confidence=1.0,
    )


def safe_result():
    return SafetyResult(
        is_safe=True,
        risk_level=SafetyRiskLevel.NONE,
        risk_stage="관심",
        similarity_score=0.0,
        action="allow",
    )


def crisis_safety_result():
    return SafetyResult(
        is_safe=False,
        risk_level=SafetyRiskLevel.HIGH,
        risk_stage="위험",
        matched_pattern="self_harm",
        matched_category="self_harm",
        similarity_score=1.0,
        response="Please contact emergency support now.",
        resources=[],
        action="crisis_response",
    )


def build_agent(memory_context=None, memory_side_effect=None, safety=None, risk_side_effect=None):
    agent = PsychologistAgent(mock_mode=True)
    agent.mock_mode = False
    agent._initialized = True
    agent.audit_logger = None

    agent.safety_gateway = SimpleNamespace(check=AsyncMock(return_value=safety or safe_result()))
    agent.risk_checker = SimpleNamespace(
        assess=Mock(side_effect=risk_side_effect or [normal_risk_assessment(), normal_risk_assessment()])
    )
    agent.crisis_handler = SimpleNamespace(
        get_response=Mock(
            return_value=SimpleNamespace(
                message="Crisis response",
                response_type="crisis",
                resources=[],
                requires_escalation=True,
            )
        )
    )
    agent.counseling_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                intervention_hint="supportive hint",
                to_dict=lambda: {"kind": "supportive"},
            )
        )
    )
    agent.empathy_retriever = SimpleNamespace(
        recommend=Mock(
            return_value=SimpleNamespace(
                empathy_style_hint="warm hint",
                to_dict=lambda: {"style": "warm"},
            )
        )
    )
    agent.wellness_recommender = SimpleNamespace(recommend=Mock())
    agent.pii_redactor = SimpleNamespace(
        redact=Mock(return_value=SimpleNamespace(redacted_text="redacted input", entity_count=0, entities=[]))
    )
    agent.rag_retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[]),
        format_context=Mock(return_value=""),
    )

    get_memory_context = AsyncMock(return_value=memory_context or SimpleNamespace(
        recent_summaries=[object(), object()],
        facts=[object()],
        directives=[
            SimpleNamespace(active=True, term="raw directive should not be logged"),
            SimpleNamespace(active=False, term="inactive raw directive should not be logged"),
        ],
        emotional_trend=[SimpleNamespace(label="raw emotion should not be logged")],
    ))
    if memory_side_effect is not None:
        get_memory_context.side_effect = memory_side_effect

    agent.memory_store = SimpleNamespace(
        get_cloud_context=AsyncMock(return_value=([], None)),
        get_memory_context=get_memory_context,
        get_local_context=AsyncMock(return_value=[]),
        update_profile=AsyncMock(),
    )
    agent.session_manager = SimpleNamespace(
        add_to_history=AsyncMock(),
        update_activity=AsyncMock(),
    )
    agent.prompt_generator = PromptCapture()
    agent.cloud_client = SimpleNamespace(analyze=AsyncMock(return_value=AnalysisResult()))
    agent.local_generator = SimpleNamespace(
        create_chat_completion=AsyncMock(
            return_value=GenerationResult(
                text="Generated response",
                tokens_generated=3,
                finish_reason="stop",
                generation_time_ms=1.0,
            )
        )
    )
    return agent


def test_non_crisis_flow_fetches_and_passes_memory_context():
    memory_context = SimpleNamespace(
        recent_summaries=[object()],
        facts=[object(), object()],
        directives=[SimpleNamespace(active=True), SimpleNamespace(active=False)],
        emotional_trend=[object(), object(), object()],
    )
    agent = build_agent(memory_context=memory_context)

    result = asyncio.run(agent.process_message("I feel stressed", "session-1"))

    agent.memory_store.get_memory_context.assert_awaited_once_with("session-1")
    assert agent.prompt_generator.cloud_kwargs["memory_context"] is memory_context
    assert agent.prompt_generator.local_kwargs["memory_context"] is memory_context
    assert result["pipeline_details"]["memory_context"] == {
        "available": True,
        "recent_summaries": 1,
        "facts": 2,
        "directives": 1,
        "emotional_trend": 3,
    }


def test_non_crisis_flow_passes_processed_dataset_hints_to_local_prompt():
    agent = build_agent()

    asyncio.run(agent.process_message("I feel stressed", "session-1"))

    additional_context = agent.prompt_generator.local_kwargs["additional_context"]
    assert additional_context["counseling_hint"] == "supportive hint"
    assert additional_context["empathy_style_hint"] == "warm hint"
    assert additional_context["wellness_hint"] == ""
    assert "wellness_risk_stage" in additional_context


def test_safety_gateway_crisis_does_not_fetch_memory_context():
    agent = build_agent(safety=crisis_safety_result())

    result = asyncio.run(agent.process_message("I want to hurt myself", "session-1"))

    agent.memory_store.get_memory_context.assert_not_called()
    assert agent.prompt_generator.cloud_kwargs is None
    assert agent.prompt_generator.local_kwargs is None
    assert result["requires_crisis_response"] is True


def test_initial_risk_audit_crisis_does_not_fetch_memory_context():
    agent = build_agent(risk_side_effect=[crisis_risk_assessment()])

    result = asyncio.run(agent.process_message("crisis wording", "session-1"))

    agent.memory_store.get_memory_context.assert_not_called()
    assert agent.prompt_generator.cloud_kwargs is None
    assert agent.prompt_generator.local_kwargs is None
    assert result["requires_crisis_response"] is True


def test_memory_context_failure_falls_back_to_none_and_continues():
    agent = build_agent(memory_side_effect=RuntimeError("memory unavailable"))

    result = asyncio.run(agent.process_message("I feel stressed", "session-1"))

    agent.memory_store.get_memory_context.assert_awaited_once_with("session-1")
    assert agent.prompt_generator.cloud_kwargs["memory_context"] is None
    assert agent.prompt_generator.local_kwargs["memory_context"] is None
    assert result["response"]
    assert result["pipeline_details"]["memory_context"] == {
        "available": False,
        "error": "RuntimeError",
    }
