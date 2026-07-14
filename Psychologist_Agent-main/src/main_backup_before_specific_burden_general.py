"""
Psychologist Agent - Main Orchestrator.

This module provides the main PsychologistAgent class that orchestrates
the complete inference pipeline from user input to response generation.
"""

import os
import asyncio
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass

from src.agent.decision import decide_action
from src.agent.cause import CauseExplorationResult, explore_causes
from src.agent.followup import generate_followup_question
from src.agent.intent import classify_intent
from src.agent.models import DecisionAction, EmotionLabel, EmotionalStateVector
from src.agent.models import IntentCandidate, IntentLabel, IntentSeverity, SmallActionPlan
from src.agent.planner import generate_small_action_plan
from src.agent.recall import build_proactive_recall
from src.agent.state import summarize_emotional_state, update_emotional_state
from src.agent.summary import build_session_dream_summary
from src.safety.gateway import SafetyGateway, SafetyResult
from src.safety.patterns import RiskLevel
from src.privacy.pii_redactor import PIIRedactor, RedactionResult
from src.rag.retriever import RAGRetriever, ContextBuilder
from src.prompt.generator import PromptGenerator, PromptConfig
from src.api.deepseek_client import DeepseekClient
from src.api.models import AnalysisResult
from src.audit.risk_checker import RiskChecker, RiskAssessment
from src.audit.crisis_handler import CrisisHandler, CrisisResponse
from src.audit.logger import AuditLogger, AuditLoggerConfig
from src.inference.generator import LocalGenerator
from src.memory.store import MemoryStore
from src.counseling.dataset_loader import DEFAULT_DATASET_PATHS as COUNSELING_DEFAULT_DATASET_PATHS
from src.counseling import CounselingRetriever, CounselingRecommendation
from src.empathy.dataset_loader import DEFAULT_DATASET_PATHS as EMPATHY_DEFAULT_DATASET_PATHS
from src.empathy import EmpathyRetriever, EmpathyRecommendation
from src.wellness.dataset_loader import DEFAULT_DATASET_PATHS as WELLNESS_DEFAULT_DATASET_PATHS
from src.wellness.recommender import WellnessRecommender, WellnessRecommendation
from src.session.manager import SessionManager
from src.utils.logging_config import setup_logging

logger = setup_logging("psychologist_agent")


INTERNAL_RESPONSE_MARKERS = (
    "내담자의 표현을 반영",
    "핵심 감정을 명료화",
    "상담 참고",
    "공감 참고",
    "웰니스 참고",
    "intervention_hint",
    "empathy_style_hint",
    "support_hint",
    "guidance",
    "therapeutic_guidance",
)
COMPLETE_SENTENCE_ENDINGS = ("요", "다", "습니다", "세요", "?", "!", ".")
INCOMPLETE_KOREAN_SUFFIXES = (
    "힘드",
    "부담",
    "무겁",
    "어렵",
    "막막",
    "걱정",
    "불안",
    "속상",
    "피곤",
    "지치",
    "버겁",
    "많이",
    "너무",
    "그리고",
    "하지만",
    "그래서",
)
STUDY_EXAM_PRESSURE_MARKERS = (
    "공부",
    "시험",
    "과제",
    "성적",
    "중간",
    "기말",
    "부담",
    "힘들어",
)
ACADEMIC_OVERLOAD_ACTION_STEP = (
    "과제 1개와 시험 범위 1개만 적고, 각각 10분 안에 시작할 첫 행동을 하나씩 정해보세요."
)
ACADEMIC_PRESSURE_ACTION_STEP = (
    "타이머를 10분만 맞추고, 과제나 공부에서 가장 작은 첫 단계 하나만 시작해보세요."
)
ACADEMIC_RELIEF_ACTION_STEP = (
    "잠깐 쉬면서 남은 시험 범위 중 꼭 볼 것 3가지만 적어보세요."
)
MANAGEABLE_ACADEMIC_CONCERN_ACTION_STEP = (
    "인공지능 과목에서 꼭 외워야 할 개념 3개만 골라 체크리스트로 적어보세요."
)
SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP = (
    "인공지능 과목에서 꼭 외워야 할 개념 5개만 골라서, 아는 것과 모르는 것으로 나눠 적어보세요."
)
GENERAL_SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP = (
    "인공지능 시험 범위에서 꼭 외워야 할 개념 5개만 골라서, 아는 것과 모르는 것으로 나눠 적어보세요."
)
RECOVERY_IMPROVEMENT_ACTION_STEP = (
    "오늘은 나아지는 데 도움이 됐던 행동이나 상황을 하나만 적어두고, 내일도 5분만 반복해보세요."
)
SAFE_HINT_FIELDS = (
    "source_dataset",
    "label_or_topic",
    "intent_hint",
    "emotion_hint",
    "cause_hint",
    "action_hint",
    "safety_hint",
    "short_summary",
)
DEFAULT_SAFE_HINTS_PATH = Path(__file__).resolve().parents[1] / "data" / "derived" / "balanced_safe_hints.jsonl"


@dataclass
class AgentConfig:
    """Configuration for PsychologistAgent."""
    enable_safety_check: bool = True
    enable_pii_redaction: bool = True
    enable_rag: bool = True
    enable_cloud_analysis: bool = True
    enable_risk_audit: bool = True
    enable_audit_logging: bool = True
    max_cloud_history_turns: int = 10
    max_local_history_turns: int = 3


class PsychologistAgent:
    """
    Main orchestrator for the Psychologist Agent.

    Implements the complete inference pipeline:
    User Input → Safety Gateway → PII Redaction → RAG Retrieval
        → Cloud Analysis (Deepseek) → Risk Audit
        → Local Generation (GGUF) → Memory Update → Response

    Example:
        agent = PsychologistAgent()
        await agent.initialize()
        result = await agent.process_message("I'm feeling anxious", session_id)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize the Psychologist Agent.

        Args:
            config: Agent configuration
            mock_mode: Whether to use mock mode for all components
        """
        self.config = config or AgentConfig()
        self.mock_mode = mock_mode
        self.llm_type = os.getenv("LLM_TYPE", "MOCK").strip().upper()
        if self.mock_mode is None:
            self.mock_mode = self.llm_type == "MOCK"
        self.use_ollama_generation = self.llm_type == "OLLAMA"
        self.use_gemini_generation = self.llm_type == "GEMINI"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b").strip() or "qwen2.5-coder:3b"
        self.ollama_url = (
            os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
            or "http://localhost:11434/api/generate"
        )
        self.ollama_timeout_seconds = self._ollama_timeout_from_env()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.gemini_url = os.getenv("GEMINI_URL", "").strip() or self._default_gemini_url(self.gemini_model)
        self.gemini_timeout_seconds = self._gemini_timeout_from_env()
        self.gemini_max_output_tokens = self._gemini_max_output_tokens_from_env()
        self.safe_hints_path = Path(
            os.getenv("SAFE_HINTS_PATH", str(DEFAULT_SAFE_HINTS_PATH)).strip()
            or str(DEFAULT_SAFE_HINTS_PATH)
        )
        self.safe_hints_cache: List[Dict[str, str]] = []
        self._safe_hint_search_cache: List[Dict[str, Any]] = []
        self.safe_hint_load_ms = 0.0
        self._use_full_mock_retrievers = (
            os.getenv("MOCK_USE_DATASET_RETRIEVERS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # Initialize components
        # Safety and RAG always use real embeddings (BGE-small, CPU)
        # for meaningful semantic matching, even in MOCK mode
        self.safety_gateway = SafetyGateway(mock_mode=False)
        self.counseling_retriever = CounselingRetriever()
        self.empathy_retriever = EmpathyRetriever()
        self.pii_redactor = PIIRedactor(mock_mode=self.mock_mode)
        self.rag_retriever = RAGRetriever(mock_mode=False)
        self.prompt_generator = PromptGenerator()
        self.cloud_client = DeepseekClient(mock_mode=self.mock_mode)
        self.risk_checker = RiskChecker()
        self.crisis_handler = CrisisHandler()
        self.local_generator = LocalGenerator(mock_mode=self.mock_mode)
        self.memory_store = MemoryStore()
        self.wellness_recommender = WellnessRecommender()
        self.session_manager = SessionManager(memory_store=self.memory_store)

        if self.config.enable_audit_logging:
            self.audit_logger = AuditLogger()
        else:
            self.audit_logger = None

        self._initialized = False

        logger.info(
            "PsychologistAgent created (llm_type=%s, mock_mode=%s, ollama_generation=%s, gemini_generation=%s)",
            self.llm_type,
            self.mock_mode,
            self.use_ollama_generation,
            self.use_gemini_generation,
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing PsychologistAgent...")

        # MOCK/OLLAMA/GEMINI response modes use the agent pipeline directly and do not
        # need expensive GGUF/RAG loading before the first demo response.
        if self.config.enable_rag and not self._uses_template_response_pipeline():
            await self.rag_retriever.initialize()

        if not self._uses_template_response_pipeline():
            await self.local_generator.initialize()

        safe_hint_start = time.perf_counter()
        self.safe_hints_cache = self._load_safe_hints_once()
        self._safe_hint_search_cache = self._build_safe_hint_search_cache(self.safe_hints_cache)
        self.safe_hint_load_ms = self._elapsed_ms(safe_hint_start)
        logger.info(
            "safe_hint_load_ms=%s records=%s path=%s",
            self.safe_hint_load_ms,
            len(self.safe_hints_cache),
            self.safe_hints_path,
        )

        self._initialized = True
        logger.info("PsychologistAgent initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        if self.local_generator:
            await self.local_generator.unload()

        if self.cloud_client:
            await self.cloud_client.close()

        self._initialized = False
        logger.info("PsychologistAgent shutdown complete")

    async def process_message(
        self,
        user_input: str,
        session_id: str,
        wellness_checkin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message through the complete pipeline.

        Args:
            user_input: User's message
            session_id: Session identifier

        Returns:
            Dict containing response and metadata
        """
        total_start = time.perf_counter()
        if not self._initialized:
            initialize_start = time.perf_counter()
            await self.initialize()
            initialize_ms = self._elapsed_ms(initialize_start)
        else:
            initialize_ms = 0.0

        result = {
            "response": "",
            "risk_level": "none",
            "risk_stage": "관심",
            "requires_crisis_response": False,
            "session_id": session_id,
            "pipeline_details": {},
            "counseling_hint": "",
            "empathy_style_hint": "",
            "wellness_hint": "",
        }
        wellness_recommendation: Optional[WellnessRecommendation] = None
        memory_context = None
        intent_result = None
        emotional_state = EmotionalStateVector()
        proactive_recall = None
        decision_result = None
        followup_question = ""
        small_action_plan = None
        cause_exploration = CauseExplorationResult()
        continuity_memory = None
        previous_emotional_state = None
        previous_small_action = None
        previous_followup = ""
        timing = result["pipeline_details"].setdefault("timing", {})
        if initialize_ms:
            timing["initialize"] = initialize_ms
        timing["safe_hint_load_ms"] = self.safe_hint_load_ms

        try:
            # Step 1: Safety Gateway Check
            safety_start = time.perf_counter()
            if self.config.enable_safety_check:
                safety_result = await self.safety_gateway.check(user_input)

                result["pipeline_details"]["safety"] = {
                    "is_safe": safety_result.is_safe,
                    "risk_level": safety_result.risk_level.value,
                    "risk_stage": safety_result.risk_stage,
                    "matched_pattern": safety_result.matched_pattern,
                    "matched_category": safety_result.matched_category,
                    "similarity_score": round(safety_result.similarity_score, 4),
                    "action": safety_result.action
                }

                if self.audit_logger:
                    self.audit_logger.log_safety_check(
                        session_id=session_id,
                        risk_level=safety_result.risk_level.value,
                        is_safe=safety_result.is_safe,
                        matched_pattern=safety_result.matched_pattern,
                        action_taken=safety_result.action
                    )

                # Handle immediate crisis
                if not safety_result.is_safe:
                    result["response"] = self.sanitize_user_response(safety_result.response or "")
                    result["risk_level"] = safety_result.risk_level.value
                    result["risk_stage"] = safety_result.risk_stage
                    result["requires_crisis_response"] = True
                    result["resources"] = safety_result.resources

                    # Still save to history
                    await self.session_manager.add_to_history(
                        session_id, user_input, safety_result.response
                    )
                    await self.session_manager.update_activity(
                        session_id, risk_level=safety_result.risk_level.value
                    )

                    self._sync_final_safety_agent(result)
                    timing["safety"] = self._elapsed_ms(safety_start)
                    self._finalize_timing(result, total_start)
                    return result

            if self.config.enable_risk_audit:
                risk_assessment = self.risk_checker.assess(AnalysisResult(), user_input)
                result["pipeline_details"]["risk_audit"] = {
                    "risk_level": risk_assessment.risk_level.value,
                    "risk_stage": risk_assessment.risk_stage,
                    "requires_crisis": risk_assessment.requires_crisis_response,
                    "recommended_actions": getattr(risk_assessment, "recommended_actions", []),
                }

                if risk_assessment.requires_crisis_response:
                    crisis_response = self.crisis_handler.get_response(risk_assessment)
                    result["response"] = self.sanitize_user_response(crisis_response.message)
                    result["risk_level"] = risk_assessment.risk_level.value
                    result["risk_stage"] = risk_assessment.risk_stage
                    result["requires_crisis_response"] = True

                    await self.session_manager.add_to_history(
                        session_id, user_input, crisis_response.message
                    )

                    self._sync_final_safety_agent(result)
                    timing["safety"] = self._elapsed_ms(safety_start)
                    self._finalize_timing(result, total_start)
                    return result

                result["risk_level"] = risk_assessment.risk_level.value
                result["risk_stage"] = risk_assessment.risk_stage
                self._sync_final_safety_agent(result)
            timing["safety"] = self._elapsed_ms(safety_start)

            result["pipeline_details"].setdefault("agents", {})

            dataset_start = time.perf_counter()
            intent_result = classify_intent(user_input)
            safe_hint_start = time.perf_counter()
            selected_safe_hints = self._retrieve_safe_hints(
                user_input=user_input,
                intent_result=intent_result,
            )
            timing["safe_hint_retrieval_ms"] = self._elapsed_ms(safe_hint_start)
            result["pipeline_details"]["safe_hints"] = {
                "available": bool(self.safe_hints_cache),
                "cache_size": len(self.safe_hints_cache),
                "selected_count": len(selected_safe_hints),
                "path": str(self.safe_hints_path),
                "selected": selected_safe_hints,
            }

            (
                safe_counseling_recommendation,
                safe_empathy_recommendation,
                safe_wellness_recommendation,
            ) = self._recommend_from_safe_hints(
                user_input=user_input,
                selected_safe_hints=selected_safe_hints,
                wellness_checkin=wellness_checkin,
            )

            if self._uses_default_dataset_loader(
                self.counseling_retriever,
                CounselingRetriever,
                COUNSELING_DEFAULT_DATASET_PATHS,
            ):
                counseling_recommendation = safe_counseling_recommendation
            elif self.mock_mode:
                counseling_recommendation = self._mock_counseling_recommendation(user_input)
            else:
                counseling_recommendation = self.counseling_retriever.recommend(user_input)

            if self._uses_default_dataset_loader(
                self.empathy_retriever,
                EmpathyRetriever,
                EMPATHY_DEFAULT_DATASET_PATHS,
            ):
                empathy_recommendation = safe_empathy_recommendation
            elif self.mock_mode:
                empathy_recommendation = self._mock_empathy_recommendation(user_input)
            else:
                empathy_recommendation = self.empathy_retriever.recommend(user_input)

            if self._uses_default_dataset_loader(
                self.wellness_recommender,
                WellnessRecommender,
                WELLNESS_DEFAULT_DATASET_PATHS,
            ):
                wellness_recommendation = safe_wellness_recommendation
            else:
                wellness_recommendation = self._get_wellness_recommendation(wellness_checkin)
            result["counseling_hint"] = counseling_recommendation.intervention_hint
            result["empathy_style_hint"] = empathy_recommendation.empathy_style_hint

            if wellness_recommendation:
                result["wellness_hint"] = wellness_recommendation.support_hint
                result["pipeline_details"]["wellness"] = self._safe_wellness_details(
                    wellness_recommendation
                )

            result["pipeline_details"]["counseling"] = self._safe_counseling_details(
                counseling_recommendation
            )
            result["pipeline_details"]["empathy"] = self._safe_empathy_details(
                empathy_recommendation
            )
            timing["dataset_retrieval"] = self._elapsed_ms(dataset_start)

            memory_start = time.perf_counter()
            (memory_context, memory_details), continuity_memory = await asyncio.gather(
                self._get_memory_context_with_details(session_id),
                self._get_continuity_memory(session_id),
            )
            result["pipeline_details"]["memory_context"] = memory_details
            previous_emotional_state = self._previous_state_from_continuity(continuity_memory)
            previous_small_action = self._small_action_from_continuity(
                continuity_memory,
                session_id=session_id,
            )
            previous_followup = getattr(continuity_memory, "next_follow_up", "") if continuity_memory else ""
            timing["memory_context"] = self._elapsed_ms(memory_start)

            agent_pipeline_start = time.perf_counter()
            intent_result = self._apply_followup_continuity(
                intent_result=intent_result,
                previous_followup=previous_followup,
                user_input=user_input,
            )
            result["pipeline_details"]["agents"]["intent"] = self._serialize_intent_agent(
                intent_result
            )
            self._sync_final_safety_agent(result)

            emotional_state = update_emotional_state(
                previous_state=previous_emotional_state,
                intent_result=intent_result,
                emotion_labels=self._emotion_labels_from_empathy(empathy_recommendation),
                risk_stage=result["risk_stage"],
                wellness_checkin=wellness_checkin,
            )
            result["pipeline_details"]["agents"]["emotional_state"] = (
                self._serialize_emotional_state_agent(emotional_state)
            )

            proactive_recall = build_proactive_recall(
                memory_context=memory_context,
                last_small_action=previous_small_action,
                next_followup=previous_followup,
            )
            result["pipeline_details"]["agents"]["memory_recall"] = (
                self._serialize_memory_recall_agent(
                    proactive_recall,
                    continuity_memory=continuity_memory,
                )
            )

            decision_result = decide_action(
                risk_stage=result["risk_stage"],
                requires_crisis_response=result["requires_crisis_response"],
                intent_result=intent_result,
                emotional_state=emotional_state,
                counseling_hint=result["counseling_hint"],
                empathy_style_hint=result["empathy_style_hint"],
                wellness_hint=result["wellness_hint"],
                memory_context=memory_context,
                proactive_recall=proactive_recall,
            )
            result["pipeline_details"]["agents"]["decision"] = (
                self._serialize_decision_agent(decision_result)
            )

            cause_exploration = explore_causes(
                user_input=user_input,
                intent_result=intent_result,
                counseling_recommendation=counseling_recommendation,
                empathy_recommendation=empathy_recommendation,
                wellness_recommendation=wellness_recommendation,
                proactive_recall=proactive_recall,
                previous_followup=previous_followup,
            )
            result["pipeline_details"]["agents"]["cause_exploration"] = (
                cause_exploration.to_pipeline_dict()
            )

            if (
                decision_result.primary_action == DecisionAction.ASK_FOLLOW_UP
                or decision_result.response_constraints.get("must_include_followup") is True
                or bool(cause_exploration.exploration_question)
            ):
                exploration_question = (cause_exploration.exploration_question or "").strip()
                previous_followup_text = (previous_followup or "").strip()
                if exploration_question and exploration_question != previous_followup_text:
                    followup_question = exploration_question
                else:
                    followup_question = generate_followup_question(
                        intent_result=intent_result,
                        decision_result=decision_result,
                        emotional_state=emotional_state,
                        risk_stage=result["risk_stage"],
                        previous_followup=previous_followup,
                        avoid_topics=decision_result.response_constraints.get("avoid_topics", []),
                        prefer_previous=False,
                    )
                    if followup_question.strip() == previous_followup_text:
                        followup_question = ""

            result["pipeline_details"]["agents"]["followup"] = {
                "has_question": bool(followup_question),
                "question_type": intent_result.primary_intent.name if intent_result else "",
                "question": followup_question,
            }

            should_plan_small_action = (
                DecisionAction.SUGGEST_SMALL_ACTION in decision_result.secondary_actions
                or bool(result["wellness_hint"])
                or cause_exploration.selected_cause
                in {
                    "exam_assignment_pressure",
                    "academic_deadline_exam_overload",
                    "academic_relief",
                    "manageable_academic_concern",
                    "specific_academic_burden",
                    "specific_academic_burden_after_relief",
                    "recovery_improvement",
                }
                or (
                    intent_result
                    and intent_result.primary_intent
                    in {IntentLabel.SLEEP_PROBLEM, IntentLabel.ANXIETY_SUPPORT}
                )
            )
            if (
                should_plan_small_action
                and result["risk_stage"] != "위험"
                and decision_result.primary_action != DecisionAction.ESCALATE_SAFETY
            ):
                small_action_plan = generate_small_action_plan(
                    session_id=session_id,
                    intent_result=intent_result,
                    decision_result=decision_result,
                    emotional_state=emotional_state,
                    wellness_hint=result["wellness_hint"],
                    counseling_hint=result["counseling_hint"],
                    risk_stage=result["risk_stage"],
                )
                if cause_exploration.selected_cause == "exam_assignment_pressure":
                    small_action_plan.action_text = ACADEMIC_PRESSURE_ACTION_STEP
                    small_action_plan.steps = [ACADEMIC_PRESSURE_ACTION_STEP]
                    small_action_plan.rationale_label = "exam_assignment_pressure"
                    small_action_plan.rationale_tags = ["exam_assignment_pressure"]
                elif cause_exploration.selected_cause == "academic_relief":
                    small_action_plan.action_text = ACADEMIC_RELIEF_ACTION_STEP
                    small_action_plan.steps = [ACADEMIC_RELIEF_ACTION_STEP]
                    small_action_plan.rationale_label = "academic_relief"
                    small_action_plan.rationale_tags = ["academic_relief"]
                elif cause_exploration.selected_cause == "manageable_academic_concern":
                    small_action_plan.action_text = MANAGEABLE_ACADEMIC_CONCERN_ACTION_STEP
                    small_action_plan.steps = [MANAGEABLE_ACADEMIC_CONCERN_ACTION_STEP]
                    small_action_plan.rationale_label = "manageable_academic_concern"
                    small_action_plan.rationale_tags = ["manageable_academic_concern"]
                elif cause_exploration.selected_cause == "specific_academic_burden_after_relief":
                    small_action_plan.action_text = SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP
                    small_action_plan.steps = [SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP]
                    small_action_plan.rationale_label = "specific_academic_burden_after_relief"
                    small_action_plan.rationale_tags = ["specific_academic_burden_after_relief"]
                elif cause_exploration.selected_cause == "specific_academic_burden":
                    small_action_plan.action_text = GENERAL_SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP
                    small_action_plan.steps = [GENERAL_SPECIFIC_ACADEMIC_BURDEN_ACTION_STEP]
                    small_action_plan.rationale_label = "specific_academic_burden"
                    small_action_plan.rationale_tags = ["specific_academic_burden"]
                elif cause_exploration.selected_cause == "academic_deadline_exam_overload":
                    small_action_plan.action_text = ACADEMIC_OVERLOAD_ACTION_STEP
                    small_action_plan.steps = [ACADEMIC_OVERLOAD_ACTION_STEP]
                    small_action_plan.rationale_label = "academic_deadline_exam_overload"
                    small_action_plan.rationale_tags = ["academic_deadline_exam_overload"]
                elif cause_exploration.selected_cause == "recovery_improvement":
                    small_action_plan.action_text = RECOVERY_IMPROVEMENT_ACTION_STEP
                    small_action_plan.steps = [RECOVERY_IMPROVEMENT_ACTION_STEP]
                    small_action_plan.rationale_label = "recovery_improvement"
                    small_action_plan.rationale_tags = ["recovery_improvement"]
                self._avoid_repeated_small_action(
                    small_action_plan,
                    previous_small_action,
                    selected_cause=cause_exploration.selected_cause,
                )

            result["pipeline_details"]["agents"]["small_action"] = (
                self._serialize_small_action_agent(small_action_plan)
            )
            agent_context = self._build_agent_context(
                decision_result=decision_result,
                emotional_state=emotional_state,
                proactive_recall=proactive_recall,
                followup_question=followup_question,
                small_action_plan=small_action_plan,
            )
            result["pipeline_details"]["agents"]["prompt_context"] = agent_context
            timing["agent_pipeline"] = self._elapsed_ms(agent_pipeline_start)
            timing["agent_pipeline_ms"] = timing["agent_pipeline"]

            response_start = time.perf_counter()
            if self._uses_template_response_pipeline():
                fallback_response_text = self._compose_mock_response(
                    counseling_recommendation.intervention_hint,
                    empathy_recommendation.empathy_style_hint,
                    wellness_recommendation.support_hint if wellness_recommendation else "",
                    intent_result=intent_result,
                    emotional_state=emotional_state,
                    followup_question=followup_question,
                    small_action_text=small_action_plan.action_text if small_action_plan else "",
                    cause_exploration=cause_exploration,
                    selected_safe_hints=selected_safe_hints,
                    user_input=user_input,
                )
                response_text = fallback_response_text
                if self.use_gemini_generation:
                    llm_start = time.perf_counter()
                    logger.info(
                        "Attempting Gemini response naturalization (model=%s, timeout=%ss)",
                        self.gemini_model,
                        self.gemini_timeout_seconds,
                    )
                    gemini_text = await self._naturalize_response_with_gemini(
                        user_input=user_input,
                        intent_result=intent_result,
                        emotional_state=emotional_state,
                        cause_exploration=cause_exploration,
                        followup_question=followup_question,
                        small_action_text=small_action_plan.action_text if small_action_plan else "",
                        selected_safe_hints=selected_safe_hints,
                    )
                    llm_generation_ms = self._elapsed_ms(llm_start)
                    if gemini_text and not self._is_valid_cloud_llm_response(gemini_text):
                        logger.warning("Gemini response invalid; using fallback")
                        gemini_text = ""
                    if gemini_text:
                        response_text = gemini_text
                        logger.info("Gemini response generation succeeded")
                    else:
                        logger.info("Gemini response generation fell back to template response")
                    result["pipeline_details"]["response_generation"] = {
                        "llm_type": "GEMINI",
                        "provider": "GEMINI",
                        "model": self.gemini_model,
                        "timeout_seconds": self.gemini_timeout_seconds,
                        "generation_strategy": "agent_pipeline_with_gemini_naturalization",
                        "attempted_cloud_llm": bool(self.gemini_api_key),
                        "used_cloud_llm": bool(gemini_text),
                        "fallback_used": not bool(gemini_text),
                        "latency_ms": llm_generation_ms,
                    }
                    timing["llm_generation_ms"] = llm_generation_ms
                elif self.use_ollama_generation:
                    llm_start = time.perf_counter()
                    logger.info(
                        "Attempting Ollama response naturalization (model=%s, url=%s, timeout=%ss)",
                        self.ollama_model,
                        self.ollama_url,
                        self.ollama_timeout_seconds,
                    )
                    ollama_text = await self._naturalize_response_with_ollama(
                        user_input=user_input,
                        intent_result=intent_result,
                        emotional_state=emotional_state,
                        cause_exploration=cause_exploration,
                        followup_question=followup_question,
                        small_action_text=small_action_plan.action_text if small_action_plan else "",
                        selected_safe_hints=selected_safe_hints,
                    )
                    llm_generation_ms = self._elapsed_ms(llm_start)
                    if ollama_text:
                        response_text = ollama_text
                        logger.info("Ollama response generation succeeded")
                    else:
                        logger.info("Ollama response generation fell back to template response")
                    result["pipeline_details"]["response_generation"] = {
                        "llm_type": "OLLAMA",
                        "model": self.ollama_model,
                        "timeout_seconds": self.ollama_timeout_seconds,
                        "generation_strategy": "agent_pipeline_with_optional_ollama_naturalization",
                        "attempted_ollama": True,
                        "used_ollama": bool(ollama_text),
                        "fallback_used": not bool(ollama_text),
                        "latency_ms": llm_generation_ms,
                    }
                    timing["llm_generation_ms"] = llm_generation_ms
                else:
                    result["pipeline_details"]["response_generation"] = {
                        "llm_type": "MOCK",
                        "provider": "MOCK",
                        "model": "",
                        "timeout_seconds": 0.0,
                        "generation_strategy": "agent_pipeline_with_optional_ollama_naturalization",
                        "used_cloud_llm": False,
                        "attempted_ollama": False,
                        "used_ollama": False,
                        "fallback_used": False,
                        "latency_ms": 0.0,
                    }
                    timing["llm_generation_ms"] = 0.0
                result["response"] = self._add_safety_notice(
                    self.sanitize_user_response(response_text)
                )
                self._sync_final_safety_agent(result)

                await self.session_manager.add_to_history(
                    session_id, user_input, result["response"]
                )
                await self._store_structured_continuity(
                    session_id=session_id,
                    intent_result=intent_result,
                    emotional_state=emotional_state,
                    risk_stage=result["risk_stage"],
                    small_action_plan=small_action_plan,
                    followup_question=followup_question,
                    memory_context=memory_context,
                    previous_continuity=continuity_memory,
                    result=result,
                )

                timing["response_generation"] = self._elapsed_ms(response_start)
                self._finalize_timing(result, total_start)
                return result

            timing["response_generation"] = self._elapsed_ms(response_start)
            # Step 2: PII Redaction
            non_mock_response_start = time.perf_counter()
            if self.config.enable_pii_redaction:
                redaction_result = self.pii_redactor.redact(user_input)
                sanitized_input = redaction_result.redacted_text

                result["pipeline_details"]["pii"] = {
                    "entity_count": redaction_result.entity_count,
                    "entities": [
                        {"type": e.entity_type.value, "replacement": e.replacement}
                        for e in redaction_result.entities
                    ],
                    "redacted_text": redaction_result.redacted_text
                }

                if self.audit_logger and redaction_result.entity_count > 0:
                    self.audit_logger.log_pii_redaction(
                        session_id=session_id,
                        entity_count=redaction_result.entity_count,
                        entity_types=[e.entity_type.value for e in redaction_result.entities]
                    )
            else:
                sanitized_input = user_input

            # Step 3: RAG Retrieval
            rag_context = ""
            if self.config.enable_rag:
                rag_results = await self.rag_retriever.retrieve(sanitized_input)
                rag_context = self.rag_retriever.format_context(rag_results)

                result["pipeline_details"]["rag"] = {
                    "num_chunks": len(rag_results),
                    "chunks": [
                        {
                            "source": r.source,
                            "source_type": r.source_type,
                            "score": round(r.score, 4),
                            "text_preview": r.content[:150] + "..."
                            if len(r.content) > 150
                            else r.content
                        }
                        for r in rag_results[:3]
                    ]
                }

            # Step 4: Get conversation history (separate for cloud and local)
            cloud_history, user_profile = await self.memory_store.get_cloud_context(session_id)

            # Step 5: Cloud Analysis (Deepseek) with profile
            if self.config.enable_cloud_analysis:
                cloud_prompt = self.prompt_generator.gen_cloud_prompt(
                    sanitized_input=sanitized_input,
                    rag_context=rag_context,
                    history=cloud_history,
                    user_profile=user_profile.to_json() if user_profile else None,
                    memory_context=memory_context,
                )

                cloud_analysis = await self.cloud_client.analyze(
                    system_message=cloud_prompt.system_message,
                    user_message=cloud_prompt.user_message
                )

                result["pipeline_details"]["cloud_analysis"] = {
                    "risk_level": cloud_analysis.risk_level.value,
                    "primary_concern": cloud_analysis.primary_concern,
                    "suggested_approach": cloud_analysis.suggested_approach.value,
                    "suggested_technique": cloud_analysis.suggested_technique,
                    "guidance": cloud_analysis.guidance_for_local_model,
                    "key_points": cloud_analysis.key_points
                }

                # Update profile if cloud analysis provides updates
                if cloud_analysis.updated_user_profile:
                    await self.memory_store.update_profile(
                        session_id, cloud_analysis.updated_user_profile
                    )
                    result["pipeline_details"]["profile_update"] = cloud_analysis.updated_user_profile
            else:
                # Default analysis if cloud disabled
                cloud_analysis = AnalysisResult()

            # Step 6: Risk Audit
            if self.config.enable_risk_audit:
                risk_assessment = self.risk_checker.assess(
                    cloud_analysis, user_input
                )

                if self.audit_logger:
                    self.audit_logger.log_risk_assessment(
                        session_id=session_id,
                        risk_level=risk_assessment.risk_level.value,
                        primary_concern=cloud_analysis.primary_concern,
                        approach=cloud_analysis.suggested_approach.value,
                        key_points=cloud_analysis.key_points
                    )

                result["pipeline_details"]["risk_audit"] = {
                    "risk_level": risk_assessment.risk_level.value,
                    "risk_stage": risk_assessment.risk_stage,
                    "requires_crisis": risk_assessment.requires_crisis_response,
                    "recommended_actions": getattr(risk_assessment, "recommended_actions", [])
                }

                # Handle crisis from risk audit
                if risk_assessment.requires_crisis_response:
                    crisis_response = self.crisis_handler.get_response(risk_assessment)
                    result["response"] = self.sanitize_user_response(crisis_response.message)
                    result["risk_level"] = risk_assessment.risk_level.value
                    result["risk_stage"] = risk_assessment.risk_stage
                    result["requires_crisis_response"] = True

                    if self.audit_logger:
                        self.audit_logger.log_crisis_intervention(
                            session_id=session_id,
                            trigger=crisis_response.response_type,
                            resources_provided=[r.name for r in crisis_response.resources],
                            escalated=crisis_response.requires_escalation
                        )

                    await self.session_manager.add_to_history(
                        session_id, user_input, crisis_response.message
                    )

                    self._sync_final_safety_agent(result)
                    return result

                result["risk_level"] = risk_assessment.risk_level.value
                result["risk_stage"] = risk_assessment.risk_stage
                self._sync_final_safety_agent(result)

            # Step 7: Local Generation (GGUF) with 3-turn history and messages list
            local_history = await self.memory_store.get_local_context(session_id)

            local_prompt = self.prompt_generator.gen_local_prompt(
                user_input=user_input,
                cloud_analysis=cloud_analysis.to_dict(),
                rag_context=rag_context,
                history=local_history,
                therapeutic_guidance=wellness_recommendation.support_hint if wellness_recommendation else "",
                additional_context={
                    "counseling_hint": counseling_recommendation.intervention_hint,
                    "empathy_style_hint": empathy_recommendation.empathy_style_hint,
                    "wellness_hint": wellness_recommendation.support_hint if wellness_recommendation else "",
                    "wellness_risk_stage": wellness_recommendation.risk_stage if wellness_recommendation else "",
                },
                memory_context=memory_context,
                agent_context=agent_context,
            )

            # Use create_chat_completion with messages list
            llm_start = time.perf_counter()
            generation_result = await self.local_generator.create_chat_completion(
                messages=local_prompt.to_messages()
            )
            timing["llm_generation_ms"] = self._elapsed_ms(llm_start)

            response_text = self.sanitize_user_response(generation_result.text)
            if self.mock_mode and wellness_recommendation and wellness_recommendation.support_hint:
                response_text = self._merge_wellness_hint(response_text, wellness_recommendation.support_hint)

            result["response"] = self._add_safety_notice(
                self.sanitize_user_response(response_text)
            )
            self._sync_final_safety_agent(result)

            # Step 8: Update memory
            await self.session_manager.add_to_history(
                session_id, user_input, generation_result.text
            )
            await self._store_structured_continuity(
                session_id=session_id,
                intent_result=intent_result,
                emotional_state=emotional_state,
                risk_stage=result["risk_stage"],
                small_action_plan=small_action_plan,
                followup_question=followup_question,
                memory_context=memory_context,
                previous_continuity=continuity_memory,
                result=result,
            )

            timing["response_generation"] = self._elapsed_ms(non_mock_response_start)
            self._finalize_timing(result, total_start)
            return result

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            if self.audit_logger:
                self.audit_logger.log_error(
                    session_id=session_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )

            result["response"] = "I apologize, but I'm having trouble processing your message right now. If you're in crisis, please call 988 for immediate support."
            result["error"] = str(e)
            self._finalize_timing(result, total_start)

            return result

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 3)

    @staticmethod
    def _ollama_timeout_from_env() -> float:
        raw_value = os.getenv("OLLAMA_TIMEOUT_SECONDS", "3").strip()
        try:
            timeout = float(raw_value)
        except ValueError:
            logger.warning("Invalid OLLAMA_TIMEOUT_SECONDS=%r; using 3 seconds", raw_value)
            return 3.0
        if timeout <= 0:
            logger.warning("OLLAMA_TIMEOUT_SECONDS must be positive; using 3 seconds")
            return 3.0
        return timeout

    @staticmethod
    def _gemini_timeout_from_env() -> float:
        raw_value = os.getenv("GEMINI_TIMEOUT_SECONDS", "4").strip()
        try:
            timeout = float(raw_value)
        except ValueError:
            logger.warning("Invalid GEMINI_TIMEOUT_SECONDS=%r; using 4 seconds", raw_value)
            return 4.0
        if timeout <= 0:
            logger.warning("GEMINI_TIMEOUT_SECONDS must be positive; using 4 seconds")
            return 4.0
        return timeout

    @staticmethod
    def _gemini_max_output_tokens_from_env() -> int:
        raw_value = os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "256").strip()
        try:
            tokens = int(raw_value)
        except ValueError:
            logger.warning("Invalid GEMINI_MAX_OUTPUT_TOKENS=%r; using 256", raw_value)
            return 256
        if tokens < 64:
            logger.warning("GEMINI_MAX_OUTPUT_TOKENS must be at least 64; using 256")
            return 256
        return tokens

    @staticmethod
    def _default_gemini_url(model: str) -> str:
        quoted_model = urllib.parse.quote(model, safe="")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{quoted_model}:generateContent"

    def _uses_template_response_pipeline(self) -> bool:
        return bool(self.mock_mode or self.use_ollama_generation or self.use_gemini_generation)

    def _finalize_timing(self, result: Dict[str, Any], total_start: float) -> None:
        timing = result.setdefault("pipeline_details", {}).setdefault("timing", {})
        timing.setdefault("safety", 0.0)
        timing.setdefault("safe_hint_load_ms", self.safe_hint_load_ms)
        timing.setdefault("safe_hint_retrieval_ms", 0.0)
        timing.setdefault("dataset_retrieval", 0.0)
        timing.setdefault("memory_context", 0.0)
        timing.setdefault("agent_pipeline", 0.0)
        timing.setdefault("agent_pipeline_ms", timing.get("agent_pipeline", 0.0))
        timing.setdefault("response_generation", 0.0)
        timing.setdefault("llm_generation_ms", 0.0)
        timing["total"] = self._elapsed_ms(total_start)
        timing["total_response_ms"] = timing["total"]
        logger.info(
            "chat_latency safe_hint_load_ms=%s safe_hint_retrieval_ms=%s agent_pipeline_ms=%s llm_generation_ms=%s total_response_ms=%s",
            timing.get("safe_hint_load_ms", 0.0),
            timing.get("safe_hint_retrieval_ms", 0.0),
            timing.get("agent_pipeline_ms", timing.get("agent_pipeline", 0.0)),
            timing.get("llm_generation_ms", 0.0),
            timing.get("total_response_ms", 0.0),
        )

    def _load_safe_hints_once(self) -> List[Dict[str, str]]:
        if not self.safe_hints_path.exists():
            logger.warning(
                "Live safe hint file is missing: %s. Run: PYTHONPATH=. python3 scripts/analyze_and_balance_datasets.py",
                self.safe_hints_path,
            )
            return []

        records: List[Dict[str, str]] = []
        try:
            with self.safe_hints_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        continue
                    safe_record = {
                        field: " ".join(str(parsed.get(field, "") or "").split())
                        for field in SAFE_HINT_FIELDS
                    }
                    records.append(safe_record)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Live safe hint loading failed from %s: %s. Run: PYTHONPATH=. python3 scripts/analyze_and_balance_datasets.py",
                self.safe_hints_path,
                exc,
            )
            return []
        return records

    def _build_safe_hint_search_cache(self, records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        search_cache: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            searchable = " ".join(
                record.get(field, "")
                for field in (
                    "label_or_topic",
                    "intent_hint",
                    "emotion_hint",
                    "cause_hint",
                    "action_hint",
                    "safety_hint",
                    "short_summary",
                )
            )
            search_cache.append(
                {
                    "index": index,
                    "record": record,
                    "label": record.get("label_or_topic", "").lower(),
                    "searchable_lower": searchable.lower(),
                    "tokens": self._safe_hint_tokens(searchable),
                }
            )
        return search_cache

    def _uses_default_dataset_loader(
        self,
        component: Any,
        component_type: type,
        default_paths: List[Path],
    ) -> bool:
        if not isinstance(component, component_type):
            return False
        loader = getattr(component, "dataset_loader", None)
        dataset_path = getattr(loader, "dataset_path", None)
        if dataset_path is None:
            return True
        try:
            resolved_path = Path(dataset_path).resolve()
        except OSError:
            resolved_path = Path(dataset_path)
        return any(resolved_path == Path(path).resolve() for path in default_paths)

    def _retrieve_safe_hints(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        if not self.safe_hints_cache:
            return []

        query_tokens = self._safe_hint_tokens(user_input)
        intent_name = self._enum_name(getattr(intent_result, "primary_intent", ""))
        scored: List[tuple[float, int, Dict[str, str]]] = []
        for entry in self._safe_hint_search_cache:
            index = int(entry["index"])
            record = entry["record"]
            record_tokens = entry["tokens"]
            score = float(len(query_tokens & record_tokens))
            label = entry["label"]
            if intent_name and intent_name.lower() in entry["searchable_lower"]:
                score += 2.0
            if any(keyword in label for keyword in ("불안", "anxiety")) and any(
                keyword in user_input for keyword in ("불안", "걱정", "초조")
            ):
                score += 3.0
            if any(keyword in label for keyword in ("sleep", "수면", "잠")) and any(
                keyword in user_input for keyword in ("잠", "수면", "불면")
            ):
                score += 3.0
            if any(keyword in label for keyword in ("슬픔", "우울", "depression")) and any(
                keyword in user_input for keyword in ("슬프", "우울", "무기력", "눈물")
            ):
                score += 2.0
            if score > 0:
                scored.append((score, -index, record))

        if not scored:
            return []

        selected = [
            self._compact_safe_hint(record)
            for _, _, record in sorted(scored, reverse=True)[:limit]
        ]
        return selected

    def _safe_hint_tokens(self, text: str) -> set[str]:
        import re

        return {
            token
            for token in re.findall(r"[\w가-힣]+", (text or "").lower())
            if len(token) > 1
        }

    def _compact_safe_hint(self, record: Dict[str, str]) -> Dict[str, str]:
        return {
            "source_dataset": record.get("source_dataset", "")[:80],
            "label_or_topic": record.get("label_or_topic", "")[:80],
            "intent_hint": record.get("intent_hint", "")[:180],
            "emotion_hint": record.get("emotion_hint", "")[:180],
            "cause_hint": record.get("cause_hint", "")[:180],
            "action_hint": record.get("action_hint", "")[:180],
            "safety_hint": record.get("safety_hint", "")[:180],
            "short_summary": record.get("short_summary", "")[:180],
        }

    def _recommend_from_safe_hints(
        self,
        *,
        user_input: str,
        selected_safe_hints: List[Dict[str, str]],
        wellness_checkin: Optional[Dict[str, Any]],
    ) -> tuple[CounselingRecommendation, EmpathyRecommendation, Optional[WellnessRecommendation]]:
        if not selected_safe_hints:
            return (
                self._lightweight_counseling_recommendation(user_input),
                self._lightweight_empathy_recommendation(user_input),
                self._lightweight_wellness_recommendation(wellness_checkin) if wellness_checkin else None,
            )

        first = selected_safe_hints[0]
        counseling = CounselingRecommendation(
            intervention_hint=first.get("action_hint") or first.get("cause_hint") or "감정을 확인하고 오늘 가능한 작은 실행 단계를 하나만 제안하세요.",
            matched_record_id="safe-hints-cache",
            category=first.get("label_or_topic", "safe_hint"),
            score=1.0,
        )
        empathy = EmpathyRecommendation(
            empathy_style_hint=first.get("emotion_hint") or "감정을 먼저 확인하고 차분하게 공감하세요.",
            emotion_label=self._emotion_label_from_safe_hint(first),
            empathy_label="safe_hint",
            matched_record_id="safe-hints-cache",
            score=1.0,
        )
        wellness = None
        if wellness_checkin:
            wellness = self._lightweight_wellness_recommendation(wellness_checkin)
        return counseling, empathy, wellness

    def _emotion_label_from_safe_hint(self, record: Dict[str, str]) -> str:
        text = " ".join(record.values())
        for label in ("불안", "슬픔", "분노", "상처", "당황", "기쁨"):
            if label in text:
                return label
        return ""

    def _lightweight_counseling_recommendation(self, user_input: str) -> CounselingRecommendation:
        text = user_input or ""
        if any(keyword in text for keyword in ("잠", "수면", "불면")):
            return CounselingRecommendation(
                intervention_hint="수면 부담을 낮추는 작은 행동을 하나만 제안하세요.",
                matched_record_id="mock-sleep",
                category="sleep",
                score=1.0,
            )
        if any(keyword in text for keyword in ("불안", "걱정", "초조")):
            return CounselingRecommendation(
                intervention_hint="불안을 낮추는 안정화 행동을 하나만 제안하세요.",
                matched_record_id="mock-anxiety",
                category="anxiety",
                score=1.0,
            )
        if any(keyword in text for keyword in ("외로", "혼자", "고립")):
            return CounselingRecommendation(
                intervention_hint="고립감을 줄이는 부담 낮은 연결 행동을 제안하세요.",
                matched_record_id="mock-loneliness",
                category="loneliness",
                score=1.0,
            )
        return CounselingRecommendation(
            intervention_hint="감정을 확인하고 오늘 가능한 작은 실행 단계를 하나만 제안하세요.",
            matched_record_id="mock-general",
            category="general",
            score=0.5,
        )

    def _mock_counseling_recommendation(self, user_input: str) -> CounselingRecommendation:
        if (
            not self._use_full_mock_retrievers
            and self._uses_default_dataset_loader(
                self.counseling_retriever,
                CounselingRetriever,
                COUNSELING_DEFAULT_DATASET_PATHS,
            )
        ):
            return self._lightweight_counseling_recommendation(user_input)

        try:
            recommendation = self.counseling_retriever.recommend(user_input)
            if getattr(recommendation, "matched_record_id", ""):
                return recommendation
        except Exception as exc:
            logger.warning("Mock counseling dataset recommendation failed: %s", exc)
        return self._lightweight_counseling_recommendation(user_input)

    def _lightweight_empathy_recommendation(self, user_input: str) -> EmpathyRecommendation:
        text = user_input or ""
        if any(keyword in text for keyword in ("불안", "걱정", "초조")):
            return EmpathyRecommendation(
                empathy_style_hint="불안이 이어지는 부담을 먼저 확인하고 차분하게 공감하세요.",
                emotion_label="불안",
                empathy_label="위로",
                matched_record_id="mock-anxiety",
                score=1.0,
            )
        if any(keyword in text for keyword in ("외로", "혼자", "고립")):
            return EmpathyRecommendation(
                empathy_style_hint="혼자 버티는 느낌을 먼저 알아주고 연결감을 회복하도록 돕습니다.",
                emotion_label="슬픔",
                empathy_label="위로",
                matched_record_id="mock-loneliness",
                score=1.0,
            )
        if any(keyword in text for keyword in ("무기력", "기운", "소진", "지쳤")):
            return EmpathyRecommendation(
                empathy_style_hint="무기력과 소진감을 먼저 인정하고 부담을 낮춰 공감하세요.",
                emotion_label="슬픔",
                empathy_label="위로",
                matched_record_id="mock-low-mood",
                score=1.0,
            )
        return EmpathyRecommendation(
            empathy_style_hint="감정을 먼저 확인하고, 차분하게 공감한 뒤 다음 한 걸음을 제안하세요.",
            emotion_label="",
            empathy_label="위로",
            matched_record_id="mock-general",
            score=0.5,
        )

    def _mock_empathy_recommendation(self, user_input: str) -> EmpathyRecommendation:
        if (
            not self._use_full_mock_retrievers
            and self._uses_default_dataset_loader(
                self.empathy_retriever,
                EmpathyRetriever,
                EMPATHY_DEFAULT_DATASET_PATHS,
            )
        ):
            return self._lightweight_empathy_recommendation(user_input)

        try:
            recommendation = self.empathy_retriever.recommend(user_input)
            if getattr(recommendation, "matched_record_id", ""):
                return recommendation
        except Exception as exc:
            logger.warning("Mock empathy dataset recommendation failed: %s", exc)
        return self._lightweight_empathy_recommendation(user_input)

    def _risk_stage_from_level(self, risk_level: str) -> str:
        """Convert technical risk levels into the Korean-facing stage labels."""
        normalized = (risk_level or "").strip().lower()
        if normalized in {"high", "critical"}:
            return "위험"
        if normalized in {"moderate", "medium"}:
            return "주의"
        return "관심"

    def _enum_name(self, value: Any) -> str:
        return getattr(value, "name", str(value))

    def _safe_counseling_details(
        self,
        recommendation: CounselingRecommendation,
    ) -> Dict[str, Any]:
        return {
            "matched_record_id": getattr(recommendation, "matched_record_id", ""),
            "category": getattr(recommendation, "category", "general"),
            "score": getattr(recommendation, "score", 0.0),
            "hint_present": bool(getattr(recommendation, "intervention_hint", "")),
        }

    def _safe_empathy_details(
        self,
        recommendation: EmpathyRecommendation,
    ) -> Dict[str, Any]:
        return {
            "emotion_label": getattr(recommendation, "emotion_label", ""),
            "empathy_label": getattr(recommendation, "empathy_label", ""),
            "matched_record_id": getattr(recommendation, "matched_record_id", ""),
            "score": getattr(recommendation, "score", 0.0),
            "hint_present": bool(getattr(recommendation, "empathy_style_hint", "")),
        }

    def _safe_wellness_details(
        self,
        recommendation: WellnessRecommendation,
    ) -> Dict[str, Any]:
        return {
            "risk_stage": getattr(recommendation, "risk_stage", "관심"),
            "matched_record_id": getattr(recommendation, "matched_record_id", ""),
            "matched_topic": getattr(recommendation, "matched_topic", ""),
            "distance": getattr(recommendation, "distance", 0.0),
            "hint_present": bool(getattr(recommendation, "support_hint", "")),
        }

    def _emotion_labels_from_empathy(
        self,
        recommendation: EmpathyRecommendation,
    ) -> List[EmotionLabel]:
        label = str(getattr(recommendation, "emotion_label", "") or "").strip()
        mapping = {
            "불안": EmotionLabel.ANXIETY,
            "슬픔": EmotionLabel.SADNESS,
            "분노": EmotionLabel.ANGER,
            "상처": EmotionLabel.SADNESS,
            "당황": EmotionLabel.STRESS,
            "기쁨": EmotionLabel.RELIEF,
        }
        emotion = mapping.get(label)
        return [emotion] if emotion else []

    def _serialize_intent_agent(self, intent_result: Any) -> Dict[str, Any]:
        labels = []
        for candidate in getattr(intent_result, "candidates", []) or []:
            label = getattr(candidate, "label", None)
            label_name = self._enum_name(label)
            if label_name and label_name not in labels:
                labels.append(label_name)

        primary = getattr(intent_result, "primary_intent", "")
        primary_name = self._enum_name(primary)
        if primary_name and primary_name not in labels:
            labels.insert(0, primary_name)

        return {
            "primary_intent": primary_name,
            "labels": labels,
            "s2_suspected": bool(getattr(intent_result, "s2_suspected", False)),
            "s3_sos": bool(getattr(intent_result, "s3_sos", False)),
            "confidence": round(float(getattr(intent_result, "confidence", 0.0)), 4),
        }

    def _serialize_emotional_state_agent(
        self,
        emotional_state: EmotionalStateVector,
    ) -> Dict[str, Any]:
        data = {
            key: round(float(value), 4)
            for key, value in emotional_state.to_dict().items()
        }
        data["state_summary"] = summarize_emotional_state(emotional_state)
        return data

    async def _get_memory_context_with_details(self, session_id: str) -> tuple[Any, Dict[str, Any]]:
        try:
            memory_context = await self.memory_store.get_memory_context(session_id)
            return memory_context, {
                "available": True,
                "recent_summaries": len(memory_context.recent_summaries),
                "facts": len(memory_context.facts),
                "directives": len([
                    directive for directive in memory_context.directives
                    if getattr(directive, "active", True)
                ]),
                "emotional_trend": len(memory_context.emotional_trend),
            }
        except Exception as exc:
            logger.warning("Memory context unavailable: %s", exc)
            return None, {
                "available": False,
                "error": type(exc).__name__,
            }

    async def _get_continuity_memory(self, session_id: str) -> Any:
        getter = getattr(self.memory_store, "get_conversation_continuity", None)
        if not callable(getter):
            return None
        try:
            return await getter(session_id)
        except Exception as exc:
            logger.warning("Conversation continuity unavailable: %s", exc)
            return None

    def _previous_state_from_continuity(self, continuity_memory: Any) -> Optional[EmotionalStateVector]:
        values = getattr(continuity_memory, "emotional_state_vector", None)
        if not isinstance(values, dict) or not values:
            return None
        try:
            return EmotionalStateVector(**{
                key: float(values[key])
                for key in ("mood", "anxiety", "stress", "sleep", "energy", "safety", "rapport")
                if key in values
            })
        except (TypeError, ValueError):
            return None

    def _small_action_from_continuity(
        self,
        continuity_memory: Any,
        session_id: str,
    ) -> Optional[SmallActionPlan]:
        action = getattr(continuity_memory, "last_small_action", None)
        action_text = getattr(action, "action_text", "") if action else ""
        if not isinstance(action_text, str) or not action_text.strip():
            return None
        return SmallActionPlan(
            action_id=str(getattr(action, "action_id", "")),
            title="Small action",
            session_id=session_id,
            intent_label=str(getattr(action, "intent_label", "")),
            action_text=action_text.strip(),
            rationale_label="memory_recall",
            status=str(getattr(action, "status", "suggested")),
            created_at=str(getattr(action, "created_at", "")),
            steps=[action_text.strip()],
        )

    def _apply_followup_continuity(
        self,
        intent_result: Any,
        previous_followup: str,
        user_input: str,
    ) -> Any:
        followup = (previous_followup or "").strip()
        if not followup:
            return intent_result

        sleep_followup = any(marker in followup for marker in ("잠", "수면", "잠드는", "자다가", "깨"))
        short_answer = len((user_input or "").strip()) <= 40
        sleep_context_answer = any(
            marker in (user_input or "")
            for marker in ("잠", "수면", "숙면", "잠들", "깨")
        )
        current = getattr(intent_result, "primary_intent", None)
        if not (
            sleep_followup
            and short_answer
            and (
                current in {IntentLabel.OTHER_CONCERN, IntentLabel.SUPPORT_REQUEST, IntentLabel.NEED_EMPATHY}
                or (sleep_context_answer and current == IntentLabel.ANXIETY_SUPPORT)
            )
        ):
            return intent_result

        existing = [
            candidate.label
            for candidate in getattr(intent_result, "candidates", []) or []
        ]
        if IntentLabel.SLEEP_PROBLEM not in existing:
            intent_result.candidates.insert(
                0,
                IntentCandidate(
                    label=IntentLabel.SLEEP_PROBLEM,
                    severity=IntentSeverity.S1_CONCERN,
                    confidence=0.75,
                    rationale_tags=["previous_followup_sleep_context"],
                    evidence=["previous_followup_sleep_context"],
                ),
            )
        intent_result.primary_intent = IntentLabel.SLEEP_PROBLEM
        intent_result.needs_follow_up = True
        intent_result.chat_label_hint["question"] = True
        return intent_result

    def _should_build_session_summary(self, previous_continuity: Any) -> bool:
        turn_count = int(getattr(previous_continuity, "turn_count", 0) or 0) + 1
        return turn_count > 0 and turn_count % 5 == 0

    async def _store_structured_continuity(
        self,
        session_id: str,
        intent_result: Any,
        emotional_state: EmotionalStateVector,
        risk_stage: str,
        small_action_plan: Any,
        followup_question: str,
        memory_context: Any,
        previous_continuity: Any,
        result: Dict[str, Any],
    ) -> None:
        updater = getattr(self.memory_store, "update_conversation_continuity", None)
        if not callable(updater):
            return

        latest_summary = None
        if self._should_build_session_summary(previous_continuity):
            latest_summary = build_session_dream_summary(
                session_id=session_id,
                memory_context=memory_context,
                intent_results=[intent_result] if intent_result else [],
                risk_stages=[
                    getattr(previous_continuity, "risk_stage_start", risk_stage) if previous_continuity else risk_stage,
                    risk_stage,
                ],
                last_small_action=small_action_plan,
                next_followup=followup_question,
            )
            result.setdefault("pipeline_details", {}).setdefault("agents", {})[
                "session_summary"
            ] = {
                "created": True,
                "summary_id": latest_summary.summary_id,
                "main_issue": latest_summary.main_issue,
                "emotional_trend": latest_summary.emotional_trend,
                "risk_stage_start": latest_summary.risk_stage_start,
                "risk_stage_end": latest_summary.risk_stage_end,
                "has_last_small_action": bool(latest_summary.last_small_action),
                "has_next_follow_up": bool(latest_summary.next_follow_up),
            }

        await updater(
            session_id=session_id,
            last_small_action=small_action_plan,
            next_follow_up=followup_question,
            emotional_state_vector=emotional_state.to_dict(),
            risk_stage=risk_stage,
            intent_label=getattr(getattr(intent_result, "primary_intent", ""), "value", ""),
            latest_structured_summary=latest_summary,
        )

    def _serialize_memory_recall_agent(
        self,
        proactive_recall: Any,
        continuity_memory: Any = None,
    ) -> Dict[str, Any]:
        recalled_keys = list(getattr(proactive_recall, "recalled_keys", []) or [])
        if continuity_memory is not None:
            if getattr(continuity_memory, "emotional_state_vector", None):
                if "previous_emotional_state" not in recalled_keys:
                    recalled_keys.append("previous_emotional_state")
            if getattr(continuity_memory, "latest_structured_summary", None):
                if "session_summary" not in recalled_keys:
                    recalled_keys.append("session_summary")
        return {
            "recalled_keys": recalled_keys,
            "repeated_concerns": list(getattr(proactive_recall, "repeated_concerns", []) or []),
            "preferred_response_style": list(getattr(proactive_recall, "preferred_response_style", []) or []),
            "avoid_topics": list(getattr(proactive_recall, "avoid_topics", []) or []),
            "has_last_small_action": bool(getattr(proactive_recall, "last_small_action", "")),
            "has_next_follow_up": bool(getattr(proactive_recall, "next_follow_up", "")),
            "has_previous_emotional_state": bool(
                getattr(continuity_memory, "emotional_state_vector", None)
            ),
            "has_session_summary": bool(
                getattr(continuity_memory, "latest_structured_summary", None)
            ),
            "stale": bool(getattr(proactive_recall, "stale", False)),
        }

    def _serialize_decision_agent(self, decision_result: Any) -> Dict[str, Any]:
        return {
            "primary_action": self._enum_name(getattr(decision_result, "primary_action", "")),
            "secondary_actions": [
                self._enum_name(action)
                for action in getattr(decision_result, "secondary_actions", []) or []
            ],
            "reason_codes": list(getattr(decision_result, "reason_codes", []) or []),
            "response_constraints": dict(getattr(decision_result, "response_constraints", {}) or {}),
        }

    def _serialize_small_action_agent(
        self,
        small_action_plan: Any,
    ) -> Dict[str, Any]:
        if small_action_plan is None:
            return {
                "has_action": False,
                "action_id": "",
                "intent_label": "",
                "status": "",
            }
        return {
            "has_action": bool(getattr(small_action_plan, "action_text", "")),
            "action_id": getattr(small_action_plan, "action_id", ""),
            "intent_label": str(getattr(small_action_plan, "intent_label", "")).upper(),
            "action_text": getattr(small_action_plan, "action_text", ""),
            "status": getattr(small_action_plan, "status", ""),
        }

    def _avoid_repeated_small_action(
        self,
        small_action_plan: Any,
        previous_small_action: Any,
        *,
        selected_cause: str,
    ) -> None:
        if small_action_plan is None or previous_small_action is None:
            return

        current = " ".join((getattr(small_action_plan, "action_text", "") or "").split())
        previous = " ".join((getattr(previous_small_action, "action_text", "") or "").split())
        if not current or current != previous:
            return

        alternatives = {
            "exam_assignment_pressure": "해야 할 일을 전부 보지 말고, 과제 하나나 시험 범위 하나를 골라 첫 10분만 시작해보세요.",
            "academic_relief": ACADEMIC_RELIEF_ACTION_STEP,
            "manageable_academic_concern": MANAGEABLE_ACADEMIC_CONCERN_ACTION_STEP,
            "academic_deadline_exam_overload": ACADEMIC_OVERLOAD_ACTION_STEP,
            "authority_criticism": "지적받은 내용, 말투, 평가받는 느낌을 세 칸으로 나눠 한 줄씩만 적어보세요.",
            "crying_urge": "안전한 자리에 앉아 물을 한 모금 마시고, 떠오르는 일을 한 단어로만 적어보세요.",
            "sadness_burden": "속상함이 생긴 사건 하나와 피로가 쌓인 부분 하나를 각각 한 단어로 적어보세요.",
            "anger_frustration": "가장 억울한 지점 하나와 지금 당장 미룰 수 있는 일 하나를 나눠 적어보세요.",
            "recovery_improvement": RECOVERY_IMPROVEMENT_ACTION_STEP,
        }
        replacement = alternatives.get(
            selected_cause,
            "방금 떠오른 문제를 한 문장으로 적고, 지금 바로 할 수 있는 첫 행동 하나만 골라보세요.",
        )
        if replacement == current:
            replacement = "이번에는 같은 행동을 반복하기보다, 지금 부담을 키우는 요소 하나만 적어보세요."

        small_action_plan.action_text = replacement
        small_action_plan.steps = [replacement]
        small_action_plan.rationale_label = "avoid_repeated_action"
        small_action_plan.rationale_tags = ["avoid_repeated_action", selected_cause]

    def _sync_final_safety_agent(self, result: Dict[str, Any]) -> None:
        """Mirror final risk fields into the agent-facing safety summary."""
        details = result.setdefault("pipeline_details", {})
        agents = details.setdefault("agents", {})
        existing_safety = {}
        if isinstance(agents.get("safety"), dict):
            existing_safety.update(agents["safety"])
        if isinstance(details.get("safety"), dict):
            existing_safety.update(details["safety"])

        existing_safety.update(
            {
                "risk_stage": result.get("risk_stage", "관심"),
                "risk_level": result.get("risk_level", "none"),
                "requires_crisis_response": result.get("requires_crisis_response", False),
            }
        )
        agents["safety"] = existing_safety

    def _build_agent_context(
        self,
        decision_result: Any,
        emotional_state: EmotionalStateVector,
        proactive_recall: Any,
        followup_question: str,
        small_action_plan: Any,
    ) -> Dict[str, Any]:
        return {
            "decision": self._serialize_decision_agent(decision_result),
            "emotional_state": self._serialize_emotional_state_agent(emotional_state),
            "proactive_recall": self._serialize_memory_recall_agent(proactive_recall),
            "followup": {"question": followup_question},
            "small_action": self._serialize_small_action_agent(small_action_plan),
        }

    def _add_safety_notice(self, response_text: str) -> str:
        """Append a short safety notice to normal responses."""
        notice = (
            "\n\n이 AI는 의료 진단이나 치료를 하지 않으며 전문 상담사를 대체하지 않습니다. "
            "위험 신호가 있으면 109, 119, 112 또는 가까운 응급실/지역 정신건강복지센터에 바로 연결하세요."
        )
        if not response_text:
            return notice.strip()
        if "의료 진단이나 치료" in response_text:
            return response_text
        return f"{response_text}{notice}"

    def sanitize_user_response(self, response_text: str) -> str:
        """Remove internal guidance fragments from user-facing responses."""
        if not isinstance(response_text, str):
            return ""

        normalized = response_text.replace("\r\n", "\n").replace("\r", "\n")
        safe_blocks: List[str] = []
        for block in normalized.split("\n\n"):
            lines = []
            for line in block.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if any(marker in stripped for marker in INTERNAL_RESPONSE_MARKERS):
                    continue
                lines.append(line)
            if lines:
                safe_blocks.append("\n".join(lines).strip())

        sanitized = "\n\n".join(safe_blocks).strip()
        if sanitized:
            return sanitized

        if any(number in normalized for number in ("109", "119", "112", "988", "911")):
            return normalized.strip()
        return "지금 느끼는 부담을 혼자 다 감당하지 않아도 괜찮아요. 오늘은 가장 작은 한 가지부터 같이 정리해볼게요."

    def _intent_labels(self, intent_result: Any) -> List[str]:
        labels = []
        primary = getattr(intent_result, "primary_intent", "")
        primary_name = self._enum_name(primary)
        if primary_name:
            labels.append(primary_name)
        for candidate in getattr(intent_result, "candidates", []) or []:
            label_name = self._enum_name(getattr(candidate, "label", ""))
            if label_name and label_name not in labels:
                labels.append(label_name)
        return labels

    def _is_low_mood_context(self, intent_result: Any, emotional_state: EmotionalStateVector) -> bool:
        labels = set(self._intent_labels(intent_result))
        return (
            "LOW_MOOD_SUPPORT" in labels
            or getattr(emotional_state, "mood", 1.0) <= 0.35
        )

    def _is_sleep_or_anxiety_context(self, intent_result: Any, emotional_state: EmotionalStateVector) -> bool:
        labels = set(self._intent_labels(intent_result))
        return (
            bool(labels.intersection({"SLEEP_PROBLEM", "ANXIETY_SUPPORT"}))
            or getattr(emotional_state, "anxiety", 0.0) >= 0.55
            or getattr(emotional_state, "sleep", 1.0) <= 0.45
        )

    def _safe_response_hint(
        self,
        hint: str,
        *,
        allow_low_mood: bool,
        require_action: bool = False,
    ) -> str:
        compact = " ".join((hint or "").split())
        if not compact:
            return ""

        blocked_markers = (
            "상담 참고",
            "공감 참고",
            "웰니스 참고",
            "심리상담 데이터 기반 힌트",
            "공감형 대화 기반 힌트",
            "웰니스 기반 힌트",
            "내담자의 표현을 반영",
            "핵심 감정을 명료화",
            "intervention_hint",
            "empathy_style_hint",
            "support_hint",
            "guidance",
            "therapeutic_guidance",
        )
        if any(marker in compact for marker in blocked_markers):
            return ""
        if not allow_low_mood and any(marker in compact for marker in ("기분이 우울", "우울하시군요", "우울")):
            return ""
        if "제안하세요" in compact or "공감하세요" in compact:
            return ""

        if require_action and not self._looks_like_action_text(compact):
            return ""

        return compact

    def _looks_like_action_text(self, text: str) -> bool:
        compact = " ".join((text or "").split())
        if not compact:
            return False
        blocked = (
            "감정 확인",
            "공감",
            "상담",
            "힌트",
            "기분이 우울",
            "우울하시군요",
            "반응이 도움이",
            "도움이 됩니다",
        )
        if any(marker in compact for marker in blocked):
            return False
        action_markers = (
            "보세요",
            "해보세요",
            "챙겨",
            "낮춰",
            "적어",
            "느껴",
            "마시",
            "쉬",
            "정해",
            "내려놓",
            "집중",
            "연락",
            "걸어",
        )
        return any(marker in compact for marker in action_markers)

    def _get_wellness_recommendation(
        self,
        wellness_checkin: Optional[Dict[str, Any]],
    ) -> Optional[WellnessRecommendation]:
        if not wellness_checkin:
            return None

        try:
            if self.mock_mode:
                if (
                    not self._use_full_mock_retrievers
                    and self._uses_default_dataset_loader(
                        self.wellness_recommender,
                        WellnessRecommender,
                        WELLNESS_DEFAULT_DATASET_PATHS,
                    )
                ):
                    return self._lightweight_wellness_recommendation(wellness_checkin)
                recommendation = self.wellness_recommender.recommend(wellness_checkin)
                if recommendation:
                    return recommendation
                return self._lightweight_wellness_recommendation(wellness_checkin)
            return self.wellness_recommender.recommend(wellness_checkin)
        except Exception as exc:
            logger.warning("Wellness recommender failed: %s", exc)
            return None

    def _lightweight_wellness_recommendation(
        self,
        wellness_checkin: Dict[str, Any],
    ) -> WellnessRecommendation:
        sleep = int(wellness_checkin.get("sleep_quality", 3) or 3)
        anxiety = int(wellness_checkin.get("anxiety_score", 3) or 3)
        loneliness = int(wellness_checkin.get("loneliness_score", 3) or 3)
        stress = int(wellness_checkin.get("stress_score", 3) or 3)

        if sleep <= 2:
            return WellnessRecommendation(
                support_hint="잠들기 전 화면 밝기를 낮추고, 발바닥 감각을 30초만 느껴보세요.",
                risk_stage="주의",
                matched_record_id="mock-wellness-sleep",
                matched_topic="sleep",
                distance=0.0,
            )
        if anxiety >= 4:
            return WellnessRecommendation(
                support_hint="숨을 천천히 내쉬며 지금 보이는 물건 세 가지를 확인해보세요.",
                risk_stage="주의",
                matched_record_id="mock-wellness-anxiety",
                matched_topic="anxiety",
                distance=0.0,
            )
        if loneliness >= 4:
            return WellnessRecommendation(
                support_hint="부담이 낮은 사람 한 명에게 짧은 안부 메시지를 보내보세요.",
                risk_stage="주의",
                matched_record_id="mock-wellness-loneliness",
                matched_topic="loneliness",
                distance=0.0,
            )
        if stress >= 4:
            return WellnessRecommendation(
                support_hint="오늘 해야 할 일을 한 줄로만 적고 하나만 고르세요.",
                risk_stage="주의",
                matched_record_id="mock-wellness-stress",
                matched_topic="stress",
                distance=0.0,
            )
        return WellnessRecommendation(
            support_hint="지금은 숨을 고르고, 오늘 할 수 있는 가장 작은 한 가지를 선택해 보세요.",
            risk_stage="관심",
            matched_record_id="mock-wellness-general",
            matched_topic="general",
            distance=0.0,
        )

    def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
        return response_text

    def _compose_mock_response(
        self,
        counseling_hint: str,
        empathy_style_hint: str,
        wellness_hint: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        followup_question: str = "",
        small_action_text: str = "",
        cause_exploration: Optional[CauseExplorationResult] = None,
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
        user_input: str = "",
    ) -> str:
        emotional_state = emotional_state or EmotionalStateVector()
        allow_low_mood = self._is_low_mood_context(intent_result, emotional_state)
        intent_labels = set(self._intent_labels(intent_result))
        primary_intent = self._enum_name(getattr(intent_result, "primary_intent", ""))
        selected_cause = getattr(cause_exploration, "selected_cause", "") or ""
        empathy_only = primary_intent == "NEED_EMPATHY" and not intent_labels.intersection(
            {"SLEEP_PROBLEM", "ANXIETY_SUPPORT", "LOW_MOOD_SUPPORT"}
        )
        advice_requested = "NEED_ADVICE" in intent_labels and not empathy_only

        segments = [
            "지금 느끼는 부담이 꽤 컸을 것 같아요.",
            "이런 상태에서는 마음이 복잡해지고, 무엇부터 해야 할지 막막하게 느껴질 수 있습니다.",
        ]
        if selected_cause == "authority_criticism":
            segments = [
                "혼나거나 지적받은 뒤에는 창피함과 화가 같이 올라와서 한동안 마음에 남을 수 있어요.",
                "그 반응은 이상한 게 아니고, 지적 내용뿐 아니라 말투나 평가받는 느낌까지 한꺼번에 건드려졌을 때 더 크게 느껴질 수 있습니다.",
            ]
        elif selected_cause == "academic_relief":
            segments = [
                "시험이 하나만 남았다는 말에서 정말 숨이 조금 트이는 느낌이 전해져요.",
                "그동안 많이 버텼고, 이제 끝이 보이기 시작해서 기쁜 마음이 올라온 것 같아요.",
            ]
        elif selected_cause == "manageable_academic_concern":
            segments = [
                "인공지능 과목이 조금 어렵고 암기할 내용이 많다는 건 부담될 수 있어요.",
                "그래도 주말에 시간이 있다고 느끼는 걸 보면, 지금은 완전히 막막하다기보다 정리해서 마무리할 수 있다고 보는 상태에 가까워 보여요.",
            ]
        elif selected_cause == "specific_academic_burden_after_relief":
            subject = self._specific_academic_subject(user_input)
            burden = self._specific_academic_burden(user_input)
            segments = [
                f"{subject} 과목이라 {burden}, 시험이 하나만 남았어도 부담이 크게 느껴질 수 있어요.",
                "그래도 종강했고 남은 시험이 하나라는 점에서는 끝이 보이는 상태라, 지금은 완전히 막힌 상황이라기보다 남은 암기량을 어떻게 정리할지가 핵심인 것 같아요.",
            ]
        elif selected_cause == "specific_academic_burden":
            subject = self._specific_academic_subject(user_input)
            academic_target = f"{subject} 시험" if "시험" in user_input else f"{subject} 과목"
            burden = self._specific_academic_burden(user_input)
            segments = [
                f"{academic_target}에서 {burden} 막막하게 느껴질 수 있어요.",
                "특히 개념, 용어, 알고리즘 흐름이 한꺼번에 나오면 어디서부터 외워야 할지 잡기 어려울 수 있습니다.",
            ]
        elif selected_cause == "recovery_improvement":
            segments = [
                "조금 괜찮아지거나 나아졌다고 느낀 건 중요한 변화예요.",
                "좋아진 이유를 단정할 필요는 없지만, 휴식이나 거리두기, 누군가의 말처럼 무엇이 회복에 도움이 됐는지 살펴보면 다음에도 붙잡을 단서가 됩니다.",
            ]
        elif selected_cause == "academic_deadline_exam_overload":
            segments = [
                "과제 마감과 기말 준비가 동시에 겹친 상황이군요.",
                "이 압박은 의지가 약해서라기보다 마감, 시험 범위, 성적 걱정, 피로가 동시에 몰릴 때 자연스럽게 커질 수 있습니다.",
            ]
        elif selected_cause == "exam_assignment_pressure" or self._is_study_exam_pressure(user_input):
            segments = [
                "공부나 시험, 과제 부담을 말할 정도면 압박이 꽤 크게 쌓인 상태일 수 있어요.",
                "이 반응은 이상한 게 아니고, 공부량 자체, 성적 걱정, 마감 압박, 지쳐서 버티기 어려운 느낌이 겹치면 누구라도 막막해질 수 있습니다.",
            ]
        elif selected_cause == "self_blame":
            segments = [
                "자꾸 내 탓처럼 느껴질 때는 상황 전체를 보기보다 나만 문제였다는 결론으로 마음이 빨리 좁아질 수 있어요.",
                "단정하기보다 내가 책임질 수 있는 부분과 내 책임이 아닌 부분을 천천히 나눠보면 좋겠습니다.",
            ]
        elif selected_cause == "anger_frustration":
            segments = [
                "짜증이나 화가 올라올 만큼 뭔가가 많이 건드려진 상태로 보여요.",
                "그 반응은 참을성이 부족해서라기보다 억울함, 지친 상태, 계속 밀리는 압박이 겹칠 때 강해질 수 있습니다.",
            ]
        elif selected_cause == "sleep_maintenance":
            segments = [
                "중간에 자주 깨는 쪽이면 자꾸 깨는 밤이 반복되고, 잠을 자도 몸이 충분히 쉬지 못한 느낌이 남을 수 있어요.",
                "원인을 단정하기보다, 깨고 난 뒤 다시 잠들기 어려운지부터 같이 확인해보면 좋겠습니다.",
            ]
        elif selected_cause == "worry_or_anxiety":
            segments = [
                "잠자리에서 걱정이 커지거나 불안이 이어지면, 쉬어야 할 시간에도 머릿속이 계속 켜져 있는 것처럼 느껴질 수 있어요.",
                "원인을 단정하기보다, 어떤 생각이 밤에 더 크게 올라오는지부터 같이 작게 확인해보면 좋겠습니다.",
            ]
        elif selected_cause == "physical_tension_chest":
            segments = [
                "가슴이 답답하고 속상한 느낌이 같이 오면, 마음의 부담이 몸의 긴장으로도 느껴질 수 있어요.",
                "다만 이것만으로 원인을 단정하거나 진단할 수는 없어서, 지금은 몸의 긴장과 속상함을 나눠서 천천히 살펴보면 좋겠습니다.",
            ]
        elif selected_cause == "crying_urge":
            segments = [
                "울고 싶은 마음이나 눈물이 올라오는 상태라면 지금 감정이 꽤 벅찬 것 같아요.",
                "그건 약해서가 아니라 특정한 일이 마음에 남았거나, 피로와 감정이 쌓여 더는 안쪽에만 머물기 어려워졌다는 신호일 수 있습니다.",
            ]
        elif selected_cause == "sadness_burden":
            segments = [
                "속상하다고 말할 정도면 마음이 먼저 무거워진 시간이 있었을 것 같아요.",
                "이 감정은 특정한 사건에서 올라왔을 수도 있고, 피로와 서운함이 누적되면서 더 크게 느껴졌을 수도 있습니다.",
            ]
        elif primary_intent == "SLEEP_PROBLEM":
            segments = [
                "잠이 잘 오지 않고 불안까지 겹치면 몸과 마음이 계속 긴장한 채로 버티는 느낌이 들 수 있어요.",
                "원인을 단정하기보다, 머릿속 걱정이나 생활 리듬, 몸의 피로가 함께 영향을 줄 가능성을 같이 조금씩 좁혀보면 좋겠습니다.",
            ]
        elif primary_intent == "ANXIETY_SUPPORT":
            segments = [
                "불안이 계속 올라오면 몸도 마음도 쉬지 못하고 긴장한 채로 버티는 느낌이 들 수 있어요.",
                "원인을 단정하기보다, 해야 할 일의 압박이나 관계 긴장, 앞으로의 불확실성이 함께 영향을 주는지 같이 좁혀보면 좋겠습니다.",
            ]
        elif allow_low_mood:
            segments = [
                "무기력하거나 기운이 없는 상태로 하루를 버티는 일이 꽤 무겁게 느껴졌을 것 같아요.",
                "원인을 단정할 수는 없지만, 소진감이나 고립감, 스스로를 낮게 보는 생각이 겹쳤는지 같이 살펴볼 수 있습니다.",
            ]
        elif primary_intent in {"STRESS_SUPPORT", "WORK_OR_STUDY_STRESS"}:
            segments = [
                "부담이 계속 쌓이면 몸과 마음이 계속 긴장한 상태로 버티게 될 수 있어요.",
                "원인을 단정하기보다, 일이 많은지, 시작점이 막막한지, 끝내야 한다는 압박이 큰지 같이 좁혀보면 좋겠습니다.",
            ]
        elif primary_intent == "RELATIONSHIP_STRESS":
            segments = [
                "관계에서 생긴 긴장은 혼자 정리하려 할수록 더 크게 느껴질 수 있어요.",
                "원인을 단정하기보다, 소통이 막힌 느낌인지 혼자 감당하는 느낌인지 같이 살펴볼 수 있습니다.",
            ]
        elif empathy_only:
            segments = [
                "지금은 해결책을 서둘러 찾기보다, 그 마음이 얼마나 버거웠는지 먼저 알아주는 게 필요해 보여요.",
                "여기서는 판단하거나 몰아붙이지 않고, 지금 느끼는 감정을 차분히 함께 확인해볼게요.",
            ]
        elif empathy_style_hint:
            segments[1] = (
                "지금의 반응은 이상하거나 약한 것이 아니라, "
                "많이 버텨온 마음이 보내는 신호일 수 있어요."
            )

        action_step = ""
        safe_small_action = self._safe_response_hint(
            small_action_text,
            allow_low_mood=allow_low_mood,
            require_action=True,
        )

        tailored_action_step = ""
        if selected_cause == "authority_criticism":
            tailored_action_step = "지적 내용, 말투, 평가받는 느낌을 세 칸으로 나눠 각각 한 단어씩만 적어보세요."
        elif selected_cause == "academic_relief":
            tailored_action_step = ACADEMIC_RELIEF_ACTION_STEP
        elif selected_cause == "manageable_academic_concern":
            tailored_action_step = MANAGEABLE_ACADEMIC_CONCERN_ACTION_STEP
        elif selected_cause == "recovery_improvement":
            tailored_action_step = RECOVERY_IMPROVEMENT_ACTION_STEP
        elif selected_cause == "academic_deadline_exam_overload":
            tailored_action_step = ACADEMIC_OVERLOAD_ACTION_STEP
        elif selected_cause == "exam_assignment_pressure":
            tailored_action_step = ACADEMIC_PRESSURE_ACTION_STEP
        elif selected_cause == "self_blame":
            tailored_action_step = "지금 떠오르는 자책 문장을 하나 적고, 그 옆에 사실로 확인된 것만 짧게 다시 써보세요."
        elif selected_cause == "anger_frustration":
            tailored_action_step = "바로 답하거나 결정하기 전에 숨을 세 번 내쉬고, 가장 화난 이유를 한 문장으로만 적어보세요."
        elif selected_cause == "sleep_maintenance":
            tailored_action_step = "밤에 깨면 바로 시간을 확인하지 말고, 눈을 감은 채 호흡을 천천히 세 번만 해보세요."
        elif selected_cause == "worry_or_anxiety":
            tailored_action_step = "잠들기 전 떠오르는 걱정 하나만 짧게 적고, 지금 해결할 일과 내일 볼 일을 나눠보세요."
        elif selected_cause == "physical_tension_chest":
            tailored_action_step = "어깨를 한번 아래로 내려놓고, 숨을 천천히 내쉬는 호흡을 세 번만 해보세요."
        elif selected_cause == "crying_urge":
            tailored_action_step = "울음을 참으려 애쓰기보다 안전한 자리에 앉아 물을 한 모금 마시고 감정이 지나갈 시간을 조금 주세요."
        elif selected_cause == "sadness_burden":
            tailored_action_step = "지금 속상함을 한 단어로만 적고, 몸에 힘이 들어간 곳 하나를 천천히 풀어보세요."

        if empathy_only:
            action_step = "지금 감정을 고치려 하지 말고, 가장 가까운 감정 단어 하나만 천천히 떠올려보세요."
        elif advice_requested:
            action_step = "문제를 한 번에 해결하려 하지 말고, 지금 바로 할 수 있는 작은 실행 단계 하나만 정해보세요."
        elif safe_small_action:
            action_step = safe_small_action
        elif tailored_action_step:
            action_step = tailored_action_step
        else:
            for hint in (wellness_hint, counseling_hint):
                safe_hint = self._safe_response_hint(hint, allow_low_mood=allow_low_mood)
                if safe_hint:
                    action_step = safe_hint
                    break

        if not action_step:
            action_step = "지금 당장 해결하려 하기보다, 오늘 할 수 있는 가장 작은 한 가지를 정해보세요."

        if followup_question:
            segments.append(followup_question)
        if selected_cause == "academic_deadline_exam_overload":
            segments.append(f"오늘은 {action_step}")
        elif safe_small_action and action_step == safe_small_action:
            segments.append(f"오늘의 작은 행동으로는 {action_step}")
        elif small_action_text:
            segments.append(f"오늘의 작은 행동으로는 {action_step}")
        else:
            action_prefix = "" if action_step.startswith("오늘") else "오늘은 "
            segments.append(f"{action_prefix}{action_step}")
        return "\n\n".join(segments)

    def _is_study_exam_pressure(self, user_input: str) -> bool:
        return any(marker in (user_input or "") for marker in STUDY_EXAM_PRESSURE_MARKERS)

    def _specific_academic_subject(self, user_input: str) -> str:
        current = user_input or ""
        for marker in ("인공지능", "AI", "머신러닝", "딥러닝", "알고리즘", "자료구조", "전공"):
            if marker in current:
                return marker
        return "남은"

    def _specific_academic_burden(self, user_input: str) -> str:
        current = user_input or ""
        if "암기" in current or "외울" in current:
            return "암기할 내용이 많으면"
        if "용어" in current:
            return "구분할 용어가 많으면"
        if "공식" in current:
            return "정리하고 외울 공식이 많으면"
        if "개념" in current:
            return "정리할 개념이 많으면"
        return "정리할 내용이 많으면"

    async def _naturalize_response_with_ollama(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        try:
            return await asyncio.wait_for(
                self._generate_ollama_response(
                    user_input=user_input,
                    intent_result=intent_result,
                    emotional_state=emotional_state,
                    cause_exploration=cause_exploration,
                    followup_question=followup_question,
                    small_action_text=small_action_text,
                    selected_safe_hints=selected_safe_hints or [],
                ),
                timeout=self.ollama_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Ollama naturalization timed out after %ss",
                self.ollama_timeout_seconds,
            )
            return ""
        except Exception as exc:
            logger.warning("Ollama naturalization failed: %s", exc)
            return ""

    async def _naturalize_response_with_gemini(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        try:
            return await asyncio.wait_for(
                self._generate_gemini_response(
                    user_input=user_input,
                    intent_result=intent_result,
                    emotional_state=emotional_state,
                    cause_exploration=cause_exploration,
                    followup_question=followup_question,
                    small_action_text=small_action_text,
                    selected_safe_hints=selected_safe_hints or [],
                ),
                timeout=self.gemini_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Gemini naturalization timed out after %ss",
                self.gemini_timeout_seconds,
            )
            return ""
        except Exception as exc:
            logger.warning("Gemini naturalization failed: %s", exc)
            return ""

    async def _generate_gemini_response(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set; using template fallback")
            return ""

        prompt = self._build_llm_naturalization_prompt(
            provider="Gemini",
            user_input=user_input,
            intent_result=intent_result,
            emotional_state=emotional_state,
            cause_exploration=cause_exploration,
            followup_question=followup_question,
            small_action_text=small_action_text,
            selected_safe_hints=selected_safe_hints or [],
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.75,
                "topP": 0.9,
                "maxOutputTokens": self.gemini_max_output_tokens,
            },
        }

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._call_gemini(payload))
        except Exception as exc:
            logger.warning("Gemini response generation unavailable: %s", exc)
            return ""

    async def _generate_ollama_response(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        prompt = self._build_ollama_prompt(
            user_input=user_input,
            intent_result=intent_result,
            emotional_state=emotional_state,
            cause_exploration=cause_exploration,
            followup_question=followup_question,
            small_action_text=small_action_text,
            selected_safe_hints=selected_safe_hints or [],
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.75,
                "top_p": 0.9,
                "num_predict": 160,
            },
        }

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._call_ollama(payload))
        except Exception as exc:
            logger.warning("Ollama response generation unavailable: %s", exc)
            return ""

    def _build_ollama_prompt(
        self,
        *,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return self._build_llm_naturalization_prompt(
            provider="Ollama",
            user_input=user_input,
            intent_result=intent_result,
            emotional_state=emotional_state,
            cause_exploration=cause_exploration,
            followup_question=followup_question,
            small_action_text=small_action_text,
            selected_safe_hints=selected_safe_hints or [],
        )

    def _build_llm_naturalization_prompt(
        self,
        *,
        provider: str,
        user_input: str,
        intent_result: Any = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        cause_exploration: Optional[CauseExplorationResult] = None,
        followup_question: str = "",
        small_action_text: str = "",
        selected_safe_hints: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        emotional_summary = []
        if emotional_state is not None:
            emotional_summary = summarize_emotional_state(emotional_state)

        safe_context = {
            "user_input": " ".join((user_input or "").split())[:600],
            "detected_intent": self._enum_name(getattr(intent_result, "primary_intent", "")),
            "emotional_state": emotional_summary,
            "selected_cause": str(getattr(cause_exploration, "selected_cause", "") or ""),
            "followup_question": " ".join((followup_question or "").split())[:240],
            "action_step": " ".join((small_action_text or "").split())[:240],
            "selected_safe_hints": [
                {
                    "label_or_topic": hint.get("label_or_topic", "")[:80],
                    "short_summary": hint.get("short_summary", "")[:160],
                    "intent_hint": hint.get("intent_hint", "")[:140],
                    "emotion_hint": hint.get("emotion_hint", "")[:140],
                    "cause_hint": hint.get("cause_hint", "")[:140],
                    "action_hint": hint.get("action_hint", "")[:140],
                    "safety_hint": hint.get("safety_hint", "")[:140],
                }
                for hint in (selected_safe_hints or [])[:3]
            ],
            "safety_constraints": [
                "do_not_diagnose",
                "do_not_claim_doctor_or_therapist",
                "crisis_flow_already_handled_before_this_call",
                "do_not_include_raw_dataset_memory_or_internal_debug_details",
            ],
        }
        compact_context = {
            key: value
            for key, value in safe_context.items()
            if value not in ("", [], None)
        }

        return (
            f"너는 {provider} 기반 데모용 한국어 정서 지원 응답을 자연스럽게 다듬는 역할이다.\n"
            "Agent Pipeline이 이미 안전한 응답 재료를 골랐다. 아래 안전하게 요약된 정보만 사용하라.\n"
            "원문 데이터셋/기억/내부 추론을 언급하지 마라.\n"
            "진단하지 말고, 의사나 치료자라고 주장하지 마라.\n"
            "따뜻하지만 간결하게 3~5개의 짧은 한국어 완결 문장으로 답하라.\n"
            "정상 비위기 상담 응답은 반드시 다음 흐름을 지켜라: "
            "1) 사용자의 감정이나 상황을 먼저 공감하고 가능한 경우 구체 맥락을 반영한다. "
            "2) 반응이 이해될 수 있음을 부드럽게 검증하되 진단하지 않는다. "
            "3) 원인을 단정하지 말고 safe_context의 selected_cause와 followup_question을 바탕으로 "
            "가능한 원인 후보 1~2개를 조심스럽게 탐색하며 질문 하나를 포함한다. "
            "4) 마지막에는 맥락에 맞는 작은 행동 하나만 제안한다.\n"
            "모든 문장은 반드시 자연스럽게 끝내고, 문장 중간에서 멈추지 마라.\n"
            "마지막 문장도 반드시 요, 다, 습니다, 세요, ?, !, . 중 하나로 끝내라.\n"
            "followup_question과 action_step이 있으면 자연스럽게 포함하라.\n"
            "매번 같은 일반 공감 문장을 반복하지 말고, 사용자의 표현에 맞춰 다르게 말하라.\n\n"
            f"safe_context: {json.dumps(compact_context, ensure_ascii=False)}\n\n"
            "응답:"
        )

    def _call_gemini(self, payload: Dict[str, Any]) -> str:
        url = self._gemini_url_with_key()
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.gemini_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid Gemini JSON response")
            return ""

        text = self._extract_gemini_text(parsed)
        sanitized = self.sanitize_user_response(text)
        if not self._is_valid_cloud_llm_response(sanitized):
            logger.warning("Gemini response invalid; using fallback")
            return ""
        return sanitized

    def _gemini_url_with_key(self) -> str:
        if "key=" in self.gemini_url:
            return self.gemini_url
        separator = "&" if "?" in self.gemini_url else "?"
        return f"{self.gemini_url}{separator}{urllib.parse.urlencode({'key': self.gemini_api_key})}"

    def _extract_gemini_text(self, parsed: Dict[str, Any]) -> str:
        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not isinstance(parts, list):
            return ""
        text_parts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text", ""), str)
        ]
        return "\n".join(part for part in text_parts if part).strip()

    def _call_ollama(self, payload: Dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.ollama_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid Ollama JSON response")
            return ""

        text = parsed.get("response", "")
        if not isinstance(text, str):
            return ""

        sanitized = self.sanitize_user_response(text)
        if not self._is_valid_cloud_llm_response(sanitized):
            return ""
        return sanitized

    def _is_valid_cloud_llm_response(self, response_text: str) -> bool:
        compact = " ".join((response_text or "").split())
        if len(compact) < 16:
            return False
        if len(compact) > 1200:
            return False
        blocked = (
            *INTERNAL_RESPONSE_MARKERS,
            "safe_context",
            "raw_text",
            "raw memory",
            "dataset",
            "데이터셋 원문",
        )
        if any(marker in compact for marker in blocked):
            return False
        if not compact.endswith(COMPLETE_SENTENCE_ENDINGS):
            return False

        clauses = [part.strip() for part in re.split(r"[.!?。！？\n]+", compact) if part.strip()]
        if len(clauses) <= 1 and not compact.endswith(("요", "다", "습니다", "세요")):
            return False

        last_clause = clauses[-1] if clauses else compact
        if self._looks_like_incomplete_korean_fragment(last_clause):
            return False
        return True

    def _looks_like_incomplete_korean_fragment(self, text: str) -> bool:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return True
        trimmed = compact.rstrip(".!?。！？")
        if trimmed.endswith(INCOMPLETE_KOREAN_SUFFIXES):
            return True
        tokens = trimmed.split()
        if len(tokens) <= 4 and not trimmed.endswith(("요", "다", "습니다", "세요")):
            return True
        return False

    async def process_message_stream(
        self,
        user_input: str,
        session_id: str
    ) -> AsyncIterator[str]:
        """
        Process message with streaming response.

        Args:
            user_input: User's message
            session_id: Session identifier

        Yields:
            Response tokens
        """
        if not self._initialized:
            await self.initialize()

        result = await self.process_message(user_input, session_id)
        yield result["response"]


async def main():
    """Main entry point for running the agent."""
    agent = PsychologistAgent()
    await agent.initialize()

    # Create a session
    session = await agent.session_manager.create_session()
    print(f"Created session: {session.session_id}")

    # Example conversation
    messages = [
        "Hi, I've been feeling really anxious lately about work.",
        "It's hard to concentrate and I feel overwhelmed.",
        "What can I do to feel better?"
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        result = await agent.process_message(msg, session.session_id)
        print(f"Agent: {result['response']}")
        print(f"Risk Level: {result['risk_level']}")

    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
