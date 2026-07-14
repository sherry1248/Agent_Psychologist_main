"""
Rule-based deterministic small action planner.

The planner emits one low-burden action from structured agent signals. It never
stores user raw text fields.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    EmotionalStateVector,
    IntentAgentResult,
    IntentCandidate,
    IntentLabel,
    SmallActionPlan,
)


ACTION_TEMPLATES = {
    IntentLabel.SLEEP_PROBLEM: "오늘 잠들기 전 10분만 화면을 내려놓고, 방 조명을 조금 낮춰보세요.",
    IntentLabel.ANXIETY_SUPPORT: "지금 자리에서 발바닥 감각을 30초만 느껴보세요.",
    IntentLabel.STRESS_SUPPORT: "해야 할 일을 한 줄로만 적고, 가장 작은 첫 단계에 동그라미를 쳐보세요.",
    IntentLabel.WORK_OR_STUDY_STRESS: "지금 해야 할 일 중 가장 작은 것 하나만 5분 동안 시작해보세요.",
    IntentLabel.LOW_MOOD_SUPPORT: "물 한 잔을 마시고 창문 근처에서 1분만 서 있어보세요.",
    IntentLabel.RELATIONSHIP_STRESS: "바로 답장하기 전에, 하고 싶은 말을 메모장에 한 문장만 적어보세요.",
    IntentLabel.FAMILY_CONFLICT: "지금 당장 대화하려 하기보다, 내가 느낀 감정을 한 단어로만 적어보세요.",
    IntentLabel.LOW_SELF_ESTEEM: "오늘 버틴 일 하나를 아주 작게라도 적어보세요.",
    IntentLabel.NEED_EMPATHY: "지금 감정을 고치려 하지 말고, 가장 가까운 감정 단어 하나만 골라보세요.",
    IntentLabel.NEED_ADVICE: "문제를 한 번에 해결하려 하지 말고, 지금 가장 먼저 할 수 있는 행동 하나만 적어보세요.",
    IntentLabel.OTHER_CONCERN: "오늘 할 수 있는 가장 작은 행동 하나를 정해보세요.",
}

MAX_HINT_ACTION_CHARS = 80
NON_ACTION_HINT_MARKERS = (
    "상담 참고",
    "공감 참고",
    "웰니스 참고",
    "감정 확인",
    "공감",
    "상담",
    "힌트",
    "제안하세요",
    "반응이 도움이",
    "도움이 됩니다",
    "기분이 우울",
    "우울하시군요",
)

ACTION_MARKERS = (
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


def _new_action_id() -> str:
    return f"action_{uuid4().hex[:8]}"


def _created_at() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _blocked_by_safety(
    risk_stage: str,
    decision_result: Optional[DecisionAgentResult],
) -> bool:
    return (
        (risk_stage or "").strip() == "위험"
        or bool(
            decision_result
            and decision_result.primary_action == DecisionAction.ESCALATE_SAFETY
        )
    )


def _best_candidate(intent_result: IntentAgentResult) -> Optional[IntentCandidate]:
    if not intent_result.candidates:
        return None
    return max(intent_result.candidates, key=lambda candidate: candidate.confidence)


def _select_intent(intent_result: Optional[IntentAgentResult]) -> IntentLabel:
    if intent_result is None:
        return IntentLabel.OTHER_CONCERN

    if intent_result.primary_intent:
        return intent_result.primary_intent

    candidate = _best_candidate(intent_result)
    if candidate:
        return candidate.label

    return IntentLabel.OTHER_CONCERN


def _clean_hint_action(wellness_hint: str) -> str:
    compact = " ".join((wellness_hint or "").split())
    if not compact:
        return ""

    if any(marker in compact for marker in NON_ACTION_HINT_MARKERS):
        return ""

    if not any(marker in compact for marker in ACTION_MARKERS):
        return ""

    if len(compact) <= MAX_HINT_ACTION_CHARS:
        return compact

    return compact[:MAX_HINT_ACTION_CHARS].rstrip() + "..."


def _rationale_label(
    intent_label: IntentLabel,
    wellness_hint: str,
    counseling_hint: str,
    emotional_state: Optional[EmotionalStateVector],
) -> str:
    if wellness_hint.strip():
        return "wellness_hint_action"
    if counseling_hint.strip():
        return "counseling_hint_considered"
    if emotional_state and emotional_state.anxiety >= 0.65:
        return "elevated_anxiety_state"
    if emotional_state and emotional_state.sleep < 0.4:
        return "low_sleep_state"
    return f"{intent_label.value}_template"


def _make_plan(
    session_id: str,
    intent_label: IntentLabel,
    action_text: str,
    rationale_label: str,
    status: str = "suggested",
) -> SmallActionPlan:
    return SmallActionPlan(
        action_id=_new_action_id(),
        title="Small action" if action_text else "Safety priority",
        session_id=session_id,
        intent_label=intent_label.value,
        action_text=action_text,
        rationale_label=rationale_label,
        status=status,
        created_at=_created_at(),
        check_after_turns=1,
        steps=[action_text] if action_text else [],
        estimated_minutes=5,
        difficulty="easy",
        rationale_tags=[rationale_label],
    )


def generate_small_action_plan(
    session_id: str = "",
    intent_result: Optional[IntentAgentResult] = None,
    decision_result: Optional[DecisionAgentResult] = None,
    emotional_state: Optional[EmotionalStateVector] = None,
    wellness_hint: str = "",
    counseling_hint: str = "",
    risk_stage: str = "관심",
) -> SmallActionPlan:
    """Generate one small action plan for the current turn."""
    intent_label = _select_intent(intent_result)

    if _blocked_by_safety(risk_stage, decision_result):
        return _make_plan(
            session_id=session_id,
            intent_label=intent_label,
            action_text="",
            rationale_label="safety_priority",
            status="safety_blocked",
        )

    hint_action = _clean_hint_action(wellness_hint)
    action_text = hint_action or ACTION_TEMPLATES.get(
        intent_label,
        ACTION_TEMPLATES[IntentLabel.OTHER_CONCERN],
    )

    return _make_plan(
        session_id=session_id,
        intent_label=intent_label,
        action_text=action_text,
        rationale_label=_rationale_label(
            intent_label=intent_label,
            wellness_hint=wellness_hint,
            counseling_hint=counseling_hint,
            emotional_state=emotional_state,
        ),
    )


@dataclass
class SmallActionPlanner:
    """Small wrapper class for callers that prefer an agent object."""

    def generate(
        self,
        session_id: str = "",
        intent_result: Optional[IntentAgentResult] = None,
        decision_result: Optional[DecisionAgentResult] = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        wellness_hint: str = "",
        counseling_hint: str = "",
        risk_stage: str = "관심",
    ) -> SmallActionPlan:
        return generate_small_action_plan(
            session_id=session_id,
            intent_result=intent_result,
            decision_result=decision_result,
            emotional_state=emotional_state,
            wellness_hint=wellness_hint,
            counseling_hint=counseling_hint,
            risk_stage=risk_stage,
        )
