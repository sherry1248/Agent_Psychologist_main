"""
Rule-based deterministic session dream summary builder.

This module creates compact session summaries from structured agent outputs
without storing raw conversation text.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Optional
from uuid import uuid4

from src.agent.models import (
    EmotionAgentResult,
    EmotionLabel,
    IntentAgentResult,
    IntentLabel,
    SessionDreamSummary,
    SmallActionPlan,
)


INTENT_ISSUE_LABELS = {
    IntentLabel.SLEEP_PROBLEM: "sleep",
    IntentLabel.ANXIETY_SUPPORT: "anxiety",
    IntentLabel.STRESS_SUPPORT: "stress",
    IntentLabel.WORK_OR_STUDY_STRESS: "work_or_study",
    IntentLabel.RELATIONSHIP_STRESS: "relationship",
    IntentLabel.LOW_MOOD_SUPPORT: "low_mood",
    IntentLabel.LOW_SELF_ESTEEM: "low_self_esteem",
    IntentLabel.CRISIS_SIGNAL: "crisis_signal",
    IntentLabel.FAMILY_CONFLICT: "family_conflict",
    IntentLabel.NEED_EMPATHY: "need_empathy",
    IntentLabel.NEED_ADVICE: "need_advice",
    IntentLabel.OTHER_CONCERN: "other_concern",
}

ALLOWED_ISSUES = set(INTENT_ISSUE_LABELS.values())
ALLOWED_EMOTIONS = {item.value for item in EmotionLabel}
RAW_VALUE_MARKERS = ("\n", ".", "요", "다", "습니다", "했", "말했", "사용자")
MAX_LABEL_LENGTH = 48


def _created_at() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _summary_id() -> str:
    return f"dream_{uuid4().hex[:8]}"


def _append_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _safe_label(value: Any, allowed: Optional[Iterable[str]] = None) -> str:
    if not isinstance(value, str):
        return ""

    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_LABEL_LENGTH:
        return ""
    if any(marker in cleaned for marker in RAW_VALUE_MARKERS):
        return ""
    if allowed is not None and cleaned not in set(allowed):
        return ""
    return cleaned


def _intent_labels(intent_result: IntentAgentResult) -> List[IntentLabel]:
    labels = [intent_result.primary_intent]
    labels.extend(candidate.label for candidate in intent_result.candidates)
    return labels


def _build_main_issue(intent_results: Optional[List[IntentAgentResult]]) -> List[str]:
    issues: List[str] = []
    for intent_result in intent_results or []:
        for label in _intent_labels(intent_result):
            issue = INTENT_ISSUE_LABELS.get(label)
            safe_issue = _safe_label(issue, ALLOWED_ISSUES)
            if safe_issue:
                _append_unique(issues, safe_issue)
    return issues


def _intensity_suffix(intensity: float) -> str:
    if intensity >= 0.75:
        return "high"
    if intensity >= 0.45:
        return "moderate"
    return "low"


def _build_emotional_trend(
    emotion_results: Optional[List[EmotionAgentResult]],
) -> List[str]:
    trend: List[str] = []
    for emotion_result in emotion_results or []:
        labels = [emotion_result.primary_label]
        labels.extend(emotion_result.secondary_labels)
        for label in labels:
            emotion = label.value if isinstance(label, EmotionLabel) else str(label)
            safe_emotion = _safe_label(emotion, ALLOWED_EMOTIONS)
            if safe_emotion:
                _append_unique(
                    trend,
                    f"{safe_emotion}_{_intensity_suffix(emotion_result.intensity)}",
                )
    return trend


def _risk_start(risk_stages: Optional[List[str]]) -> str:
    if not risk_stages:
        return "관심"
    return risk_stages[0] or "관심"


def _risk_end(risk_stages: Optional[List[str]]) -> str:
    if not risk_stages:
        return "관심"
    return risk_stages[-1] or "관심"


def _safe_action_text(last_small_action: Optional[SmallActionPlan]) -> str:
    if last_small_action is None:
        return ""

    action_text = getattr(last_small_action, "action_text", "")
    if not isinstance(action_text, str):
        return ""
    if len(action_text.strip()) > 160:
        return ""
    return action_text.strip()


def _safe_next_followup(next_followup: str) -> str:
    if not isinstance(next_followup, str):
        return ""
    if len(next_followup.strip()) > 160:
        return ""
    return next_followup.strip()


def _explicit_directives(values: Optional[List[str]]) -> List[str]:
    directives: List[str] = []
    for value in values or []:
        safe_value = _safe_label(value)
        if safe_value:
            _append_unique(directives, safe_value)
    return directives


def _memory_directives(memory_context: Any) -> List[str]:
    directives: List[str] = []
    if memory_context is None:
        return directives

    for directive in getattr(memory_context, "directives", []) or []:
        if getattr(directive, "active", True) is False:
            continue
        kind = _safe_label(getattr(directive, "kind", ""))
        term = _safe_label(getattr(directive, "term", ""))
        if kind and term:
            _append_unique(directives, f"{kind}:{term}")
    return directives


def build_session_dream_summary(
    session_id: str,
    memory_context: Any = None,
    intent_results: Optional[List[IntentAgentResult]] = None,
    emotion_results: Optional[List[EmotionAgentResult]] = None,
    risk_stages: Optional[List[str]] = None,
    last_small_action: Optional[SmallActionPlan] = None,
    next_followup: str = "",
    important_user_directives: Optional[List[str]] = None,
) -> SessionDreamSummary:
    """Build a structured session-level summary without raw conversation text."""
    main_issue = _build_main_issue(intent_results)
    emotional_trend = _build_emotional_trend(emotion_results)
    directives = _explicit_directives(important_user_directives)
    if not directives:
        directives = _memory_directives(memory_context)

    return SessionDreamSummary(
        session_id=session_id,
        summary_id=_summary_id(),
        main_issue=main_issue,
        emotional_trend=emotional_trend,
        risk_stage_start=_risk_start(risk_stages),
        risk_stage_end=_risk_end(risk_stages),
        last_small_action=_safe_action_text(last_small_action),
        next_follow_up=_safe_next_followup(next_followup),
        important_user_directives=directives,
        created_at=_created_at(),
        emotional_arc=[
            result.primary_label for result in emotion_results or []
            if isinstance(result.primary_label, EmotionLabel)
        ],
        recurring_themes=main_issue,
        memory_updates=directives,
        unresolved_needs=[],
        safety_notes=[
            stage for stage in (risk_stages or []) if stage == "위험"
        ],
        next_session_focus=main_issue[:2],
    )


@dataclass
class SessionDreamSummaryBuilder:
    """Small wrapper class for callers that prefer an agent object."""

    def build(
        self,
        session_id: str,
        memory_context: Any = None,
        intent_results: Optional[List[IntentAgentResult]] = None,
        emotion_results: Optional[List[EmotionAgentResult]] = None,
        risk_stages: Optional[List[str]] = None,
        last_small_action: Optional[SmallActionPlan] = None,
        next_followup: str = "",
        important_user_directives: Optional[List[str]] = None,
    ) -> SessionDreamSummary:
        return build_session_dream_summary(
            session_id=session_id,
            memory_context=memory_context,
            intent_results=intent_results,
            emotion_results=emotion_results,
            risk_stages=risk_stages,
            last_small_action=last_small_action,
            next_followup=next_followup,
            important_user_directives=important_user_directives,
        )
