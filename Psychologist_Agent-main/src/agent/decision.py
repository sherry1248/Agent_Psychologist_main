"""
Rule-based deterministic decision agent for turn-level action selection.

The agent consumes structured safety, intent, emotion, dataset, and memory
signals only. It stores rule labels and constraints, never raw conversation text.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set

from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    EmotionAgentResult,
    EmotionLabel,
    EmotionalStateVector,
    IntentAgentResult,
    IntentLabel,
    IntentSeverity,
    SafetyAgentResult,
)


CLARIFICATION_INTENTS = {
    IntentLabel.SLEEP_PROBLEM,
    IntentLabel.ANXIETY_SUPPORT,
    IntentLabel.RELATIONSHIP_STRESS,
    IntentLabel.WORK_OR_STUDY_STRESS,
}

CONCERN_INTENTS = {
    IntentLabel.SLEEP_PROBLEM,
    IntentLabel.ANXIETY_SUPPORT,
    IntentLabel.LOW_MOOD_SUPPORT,
    IntentLabel.STRESS_SUPPORT,
    IntentLabel.RELATIONSHIP_STRESS,
    IntentLabel.WORK_OR_STUDY_STRESS,
    IntentLabel.FAMILY_CONFLICT,
    IntentLabel.LOW_SELF_ESTEEM,
    IntentLabel.SUBSTANCE_OR_ADDICTION,
    IntentLabel.GRIEF_SUPPORT,
    IntentLabel.OTHER_CONCERN,
}


def _append_unique(items: List[DecisionAction], action: DecisionAction) -> None:
    if action not in items:
        items.append(action)


def _intent_labels(intent_result: Optional[IntentAgentResult]) -> Set[IntentLabel]:
    if intent_result is None:
        return set()

    labels = {candidate.label for candidate in intent_result.candidates}
    labels.add(intent_result.primary_intent)
    return labels


def _has_s1_clarification_intent(intent_result: Optional[IntentAgentResult]) -> bool:
    if intent_result is None:
        return False

    if intent_result.s2_suspected or intent_result.s3_sos:
        return False

    if intent_result.severity not in {IntentSeverity.S1_CONCERN, IntentSeverity.LOW}:
        return False

    return bool(_intent_labels(intent_result).intersection(CLARIFICATION_INTENTS))


def _effective_state(
    emotional_state: Optional[EmotionalStateVector],
    emotion_result: Optional[EmotionAgentResult],
) -> EmotionalStateVector:
    if emotional_state is not None:
        return emotional_state
    if emotion_result is not None:
        return emotion_result.state_vector
    return EmotionalStateVector()


def _notable_state_labels(state: EmotionalStateVector) -> List[str]:
    labels: List[str] = []

    if state.sleep < 0.4:
        labels.append("low_sleep_state")
    if state.anxiety >= 0.65:
        labels.append("elevated_anxiety_state")
    if state.stress >= 0.65:
        labels.append("elevated_stress_state")
    if state.energy < 0.35:
        labels.append("low_energy_state")
    if state.mood < 0.35:
        labels.append("low_mood_state")
    if state.safety < 0.7:
        labels.append("reduced_safety_state")

    return labels


def _emotion_labels(emotion_result: Optional[EmotionAgentResult]) -> Set[EmotionLabel]:
    if emotion_result is None:
        return set()

    labels = set(emotion_result.secondary_labels)
    labels.add(emotion_result.primary_label)
    return labels


def _has_actionable_state(
    state: EmotionalStateVector,
    emotion_result: Optional[EmotionAgentResult],
) -> bool:
    labels = _emotion_labels(emotion_result)
    return (
        state.sleep < 0.4
        or state.anxiety >= 0.65
        or state.stress >= 0.65
        or state.energy < 0.35
        or bool(labels.intersection({EmotionLabel.ANXIETY, EmotionLabel.STRESS, EmotionLabel.FATIGUE}))
    )


def _has_memory_signal(
    labels: Iterable[IntentLabel],
    memory_context: Any,
    proactive_recall: Any,
    has_followup: bool,
    has_small_action: bool,
    intent_result: Optional[IntentAgentResult],
) -> bool:
    label_set = set(labels)

    if memory_context is not None or proactive_recall is not None:
        return True
    if has_followup or has_small_action:
        return True
    if label_set.intersection({IntentLabel.MEMORY_UPDATE, IntentLabel.SMALL_ACTION, IntentLabel.CLARIFICATION}):
        return True
    return bool(intent_result and intent_result.s2_suspected)


def _tone_for_turn(
    crisis: bool,
    state: EmotionalStateVector,
    emotion_result: Optional[EmotionAgentResult],
) -> str:
    if crisis:
        return "safety_first"

    labels = _emotion_labels(emotion_result)
    if (
        state.anxiety >= 0.65
        or state.stress >= 0.65
        or labels.intersection({EmotionLabel.ANXIETY, EmotionLabel.STRESS})
    ):
        return "calm"

    return "empathic"


def decide_action(
    risk_stage: str = "관심",
    requires_crisis_response: bool = False,
    intent_result: Optional[IntentAgentResult] = None,
    emotion_result: Optional[EmotionAgentResult] = None,
    emotional_state: Optional[EmotionalStateVector] = None,
    counseling_hint: str = "",
    empathy_style_hint: str = "",
    wellness_hint: str = "",
    memory_context: Any = None,
    proactive_recall: Any = None,
) -> DecisionAgentResult:
    """Choose this turn's deterministic action from structured signals."""
    state = _effective_state(emotional_state, emotion_result)
    labels = _intent_labels(intent_result)
    secondary_actions: List[DecisionAction] = []
    reason_codes: List[str] = []

    crisis = (
        (risk_stage or "").strip() == "위험"
        or requires_crisis_response
        or bool(intent_result and intent_result.s3_sos)
    )

    if crisis:
        primary_action = DecisionAction.ESCALATE_SAFETY
        reason_codes.append("crisis_priority")
    elif _has_s1_clarification_intent(intent_result):
        primary_action = DecisionAction.ASK_FOLLOW_UP
        _append_unique(secondary_actions, DecisionAction.RESPOND_SUPPORTIVELY)
        if IntentLabel.SLEEP_PROBLEM in labels:
            reason_codes.append("sleep_problem_needs_clarification")
        elif IntentLabel.ANXIETY_SUPPORT in labels:
            reason_codes.append("anxiety_support_needs_clarification")
        elif IntentLabel.RELATIONSHIP_STRESS in labels:
            reason_codes.append("relationship_stress_needs_clarification")
        else:
            reason_codes.append("work_or_study_stress_needs_clarification")
    else:
        primary_action = DecisionAction.RESPOND_SUPPORTIVELY
        reason_codes.append("supportive_default")

    actionable_state = _has_actionable_state(state, emotion_result)
    has_wellness_hint = bool((wellness_hint or "").strip())
    if not crisis and (has_wellness_hint or actionable_state):
        _append_unique(secondary_actions, DecisionAction.SUGGEST_SMALL_ACTION)
        state_reason = (
            _notable_state_labels(state)[0]
            if _notable_state_labels(state)
            else "actionable_emotion_state"
        )
        reason_codes.append(
            "actionable_wellness_hint" if has_wellness_hint else state_reason
        )

    notable_state_labels = _notable_state_labels(state)
    for label in notable_state_labels:
        if label not in reason_codes:
            reason_codes.append(label)

    concern_count = len(labels.intersection(CONCERN_INTENTS))
    if not crisis and (concern_count >= 2 or len(notable_state_labels) >= 2):
        _append_unique(secondary_actions, DecisionAction.SUMMARIZE_STATE)
        reason_codes.append("mixed_or_accumulated_state")

    has_followup = primary_action == DecisionAction.ASK_FOLLOW_UP
    has_small_action = DecisionAction.SUGGEST_SMALL_ACTION in secondary_actions
    if _has_memory_signal(
        labels=labels,
        memory_context=memory_context,
        proactive_recall=proactive_recall,
        has_followup=has_followup,
        has_small_action=has_small_action,
        intent_result=intent_result,
    ):
        _append_unique(secondary_actions, DecisionAction.UPDATE_MEMORY)
        reason_codes.append("update_memory_candidate")

    if crisis:
        secondary_actions = [
            action
            for action in secondary_actions
            if action
            not in {
                DecisionAction.ASK_FOLLOW_UP,
                DecisionAction.SUGGEST_SMALL_ACTION,
            }
        ]

    response_constraints = {
        "must_include_followup": primary_action == DecisionAction.ASK_FOLLOW_UP,
        "must_include_small_action": DecisionAction.SUGGEST_SMALL_ACTION in secondary_actions,
        "max_questions": 1,
        "avoid_topics": ["non_safety_followup", "small_action"] if crisis else [],
        "tone": _tone_for_turn(crisis, state, emotion_result),
    }

    safety_escalation = None
    if crisis:
        safety_escalation = SafetyAgentResult(
            is_safe=False,
            risk_stage="위험" if (risk_stage or "").strip() == "위험" else risk_stage,
            requires_escalation=True,
            safety_topics=["crisis_priority"],
            confidence=1.0,
        )

    return DecisionAgentResult(
        action=primary_action,
        primary_action=primary_action,
        secondary_actions=secondary_actions,
        response_constraints=response_constraints,
        reason_codes=reason_codes,
        confidence=1.0,
        rationale_tags=list(reason_codes),
        state_vector=state,
        safety_escalation=safety_escalation,
    )


@dataclass
class DecisionAgent:
    """Small wrapper class for callers that prefer an agent object."""

    def decide(
        self,
        risk_stage: str = "관심",
        requires_crisis_response: bool = False,
        intent_result: Optional[IntentAgentResult] = None,
        emotion_result: Optional[EmotionAgentResult] = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        counseling_hint: str = "",
        empathy_style_hint: str = "",
        wellness_hint: str = "",
        memory_context: Any = None,
        proactive_recall: Any = None,
    ) -> DecisionAgentResult:
        return decide_action(
            risk_stage=risk_stage,
            requires_crisis_response=requires_crisis_response,
            intent_result=intent_result,
            emotion_result=emotion_result,
            emotional_state=emotional_state,
            counseling_hint=counseling_hint,
            empathy_style_hint=empathy_style_hint,
            wellness_hint=wellness_hint,
            memory_context=memory_context,
            proactive_recall=proactive_recall,
        )
