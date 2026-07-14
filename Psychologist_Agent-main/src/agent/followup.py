"""
Rule-based deterministic follow-up question generator.

The generator returns at most one template-based question. It does not store or
return user raw text.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.agent.models import (
    DecisionAction,
    DecisionAgentResult,
    EmotionalStateVector,
    IntentAgentResult,
    IntentCandidate,
    IntentLabel,
)


GENERAL_FOLLOWUP_QUESTION = "지금 이야기에서 가장 먼저 다뤄보고 싶은 부분은 무엇인가요?"

FOLLOWUP_TEMPLATES = {
    IntentLabel.SLEEP_PROBLEM: "잠드는 데 오래 걸리는 편인가요, 아니면 자다가 자주 깨는 편인가요?",
    IntentLabel.ANXIETY_SUPPORT: "불안이 가장 크게 올라오는 순간은 혼자 있을 때인가요, 아니면 해야 할 일을 마주할 때인가요?",
    IntentLabel.LOW_MOOD_SUPPORT: "기운이 떨어진 느낌이 몸의 피곤함에 가까운가요, 아니면 마음의 무거움에 가까운가요?",
    IntentLabel.STRESS_SUPPORT: "최근 부담은 해야 할 일이 많은 쪽에 가까운가요, 아니면 어디서부터 시작할지 막막한 쪽에 가까운가요?",
    IntentLabel.WORK_OR_STUDY_STRESS: "지금 제일 부담되는 건 시작하기 어려움인가요, 끝내야 한다는 압박인가요?",
    IntentLabel.RELATIONSHIP_STRESS: "그 관계에서 힘든 지점은 말이 통하지 않는 느낌인가요, 아니면 혼자 감당하는 느낌인가요?",
    IntentLabel.FAMILY_CONFLICT: "가족 문제에서 가장 힘든 건 반복되는 말다툼인가요, 아니면 내 마음을 이해받지 못하는 느낌인가요?",
    IntentLabel.LOW_SELF_ESTEEM: "그 생각이 가장 강하게 드는 순간은 실수했을 때인가요, 아니면 다른 사람과 비교할 때인가요?",
    IntentLabel.NEED_EMPATHY: "지금은 해결책보다 먼저 그냥 들어주는 쪽이 더 필요할까요?",
    IntentLabel.NEED_ADVICE: "지금 필요한 건 바로 해볼 수 있는 방법인가요, 아니면 상황을 같이 정리하는 것인가요?",
}


def _blocked_by_safety(
    risk_stage: str,
    intent_result: Optional[IntentAgentResult],
    decision_result: Optional[DecisionAgentResult],
) -> bool:
    return (
        (risk_stage or "").strip() == "위험"
        or bool(
            decision_result
            and decision_result.primary_action == DecisionAction.ESCALATE_SAFETY
        )
        or bool(intent_result and intent_result.s3_sos)
    )


def _matches_avoid_topics(question: str, avoid_topics: Optional[List[str]]) -> bool:
    if not question or not avoid_topics:
        return False

    return any(topic and topic in question for topic in avoid_topics)


def _best_candidate(candidates: List[IntentCandidate]) -> Optional[IntentCandidate]:
    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate.confidence)


def _select_intent(intent_result: Optional[IntentAgentResult]) -> Optional[IntentLabel]:
    if intent_result is None:
        return None

    if intent_result.primary_intent and intent_result.primary_intent != IntentLabel.OTHER_CONCERN:
        return intent_result.primary_intent

    candidate = _best_candidate(intent_result.candidates)
    if candidate and candidate.label != IntentLabel.OTHER_CONCERN:
        return candidate.label

    return None


def _fallback_question(avoid_topics: Optional[List[str]]) -> str:
    if _matches_avoid_topics(GENERAL_FOLLOWUP_QUESTION, avoid_topics):
        return ""
    return GENERAL_FOLLOWUP_QUESTION


def generate_followup_question(
    intent_result: Optional[IntentAgentResult] = None,
    decision_result: Optional[DecisionAgentResult] = None,
    emotional_state: Optional[EmotionalStateVector] = None,
    risk_stage: str = "관심",
    previous_followup: str = "",
    avoid_topics: Optional[List[str]] = None,
    prefer_previous: bool = True,
) -> str:
    """Generate at most one deterministic follow-up question."""
    del emotional_state

    if _blocked_by_safety(risk_stage, intent_result, decision_result):
        return ""

    previous = (previous_followup or "").strip()
    if prefer_previous and previous and not _matches_avoid_topics(previous, avoid_topics):
        return previous

    selected_intent = _select_intent(intent_result)
    question = FOLLOWUP_TEMPLATES.get(selected_intent) if selected_intent else None
    if not question:
        return _fallback_question(avoid_topics)

    if _matches_avoid_topics(question, avoid_topics):
        return _fallback_question(avoid_topics)

    if previous and previous == question:
        fallback = _fallback_question(avoid_topics)
        return "" if fallback == previous else fallback

    return question


@dataclass
class FollowUpQuestionGenerator:
    """Small wrapper class for callers that prefer an agent object."""

    def generate(
        self,
        intent_result: Optional[IntentAgentResult] = None,
        decision_result: Optional[DecisionAgentResult] = None,
        emotional_state: Optional[EmotionalStateVector] = None,
        risk_stage: str = "관심",
        previous_followup: str = "",
        avoid_topics: Optional[List[str]] = None,
        prefer_previous: bool = True,
    ) -> str:
        return generate_followup_question(
            intent_result=intent_result,
            decision_result=decision_result,
            emotional_state=emotional_state,
            risk_stage=risk_stage,
            previous_followup=previous_followup,
            avoid_topics=avoid_topics,
            prefer_previous=prefer_previous,
        )
