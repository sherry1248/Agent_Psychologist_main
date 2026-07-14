"""
Rule-based deterministic intent classifier for the agent pipeline.

The classifier stores only intent labels, severity flags, and rule evidence
labels. It never copies raw user text into result schemas.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from src.agent.models import (
    IntentAgentResult,
    IntentCandidate,
    IntentLabel,
    IntentSeverity,
)


Rule = Tuple[IntentLabel, str, Sequence[str]]


INTENT_RULES: Sequence[Rule] = (
    (
        IntentLabel.CRISIS_SIGNAL,
        "crisis_keyword",
        ("죽고 싶", "죽고싶", "자살", "자해", "해치고 싶", "죽여", "끝내고 싶"),
    ),
    (IntentLabel.SLEEP_PROBLEM, "sleep_keyword", ("잠", "수면", "불면", "못 자", "못자", "잠이 안")),
    (IntentLabel.ANXIETY_SUPPORT, "anxiety_keyword", ("불안", "걱정", "공황", "초조", "두려")),
    (IntentLabel.LOW_MOOD_SUPPORT, "low_mood_keyword", ("우울", "무기력", "슬프", "외롭", "허무")),
    (IntentLabel.STRESS_SUPPORT, "stress_keyword", ("스트레스", "압박", "버거", "힘들", "지쳐")),
    (IntentLabel.RELATIONSHIP_STRESS, "relationship_keyword", ("연애", "친구", "인간관계", "관계", "헤어졌", "헤어진")),
    (IntentLabel.WORK_OR_STUDY_STRESS, "work_study_keyword", ("회사", "직장", "출근", "업무", "공부", "학교", "시험", "과제")),
    (IntentLabel.FAMILY_CONFLICT, "family_keyword", ("가족", "부모", "엄마", "아빠", "형제", "자매")),
    (IntentLabel.LOW_SELF_ESTEEM, "self_esteem_keyword", ("자존감", "내가 싫", "쓸모없", "못난", "가치 없")),
    (IntentLabel.NEED_EMPATHY, "empathy_keyword", ("들어줬으면", "들어줘", "공감", "위로", "그냥 들어")),
    (IntentLabel.NEED_ADVICE, "advice_keyword", ("어떻게 해야", "어떡해", "방법", "조언", "알려줘", "설명")),
    (IntentLabel.SUBSTANCE_OR_ADDICTION, "addiction_keyword", ("술", "알코올", "마약", "약물", "도박", "중독")),
    (IntentLabel.GRIEF_SUPPORT, "grief_keyword", ("상실", "사별", "돌아가셨", "세상을 떠", "장례", "애도")),
)

S2_DURATION_KEYWORDS = (
    "몇 달",
    "몇달",
    "몇 주",
    "몇주",
    "계속",
    "반복",
    "매일",
    "오랫동안",
    "장기간",
)

S2_IMPAIRMENT_KEYWORDS = (
    "못 해",
    "못해",
    "못 가",
    "못가",
    "출근을 못",
    "학교를 못",
    "일을 못",
    "생활이 안",
    "기능",
    "무너",
)

CONCERN_LABELS = {
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


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _append_candidate(
    candidates: List[IntentCandidate],
    seen: Dict[IntentLabel, IntentCandidate],
    label: IntentLabel,
    evidence_label: str,
    severity: IntentSeverity,
) -> None:
    existing = seen.get(label)
    if existing:
        if evidence_label not in existing.evidence:
            existing.evidence.append(evidence_label)
        return

    candidate = IntentCandidate(
        label=label,
        severity=severity,
        confidence=0.95 if label == IntentLabel.CRISIS_SIGNAL else 0.8,
        rationale_tags=[evidence_label],
        evidence=[evidence_label],
    )
    candidates.append(candidate)
    seen[label] = candidate


def _detect_s2(text: str, labels: Iterable[IntentLabel]) -> bool:
    has_clinical_signal = any(
        label
        in {
            IntentLabel.ANXIETY_SUPPORT,
            IntentLabel.LOW_MOOD_SUPPORT,
            IntentLabel.SLEEP_PROBLEM,
            IntentLabel.SUBSTANCE_OR_ADDICTION,
        }
        for label in labels
    )
    return (
        has_clinical_signal
        and _contains_any(text, S2_DURATION_KEYWORDS)
        and _contains_any(text, S2_IMPAIRMENT_KEYWORDS)
    )


def _chat_label_hint(labels: Iterable[IntentLabel]) -> Dict[str, bool]:
    label_set = set(labels)
    knowledge = IntentLabel.NEED_ADVICE in label_set
    question = bool(label_set.intersection(CONCERN_LABELS)) and not knowledge
    return {
        "question": question,
        "knowledge": knowledge,
        "negative": False,
    }


def classify_intent(user_text: str) -> IntentAgentResult:
    """Classify user text into deterministic agent intent labels."""
    normalized = (user_text or "").strip().lower()
    candidates: List[IntentCandidate] = []
    seen: Dict[IntentLabel, IntentCandidate] = {}

    for label, evidence_label, keywords in INTENT_RULES:
        if _contains_any(normalized, keywords):
            _append_candidate(
                candidates=candidates,
                seen=seen,
                label=label,
                evidence_label=evidence_label,
                severity=IntentSeverity.S1_CONCERN,
            )

    if not candidates:
        _append_candidate(
            candidates=candidates,
            seen=seen,
            label=IntentLabel.OTHER_CONCERN,
            evidence_label="other_concern_fallback",
            severity=IntentSeverity.S1_CONCERN,
        )

    labels = [candidate.label for candidate in candidates]
    s3_sos = IntentLabel.CRISIS_SIGNAL in labels
    s2_suspected = False if s3_sos else _detect_s2(normalized, labels)
    severity = IntentSeverity.S1_CONCERN
    if s3_sos:
        severity = IntentSeverity.S3_SOS
    elif s2_suspected:
        severity = IntentSeverity.S2_SUSPECTED_CONDITION

    for candidate in candidates:
        candidate.severity = severity if candidate.label == IntentLabel.CRISIS_SIGNAL else IntentSeverity.S1_CONCERN
        if s2_suspected and candidate.label in {
            IntentLabel.ANXIETY_SUPPORT,
            IntentLabel.LOW_MOOD_SUPPORT,
            IntentLabel.SLEEP_PROBLEM,
            IntentLabel.SUBSTANCE_OR_ADDICTION,
        }:
            candidate.severity = IntentSeverity.S2_SUSPECTED_CONDITION
            if "duration_and_impairment_signal" not in candidate.evidence:
                candidate.evidence.append("duration_and_impairment_signal")
                candidate.rationale_tags.append("duration_and_impairment_signal")

    return IntentAgentResult(
        primary_intent=candidates[0].label,
        severity=severity,
        candidates=candidates,
        needs_follow_up=_chat_label_hint(labels)["question"],
        confidence=max(candidate.confidence for candidate in candidates),
        s2_suspected=s2_suspected,
        s3_sos=s3_sos,
        chat_label_hint=_chat_label_hint(labels),
    )


@dataclass
class IntentAgent:
    """Small wrapper class for callers that prefer an agent object."""

    def classify(self, user_text: str) -> IntentAgentResult:
        return classify_intent(user_text)
