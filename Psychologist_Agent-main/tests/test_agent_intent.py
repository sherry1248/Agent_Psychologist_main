"""
Tests for the deterministic rule-based intent agent.
"""

from dataclasses import fields

from src.agent.intent import IntentAgent, classify_intent
from src.agent.models import IntentAgentResult, IntentCandidate, IntentLabel, RAW_TEXT_FIELD_NAMES


def _labels(result):
    return {candidate.label for candidate in result.candidates}


def test_sleep_problem_is_not_overclassified_as_s2_or_s3():
    result = classify_intent("요즘 잠을 못 자요")

    assert result.primary_intent == IntentLabel.SLEEP_PROBLEM
    assert result.s2_suspected is False
    assert result.s3_sos is False


def test_sleep_and_anxiety_are_both_included():
    result = classify_intent("요즘 잠을 못 자고 불안해요")

    assert result.primary_intent == IntentLabel.SLEEP_PROBLEM
    assert IntentLabel.SLEEP_PROBLEM in _labels(result)
    assert IntentLabel.ANXIETY_SUPPORT in _labels(result)
    assert result.s2_suspected is False
    assert result.s3_sos is False


def test_empathy_request_is_detected():
    result = IntentAgent().classify("그냥 들어줬으면 좋겠어요")

    assert result.primary_intent == IntentLabel.NEED_EMPATHY
    assert result.chat_label_hint["negative"] is False


def test_advice_request_sets_knowledge_hint():
    result = classify_intent("어떻게 해야 할지 모르겠어요")

    assert result.primary_intent == IntentLabel.NEED_ADVICE
    assert result.chat_label_hint["knowledge"] is True
    assert result.chat_label_hint["negative"] is False


def test_duration_and_impairment_raise_s2_suspected_condition():
    result = classify_intent("몇 달째 불안해서 출근을 못 해요")

    assert IntentLabel.ANXIETY_SUPPORT in _labels(result)
    assert result.s2_suspected is True
    assert result.s3_sos is False


def test_crisis_signal_raises_s3_sos():
    for text in ("죽고 싶어요", "자해하고 싶어요"):
        result = classify_intent(text)

        assert IntentLabel.CRISIS_SIGNAL in _labels(result)
        assert result.primary_intent == IntentLabel.CRISIS_SIGNAL
        assert result.s3_sos is True


def test_intent_result_and_candidate_have_no_raw_text_fields():
    for schema_class in (IntentAgentResult, IntentCandidate):
        names = {item.name for item in fields(schema_class)}
        assert names.isdisjoint(RAW_TEXT_FIELD_NAMES)


def test_evidence_uses_rule_labels_not_full_user_text():
    user_text = "요즘 잠을 못 자고 불안해요"
    result = classify_intent(user_text)

    evidence = [
        evidence_label
        for candidate in result.candidates
        for evidence_label in candidate.evidence
    ]

    assert evidence
    assert user_text not in evidence
    assert "sleep_keyword" in evidence
    assert "anxiety_keyword" in evidence

