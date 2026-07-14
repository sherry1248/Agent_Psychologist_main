"""
Rule-based emotional state updater for the agent pipeline.

The state agent consumes structured intent, emotion, risk, and check-in
signals. It does not accept or store raw conversation text.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.agent.models import (
    EmotionLabel,
    EmotionalStateVector,
    IntentAgentResult,
    IntentLabel,
)


SMOOTHING_OLD_WEIGHT = 0.85
SMOOTHING_OBSERVED_WEIGHT = 0.15


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _blend(old_value: float, observed_value: float) -> float:
    return _clamp01(
        old_value * SMOOTHING_OLD_WEIGHT
        + observed_value * SMOOTHING_OBSERVED_WEIGHT
    )


def normalize_checkin_score(value: Any) -> Optional[float]:
    """
    Normalize a 1-5 or 1-10 check-in score into 0.0-1.0.

    Values above 5 are treated as 1-10 scale; values up to 5 are treated as
    1-5 scale. Already normalized 0.0-1.0 values are accepted.
    """
    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score == 0.0:
        return 0.0
    if 0.0 < score < 1.0:
        return _clamp01(score)
    if score > 5.0:
        return _clamp01((score - 1.0) / 9.0)
    return _clamp01((score - 1.0) / 4.0)


def _candidate_labels(intent_result: Optional[IntentAgentResult]) -> List[IntentLabel]:
    if intent_result is None:
        return []
    labels = [candidate.label for candidate in intent_result.candidates]
    if intent_result.primary_intent not in labels:
        labels.insert(0, intent_result.primary_intent)
    return labels


def _apply_intent_observations(
    observations: Dict[str, float],
    intent_result: Optional[IntentAgentResult],
) -> None:
    labels = set(_candidate_labels(intent_result))

    if IntentLabel.SLEEP_PROBLEM in labels:
        observations["sleep"] = min(observations["sleep"], 0.2)
    if IntentLabel.ANXIETY_SUPPORT in labels:
        observations["anxiety"] = max(observations["anxiety"], 0.8)
    if labels.intersection({IntentLabel.STRESS_SUPPORT, IntentLabel.WORK_OR_STUDY_STRESS}):
        observations["stress"] = max(observations["stress"], 0.8)
        observations["energy"] = min(observations["energy"], 0.2)
    if IntentLabel.LOW_MOOD_SUPPORT in labels:
        observations["mood"] = min(observations["mood"], 0.2)
        observations["energy"] = min(observations["energy"], 0.3)
    if IntentLabel.CRISIS_SIGNAL in labels:
        observations["safety"] = min(observations["safety"], 0.0)


def _apply_emotion_observations(
    observations: Dict[str, float],
    emotion_labels: Optional[Iterable[EmotionLabel]],
) -> None:
    if not emotion_labels:
        return

    labels = set(emotion_labels)

    if EmotionLabel.ANXIETY in labels:
        observations["anxiety"] = max(observations["anxiety"], 0.8)
    if labels.intersection({EmotionLabel.SADNESS, EmotionLabel.HOPELESSNESS}):
        observations["mood"] = min(observations["mood"], 0.2)
    if EmotionLabel.STRESS in labels:
        observations["stress"] = max(observations["stress"], 0.8)
        observations["energy"] = min(observations["energy"], 0.3)
    if EmotionLabel.FATIGUE in labels:
        observations["energy"] = min(observations["energy"], 0.2)
    if EmotionLabel.LONELINESS in labels:
        observations["mood"] = min(observations["mood"], 0.3)
        observations["rapport"] = max(observations["rapport"], 0.25)


def _apply_risk_observation(observations: Dict[str, float], risk_stage: str) -> None:
    stage = (risk_stage or "관심").strip()
    if stage == "위험":
        observations["safety"] = min(observations["safety"], 0.1)
    elif stage == "주의":
        observations["safety"] = min(observations["safety"], 0.6)
    else:
        observations["safety"] = min(observations["safety"], 0.9)


def _apply_wellness_observations(
    observations: Dict[str, float],
    wellness_checkin: Optional[Dict[str, Any]],
) -> None:
    if not wellness_checkin:
        return

    mood = normalize_checkin_score(wellness_checkin.get("mood_score"))
    anxiety = normalize_checkin_score(wellness_checkin.get("anxiety_score"))
    loneliness = normalize_checkin_score(wellness_checkin.get("loneliness_score"))
    sleep = normalize_checkin_score(wellness_checkin.get("sleep_quality"))
    energy = normalize_checkin_score(wellness_checkin.get("energy_score"))
    stress = normalize_checkin_score(wellness_checkin.get("stress_score"))

    if mood is not None:
        observations["mood"] = mood
    if anxiety is not None:
        observations["anxiety"] = anxiety
    if loneliness is not None:
        observations["mood"] = min(observations["mood"], 1.0 - loneliness)
    if sleep is not None:
        observations["sleep"] = sleep
    if energy is not None:
        observations["energy"] = energy
    if stress is not None:
        observations["stress"] = stress

    meal_status = wellness_checkin.get("meal_status")
    if isinstance(meal_status, str) and meal_status.strip().lower() in {
        "skipped",
        "poor",
        "none",
        "no",
        "missed",
    }:
        observations["energy"] = min(observations["energy"], 0.35)


def update_emotional_state(
    previous_state: Optional[EmotionalStateVector] = None,
    intent_result: Optional[IntentAgentResult] = None,
    emotion_labels: Optional[List[EmotionLabel]] = None,
    risk_stage: str = "관심",
    wellness_checkin: Optional[Dict[str, Any]] = None,
) -> EmotionalStateVector:
    """Update an emotional state vector from structured observations."""
    old_state = previous_state or EmotionalStateVector()
    old_values = old_state.to_dict()
    observations = dict(old_values)

    _apply_intent_observations(observations, intent_result)
    _apply_emotion_observations(observations, emotion_labels)
    _apply_risk_observation(observations, risk_stage)
    _apply_wellness_observations(observations, wellness_checkin)

    return EmotionalStateVector(
        mood=_blend(old_state.mood, observations["mood"]),
        anxiety=_blend(old_state.anxiety, observations["anxiety"]),
        stress=_blend(old_state.stress, observations["stress"]),
        sleep=_blend(old_state.sleep, observations["sleep"]),
        energy=_blend(old_state.energy, observations["energy"]),
        safety=_blend(old_state.safety, observations["safety"]),
        rapport=_blend(old_state.rapport, observations["rapport"]),
    )


def summarize_emotional_state(state: EmotionalStateVector) -> List[str]:
    """Return compact prompt/decision labels for notable state dimensions."""
    labels: List[str] = []

    if state.sleep < 0.5:
        labels.append("low sleep")
    if state.anxiety >= 0.35:
        labels.append("elevated anxiety")
    if state.stress >= 0.35:
        labels.append("high stress")
    if state.energy < 0.5:
        labels.append("low energy")
    if state.safety < 0.8:
        labels.append("safety concern")
    if state.mood < 0.45:
        labels.append("low mood")

    return labels


@dataclass
class EmotionalStateAgent:
    """Small wrapper class for callers that prefer an agent object."""

    def update(
        self,
        previous_state: Optional[EmotionalStateVector] = None,
        intent_result: Optional[IntentAgentResult] = None,
        emotion_labels: Optional[List[EmotionLabel]] = None,
        risk_stage: str = "관심",
        wellness_checkin: Optional[Dict[str, Any]] = None,
    ) -> EmotionalStateVector:
        return update_emotional_state(
            previous_state=previous_state,
            intent_result=intent_result,
            emotion_labels=emotion_labels,
            risk_stage=risk_stage,
            wellness_checkin=wellness_checkin,
        )
