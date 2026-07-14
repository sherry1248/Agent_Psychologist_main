"""Simple wellness recommender backed by the local wellness dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logging
from src.wellness.dataset_loader import NUMERIC_FIELDS, WellnessDatasetLoader

logger = setup_logging("wellness_recommender")

SAFE_SUPPORT_HINT = "지금은 숨을 고르고, 오늘 할 수 있는 가장 작은 한 가지를 선택해 보세요."


@dataclass
class WellnessRecommendation:
    support_hint: str
    risk_stage: str
    matched_record_id: str = ""
    matched_topic: str = ""
    distance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "support_hint": self.support_hint,
            "risk_stage": self.risk_stage,
            "matched_record_id": self.matched_record_id,
            "matched_topic": self.matched_topic,
            "distance": self.distance,
        }


class WellnessRecommender:
    """Pick the closest wellness record for a check-in dict."""

    def __init__(self, dataset_loader: Optional[WellnessDatasetLoader] = None):
        self.dataset_loader = dataset_loader or WellnessDatasetLoader()
        self._records: Optional[List[Dict[str, Any]]] = None

    def recommend(self, wellness_checkin: Optional[Dict[str, Any]]) -> WellnessRecommendation:
        if not isinstance(wellness_checkin, dict) or not wellness_checkin:
            return self._fallback_recommendation()

        records = self._load_records()
        if not records:
            return self._fallback_recommendation()

        normalized_checkin = self._normalize_checkin(wellness_checkin)
        if not normalized_checkin:
            return self._fallback_recommendation()

        best_record = None
        best_distance = float("inf")

        for record in records:
            distance = self._distance(normalized_checkin, record)
            if distance < best_distance:
                best_distance = distance
                best_record = record

        if not best_record:
            return self._fallback_recommendation()

        return WellnessRecommendation(
            support_hint=best_record.get("support_hint") or SAFE_SUPPORT_HINT,
            risk_stage=best_record.get("risk_stage") or "관심",
            matched_record_id=best_record.get("id", ""),
            matched_topic=best_record.get("topic", ""),
            distance=best_distance,
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            try:
                self._records = self.dataset_loader.load_records()
            except Exception as exc:
                logger.warning("Wellness dataset loading failed: %s", exc)
                self._records = []
        return self._records

    def _normalize_checkin(self, wellness_checkin: Dict[str, Any]) -> Dict[str, int]:
        normalized: Dict[str, int] = {}

        for field_name in NUMERIC_FIELDS:
            value = wellness_checkin.get(field_name)
            score = self._coerce_score(value)
            if score is not None:
                normalized[field_name] = score

        return normalized

    def _coerce_score(self, value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return self._bounded_score(int(round(float(value))))
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text.isdigit():
                return self._bounded_score(int(text))
            if any(marker in text for marker in ("좋", "괜찮", "양호", "충분", "잘", "good", "rested", "regular")):
                return 8
            if any(marker in text for marker in ("나쁘", "힘들", "부족", "안 좋", "못", "거르", "poor", "bad", "skip")):
                return 2
        return None

    def _bounded_score(self, value: int) -> int:
        return max(0, min(10, value))

    def _distance(self, checkin: Dict[str, int], record: Dict[str, Any]) -> float:
        total = 0.0
        compared = 0

        for field_name in NUMERIC_FIELDS:
            if field_name not in checkin:
                continue
            compared += 1
            total += abs(checkin[field_name] - int(record.get(field_name, 5)))

        if compared == 0:
            return float("inf")
        return total / compared

    def _fallback_recommendation(self) -> WellnessRecommendation:
        return WellnessRecommendation(
            support_hint=SAFE_SUPPORT_HINT,
            risk_stage="관심",
            matched_record_id="",
            matched_topic="",
            distance=float("inf"),
        )
