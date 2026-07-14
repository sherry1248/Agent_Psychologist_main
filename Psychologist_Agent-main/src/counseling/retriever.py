"""Counseling dataset retrieval helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.counseling.dataset_loader import CounselingDatasetLoader
from src.utils.logging_config import setup_logging

logger = setup_logging("counseling_retriever")

DEFAULT_INTERVENTION_HINT = "감정을 먼저 확인하고, 작은 실행 단계를 하나만 제안하세요."


@dataclass
class CounselingRecommendation:
    intervention_hint: str
    matched_record_id: str = ""
    category: str = "general"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_hint": self.intervention_hint,
            "matched_record_id": self.matched_record_id,
            "category": self.category,
            "score": self.score,
        }


class CounselingRetriever:
    """Find a relevant counseling hint for a user message."""

    def __init__(self, dataset_loader: Optional[CounselingDatasetLoader] = None):
        self.dataset_loader = dataset_loader or CounselingDatasetLoader()
        self._records: Optional[List[Dict[str, Any]]] = None

    def recommend(self, user_input: str) -> CounselingRecommendation:
        if not isinstance(user_input, str) or not user_input.strip():
            return self._fallback_recommendation()

        records = self._load_records()
        if not records:
            return self._fallback_recommendation()

        best_record = None
        best_score = float("-inf")

        for record in records:
            score = self._score(user_input, record)
            if score > best_score:
                best_score = score
                best_record = record

        if not best_record:
            return self._fallback_recommendation()

        return CounselingRecommendation(
            intervention_hint=best_record.get("intervention_hint") or DEFAULT_INTERVENTION_HINT,
            matched_record_id=best_record.get("id", ""),
            category=best_record.get("category", "general"),
            score=best_score,
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            try:
                self._records = self.dataset_loader.load_records()
            except Exception as exc:
                logger.warning("Counseling dataset loading failed: %s", exc)
                self._records = []
        return self._records

    def _score(self, user_input: str, record: Dict[str, Any]) -> float:
        input_tokens = self._tokenize(user_input)
        record_tokens = self._tokenize(
            " ".join(
                str(record.get(field, ""))
                for field in ("user_input", "counselor_response", "category")
            )
        )

        overlap = len(input_tokens & record_tokens)
        category = str(record.get("category", "")).lower()
        if category and category in user_input.lower():
            overlap += 1.5
        return float(overlap)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[\w가-힣]+", text.lower()) if token}

    def _fallback_recommendation(self) -> CounselingRecommendation:
        return CounselingRecommendation(intervention_hint=DEFAULT_INTERVENTION_HINT)
