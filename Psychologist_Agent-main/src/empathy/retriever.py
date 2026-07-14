"""Empathy dataset retrieval helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.empathy.dataset_loader import EmpathyDatasetLoader
from src.utils.logging_config import setup_logging

logger = setup_logging("empathy_retriever")

DEFAULT_EMPATHY_STYLE_HINT = "감정을 먼저 확인하고, 차분하게 공감한 뒤 다음 한 걸음을 제안하세요."

EMOTION_KEYWORDS = {
    "기쁨": ("기쁘", "행복", "좋아", "즐겁", "설레", "신나"),
    "당황": ("당황", "민망", "어쩔", "황당", "놀랐"),
    "분노": ("화나", "짜증", "분노", "억울", "열받", "분해"),
    "불안": ("불안", "걱정", "초조", "긴장", "떨리", "두렵"),
    "상처": ("상처", "서운", "배신", "무시", "아프", "속상"),
    "슬픔": ("슬프", "우울", "힘들", "외로", "눈물", "허무"),
}


@dataclass
class EmpathyRecommendation:
    empathy_style_hint: str
    emotion_label: str = ""
    empathy_label: str = ""
    matched_record_id: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "empathy_style_hint": self.empathy_style_hint,
            "emotion_label": self.emotion_label,
            "empathy_label": self.empathy_label,
            "matched_record_id": self.matched_record_id,
            "score": self.score,
        }


class EmpathyRetriever:
    """Find a suitable empathy style hint for a user message."""

    def __init__(self, dataset_loader: Optional[EmpathyDatasetLoader] = None):
        self.dataset_loader = dataset_loader or EmpathyDatasetLoader()
        self._records: Optional[List[Dict[str, Any]]] = None

    def recommend(self, user_input: str) -> EmpathyRecommendation:
        if not isinstance(user_input, str) or not user_input.strip():
            return self._fallback_recommendation()

        records = self._load_records()
        if not records:
            return self._fallback_recommendation()

        best_record = None
        best_score = float("-inf")
        normalized_input = user_input.lower()

        for record in records:
            score = self._score(normalized_input, user_input, record)
            if score > best_score:
                best_score = score
                best_record = record

        if not best_record:
            return self._fallback_recommendation()

        return EmpathyRecommendation(
            empathy_style_hint=best_record.get("empathy_style_hint") or DEFAULT_EMPATHY_STYLE_HINT,
            emotion_label=best_record.get("emotion_label", ""),
            empathy_label=best_record.get("empathy_label", ""),
            matched_record_id=best_record.get("id", ""),
            score=best_score,
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            try:
                self._records = self.dataset_loader.load_records()
            except Exception as exc:
                logger.warning("Empathy dataset loading failed: %s", exc)
                self._records = []
        return self._records

    def _score(self, normalized_input: str, original_input: str, record: Dict[str, Any]) -> float:
        input_tokens = self._tokenize(normalized_input)
        record_tokens = self._tokenize(str(record.get("user_input", "")))
        overlap = len(input_tokens & record_tokens)

        emotion_label = str(record.get("emotion_label", ""))
        if emotion_label:
            keywords = EMOTION_KEYWORDS.get(emotion_label, ())
            if any(keyword in original_input for keyword in keywords):
                overlap += 3.0

        empathy_label = str(record.get("empathy_label", ""))
        if empathy_label in {"위로", "격려"} and any(marker in original_input for marker in ("힘들", "지쳤", "외로", "슬프", "불안")):
            overlap += 1.0

        return float(overlap)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[\w가-힣]+", text.lower()) if token}

    def _fallback_recommendation(self) -> EmpathyRecommendation:
        return EmpathyRecommendation(empathy_style_hint=DEFAULT_EMPATHY_STYLE_HINT)
