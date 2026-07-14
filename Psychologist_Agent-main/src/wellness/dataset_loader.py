"""Utilities for loading the project's wellness dataset.

The loader reuses the existing preprocessing aliases so the same raw or
processed files can be used without changing the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("wellness_dataset_loader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "wellness_processed.jsonl",
    PROJECT_ROOT / "data" / "raw" / "wellness_sample.jsonl",
]

NUMERIC_FIELDS = (
    "mood_score",
    "anxiety_score",
    "loneliness_score",
    "sleep_quality",
    "meal_status",
    "energy_score",
    "stress_score",
)


WellnessRecord = Dict[str, Any]


class WellnessDatasetLoader:
    """Load and normalize wellness records from local raw or processed files."""

    def __init__(self, dataset_path: Optional[str | Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        # Delay importing heavy preprocessing utilities until actually needed
        self._preparation = None

    def load_records(self) -> List[WellnessRecord]:
        """Load wellness records from the configured dataset path or defaults."""

        for candidate in self._resolve_candidates():
            try:
                raw_records = self._load_candidate(candidate)
                normalized = [
                    self._normalize_record(record, index)
                    for index, record in enumerate(raw_records)
                ]
                normalized = [record for record in normalized if record["question"] and record["answer"]]

                if normalized:
                    logger.info("Loaded %s wellness records from %s", len(normalized), candidate)
                    return normalized
            except Exception as exc:
                logger.warning("Failed to load wellness dataset from %s: %s", candidate, exc)

        return []

    def _resolve_candidates(self) -> List[Path]:
        if self.dataset_path is not None:
            return [self.dataset_path]
        return DEFAULT_DATASET_PATHS

    def _load_candidate(self, candidate: Path) -> List[Dict[str, Any]]:
        if not candidate.exists():
            raise FileNotFoundError(candidate)

        if candidate.is_dir():
            # Import DataPreparation lazily to avoid pulling in heavy deps at module import time
            from scripts.data_preparation import DataPreparation

            preparation = DataPreparation(
                raw_dir=str(candidate),
                output_dir=str(PROJECT_ROOT / "data" / "processed"),
                min_question_len=1,
                min_answer_len=1,
            )
            return preparation.load_raw_records()

        suffix = candidate.suffix.lower()
        if suffix in {".json", ".jsonl", ".csv"}:
            # Ensure a preparation helper exists (created lazily)
            if self._preparation is None:
                from scripts.data_preparation import DataPreparation

                self._preparation = DataPreparation(
                    raw_dir=str(PROJECT_ROOT / "data" / "raw"),
                    output_dir=str(PROJECT_ROOT / "data" / "processed"),
                    min_question_len=1,
                    min_answer_len=1,
                )

            return self._preparation._load_single_raw_file(candidate)

        raise ValueError(f"Unsupported wellness dataset format: {candidate}")

    def _normalize_record(self, record: Dict[str, Any], index: int) -> WellnessRecord:
        question = self._first_text(record, ("question", "questionText", "prompt", "input"))
        answer = self._first_text(record, ("answer", "answerText", "response", "output"))
        topic = self._first_text(record, ("topic", "category", "label")) or "general"
        risk_stage = self._first_text(record, ("risk_stage", "wellness_stage", "wellnessStage")) or self._infer_risk_stage(topic)
        support_hint = self._normalize_support_hint(record, answer)

        derived_scores = self._derive_scores(topic, risk_stage)

        normalized: WellnessRecord = {
            "id": self._first_text(record, ("id",)) or f"wellness_{index:05d}",
            "question": question,
            "answer": answer,
            "topic": topic,
            "risk_stage": risk_stage,
            "support_hint": support_hint,
        }

        for field_name in NUMERIC_FIELDS:
            normalized[field_name] = self._coerce_score(
                record.get(field_name),
                derived_scores[field_name],
            )

        return normalized

    def _first_text(self, record: Dict[str, Any], keys: Iterable[str]) -> str:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                cleaned = self._preparation.normalize_text(value)
            else:
                cleaned = self._preparation.normalize_text(str(value))
            if cleaned:
                return cleaned
        return ""

    def _normalize_support_hint(self, record: Dict[str, Any], answer: str) -> str:
        hint = self._first_text(record, ("support_hint", "supportHint", "hint"))
        if hint:
            return hint

        if not answer:
            return "지금은 숨을 고르고, 오늘 할 수 있는 가장 작은 한 가지를 선택해 보세요."

        first_sentence = answer.split(".")[0].strip()
        if not first_sentence:
            return answer[:120]
        return first_sentence[:160]

    def _coerce_score(self, value: Any, fallback: int) -> int:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            return self._bounded_score(int(round(float(value))))
        if isinstance(value, str):
            parsed = self._parse_text_score(value)
            if parsed is not None:
                return parsed
        return self._bounded_score(fallback)

    def _parse_text_score(self, value: str) -> Optional[int]:
        text = value.strip().lower()
        if not text:
            return None

        if text.isdigit():
            return self._bounded_score(int(text))

        positive_markers = ("좋", "괜찮", "충분", "잘", "양호", "규칙", "식사함", "rested", "good")
        negative_markers = ("나쁘", "힘들", "부족", "안 좋", "못", "거르", "skip", "poor", "bad")

        if any(marker in text for marker in positive_markers):
            return 8
        if any(marker in text for marker in negative_markers):
            return 2
        return None

    def _bounded_score(self, value: int) -> int:
        return max(0, min(10, value))

    def _infer_risk_stage(self, topic: str) -> str:
        normalized = topic.lower()
        if any(keyword in normalized for keyword in ("risk", "crisis", "danger")):
            return "위험"
        if any(keyword in normalized for keyword in ("mood", "stress", "anxiety", "sleep")):
            return "주의"
        return "관심"

    def _derive_scores(self, topic: str, risk_stage: str) -> Dict[str, int]:
        if risk_stage == "위험":
            base = {
                "mood_score": 2,
                "anxiety_score": 8,
                "loneliness_score": 7,
                "sleep_quality": 2,
                "meal_status": 3,
                "energy_score": 2,
                "stress_score": 8,
            }
        elif risk_stage == "주의":
            base = {
                "mood_score": 4,
                "anxiety_score": 6,
                "loneliness_score": 5,
                "sleep_quality": 4,
                "meal_status": 5,
                "energy_score": 4,
                "stress_score": 6,
            }
        else:
            base = {
                "mood_score": 7,
                "anxiety_score": 3,
                "loneliness_score": 3,
                "sleep_quality": 7,
                "meal_status": 7,
                "energy_score": 6,
                "stress_score": 3,
            }

        topic_lower = topic.lower()
        if "sleep" in topic_lower:
            base.update({"sleep_quality": 3, "stress_score": max(base["stress_score"], 5)})
        if "anxiety" in topic_lower:
            base.update({"anxiety_score": 8, "stress_score": max(base["stress_score"], 6)})
        if "loneliness" in topic_lower or "relationship" in topic_lower:
            base.update({"loneliness_score": 8})
        if "mood" in topic_lower:
            base.update({"mood_score": 3, "energy_score": 3})
        if "stress" in topic_lower or "work" in topic_lower:
            base.update({"stress_score": 8})
        if "study" in topic_lower:
            base.update({"stress_score": 5, "energy_score": 5})

        return base
