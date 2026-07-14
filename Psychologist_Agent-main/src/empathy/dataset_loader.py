"""Empathy dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("empathy_dataset_loader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "empathy_processed.jsonl",
    PROJECT_ROOT / "data" / "raw" / "empathy_sample.jsonl",
]

ALLOWED_EMOTION_LABELS = {"기쁨", "당황", "분노", "불안", "상처", "슬픔"}
ALLOWED_EMPATHY_LABELS = {"동조", "조언", "위로", "격려"}

EmpathyRecord = Dict[str, Any]


class EmpathyDatasetLoader:
    """Load and normalize empathy dialogue records."""

    def __init__(self, dataset_path: Optional[str | Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None

    def load_records(self) -> List[EmpathyRecord]:
        for candidate in self._resolve_candidates():
            try:
                raw_records = self._load_candidate(candidate)
                normalized = [
                    self._normalize_record(record, index)
                    for index, record in enumerate(raw_records)
                ]
                normalized = [
                    record for record in normalized
                    if record["user_input"] and record["emotion_label"] and record["empathy_label"]
                ]

                if normalized:
                    logger.info("Loaded %s empathy records from %s", len(normalized), candidate)
                    return normalized
            except Exception as exc:
                logger.warning("Failed to load empathy dataset from %s: %s", candidate, exc)

        return []

    def validate_emotion_label(self, emotion_label: str) -> str:
        normalized = self._normalize_label(emotion_label)
        if normalized not in ALLOWED_EMOTION_LABELS:
            raise ValueError(f"Unsupported emotion label: {emotion_label}")
        return normalized

    def validate_empathy_label(self, empathy_label: str) -> str:
        normalized = self._normalize_label(empathy_label)
        if normalized not in ALLOWED_EMPATHY_LABELS:
            raise ValueError(f"Unsupported empathy label: {empathy_label}")
        return normalized

    def _resolve_candidates(self) -> List[Path]:
        if self.dataset_path is not None:
            return [self.dataset_path]
        return DEFAULT_DATASET_PATHS

    def _load_candidate(self, candidate: Path) -> List[Dict[str, Any]]:
        if not candidate.exists():
            raise FileNotFoundError(candidate)

        if candidate.is_dir():
            raise ValueError(f"Unsupported empathy dataset directory: {candidate}")

        suffix = candidate.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(candidate)
        if suffix == ".json":
            with candidate.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            if isinstance(payload, list):
                return [record for record in payload if isinstance(record, dict)]
            if isinstance(payload, dict):
                return [payload]
            raise ValueError(f"Unsupported empathy JSON payload: {candidate}")

        raise ValueError(f"Unsupported empathy dataset format: {candidate}")

    def _load_jsonl(self, candidate: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with candidate.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def _normalize_record(self, record: Dict[str, Any], index: int) -> EmpathyRecord:
        user_input = self._first_text(record, ("user_input", "input", "utterance", "question", "prompt"))
        emotion_label = self.validate_emotion_label(
            self._first_text(record, ("emotion_label", "emotion", "emotionLabel"))
        )
        empathy_label = self.validate_empathy_label(
            self._first_text(record, ("empathy_label", "empathy", "empathyLabel"))
        )
        empathy_style_hint = self._normalize_hint(record, emotion_label, empathy_label)

        return {
            "id": self._first_text(record, ("id",)) or f"empathy_{index:05d}",
            "user_input": user_input,
            "emotion_label": emotion_label,
            "empathy_label": empathy_label,
            "empathy_style_hint": empathy_style_hint,
        }

    def _first_text(self, record: Dict[str, Any], keys: Iterable[str]) -> str:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                cleaned = " ".join(value.split()).strip()
            else:
                cleaned = " ".join(str(value).split()).strip()
            if cleaned:
                return cleaned
        return ""

    def _normalize_label(self, value: str) -> str:
        return " ".join(value.split()).strip()

    def _normalize_hint(self, record: Dict[str, Any], emotion_label: str, empathy_label: str) -> str:
        hint = self._first_text(record, ("empathy_style_hint", "style_hint", "hint"))
        if hint:
            return hint

        emotion_text = {
            "기쁨": "기쁨을 함께 반영하고 자연스러운 흐름을 이어가요.",
            "당황": "당황스러움을 인정하고 천천히 정리해요.",
            "분노": "분노의 신호를 확인하고 안전한 표현을 돕고 있어요.",
            "불안": "불안을 가볍게 여기지 않고 차분하게 정리해요.",
            "상처": "상처받은 마음을 먼저 받아들이고 있어요.",
            "슬픔": "슬픔을 충분히 담아내고 조용히 지지해요.",
        }.get(emotion_label, "감정을 확인하고 편안하게 반응해요.")

        empathy_text = {
            "동조": "상대의 감정에 맞장구치며 함께하고 있어요.",
            "조언": "감정 다음의 작은 실천을 제안하고 있어요.",
            "위로": "지금의 고됨을 인정하며 위로하고 있어요.",
            "격려": "조금씩 해낼 수 있다는 점을 격려하고 있어요.",
        }.get(empathy_label, "상담 맥락에 맞는 공감 표현을 사용해요.")

        return f"{emotion_text} {empathy_text}"
