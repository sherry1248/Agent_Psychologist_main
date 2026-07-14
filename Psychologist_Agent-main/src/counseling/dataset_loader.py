"""Counseling dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("counseling_dataset_loader")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "counseling_processed.jsonl",
    PROJECT_ROOT / "data" / "raw" / "counseling_sample.jsonl",
]

CounselingRecord = Dict[str, Any]


class CounselingDatasetLoader:
    """Load and normalize counseling dataset records."""

    def __init__(self, dataset_path: Optional[str | Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None

    def load_records(self) -> List[CounselingRecord]:
        for candidate in self._resolve_candidates():
            try:
                raw_records = self._load_candidate(candidate)
                normalized = [
                    self._normalize_record(record, index)
                    for index, record in enumerate(raw_records)
                ]
                normalized = [
                    record for record in normalized
                    if record["user_input"] and record["intervention_hint"]
                ]

                if normalized:
                    logger.info("Loaded %s counseling records from %s", len(normalized), candidate)
                    return normalized
            except Exception as exc:
                logger.warning("Failed to load counseling dataset from %s: %s", candidate, exc)

        return []

    def _resolve_candidates(self) -> List[Path]:
        if self.dataset_path is not None:
            return [self.dataset_path]
        return DEFAULT_DATASET_PATHS

    def _load_candidate(self, candidate: Path) -> List[Dict[str, Any]]:
        if not candidate.exists():
            raise FileNotFoundError(candidate)

        if candidate.is_dir():
            raise ValueError(f"Unsupported counseling dataset directory: {candidate}")

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
            raise ValueError(f"Unsupported counseling JSON payload: {candidate}")

        raise ValueError(f"Unsupported counseling dataset format: {candidate}")

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

    def _normalize_record(self, record: Dict[str, Any], index: int) -> CounselingRecord:
        user_input = self._first_text(record, ("user_input", "input", "utterance", "question", "prompt"))
        counselor_response = self._first_text(record, ("assistant_response", "response", "answer", "output"))
        category = self._first_text(record, ("category", "condition", "topic", "label")) or "general"
        intervention_hint = self._normalize_hint(record, counselor_response)

        return {
            "id": self._first_text(record, ("id",)) or f"counseling_{index:05d}",
            "user_input": user_input,
            "counselor_response": counselor_response,
            "category": category,
            "intervention_hint": intervention_hint,
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

    def _normalize_hint(self, record: Dict[str, Any], response: str) -> str:
        hint = self._first_text(record, ("intervention_hint", "interventionHint", "hint"))
        if hint:
            return hint
        if not response:
            return "감정을 확인하고 작은 실행 단계부터 제안하세요."
        first_sentence = response.split(".")[0].strip()
        return first_sentence[:160] if first_sentence else response[:160]
