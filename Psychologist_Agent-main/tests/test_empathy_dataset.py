"""Tests for the empathy dataset loader and retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.empathy.dataset_loader import (
    ALLOWED_EMOTION_LABELS,
    ALLOWED_EMPATHY_LABELS,
    EmpathyDatasetLoader,
)
from src.empathy.retriever import EmpathyRetriever


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT_DIR / "data" / "raw" / "empathy_sample.jsonl"


def test_empathy_sample_loads_and_validates_labels():
    loader = EmpathyDatasetLoader(SAMPLE_FILE)
    records = loader.load_records()

    assert records
    assert all(record["emotion_label"] in ALLOWED_EMOTION_LABELS for record in records)
    assert all(record["empathy_label"] in ALLOWED_EMPATHY_LABELS for record in records)
    assert all(record["empathy_style_hint"] for record in records)

    assert loader.validate_emotion_label("불안") == "불안"
    assert loader.validate_empathy_label("위로") == "위로"

    with pytest.raises(ValueError):
        loader.validate_emotion_label("기분좋음")

    with pytest.raises(ValueError):
        loader.validate_empathy_label("공감")


def test_empathy_retriever_returns_empathy_style_hint():
    retriever = EmpathyRetriever(EmpathyDatasetLoader(SAMPLE_FILE))

    recommendation = retriever.recommend("계속 불안하고 걱정이 멈추지 않아요.")

    assert recommendation.emotion_label == "불안"
    assert recommendation.empathy_label in ALLOWED_EMPATHY_LABELS
    assert recommendation.empathy_style_hint
