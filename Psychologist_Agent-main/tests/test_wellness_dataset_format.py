"""
Tests for the standard wellness dataset format and sample raw data.
"""

from pathlib import Path
import json

import pytest

from scripts.data_preparation import DataPreparation


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_RAW_FILE = ROOT_DIR / "tests" / "fixtures" / "wellness_sample.jsonl"


class TestWellnessDatasetFormat:
    def test_sample_dataset_exists(self):
        assert SAMPLE_RAW_FILE.exists()

    def test_sample_dataset_has_canonical_fields(self):
        lines = SAMPLE_RAW_FILE.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1

        first_record = json.loads(lines[0])
        assert "questionText" in first_record
        assert "answerText" in first_record
        assert "topic" in first_record
        assert "question" not in first_record
        assert "answer" not in first_record

    def test_preprocessing_pipeline_accepts_sample_dataset(self, tmp_path):
        output_dir = tmp_path / "processed"
        prep = DataPreparation(
            raw_dir=str(SAMPLE_RAW_FILE.parent),
            output_dir=str(output_dir),
            min_question_len=10,
            min_answer_len=50,
            seed=42,
        )

        raw_records = prep.load_raw_records()
        assert len(raw_records) == 6

        cleaned_records = prep.clean_records(raw_records)
        assert len(cleaned_records) == 6

        transformed = [prep.transform_record(record, idx) for idx, record in enumerate(cleaned_records)]
        assert len(transformed) == 6
        assert all(set(record.keys()) == {"id", "question", "answer", "topic"} for record in transformed)

        stats = prep.run()
        assert stats["original_count"] == 6
        assert stats["after_dedup"] == 6

        cleaned_path = output_dir / "counsel_chat_cleaned.jsonl"
        train_path = output_dir / "counsel_chat_train.jsonl"
        eval_path = output_dir / "counsel_chat_eval.jsonl"
        test_path = output_dir / "counsel_chat_test.jsonl"

        assert cleaned_path.exists()
        assert train_path.exists()
        assert eval_path.exists()
        assert test_path.exists()

        cleaned_lines = cleaned_path.read_text(encoding="utf-8").splitlines()
        assert len(cleaned_lines) == 6

        processed_record = json.loads(cleaned_lines[0])
        assert processed_record["id"].startswith("counsel_")
        assert processed_record["question"]
        assert processed_record["answer"]
        assert processed_record["topic"]

    def test_preprocessing_accepts_alias_fields(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        alias_file = raw_dir / "alias_sample.jsonl"
        alias_file.write_text(
            "{\"prompt\":\"요즘 너무 지쳐요.\",\"output\":\"잠시 쉬고, 할 일을 작은 단위로 나누어 보세요.\",\"label\":\"stress\"}\n"
            "{\"question\":\"밤에 잠이 안 와요.\",\"response\":\"자기 전 화면을 줄이고 호흡을 천천히 맞춰보세요.\",\"category\":\"sleep\"}\n",
            encoding="utf-8",
        )

        prep = DataPreparation(
            raw_dir=str(raw_dir),
            output_dir=str(tmp_path / "processed"),
            min_question_len=5,
            min_answer_len=10,
        )

        raw_records = prep.load_raw_records()
        assert len(raw_records) == 2

        cleaned_records = prep.clean_records(raw_records)
        assert len(cleaned_records) == 2

        transformed = [prep.transform_record(record, idx) for idx, record in enumerate(cleaned_records)]
        assert transformed[0]["question"]
        assert transformed[0]["answer"]
        assert transformed[1]["topic"] in {"sleep", "stress"}
