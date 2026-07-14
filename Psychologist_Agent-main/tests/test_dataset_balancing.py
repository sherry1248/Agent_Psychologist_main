from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_and_balance_datasets.py"


def load_balancing_module():
    spec = importlib.util.spec_from_file_location("analyze_and_balance_datasets", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_dataset_balancing_downsamples_and_preserves_low_frequency_labels(tmp_path):
    module = load_balancing_module()
    processed_dir = tmp_path / "processed"
    derived_dir = tmp_path / "derived"
    report_path = tmp_path / "reports" / "dataset_processing_report.md"

    records = [
        {
            "id": f"common_{index}",
            "user_input": f"요즘 불안해서 잠이 잘 안 와요 {index}",
            "category": "ANXIETY",
            "intervention_hint": "감정을 먼저 확인하고 작은 행동을 제안하세요.",
            "source": "unit",
        }
        for index in range(5)
    ]
    records.extend(
        [
            {
                "id": "rare_1",
                "user_input": "친구와의 관계 때문에 마음이 복잡해요",
                "category": "RELATIONSHIP",
                "source": "unit",
            },
            {
                "id": "dup_1",
                "user_input": "중복처럼 보이는 기록입니다",
                "category": "DUPLICATE",
                "source": "unit",
            },
            {
                "id": "dup_2",
                "user_input": "중복처럼 보이는 기록입니다",
                "category": "DUPLICATE",
                "source": "unit",
            },
            {
                "id": "invalid_1",
                "user_input": "",
                "category": "INVALID",
                "source": "unit",
            },
        ]
    )
    write_jsonl(processed_dir / "sample.jsonl", records)

    result = module.run_analysis(
        processed_dir=processed_dir,
        derived_dir=derived_dir,
        report_path=report_path,
        profile_sample_caps={"small": 2, "recommended": 3, "broad": 4},
        output_name="balanced_safe_hints.jsonl",
    )

    output_path = derived_dir / "balanced_safe_hints.jsonl"
    assert result["output_path"] == output_path
    assert output_path.exists()
    assert result["profile_counts"] == {
        "small": 4,
        "recommended": 5,
        "broad": 6,
    }
    assert (derived_dir / "balanced_safe_hints_small.jsonl").exists()
    assert (derived_dir / "balanced_safe_hints_recommended.jsonl").exists()
    assert (derived_dir / "balanced_safe_hints_broad.jsonl").exists()

    balanced_records = read_jsonl(output_path)
    labels = Counter(record["label_or_topic"] for record in balanced_records)
    assert labels["ANXIETY/sleep_problem"] == 4
    assert labels["RELATIONSHIP/relationship_hurt"] == 1
    assert labels["DUPLICATE"] == 1
    assert result["stats"]["duplicate_like_records"] == 1
    assert result["stats"]["empty_invalid_records"] == 1

    for record in balanced_records:
        assert set(record) == {
            "source_dataset",
            "label_or_topic",
            "intent_hint",
            "emotion_hint",
            "cause_hint",
            "action_hint",
            "safety_hint",
            "short_summary",
        }
        assert len(record["short_summary"]) <= 143

    report = report_path.read_text(encoding="utf-8")
    assert "## Before Label Distribution" in report
    assert "## After Label Distribution" in report
    assert "## Kept Low-Frequency Labels" in report
    assert "Balanced safe hints path" in report
    assert "## Profile Outputs" in report
    assert "## Coverage Check" in report
    assert "## Scenario Bucket Coverage" in report
    assert "## Profile Comparison" in report
