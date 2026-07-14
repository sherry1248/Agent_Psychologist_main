"""End-to-end tests for the multi-dataset agent flow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.counseling.dataset_loader import CounselingDatasetLoader
from src.counseling.retriever import CounselingRetriever
from src.empathy.dataset_loader import EmpathyDatasetLoader
from src.empathy.retriever import EmpathyRetriever
from src.main import PsychologistAgent


def _write_raw_dataset_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    counseling_path = tmp_path / "counseling_sample.jsonl"
    empathy_path = tmp_path / "empathy_sample.jsonl"

    counseling_path.write_text(
        json.dumps(
            {
                "id": "counseling_fixture",
                "user_input": "요즘 일 때문에 지치고 외로워요.",
                "assistant_response": "부담을 확인하고 작은 실행 단계를 함께 정리합니다.",
                "category": "work_stress",
                "intervention_hint": "오늘 할 일을 작은 단위로 나누고 쉬는 시간을 먼저 정해보세요.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    empathy_path.write_text(
        json.dumps(
            {
                "id": "empathy_fixture",
                "user_input": "요즘 일 때문에 지치고 외로워요.",
                "emotion_label": "슬픔",
                "empathy_label": "위로",
                "empathy_style_hint": "지친 마음을 먼저 인정하고 차분하게 위로하세요.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return counseling_path, empathy_path


async def _run_case(
    message: str,
    wellness_checkin: dict[str, int] | None = None,
    counseling_path: Path | None = None,
    empathy_path: Path | None = None,
):
    agent = PsychologistAgent(mock_mode=True)
    if counseling_path:
        agent.counseling_retriever = CounselingRetriever(
            CounselingDatasetLoader(counseling_path)
        )
    if empathy_path:
        agent.empathy_retriever = EmpathyRetriever(
            EmpathyDatasetLoader(empathy_path)
        )
    await agent.initialize()
    session = await agent.session_manager.create_session()

    try:
        result = await agent.process_message(
            user_input=message,
            session_id=session.session_id,
            wellness_checkin=wellness_checkin,
        )
        history = await agent.session_manager.get_session_history(session.session_id)
        return result, history
    finally:
        await agent.shutdown()


def test_normal_input_returns_dataset_hints():
    result, _ = asyncio.run(
        _run_case(
            "요즘 일 때문에 지치고 외로워요.",
            {
                "mood_score": 4,
                "anxiety_score": 6,
                "loneliness_score": 7,
                "sleep_quality": 3,
                "meal_status": 5,
                "energy_score": 4,
                "stress_score": 8,
            },
        )
    )

    assert result["requires_crisis_response"] is False
    assert result["counseling_hint"]
    assert result["empathy_style_hint"]
    assert result["wellness_hint"]
    assert "상담 참고" not in result["response"]
    assert "공감 참고" not in result["response"]
    assert "웰니스 참고" not in result["response"]
    assert "지금 느끼는 부담" in result["response"]
    assert "109" in result["response"]


def test_crisis_response_takes_priority_over_dataset_hints():
    result, _ = asyncio.run(
        _run_case(
            "죽고 싶어요. 지금 자해하고 싶어요.",
            {
                "mood_score": 1,
                "anxiety_score": 10,
                "loneliness_score": 9,
                "sleep_quality": 1,
                "meal_status": 2,
                "energy_score": 1,
                "stress_score": 10,
            },
        )
    )

    assert result["requires_crisis_response"] is True
    assert result["risk_stage"] == "위험"
    assert result["response"]
    assert result["counseling_hint"] == ""
    assert result["empathy_style_hint"] == ""
    assert result["wellness_hint"] == ""
    assert "counseling" not in result.get("pipeline_details", {})
    assert "empathy" not in result.get("pipeline_details", {})


def test_raw_input_is_not_written_to_dataset_files_or_logs(tmp_path, caplog):
    user_input = "이 문장은 어디에도 저장되면 안 됩니다 918273"
    counseling_path, empathy_path = _write_raw_dataset_fixtures(tmp_path)

    with caplog.at_level("INFO"):
        result, history = asyncio.run(
            _run_case(
                user_input,
                counseling_path=counseling_path,
                empathy_path=empathy_path,
            )
        )

    dataset_files = "\n".join(
        [
            counseling_path.read_text(encoding="utf-8"),
            empathy_path.read_text(encoding="utf-8"),
        ]
    )

    assert user_input not in dataset_files
    assert user_input not in caplog.text
    assert user_input not in str(result)
    assert user_input not in str(history)
