"""Tests for wellness dataset loading and recommendation flow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.inference.generator import GenerationResult
from src.main import AgentConfig, PsychologistAgent
from src.wellness.dataset_loader import WellnessDatasetLoader
from src.wellness.recommender import SAFE_SUPPORT_HINT, WellnessRecommender


def _write_raw_sample_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "wellness_sample.jsonl"
    records = [
        {
            "id": "raw_1",
            "question": "잠을 잘 못 자고 너무 지쳐요.",
            "answer": "잠들기 전 화면 시간을 줄이고 호흡을 천천히 맞춰보세요.",
            "topic": "sleep",
            "risk_stage": "주의",
        },
        {
            "id": "raw_2",
            "question": "걱정이 많아서 불안해요.",
            "answer": "걱정을 적어보고 지금 할 수 있는 한 가지를 정해보세요.",
            "topic": "anxiety",
            "risk_stage": "주의",
        },
        {
            "id": "raw_3",
            "question": "오늘은 식사도 잘 하고 기분이 괜찮아요.",
            "answer": "지금의 균형을 이어가도록 작은 루틴을 유지해보세요.",
            "topic": "general",
            "risk_stage": "관심",
        },
        {
            "id": "raw_4",
            "question": "혼자 있는 시간이 길어 외로워요.",
            "answer": "부담이 적은 사람에게 짧은 안부 메시지부터 보내보세요.",
            "topic": "loneliness",
            "risk_stage": "주의",
        },
        {
            "id": "raw_5",
            "question": "일 때문에 스트레스가 높아요.",
            "answer": "업무를 작은 단위로 나누고 먼저 쉬는 시간을 확보해보세요.",
            "topic": "work_stress",
            "risk_stage": "주의",
        },
        {
            "id": "raw_6",
            "question": "지금 안전하지 않을까 봐 걱정돼요.",
            "answer": "혼자 버티지 말고 즉시 주변 사람이나 긴급 도움에 연결하세요.",
            "topic": "risk",
            "risk_stage": "위험",
        },
    ]
    dataset_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return dataset_path


def _write_custom_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "wellness_custom.jsonl"
    dataset_path.write_text(
        "{" \
        '"id":"wellness_a",' \
        '"question":"요즘 너무 지쳐서 잠도 잘 못 자요.",' \
        '"answer":"오늘은 해야 할 일을 아주 작은 단위로 나누고, 잠들기 전에는 화면을 줄이면서 호흡을 천천히 맞춰보세요.",' \
        '"topic":"work_stress",' \
        '"support_hint":"오늘은 해야 할 일을 아주 작은 단위로 나누고, 잠들기 전에는 화면을 줄이면서 호흡을 천천히 맞춰보세요.",' \
        '"risk_stage":"주의",' \
        '"mood_score":4,' \
        '"anxiety_score":6,' \
        '"loneliness_score":4,' \
        '"sleep_quality":3,' \
        '"meal_status":5,' \
        '"energy_score":4,' \
        '"stress_score":8' \
        "}\n" \
        "{" \
        '"id":"wellness_b",' \
        '"question":"오늘은 기분이 괜찮고 식사도 잘 했어요.",' \
        '"answer":"지금의 균형을 이어 가는 것이 중요해요.",' \
        '"topic":"general",' \
        '"support_hint":"지금의 균형을 이어 가는 것이 중요해요.",' \
        '"risk_stage":"관심",' \
        '"mood_score":8,' \
        '"anxiety_score":2,' \
        '"loneliness_score":2,' \
        '"sleep_quality":8,' \
        '"meal_status":8,' \
        '"energy_score":7,' \
        '"stress_score":2' \
        "}\n",
        encoding="utf-8",
    )
    return dataset_path


class TestWellnessRecommender:
    def test_loader_supports_raw_sample(self, tmp_path):
        loader = WellnessDatasetLoader(_write_raw_sample_dataset(tmp_path))
        records = loader.load_records()

        assert len(records) == 6
        assert all(record["support_hint"] for record in records)
        assert all(record["risk_stage"] in {"관심", "주의", "위험"} for record in records)

    def test_returns_closest_sample_hint(self, tmp_path):
        dataset_path = _write_custom_dataset(tmp_path)
        recommender = WellnessRecommender(WellnessDatasetLoader(dataset_path))

        recommendation = recommender.recommend(
            {
                "mood_score": 4,
                "anxiety_score": 6,
                "loneliness_score": 4,
                "sleep_quality": 3,
                "meal_status": 5,
                "energy_score": 4,
                "stress_score": 8,
            }
        )

        assert recommendation.support_hint.startswith("오늘은 해야 할 일을 아주 작은 단위로 나누고")
        assert recommendation.risk_stage == "주의"

    def test_falls_back_safely_when_dataset_is_missing(self):
        recommender = WellnessRecommender(WellnessDatasetLoader(Path("does-not-exist.jsonl")))

        recommendation = recommender.recommend(
            {
                "mood_score": 5,
                "anxiety_score": 5,
                "loneliness_score": 5,
                "sleep_quality": 5,
                "meal_status": 5,
                "energy_score": 5,
                "stress_score": 5,
            }
        )

        assert recommendation.support_hint == SAFE_SUPPORT_HINT
        assert recommendation.risk_stage == "관심"

    def test_recommendation_does_not_store_raw_user_text(self, tmp_path):
        dataset_path = _write_custom_dataset(tmp_path)
        user_text = "이 문장은 어디에도 저장되면 안 됩니다"

        recommender = WellnessRecommender(WellnessDatasetLoader(dataset_path))
        recommendation = recommender.recommend(
            {
                "mood_score": 4,
                "anxiety_score": 6,
                "loneliness_score": 4,
                "sleep_quality": 3,
                "meal_status": 5,
                "energy_score": 4,
                "stress_score": 8,
            }
        )

        assert user_text not in dataset_path.read_text(encoding="utf-8")
        assert user_text not in recommendation.support_hint

    def test_mock_response_uses_wellness_hint_without_internal_label(self, tmp_path, monkeypatch):
        dataset_path = _write_custom_dataset(tmp_path)
        agent = PsychologistAgent(
            config=AgentConfig(
                enable_rag=False,
                enable_cloud_analysis=False,
                enable_risk_audit=False,
            ),
            mock_mode=True,
        )
        agent.wellness_recommender = WellnessRecommender(WellnessDatasetLoader(dataset_path))

        async def fake_create_chat_completion(messages, config=None):
            return GenerationResult(
                text="기본 MOCK 응답입니다.",
                tokens_generated=2,
                finish_reason="stop",
                generation_time_ms=1.0,
            )

        monkeypatch.setattr(agent.local_generator, "create_chat_completion", fake_create_chat_completion)

        result = asyncio.run(
            agent.process_message(
                user_input="오늘 너무 지쳤어요.",
                session_id="session-wellness",
                wellness_checkin={
                    "mood_score": 4,
                    "anxiety_score": 6,
                    "loneliness_score": 4,
                    "sleep_quality": 3,
                    "meal_status": 5,
                    "energy_score": 4,
                    "stress_score": 8,
                },
            )
        )

        assert result["wellness_hint"]
        assert "웰니스 참고:" not in result["response"]
        assert "상담 참고" not in result["response"]
        assert "공감 참고" not in result["response"]
        assert "오늘은 해야 할 일을 아주 작은 단위로 나누고" in result["response"]
        assert "109" in result["response"]

    def test_danger_priority_overrides_wellness_hint(self, tmp_path):
        dataset_path = _write_custom_dataset(tmp_path)
        agent = PsychologistAgent(
            config=AgentConfig(
                enable_rag=False,
                enable_cloud_analysis=False,
                enable_risk_audit=False,
            ),
            mock_mode=True,
        )
        agent.wellness_recommender = WellnessRecommender(WellnessDatasetLoader(dataset_path))

        result = asyncio.run(
            agent.process_message(
                user_input="죽고 싶어요. 지금 혼자라서 너무 위험해요.",
                session_id="session-danger",
                wellness_checkin={
                    "mood_score": 4,
                    "anxiety_score": 6,
                    "loneliness_score": 4,
                    "sleep_quality": 3,
                    "meal_status": 5,
                    "energy_score": 4,
                    "stress_score": 8,
                },
            )
        )

        assert result["requires_crisis_response"] is True
        assert "109" in result["response"]
        assert "웰니스 참고:" not in result["response"]
