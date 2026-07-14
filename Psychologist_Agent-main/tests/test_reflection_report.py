"""Tests for the safe chat-side reflection report."""

import asyncio
import time
from types import SimpleNamespace

from demo import app as demo_app
from src.agent.action_checkin import classify_action_checkin
from src.agent.reflection_report import (
    INSUFFICIENT_HISTORY_MESSAGE,
    MAX_REPORT_RECORDS,
    REPORT_FALLBACK_MESSAGE,
    build_reflection_report,
)
from src.memory.models import ReflectionMemoryEntry
from src.memory.store import MemoryStore


def test_report_works_with_no_history():
    report = build_reflection_report([])

    assert report["has_history"] is False
    assert report["message"] == INSUFFICIENT_HISTORY_MESSAGE
    assert report["metrics"]["recent_counseling_count"] == 0


def test_report_works_with_structured_memory_and_required_continuity_fields():
    report = build_reflection_report(
        [
            ReflectionMemoryEntry(
                intent_label="WORK_OR_STUDY_STRESS",
                main_issue=["학업 부담", "암기량 부담"],
                emotion_hint="막막함",
                last_small_action="꼭 외워야 할 개념 5개 고르기",
                action_status="in_progress",
                next_follow_up="어떤 용어 구분이 가장 헷갈렸나요?",
                created_at="2026-06-19T00:00:00Z",
            ),
            ReflectionMemoryEntry(
                intent_label="WORK_OR_STUDY_STRESS",
                main_issue=["학업 부담", "시험 준비"],
                emotion_hint="안도감",
                last_small_action="시험 개념 5개를 분류하기",
                action_status="partial",
                next_follow_up="개념 정의와 알고리즘 흐름 중 무엇이 더 헷갈렸나요?",
                created_at="2026-06-20T00:00:00Z",
            ),
        ]
    )

    assert report["has_history"] is True
    assert "학업 부담" not in report["repeated_themes"]
    assert "암기량 부담" in report["repeated_themes"]
    assert "시험 준비" in report["repeated_themes"]
    assert report["last_small_action"] == "시험 개념 5개를 분류하기"
    assert report["next_follow_up"] == "개념 정의와 알고리즘 흐름 중 무엇이 더 헷갈렸나요?"
    assert report["action_status"] == "일부 진행됨"
    assert "시험 준비" in report["long_term_trend"]
    assert report["current_emotional_state"] == "안도감"


def _academic_history(checkin_status="partial"):
    return [
        ReflectionMemoryEntry(
            intent_label="WORK_OR_STUDY_STRESS",
            main_issue=["specific_academic_burden"],
            emotion_hint="막막함",
            last_small_action="인공지능 시험에서 꼭 외워야 할 개념 5개 정리하기",
            action_status="suggested",
            next_follow_up="어떤 개념이 가장 헷갈렸나요?",
        ),
        ReflectionMemoryEntry(
            intent_label="OTHER_CONCERN",
            main_issue=["other_concern"],
            emotion_hint="비교적 안정적",
            last_small_action="정리한 개념 하나에 물음표 붙이기",
            action_status=checkin_status,
        ),
    ]


def test_partial_action_reply_keeps_previous_specific_issue_and_context():
    report = build_reflection_report(_academic_history())

    assert report["main_issue"] == ["구체적인 학업 부담"]
    assert report["current_emotional_state"] == "막막함"
    assert report["next_follow_up"] == "어떤 개념이 가장 헷갈렸나요?"
    assert report["action_status"] == "일부 진행됨"
    assert "구체적인 학업 부담" in report["long_term_trend"]
    assert "진행이 시작된 흐름" in report["long_term_trend"]


def test_internal_action_checkin_issue_carries_previous_main_issue_quickly():
    started = time.perf_counter()
    report = build_reflection_report(
        [
            {
                "intent_label": "WORK_OR_STUDY_STRESS",
                "main_issue": ["학업/업무 부담"],
                "action_status": "suggested",
            },
            {
                "intent_label": "action_checkin",
                "main_issue": ["최근 작은 행동 점검"],
                "action_status": "partial",
            },
        ]
    )

    assert time.perf_counter() - started < 0.5
    assert report["main_issue"] in (["학업/업무 부담"], ["학업 부담"], ["시험 준비 부담"])
    assert "최근 작은 행동 점검" not in str(report)
    assert report["action_status"] == "일부 진행됨"


def test_failed_action_reply_keeps_previous_specific_issue():
    report = build_reflection_report(_academic_history("not_completed"))

    assert report["main_issue"] == ["구체적인 학업 부담"]
    assert report["action_status"] == "아직 실행 전"
    assert "구체적인 학업 부담" in report["long_term_trend"]
    assert "실행하기 전" in report["long_term_trend"]


def test_new_unrelated_issue_replaces_previous_issue():
    history = _academic_history()[:1]
    history.append(
        ReflectionMemoryEntry(
            intent_label="SLEEP_PROBLEM",
            main_issue=["sleep_problem"],
            emotion_hint="피로",
            last_small_action="잠들기 전 화면 밝기 낮추기",
            action_status="partial",
        )
    )

    report = build_reflection_report(history)

    assert report["main_issue"] == ["수면 유지 문제"]
    assert "구체적인 학업 부담" not in report["main_issue"]


def test_report_never_exposes_internal_or_vague_labels_with_specific_history():
    report = build_reflection_report(_academic_history())
    rendered = str(report)

    for internal in (
        "work_or_study_stress", "other_concern", "specific_academic_burden",
        "self_blame", "crisis_safety",
    ):
        assert internal not in rendered.lower()
    assert "기타 고민" not in rendered
    assert "조금 더 살펴보고 있어요" not in rendered


def test_sleep_issue_is_inherited_for_partial_action_reply():
    report = build_reflection_report(
        [
            ReflectionMemoryEntry(
                intent_label="SLEEP_PROBLEM",
                main_issue=["sleep_problem"],
                emotion_hint="피로",
                last_small_action="잠들기 전 화면 밝기 낮추기",
                action_status="suggested",
            ),
            ReflectionMemoryEntry(
                intent_label="OTHER_CONCERN",
                main_issue=["other_concern"],
                action_status="partial",
            ),
        ]
    )

    assert report["main_issue"] == ["수면 유지 문제"]
    assert report["current_emotional_state"] == "피로"
    assert "수면 유지 문제" in report["long_term_trend"]


def test_numeric_action_reply_is_partial_progress():
    result = classify_action_checkin("3개 했어", has_previous_action=True)

    assert result.detected is True
    assert result.status == "partial"


def test_numeric_action_reply_compares_with_known_target():
    completed = classify_action_checkin(
        "개념 5개 정리했어",
        has_previous_action=True,
        previous_action_text="개념 5개 정리하기",
    )
    partial = classify_action_checkin(
        "개념 3개 정리했어",
        has_previous_action=True,
        previous_action_text="개념 5개 정리하기",
    )

    assert completed.status == "completed"
    assert partial.status == "partial"


def test_report_generation_is_fast_and_bounded_to_recent_records():
    history = [
        ReflectionMemoryEntry(
            intent_label="SLEEP_PROBLEM",
            main_issue=["sleep_problem"],
            emotion_hint="피로",
            action_status="suggested",
        )
        for _ in range(500)
    ]

    started = time.perf_counter()
    report = build_reflection_report(history)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert report["metrics"]["recent_counseling_count"] == MAX_REPORT_RECORDS
    assert report["main_issue"] == ["수면 유지 문제"]


def test_mixed_action_status_report_is_fast_and_uses_latest_valid_status():
    history = [
        {"main_issue": ["sleep_problem"], "action_status": "suggested"},
        {"main_issue": ["sleep_problem"], "action_status": "completed"},
        {"main_issue": ["sleep_problem"], "action_status": "partial"},
        {"main_issue": ["sleep_problem"], "action_status": "unknown"},
        {"main_issue": ["sleep_problem"], "action_status": None},
    ]

    started = time.perf_counter()
    report = build_reflection_report(history)

    assert time.perf_counter() - started < 0.5
    assert report["action_status"] == "일부 진행됨"
    assert "sleep_problem" not in str(report)
    assert "을(를)" not in str(report)
    assert "부담를" not in str(report)


def test_completed_latest_action_status_is_completed():
    report = build_reflection_report(
        [
            {"main_issue": ["sleep_problem"], "action_status": "suggested"},
            {"main_issue": ["sleep_problem"], "action_status": "completed"},
        ]
    )

    assert report["action_status"] == "완료됨"
    assert "다음 단계로 이어갈 수 있는 흐름" in report["long_term_trend"]


def test_suggested_only_action_status_remains_suggested():
    report = build_reflection_report(
        [{"main_issue": ["sleep_problem"], "action_status": "suggested"}]
    )

    assert report["action_status"] == "제안됨"
    assert "작은 행동을 제안한 상태" in report["long_term_trend"]


def test_broad_academic_issue_is_refined_from_structured_action_context():
    report = build_reflection_report(
        [
            {
                "intent_label": "WORK_OR_STUDY_STRESS",
                "main_issue": ["학업/업무 부담"],
                "last_small_action": "시험 개념과 정의를 세 묶음으로 나누기",
                "next_follow_up": "어떤 알고리즘이 가장 헷갈렸나요?",
                "action_status": "suggested",
            }
        ]
    )

    assert report["main_issue"] == ["시험 준비 부담"]
    assert "학업/업무 부담 관련" not in report["long_term_trend"]
    assert "을(를)" not in str(report)
    assert "부담를" not in str(report)


def test_report_with_missing_fields_returns_quickly():
    started = time.perf_counter()
    report = build_reflection_report([{"created_at": None}, {"emotion_hint": "불안"}])

    assert time.perf_counter() - started < 0.5
    assert report["has_history"] is True
    assert "anxiety" not in str(report).lower()


def test_malformed_history_returns_fallback_quickly():
    class MalformedHistory:
        def __iter__(self):
            raise ValueError("malformed structured history")

    started = time.perf_counter()
    report = build_reflection_report(MalformedHistory())

    assert time.perf_counter() - started < 0.5
    assert report["has_history"] is False
    assert report["message"] == REPORT_FALLBACK_MESSAGE


def test_report_skips_invalid_records_and_keeps_valid_records():
    report = build_reflection_report(
        [
            object(),
            {"main_issue": ["sleep_problem"], "action_status": "partial"},
            object(),
        ]
    )

    assert report["has_history"] is True
    assert report["metrics"]["recent_counseling_count"] == 1
    assert report["main_issue"] == ["수면 유지 문제"]


def test_opening_report_does_not_initialize_counseling_agent(monkeypatch):
    monkeypatch.setattr(demo_app, "agent", None)
    started = time.perf_counter()

    report, _, context = asyncio.run(
        demo_app.open_reflection_report_for_service(
            {},
            {"session_id": "pending-report", "agent_session_ready": False},
            False,
        )
    )

    assert time.perf_counter() - started < 0.5
    assert demo_app.agent is None
    assert context["session_id"] == "pending-report"
    assert "아직 상담 기록이 충분하지 않아요" in report


def test_record_saving_setting_returns_short_status_not_full_report():
    status = demo_app.privacy_settings_markdown({"mode": "logged_in"}, True)

    assert "기록 저장이 켜졌어요" in status
    assert "현재 마음 상태" not in status
    assert "반복 주제" not in status
    assert "장기 흐름" not in status
    assert len(status) < 100


def test_theme_families_remove_broad_duplicates_and_keep_distinct_aspects():
    report = build_reflection_report(
        [
            ReflectionMemoryEntry(main_issue=["학업/업무 부담", "시험 준비"]),
            ReflectionMemoryEntry(main_issue=["학업 부담", "암기 부담", "자기비난/자책"]),
        ]
    )

    assert "학업/업무 부담" not in report["repeated_themes"]
    assert "학업 부담" not in report["repeated_themes"]
    assert "시험 준비" in report["repeated_themes"]
    assert "암기 부담" in report["repeated_themes"]
    assert "자기비난/자책" in report["repeated_themes"]


def test_long_term_flow_uses_specific_issue_without_vague_fallback():
    report = build_reflection_report(
        [
            ReflectionMemoryEntry(
                main_issue=["criticism_scolding"],
                last_small_action="피드백에서 사실과 평가를 나누기",
                action_status="partial",
            )
        ]
    )

    assert report["main_issue"] == ["지적/평가 스트레스"]
    assert "지적/평가 스트레스" in report["long_term_trend"]
    assert "조금 더 살펴보고 있어요" not in str(report)
    assert "기타 고민" not in str(report)


def test_previous_memory_summary_is_short_not_a_full_report():
    messages = demo_app.structured_memory_chat_messages(
        {
            "main_issue": ["관계 스트레스"],
            "action_status": "일부 진행됨",
            "next_follow_up": "다음에 어떤 경계를 세워볼까요?",
            "metrics": {"recent_risk_signal": False},
        }
    )
    content = messages[0]["content"]

    assert "지난 상담 요약" in content
    assert "주요 고민" in content
    assert "최근 행동 상태" in content
    assert "다음 질문" in content
    assert "반복 주제" not in content
    assert "최근 작은 행동" not in content
    assert "장기 흐름" not in content


def test_empty_report_path_never_calls_agent_or_llm(monkeypatch):
    async def fail_if_called():
        raise AssertionError("counseling agent must not be called")

    monkeypatch.setattr(demo_app, "get_agent", fail_if_called)
    outputs = asyncio.run(
        demo_app.open_reflection_report_for_service(
            {}, {"session_id": "empty", "agent_session_ready": False}, False
        )
    )

    assert len(outputs) == 3
    report, tab_update, context = outputs
    assert "아직 상담 기록이 충분하지 않아요" in report
    assert tab_update == {"selected": "report", "__type__": "update"}
    assert isinstance(context, dict)


def test_report_path_never_constructs_agent_or_calls_llm(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("agent or LLM path must not be called")

    monkeypatch.setattr(demo_app, "agent", None)
    monkeypatch.setattr(demo_app, "get_agent", fail_if_called)
    monkeypatch.setattr(demo_app, "get_or_create_agent", fail_if_called)

    report = asyncio.run(
        demo_app.build_chat_reflection_report(
            {"risk_stage": "관심"},
            {"session_id": "structured-only", "agent_session_ready": False},
        )
    )

    assert INSUFFICIENT_HISTORY_MESSAGE in report


def test_report_callback_returns_fallback_outputs_for_malformed_records(monkeypatch):
    class MalformedStore:
        async def get_reflection_history(self, *args, **kwargs):
            return [object()]

    monkeypatch.setattr(
        demo_app,
        "agent",
        SimpleNamespace(memory_store=MalformedStore()),
    )
    started = time.perf_counter()
    report, tab_update, context = asyncio.run(
        demo_app.open_reflection_report_for_service(
            {}, {"session_id": "malformed", "agent_session_ready": True}, False
        )
    )

    assert time.perf_counter() - started < 0.5
    assert "아직 상담 기록이 충분하지 않아요" in report
    assert tab_update["selected"] == "report"
    assert context["session_id"] == "malformed"


def test_report_callback_bounds_slow_structured_memory_read(monkeypatch):
    class SlowStore:
        async def get_reflection_history(self, *args, **kwargs):
            await asyncio.sleep(5)

    monkeypatch.setattr(demo_app, "agent", SimpleNamespace(memory_store=SlowStore()))
    started = time.perf_counter()
    report, tab_update, _ = asyncio.run(
        demo_app.open_reflection_report_for_service(
            {}, {"session_id": "slow", "agent_session_ready": True}, False
        )
    )

    assert time.perf_counter() - started < demo_app.REPORT_TIMEOUT_SECONDS
    assert "아직 상담 기록이 충분하지 않아요" in report
    assert tab_update["selected"] == "report"


def test_report_callback_returns_fallback_when_memory_read_raises(monkeypatch):
    class FailingStore:
        async def get_reflection_history(self, *args, **kwargs):
            raise RuntimeError("reflection memory unavailable")

    monkeypatch.setattr(demo_app, "agent", SimpleNamespace(memory_store=FailingStore()))
    started = time.perf_counter()
    outputs = asyncio.run(
        demo_app.open_reflection_report_for_service(
            {}, {"session_id": "failing", "agent_session_ready": True}, False
        )
    )

    assert time.perf_counter() - started < 0.5
    assert len(outputs) == 3
    report, tab_update, context = outputs
    assert "아직 상담 기록이 충분하지 않아요" in report
    assert tab_update == {"selected": "report", "__type__": "update"}
    assert isinstance(context, dict)


def test_report_ignores_raw_conversation_fields():
    raw = "이 원문 대화는 리포트에 나오면 안 됩니다"
    report = build_reflection_report(
        [{"intent_label": "NEED_EMPATHY", "emotion_hint": "불안", "raw_text": raw, "conversation": raw}]
    )

    assert raw not in str(report)


def test_memory_store_exposes_only_reflection_whitelist_fields():
    async def scenario():
        store = MemoryStore()
        await store.update_conversation_continuity(
            "reflection-session",
            last_small_action=SimpleNamespace(
                action_id="a1",
                intent_label="WORK_OR_STUDY_STRESS",
                action_text="개념 5개 고르기",
                status="suggested",
                created_at="2026-06-20T00:00:00Z",
            ),
            next_follow_up="어떤 부분이 가장 어려웠나요?",
            emotional_state_vector={"anxiety": 0.8, "relief": 0.2},
            risk_stage="관심",
            intent_label="WORK_OR_STUDY_STRESS",
        )
        return await store.get_reflection_history("reflection-session")

    history = asyncio.run(scenario())
    assert history[-1].emotion_hint == "불안"
    assert history[-1].last_small_action == "개념 5개 고르기"
    assert history[-1].next_follow_up == "어떤 부분이 가장 어려웠나요?"
    assert set(history[-1].to_dict()) == {
        "user_id", "anonymous_session_id", "session_id",
        "intent_label", "main_issue", "emotion_hint", "emotional_trend",
        "last_small_action", "action_status", "next_follow_up", "repeated_themes",
        "risk_stage", "created_at",
    }


def test_action_checkin_override_keeps_progress_status_with_new_suggested_action():
    async def scenario(status: str):
        store = MemoryStore()
        session_id = f"checkin-{status}"
        await store.update_conversation_continuity(
            session_id,
            last_small_action=SimpleNamespace(
                action_id="initial",
                intent_label="WORK_OR_STUDY_STRESS",
                action_text="개념 5개 정리하기",
                status="suggested",
            ),
            intent_label="WORK_OR_STUDY_STRESS",
        )
        await store.update_last_action_status(session_id, status)
        continuity = await store.update_conversation_continuity(
            session_id,
            last_small_action=SimpleNamespace(
                action_id="next",
                intent_label="WORK_OR_STUDY_STRESS",
                action_text="헷갈리는 개념 하나에 설명 붙이기",
                status="suggested",
            ),
            intent_label="WORK_OR_STUDY_STRESS",
            action_status_override=status,
        )
        history = await store.get_reflection_history(session_id)
        return continuity, history

    expected_labels = {"completed": "완료됨", "partial": "일부 진행됨"}
    for status, expected_label in expected_labels.items():
        continuity, history = asyncio.run(scenario(status))
        report = build_reflection_report(history)

        assert history[-1].action_status == status
        assert history[-1].last_small_action == "헷갈리는 개념 하나에 설명 붙이기"
        assert continuity.last_small_action.status == "suggested"
        assert report["action_status"] == expected_label


def test_crisis_report_is_clear_without_changing_crisis_handler(monkeypatch):
    monkeypatch.setattr(demo_app, "agent", None)
    monkeypatch.setattr(demo_app, "current_session_id", None)

    report = asyncio.run(demo_app.build_chat_reflection_report({"risk_stage": "위험"}))

    assert "현재 안전 상태" in report
    assert "109, 119, 112" in report
    assert INSUFFICIENT_HISTORY_MESSAGE in report
