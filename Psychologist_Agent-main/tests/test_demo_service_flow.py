"""Tests for the demo's entry, navigation, privacy, and reset flow."""

import asyncio
import inspect
import json
import time
from types import SimpleNamespace

from demo import app as demo_app
from src.memory.models import ReflectionMemoryEntry
from src.memory.store import MemoryStore
from src.session.manager import SessionManager


def test_main_navigation_has_service_tabs_without_emotion_diary():
    source = inspect.getsource(demo_app.create_demo)

    for label in ("상담 채팅", "마음정리 보고서", "전문가 상담 연결", "내 정보"):
        assert f'gr.TabItem("{label}"' in source
    assert 'gr.TabItem("감정일기"' not in source
    assert "마음 리포트 보기" in source
    assert 'gr.Accordion("Agent Pipeline Details", open=False)' in source


def test_entry_screen_offers_login_signup_and_anonymous_start():
    source = inspect.getsource(demo_app.create_demo)

    assert 'gr.TabItem("로그인")' in source
    assert 'gr.TabItem("회원가입")' in source
    assert 'gr.TabItem("익명 시작")' in source
    assert "로그인하거나 익명으로 시작할 수 있어요" in source
    assert "상담 기록은 동의한 경우에만 저장됩니다" in source
    assert "start-card" in source
    assert "orange-primary" in source


def test_logged_in_and_anonymous_sessions_are_isolated_by_session_and_user(monkeypatch):
    store = MemoryStore()
    manager = SessionManager(memory_store=store)
    fake_agent = SimpleNamespace(session_manager=manager, memory_store=store)

    async def fake_get_agent():
        return fake_agent

    monkeypatch.setattr(demo_app, "get_agent", fake_get_agent)

    async def scenario():
        logged_in = await demo_app.start_service_session("login", "user-a")
        anonymous = await demo_app.start_service_session("anonymous")
        logged_session = await manager.get_session(logged_in["session_id"])
        anonymous_session = await manager.get_session(anonymous["session_id"])
        return logged_in, anonymous, logged_session, anonymous_session

    logged_in, anonymous, logged_session, anonymous_session = asyncio.run(scenario())
    assert logged_in["session_id"] != anonymous["session_id"]
    assert logged_session.user_id == "user-a"
    assert anonymous_session.user_id is None
    assert anonymous["user_id"] == ""


def test_consented_user_records_never_mix_between_users_or_anonymous(monkeypatch):
    store = MemoryStore()
    manager = SessionManager(memory_store=store)
    fake_agent = SimpleNamespace(session_manager=manager, memory_store=store)

    async def fake_get_agent():
        return fake_agent

    monkeypatch.setattr(demo_app, "get_agent", fake_get_agent)
    demo_app.consented_user_sessions.clear()

    async def scenario():
        user_a = await demo_app.ensure_service_memory_scope(
            await demo_app.start_service_session("login", "user-a"), True
        )
        user_b = await demo_app.ensure_service_memory_scope(
            await demo_app.start_service_session("login", "user-b"), True
        )
        anonymous = await demo_app.start_service_session("anonymous")
        for context, issue in ((user_a, "학업 부담"), (user_b, "관계 부담"), (anonymous, "수면 부담")):
            await store.update_conversation_continuity(
                context["session_id"],
                intent_label=issue,
                emotional_state_vector={"stress": 0.8},
            )
        return (
            await store.get_reflection_history(user_a["session_id"]),
            await store.get_reflection_history(user_b["session_id"]),
            await store.get_reflection_history(anonymous["session_id"]),
        )

    user_a_history, user_b_history, anonymous_history = asyncio.run(scenario())
    assert [entry.intent_label for entry in user_a_history] == ["학업 부담"]
    assert [entry.intent_label for entry in user_b_history] == ["관계 부담"]
    assert [entry.intent_label for entry in anonymous_history] == ["수면 부담"]
    assert user_a_history[0].user_id == "user-a"
    assert user_b_history[0].user_id == "user-b"
    assert anonymous_history[0].anonymous_session_id


def test_consent_controls_logged_in_memory_scope(monkeypatch):
    store = MemoryStore()
    manager = SessionManager(memory_store=store)
    fake_agent = SimpleNamespace(session_manager=manager, memory_store=store)

    async def fake_get_agent():
        return fake_agent

    monkeypatch.setattr(demo_app, "get_agent", fake_get_agent)
    demo_app.consented_user_sessions.clear()

    async def scenario():
        first = await demo_app.start_service_session("login", "same-user")
        second = await demo_app.start_service_session("login", "same-user")
        first_consented = await demo_app.ensure_service_memory_scope(first, True)
        second_consented = await demo_app.ensure_service_memory_scope(second, True)
        return first, second, first_consented, second_consented

    first, second, first_consented, second_consented = asyncio.run(scenario())
    assert first["session_id"] != second["session_id"]
    assert first["persistence_scope"] == "session"
    assert second["persistence_scope"] == "session"
    assert first_consented["session_id"] == second_consented["session_id"]
    assert first_consented["persistence_scope"] == "user"


def test_report_reset_does_not_clear_other_structured_counseling_memory():
    async def scenario():
        store = MemoryStore()
        store._reflection_entries["session-1"].append(
            ReflectionMemoryEntry(intent_label="ANXIETY_SUPPORT", emotion_hint="불안")
        )
        await store.set_session_metadata("session-1", "structured_marker", True)
        await store.clear_reflection_memory("session-1")
        return (
            await store.get_reflection_history("session-1"),
            await store.get_session_metadata("session-1", "structured_marker"),
        )

    report_history, counseling_marker = asyncio.run(scenario())
    assert report_history == []
    assert counseling_marker is True


def test_nearby_resources_require_explicit_location_consent():
    denied = demo_app.nearby_resource_placeholder("서울시 마포구", False)
    allowed = demo_app.nearby_resource_placeholder("서울시 마포구", True)

    expected = "위치 기반 추천은 사용자가 명시적으로 동의한 경우에만 사용할 수 있으며"
    assert expected in denied
    assert expected in allowed
    assert "지역 입력 기반 안내" in allowed


def test_report_maps_internal_labels_to_user_facing_korean():
    rendered = demo_app.render_reflection_report(
        {
            "has_history": True,
            "current_emotional_state": "불안",
            "main_issue": ["work_or_study_stress", "specific_academic_burden"],
            "repeated_themes": ["self_blame", "other_concern"],
            "last_small_action": "개념 세 개 정리하기",
            "action_status": "일부 진행됨",
            "next_follow_up": "어떤 부분이 어려웠나요?",
            "risk_stage": "관심",
            "long_term_trend": "조금씩 정리하는 흐름입니다.",
            "metrics": {
                "recent_counseling_count": 2,
                "main_topic_count": 4,
                "recent_risk_signal": False,
                "action_status": "일부 진행됨",
                "top_repeated_themes": ["academic_pressure", "crisis_safety"],
            },
        }
    )

    for internal in (
        "work_or_study_stress", "specific_academic_burden", "self_blame",
        "other_concern", "academic_pressure", "crisis_safety",
    ):
        assert internal not in rendered
    for display in ("학업/업무 부담", "구체적인 학업 부담", "자기비난/자책", "기타 고민", "학업 부담", "위기 신호"):
        assert display in rendered


def test_expert_contact_cards_include_all_required_numbers():
    contacts = demo_app.build_expert_contacts_markdown()

    for title, number in (
        ("자살예방 상담전화", "109"),
        ("청소년 상담", "1388"),
        ("긴급 신고", "112"),
        ("응급 상황", "119"),
    ):
        assert title in contacts
        assert number in contacts


def test_my_info_guidance_changes_by_mode_and_consent():
    anonymous = demo_app.privacy_settings_markdown({"mode": "anonymous"}, False)
    saving = demo_app.privacy_settings_markdown({"mode": "logged_in"}, True)
    not_saving = demo_app.privacy_settings_markdown({"mode": "logged_in"}, False)

    assert "임시 세션에서만" in anonymous
    assert "이 계정의 마음 리포트" in saving
    assert "장기 리포트에는 저장되지 않습니다" in not_saving


def test_main_header_shows_logged_in_or_anonymous_badge():
    logged_in = demo_app.service_header_markdown(
        {"mode": "logged_in", "user_id": "user-a", "nickname": "마음친구"}
    )
    anonymous = demo_app.service_header_markdown({"mode": "anonymous"})

    assert "로그인: 마음친구" in logged_in
    assert "익명 모드" in anonymous


def test_recent_trend_needs_at_least_two_snapshots(monkeypatch):
    monkeypatch.setattr(demo_app, "last_agent_result", None)
    snapshots = [{"risk_stage": "관심", "anxiety": 0.5, "stress": 0.5}]

    report = demo_app.build_service_report({}, {}, snapshots)

    assert "최근 변화 방향" in report
    assert demo_app.TREND_INSUFFICIENT_MESSAGE in report


def test_recent_trend_detects_decreasing_burden():
    trend = demo_app.build_recent_trend_markdown(
        [
            {"risk_stage": "위험", "anxiety": 1.0, "stress": 1.0, "sleep": 0.0, "energy": 0.0},
            {"risk_stage": "관심", "anxiety": 0.2, "stress": 0.2, "sleep": 0.8, "energy": 0.8, "action_status": "completed"},
        ]
    )

    assert "회복 방향으로 움직이는 흐름" in trend


def test_recent_trend_detects_increasing_burden():
    trend = demo_app.build_recent_trend_markdown(
        [
            {"risk_stage": "관심", "anxiety": 0.2, "stress": 0.2, "sleep": 0.8, "energy": 0.8},
            {"risk_stage": "위험", "anxiety": 1.0, "stress": 1.0, "sleep": 0.0, "energy": 0.0, "action_status": "not_done"},
        ]
    )

    assert "안정화와 도움 요청을 우선" in trend


def test_recent_trend_detects_similar_burden():
    snapshot = {
        "risk_stage": "주의",
        "anxiety": 0.5,
        "stress": 0.5,
        "sleep": 0.5,
        "energy": 0.5,
    }

    trend = demo_app.build_recent_trend_markdown([snapshot, dict(snapshot)])

    assert "비슷하게 유지" in trend


def test_chat_appends_bounded_structured_trend_snapshot(monkeypatch):
    async def keep_context(context, save_consent):
        return dict(context or {})

    async def fake_chat_ui(*args, **kwargs):
        summary = {"risk_stage": "주의", "wellness_checkin": {}}
        return [], [], "", "", summary, "응답이 준비됐어요."

    monkeypatch.setattr(demo_app, "ensure_service_memory_scope", keep_context)
    monkeypatch.setattr(demo_app, "handle_chat_ui", fake_chat_ui)
    monkeypatch.setattr(
        demo_app,
        "last_agent_result",
        {
            "risk_stage": "주의",
            "pipeline_details": {
                "agents": {
                    "intent": {"primary_intent": "ANXIETY_SUPPORT"},
                    "cause_exploration": {"selected_cause": "worry_or_anxiety"},
                    "action_checkin": {"status": "partial"},
                    "emotional_state": {
                        "anxiety": 0.7,
                        "stress": 0.6,
                        "sleep": 0.4,
                        "energy": 0.3,
                    },
                }
            },
        },
    )
    existing = [{"risk_stage": "관심"} for _ in range(10)]

    outputs = asyncio.run(
        demo_app.handle_chat_ui_for_service(
            "structured only",
            [],
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            {"trend_snapshots": existing},
            False,
        )
    )
    snapshot = outputs[-1]["trend_snapshots"][-1]

    assert len(outputs[-1]["trend_snapshots"]) == demo_app.TREND_SNAPSHOT_LIMIT
    assert snapshot == {
        "risk_stage": "주의",
        "primary_intent": "ANXIETY_SUPPORT",
        "selected_cause": "worry_or_anxiety",
        "action_status": "partial",
        "anxiety": 0.7,
        "stress": 0.6,
        "sleep": 0.4,
        "energy": 0.3,
    }


def test_report_button_uses_only_context_snapshots_and_returns_immediately(monkeypatch):
    class ForbiddenStore:
        def __getattr__(self, name):
            raise AssertionError(f"memory store must not be accessed: {name}")

    async def forbidden_report(*args, **kwargs):
        raise AssertionError("memory-backed report must not be called")

    def forbidden_builder(*args, **kwargs):
        raise AssertionError("reflection report builder must not be called")

    monkeypatch.setattr(demo_app, "agent", SimpleNamespace(memory_store=ForbiddenStore()))
    monkeypatch.setattr(demo_app, "last_agent_result", None)
    monkeypatch.setattr(demo_app, "build_chat_reflection_report", forbidden_report)
    monkeypatch.setattr(demo_app, "build_reflection_report", forbidden_builder)
    context = {
        "trend_snapshots": [
            {"risk_stage": "주의", "anxiety": 0.8, "stress": 0.8},
            {"risk_stage": "관심", "anxiety": 0.3, "stress": 0.3, "action_status": "completed"},
        ]
    }

    started = time.perf_counter()
    outputs = asyncio.run(
        demo_app.open_reflection_report_for_service({}, context, False)
    )

    assert time.perf_counter() - started < 0.5
    assert len(outputs) == 3
    assert "최근 변화 방향" in outputs[0]
    assert "회복 방향으로 움직이는 흐름" in outputs[0]


def test_logged_in_report_snapshots_persist_in_demo_cache(monkeypatch, tmp_path):
    path = tmp_path / "demo-cache.json"
    monkeypatch.setattr(demo_app, "DEMO_REPORT_CACHE_PATH", path)

    demo_app.append_demo_report_snapshot(
        "saved-user",
        {"risk_stage": "관심", "primary_intent": "ANXIETY_SUPPORT"},
    )

    snapshots = demo_app.get_demo_report_snapshots("saved-user")
    assert snapshots[-1]["intent_label"] == "ANXIETY_SUPPORT"


def test_anonymous_snapshot_is_not_persisted(monkeypatch, tmp_path):
    path = tmp_path / "demo-cache.json"
    monkeypatch.setattr(demo_app, "DEMO_REPORT_CACHE_PATH", path)

    assert demo_app.get_demo_report_snapshots("") == []
    assert not path.exists()


def test_demo_report_cache_excludes_raw_conversation_text(monkeypatch, tmp_path):
    path = tmp_path / "demo-cache.json"
    monkeypatch.setattr(demo_app, "DEMO_REPORT_CACHE_PATH", path)
    demo_app.append_demo_report_snapshot(
        "safe-user",
        {
            "primary_intent": "ANXIETY_SUPPORT",
            "raw_text": "SECRET RAW CONVERSATION",
            "assistant_response": "FULL RESPONSE",
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "SECRET RAW CONVERSATION" not in serialized
    assert "FULL RESPONSE" not in serialized
    assert "raw_text" not in serialized


def test_login_restore_returns_short_summary_and_sanitized_context(monkeypatch, tmp_path):
    path = tmp_path / "demo-cache.json"
    monkeypatch.setattr(demo_app, "DEMO_REPORT_CACHE_PATH", path)
    demo_app.append_demo_report_snapshot(
        "returning-user",
        {"intent_label": "WORK_OR_STUDY_STRESS", "risk_stage": "관심"},
    )
    context = {
        "mode": "logged_in",
        "user_id": "returning-user",
        "session_id": "current",
        "record_saving_enabled": True,
    }

    restored_context, messages, _, restored = asyncio.run(
        demo_app.restore_logged_in_structured_memory(context)
    )

    assert restored is True
    assert restored_context["restored_report_snapshots"]
    assert "지난 상담에서는" in messages[0]["content"]
    assert "요즘은 어떤가요?" in messages[0]["content"]
    assert len(messages[0]["content"]) < 120


def test_login_restore_cache_failure_does_not_block(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_app, "DEMO_REPORT_CACHE_PATH", tmp_path)
    context = {
        "mode": "logged_in",
        "user_id": "slow-user",
        "record_saving_enabled": True,
    }
    started = time.perf_counter()
    restored_context, messages, _, restored = asyncio.run(
        demo_app.restore_logged_in_structured_memory(context)
    )

    assert time.perf_counter() - started < 0.5
    assert restored is False
    assert restored_context == context
    assert messages == demo_app.initial_chat_messages()


def test_report_button_uses_restored_records_for_trend_without_memory(monkeypatch):
    class ForbiddenStore:
        def __getattr__(self, name):
            raise AssertionError(f"memory store must not be accessed: {name}")

    monkeypatch.setattr(demo_app, "agent", SimpleNamespace(memory_store=ForbiddenStore()))
    monkeypatch.setattr(demo_app, "last_agent_result", None)
    context = {
        "restored_report_snapshots": [
            {
                "intent_label": "ANXIETY_SUPPORT",
                "anxiety": 0.8,
                "action_status": "not_done",
                "risk_stage": "주의",
            },
            {
                "intent_label": "ANXIETY_SUPPORT",
                "anxiety": 0.2,
                "action_status": "completed",
                "risk_stage": "관심",
            },
        ]
    }
    started = time.perf_counter()
    report, _, _ = asyncio.run(
        demo_app.open_reflection_report_for_service({}, context, False)
    )

    assert time.perf_counter() - started < 0.5
    assert "최근 변화 방향" in report
    assert "회복 방향으로 움직이는 흐름" in report
    assert "이전 상담 흐름" in report
