"""Tests for the Gradio chat UI message contract."""

import asyncio
import inspect
from types import SimpleNamespace

from demo import app as demo_app


def test_normalize_chat_history_converts_tuple_history_to_messages():
    history = demo_app.normalize_chat_history([("안녕하세요", "반가워요")])

    assert history == [
        {"role": "user", "content": "안녕하세요"},
        {"role": "assistant", "content": "반가워요"},
    ]


def test_initial_chat_messages_has_assistant_greeting():
    history = demo_app.initial_chat_messages()

    assert history == [
        {
            "role": "assistant",
            "content": "안녕하세요. 오늘 마음 상태는 어떤가요? 편하게 한 문장으로 이야기해도 괜찮아요.",
        }
    ]
    assert history is not demo_app.INITIAL_CHAT_HISTORY


def test_handle_chat_ui_returns_messages_format(monkeypatch):
    async def fake_handle_chat(*args, **kwargs):
        return (
            "수면과 불안이 겹쳐서 많이 긴장됐을 수 있어요.",
            {
                "risk_stage": "관심",
                "response_preview": "수면과 불안이 겹쳐서 많이 긴장됐을 수 있어요.",
            },
        )

    monkeypatch.setattr(demo_app, "handle_chat", fake_handle_chat)
    demo_app.last_agent_result = {
        "pipeline_details": {
            "agents": {
                "safety": {"risk_stage": "관심"},
                "response": {"response_generated": True},
            }
        }
    }

    history, state_history, cleared_message, pipeline, summary, status = asyncio.run(
        demo_app.handle_chat_ui("요즘 잠을 못 자고 불안해요", [], 3, 3, 3, 3, 3, 3, 3)
    )

    assert history == [
        {
            "role": "assistant",
            "content": "안녕하세요. 오늘 마음 상태는 어떤가요? 편하게 한 문장으로 이야기해도 괜찮아요.",
        },
        {"role": "user", "content": "요즘 잠을 못 자고 불안해요"},
        {"role": "assistant", "content": "수면과 불안이 겹쳐서 많이 긴장됐을 수 있어요."},
    ]
    assert all(isinstance(message, dict) for message in history)
    assert all(set(message) == {"role", "content"} for message in history)
    assert all(message["role"] in {"user", "assistant"} for message in history)
    assert state_history == history
    assert cleared_message == ""
    assert "Agent Pipeline View" in pipeline
    assert summary["risk_stage"] == "관심"
    assert status == "응답이 준비됐어요."


def test_handle_chat_ui_does_not_return_tuple_history(monkeypatch):
    async def fake_handle_chat(*args, **kwargs):
        return (
            "괜찮아요. 오늘은 호흡을 조금 늦춰볼게요.",
            {
                "risk_stage": "관심",
                "response_preview": "괜찮아요. 오늘은 호흡을 조금 늦춰볼게요.",
            },
        )

    monkeypatch.setattr(demo_app, "handle_chat", fake_handle_chat)
    demo_app.last_agent_result = None

    history, state_history, *_ = asyncio.run(
        demo_app.handle_chat_ui("불안해요", [("이전 말", "이전 답")], 3, 3, 3, 3, 3, 3, 3)
    )

    assert history == [
        {"role": "user", "content": "이전 말"},
        {"role": "assistant", "content": "이전 답"},
        {"role": "user", "content": "불안해요"},
        {"role": "assistant", "content": "괜찮아요. 오늘은 호흡을 조금 늦춰볼게요."},
    ]
    assert state_history == history
    assert not any(isinstance(message, tuple) for message in history)


def test_handle_chat_ui_preserves_initial_greeting_and_appends_turns(monkeypatch):
    async def fake_handle_chat(message, *args, **kwargs):
        return (
            f"{message}에 대한 응답",
            {"risk_stage": "관심", "response_preview": f"{message}에 대한 응답"},
        )

    monkeypatch.setattr(demo_app, "handle_chat", fake_handle_chat)
    demo_app.last_agent_result = None

    first_history, first_state_history, *_ = asyncio.run(
        demo_app.handle_chat_ui("첫 번째 말", [], 3, 3, 3, 3, 3, 3, 3)
    )
    second_history, second_state_history, *_ = asyncio.run(
        demo_app.handle_chat_ui("두 번째 말", first_state_history, 3, 3, 3, 3, 3, 3, 3)
    )

    assert len(first_history) == len(demo_app.INITIAL_CHAT_HISTORY) + 2
    assert len(second_history) == len(first_history) + 2
    assert second_history[0] == demo_app.INITIAL_CHAT_HISTORY[0]
    assert {"role": "user", "content": "첫 번째 말"} in second_history
    assert {"role": "assistant", "content": "첫 번째 말에 대한 응답"} in second_history
    assert second_history[-2:] == [
        {"role": "user", "content": "두 번째 말"},
        {"role": "assistant", "content": "두 번째 말에 대한 응답"},
    ]
    assert first_state_history == first_history
    assert second_state_history == second_history


def test_handle_chat_ui_does_not_overwrite_with_initial_history_each_send(monkeypatch):
    responses = {
        "첫 질문": "첫 답변",
        "둘째 질문": "둘째 답변",
    }

    async def fake_handle_chat(message, *args, **kwargs):
        return responses[message], {"risk_stage": "관심", "response_preview": responses[message]}

    monkeypatch.setattr(demo_app, "handle_chat", fake_handle_chat)
    demo_app.last_agent_result = None

    first_display, first_state, *_ = asyncio.run(
        demo_app.handle_chat_ui("첫 질문", None, 3, 3, 3, 3, 3, 3, 3)
    )
    second_display, second_state, *_ = asyncio.run(
        demo_app.handle_chat_ui("둘째 질문", first_state, 3, 3, 3, 3, 3, 3, 3)
    )

    assert first_display == first_state
    assert second_display == second_state
    assert len(second_state) == len(demo_app.initial_chat_messages()) + 4
    assert second_state == [
        demo_app.initial_chat_messages()[0],
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
        {"role": "user", "content": "둘째 질문"},
        {"role": "assistant", "content": "둘째 답변"},
    ]


def test_handle_chat_ui_empty_message_returns_existing_history(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("handle_chat should not be called for empty messages")

    existing_history = [{"role": "user", "content": "이미 보낸 말"}]
    monkeypatch.setattr(demo_app, "handle_chat", fail_if_called)

    history, state_history, cleared_message, pipeline, summary, status = asyncio.run(
        demo_app.handle_chat_ui("   ", existing_history, 3, 3, 3, 3, 3, 3, 3)
    )

    assert history == existing_history
    assert state_history == existing_history
    assert cleared_message == ""
    assert pipeline == ""
    assert summary == {"empty_message": True}
    assert status == ""


def test_reset_chat_history_is_the_only_initializer():
    display_history, state_history = demo_app.reset_chat_history()

    assert display_history == demo_app.initial_chat_messages()
    assert state_history == demo_app.initial_chat_messages()
    assert display_history is not state_history


def test_create_demo_does_not_pass_unsupported_chatbot_type_argument():
    source = inspect.getsource(demo_app.create_demo)

    assert "gr.Chatbot" in source
    assert "type" not in source


def test_initial_greeting_is_assistant_message_format():
    assert demo_app.INITIAL_CHAT_HISTORY == [
        {
            "role": "assistant",
            "content": "안녕하세요. 오늘 마음 상태는 어떤가요? 편하게 한 문장으로 이야기해도 괜찮아요.",
        }
    ]


def test_create_demo_updates_chatbot_and_state_from_send_handler():
    source = inspect.getsource(demo_app.create_demo)

    assert "chat_history_state = gr.State(initial_chat_messages())" in source
    assert "inputs=[\n                message,\n                chat_history_state," in source
    assert "outputs=[\n                chatbot,\n                chat_history_state," in source
    assert "outputs=chat_state" not in source


def test_anonymous_checkbox_is_interactive_and_not_fixed():
    source = inspect.getsource(demo_app.create_demo)

    assert "익명으로 시작하기" in source
    assert "interactive=True" in source
    assert "닉네임" in source
    assert "기록 저장 동의" in source
    assert "toggle_nickname_input" in source


def test_chatbot_has_fixed_scrollable_height_without_duplicate_chat_buttons():
    source = inspect.getsource(demo_app.create_demo)

    assert "height=460" in source
    assert "overflow-y:auto" in source
    assert "외로움 예시" not in source
    assert "불안 예시" not in source
    assert "위험 예시" not in source
    assert 'gr.Button("상태 체크하기"' not in source
    assert 'gr.Button("감정일기 쓰기"' not in source
    assert 'gr.Button("마음정리 보고서 보기"' not in source
    assert 'gr.Button("전문가 상담 연결"' not in source


def test_nickname_toggle_disables_input_when_anonymous():
    anonymous_update = demo_app.toggle_nickname_input(True)
    named_update = demo_app.toggle_nickname_input(False)

    if isinstance(anonymous_update, dict):
        assert anonymous_update["interactive"] is False
        assert named_update["interactive"] is True
    else:
        assert getattr(anonymous_update, "interactive", False) is False
        assert getattr(named_update, "interactive", True) is True


def test_service_report_deduplicates_and_translates_intent_labels(monkeypatch):
    monkeypatch.setattr(
        demo_app,
        "last_agent_result",
        {
            "risk_stage": "관심",
            "pipeline_details": {
                "agents": {
                    "intent": {
                        "primary_intent": "SLEEP_PROBLEM",
                        "labels": ["SLEEP_PROBLEM", "SLEEP_PROBLEM", "ANXIETY_SUPPORT"],
                    },
                    "emotional_state": {"state_summary": ["불안"]},
                    "followup": {"question": "잠드는 데 오래 걸리는 편인가요?"},
                    "small_action": {"action_text": "잠들기 전 화면 밝기를 낮춰보세요."},
                }
            },
        },
    )

    report = demo_app.build_service_report({"risk_stage": "관심"}, {})

    assert "수면 문제" in report
    assert "불안" in report
    assert "SLEEP_PROBLEM" not in report
    assert report.count("수면 문제") == 1


def test_service_report_translates_other_concern_without_raw_label(monkeypatch):
    monkeypatch.setattr(
        demo_app,
        "last_agent_result",
        {
            "risk_stage": "관심",
            "pipeline_details": {
                "agents": {
                    "intent": {
                        "primary_intent": "SLEEP_PROBLEM",
                        "labels": ["SLEEP_PROBLEM", "SLEEP_PROBLEM", "OTHER_CONCERN"],
                    },
                    "emotional_state": {"state_summary": ["불안"]},
                }
            },
        },
    )

    report = demo_app.build_service_report({"risk_stage": "관심"}, {})

    assert "수면 문제" in report
    assert "기타 고민" in report
    assert "OTHER_CONCERN" not in report
    assert report.count("수면 문제") == 1


def test_expert_guidance_is_soft_for_attention_and_caution():
    assert "필요하면 가까운 사람이나 상담센터" in demo_app._expert_guidance_for_stage("관심")
    assert "필요하면 가까운 사람이나 상담센터" in demo_app._expert_guidance_for_stage("주의")
    assert "즉시 109, 119, 112" in demo_app._expert_guidance_for_stage("위험")


def test_get_agent_caches_initialized_instance(monkeypatch):
    class FakeSessionManager:
        async def create_session(self):
            return SimpleNamespace(session_id="session-1")

    class FakeAgent:
        created = 0
        initialized = 0

        def __init__(self):
            FakeAgent.created += 1
            self.session_manager = FakeSessionManager()

        async def initialize(self):
            FakeAgent.initialized += 1

    async def scenario():
        demo_app.agent = None
        demo_app.current_session_id = None
        monkeypatch.setattr("src.main.PsychologistAgent", FakeAgent)
        first = await demo_app.get_agent()
        second = await demo_app.get_agent()
        return first, second

    first, second = asyncio.run(scenario())

    assert first is second
    assert FakeAgent.created == 1
    assert FakeAgent.initialized == 1
    demo_app.agent = None
    demo_app.current_session_id = None
