"""Tests for structured cause exploration in MOCK responses."""

import asyncio

from src.main import AgentConfig, PsychologistAgent


async def _run_turns(messages, wellness_checkin=None):
    agent = PsychologistAgent(
        config=AgentConfig(
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=True,
    )
    await agent.initialize()
    session = await agent.session_manager.create_session()
    try:
        result = None
        for message in messages:
            result = await agent.process_message(
                message,
                session.session_id,
                wellness_checkin=wellness_checkin,
            )
        return result
    finally:
        await agent.shutdown()


def test_sleep_input_adds_cause_candidates_and_question():
    result = asyncio.run(_run_turns(["요즘 잠을 잘 못 자"]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    response = result["response"]

    assert "worry_or_anxiety" in cause["cause_candidates"]
    assert "sleep_maintenance" in cause["cause_candidates"]
    assert cause["selected_cause"] in cause["cause_candidates"]
    assert "같이" in response
    assert "편인가요" in response


def test_dataset_metadata_influences_cause_selection():
    result = asyncio.run(
        _run_turns(
            ["요즘 잠을 잘 못 자"],
            wellness_checkin={
                "mood_score": 3,
                "anxiety_score": 4,
                "loneliness_score": 3,
                "sleep_quality": 2,
                "meal_status": 3,
                "energy_score": 3,
                "stress_score": 3,
            },
        )
    )

    cause = result["pipeline_details"]["agents"]["cause_exploration"]

    assert cause["dataset_signals"]["counseling_category"]
    assert cause["dataset_signals"]["empathy_emotion"] or cause["dataset_signals"]["wellness_topic"]
    assert any(code.startswith("dataset_") for code in cause["reason_codes"])


def test_anxiety_response_asks_about_trigger_context_without_diagnosis():
    result = asyncio.run(_run_turns(["요즘 계속 불안해"]))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]

    assert cause["selected_cause"] in {
        "task_pressure",
        "future_uncertainty",
        "accumulated_fatigue",
        "relationship_stress",
    }
    assert "불안이" in response
    assert "상황" in response or "이유 없이" in response
    assert "원인은" not in response
    assert "때문입니다" not in response


def test_sleep_followup_answer_does_not_repeat_same_question():
    first = asyncio.run(_run_turns(["요즘 잠을 잘 못 자"]))
    second = asyncio.run(_run_turns(["요즘 잠을 잘 못 자", "자주 깨는 편이야"]))

    first_question = first["pipeline_details"]["agents"]["followup"]["question"]
    second_question = second["pipeline_details"]["agents"]["followup"]["question"]
    second_response = second["response"]

    assert second_question != first_question
    assert "자주 깨는 편" in second_response or "중간에 자주 깨는" in second_response
    assert "깨고 난 뒤" in second_response


def test_sleep_followup_answer_generates_deeper_question():
    result = asyncio.run(
        _run_turns(
            [
                "요즘 잠을 잘 못 자",
                "자주 깨는 편이야",
                "깨고 난 뒤 걱정이 떠올라",
            ]
        )
    )

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]

    assert cause["selected_cause"] == "sleep_maintenance"
    assert "previous_followup_answer_detected" in cause["reason_codes"]
    assert "deeper_exploration_question_selected" in cause["reason_codes"]
    assert "깨고 난 뒤 걱정이 떠올라" not in str(cause)
    assert "가장 먼저 떠오르는 생각" in followup


def test_bedtime_worry_followup_answer_narrows_worry_cause():
    result = asyncio.run(_run_turns(["요즘 잠을 잘 못 자", "잠들기 전 걱정이 많아져"]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]

    assert cause["selected_cause"] == "worry_or_anxiety"
    assert "previous_followup_narrows_worry_or_anxiety" in cause["reason_codes"]
    assert "오늘 있었던 일" in followup


def test_sleep_maintenance_markers_boost_selected_cause():
    result = asyncio.run(_run_turns(["잠자다가 눈이 떠져서 숙면이 안 돼"]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    response = result["response"]

    assert cause["selected_cause"] == "sleep_maintenance"
    assert "followup_answer_sleep_maintenance" in cause["reason_codes"]
    assert "자꾸 깨는 밤" in response


def test_academic_relief_markers_take_priority_over_exam_pressure():
    result = asyncio.run(_run_turns(["드디어 시험이 1개 남아서 기뻐\n숨통이 풀리기 시작한다."]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    response = result["response"]

    assert cause["selected_cause"] == "academic_relief"
    assert "input_academic_relief_signal" in cause["reason_codes"]
    assert "exam_assignment_pressure" not in cause["cause_candidates"]
    assert "공부나 시험, 과제 부담" not in response
    assert "숨이 조금 트" in response or "끝이 보" in response


def test_mixed_academic_concern_uses_previous_academic_context():
    result = asyncio.run(
        _run_turns(
            [
                "드디어 시험이 1개 남아서 기뻐\n숨통이 풀리기 시작한다.",
                "근데 인공지능 과목이라 조금은 어려워 암기를 많이 해야할 것 같은데 주말에 시간이 있으니 괜찮을 것 같아",
            ]
        )
    )

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    response = result["response"]

    assert cause["selected_cause"] == "manageable_academic_concern"
    assert "input_manageable_academic_concern_signal" in cause["reason_codes"]
    assert "previous_academic_context_preserved" in cause["reason_codes"]
    assert "공부나 시험, 과제 부담" not in response
    assert "인공지능" in response or "암기" in response


def test_specific_academic_burden_after_relief_preserves_context():
    result = asyncio.run(
        _run_turns(
            [
                "다음 주 시험이지만 이제 종강해서 기뻐",
                "부담은 크기 왜냐면 인공지능 과목이라 암기할 것이 많아",
            ]
        )
    )

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "specific_academic_burden_after_relief"
    assert "input_specific_academic_burden_signal" in cause["reason_codes"]
    assert "previous_academic_relief_context_preserved" in cause["reason_codes"]


def test_specific_academic_burden_without_history_beats_broad_pressure():
    result = asyncio.run(
        _run_turns(["인공지능 시험 때문에 암기할 게 너무 많아서 막막해"])
    )

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    response = result["response"]
    assert cause["selected_cause"] == "specific_academic_burden"
    assert "input_specific_academic_burden_signal" in cause["reason_codes"]
    assert "공부나 시험, 과제 부담" not in response


def test_self_judgment_marker_selects_self_blame():
    result = asyncio.run(_run_turns(["내가 너무 한심해"]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "self_blame"
    assert "input_self_blame_signal" in cause["reason_codes"]


def test_cause_exploration_does_not_store_raw_user_input():
    raw_message = "요즘 잠을 잘 못 자"
    result = asyncio.run(_run_turns([raw_message]))

    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    joined = "\n".join(str(value) for value in cause.values())

    assert raw_message not in joined
    assert "user_input" not in joined
    assert "raw_text" not in joined


def test_dataset_hints_are_not_exposed_as_response_text():
    result = asyncio.run(_run_turns(["요즘 잠을 잘 못 자"]))

    response = result["response"]

    assert "intervention_hint" not in response
    assert "empathy_style_hint" not in response
    assert "support_hint" not in response
    assert "상담 참고" not in response
    assert "공감 참고" not in response
    assert "웰니스 참고" not in response
