"""
Response quality tests for deterministic MOCK-mode agent responses.
"""

import asyncio
from types import SimpleNamespace

from src.main import AgentConfig, PsychologistAgent
from src.agent.action_checkin import classify_action_checkin


SAFETY_NOTICE_MARKER = "이 AI는 의료 진단이나 치료"
RAW_USER_INPUT = "내 비밀 원문 ABC123을 그대로 말하지 마"
RAW_DATASET_TEXT = "상담 참고: 원본 데이터셋 문장을 그대로 노출하면 안 됩니다"
RAW_MEMORY_TEXT = "raw memory transcript should never appear"
NON_ACTION_HINT = "상담 참고: 기분이 우울하시군요. 공감 참고: 그대로 말하세요."
ACTION_MARKERS = (
    "보세요",
    "해보세요",
    "정해",
    "적어",
    "느껴",
    "마시",
    "낮춰",
    "내려놓",
)


async def _run_message(message, *, counseling_hint=NON_ACTION_HINT, empathy_hint="공감 참고: 감정을 먼저 확인하세요."):
    results = await _run_messages(
        [message],
        counseling_hint=counseling_hint,
        empathy_hint=empathy_hint,
    )
    return results[-1]


async def _run_messages(
    messages,
    *,
    counseling_hint=NON_ACTION_HINT,
    empathy_hint="공감 참고: 감정을 먼저 확인하세요.",
    include_reflection=False,
):
    agent = PsychologistAgent(
        config=AgentConfig(
            enable_rag=False,
            enable_audit_logging=False,
        ),
        mock_mode=True,
    )
    agent.counseling_retriever = SimpleNamespace(
        recommend=lambda _: SimpleNamespace(
            intervention_hint=counseling_hint,
            matched_record_id="counseling-test",
            category="test",
            score=1.0,
        )
    )
    agent.empathy_retriever = SimpleNamespace(
        recommend=lambda _: SimpleNamespace(
            empathy_style_hint=empathy_hint,
            emotion_label="",
            empathy_label="test",
            matched_record_id="empathy-test",
            score=1.0,
        )
    )
    agent.wellness_recommender = SimpleNamespace(recommend=lambda _: None)

    await agent.initialize()
    session = await agent.session_manager.create_session()
    try:
        results = []
        for message in messages:
            results.append(await agent.process_message(message, session.session_id))
        if include_reflection:
            reflection = await agent.memory_store.get_reflection_history(session.session_id)
            return results, reflection
        return results
    finally:
        await agent.shutdown()


def _flatten_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_strings(item)
    elif isinstance(value, str):
        yield value


def test_sleep_problem_response_includes_sleep_followup():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    followup = result["pipeline_details"]["agents"]["followup"]["question"]
    assert "편인가요" in followup
    assert followup in result["response"]
    assert "원인을 단정" in result["response"]


def test_sleep_problem_response_does_not_mix_low_mood_sentence():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    assert "기분이 우울" not in result["response"]
    assert "우울하시군요" not in result["response"]


def test_internal_counseling_instruction_is_not_exposed():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            counseling_hint="내담자의 표현을 반영하고 핵심 감정을 명료화하세요.",
        )
    )

    assert "내담자의 표현을 반영" not in result["response"]
    assert "핵심 감정을 명료화" not in result["response"]


def test_low_mood_response_centers_low_energy_empathy():
    result = asyncio.run(_run_message("요즘 너무 무기력하고 아무 기운이 없어요"))

    assert "무기력" in result["response"] or "기운이 없는" in result["response"]
    assert "소진" in result["response"]


def test_need_empathy_response_centers_empathy_over_advice():
    result = asyncio.run(_run_message("조언보다 그냥 들어주고 공감해줬으면 해요"))

    response = result["response"]
    assert "해결책을 서둘러" in response
    assert "판단하거나 몰아붙이지" in response
    assert "작은 실행 단계" not in response


def test_need_advice_response_centers_small_execution_step():
    result = asyncio.run(_run_message("어떻게 해야 할지 방법을 알려줘"))

    response = result["response"]
    assert "작은 실행 단계" in response or "가장 작은" in response
    assert any(marker in response for marker in ACTION_MARKERS)


def test_chest_tightness_and_sadness_response_is_specific():
    result = asyncio.run(_run_message("가슴이 답답하고 속상해"))

    response = result["response"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "physical_tension_chest"
    assert any(marker in response for marker in ("답답", "긴장", "몸"))
    assert "진단" in response or "단정" in response
    assert "몸의 긴장" in followup or "속상함" in followup
    assert followup in response
    assert "어깨" in response or "호흡" in response


def test_crying_urge_response_is_specific_and_not_same_generic_response():
    chest_result = asyncio.run(_run_message("가슴이 답답하고 속상해"))
    crying_result = asyncio.run(_run_message("그냥 울고 싶어"))

    response = crying_result["response"]
    followup = crying_result["pipeline_details"]["agents"]["followup"]["question"]
    cause = crying_result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "crying_urge"
    assert any(marker in response for marker in ("울", "눈물", "감정"))
    assert "특정한 일" in response or "피로" in response
    assert "특정한 일" in followup or "쌓인" in followup
    assert followup in response
    assert "물" in response or "안전한" in response
    assert response != chest_result["response"]
    assert "지금 느끼는 부담이 꽤 컸을 것 같아요" not in response


def test_authority_criticism_response_is_specific():
    result = asyncio.run(_run_message("교수님한테 발표를 심하게 지적받아서 너무 위축돼"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "authority_criticism"
    assert any(marker in response for marker in ("교수님", "선생님", "상사", "비판", "지적"))
    assert any(marker in response for marker in ("창피", "화", "마음에 남"))
    assert all(marker in response for marker in ("내용", "말투", "평가"))


def test_exam_assignment_pressure_response_is_specific():
    result = asyncio.run(_run_message("시험이랑 과제 마감 때문에 너무 압박감이 심해"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "exam_assignment_pressure"
    assert any(marker in response for marker in ("시험", "과제", "마감", "압박"))
    assert "10분" in response or "첫 단계" in response


def test_first_academic_stress_input_produces_academic_response():
    result = asyncio.run(_run_message("공부하느라 너무 힘들어"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "exam_assignment_pressure"
    assert "공부" in response or "시험" in response or "과제" in response
    assert "압박" in response or "막막" in response
    assert all(marker in response for marker in ("공부량", "성적", "마감"))
    assert "지쳐서" in response or "버티기 어려운" in response
    assert "원인을 단정" in response or "어디에 가장 가까울까요" in response


def test_negative_academic_stress_still_triggers_pressure_response():
    for message in (
        "시험 때문에 너무 부담돼",
        "암기할 게 너무 많아서 못 하겠어",
        "과제 마감 때문에 미치겠어",
        "기말 준비랑 과제가 겹쳐서 힘들어",
    ):
        result = asyncio.run(_run_message(message))
        response = result["response"]
        cause = result["pipeline_details"]["agents"]["cause_exploration"]
        assert cause["selected_cause"] == "exam_assignment_pressure"
        assert "공부 때문에 많이 지쳐" in response
        assert "네가 약해서 그런 게 아니야" in response


def test_academic_stress_response_includes_empathy_and_cause_exploration():
    result = asyncio.run(_run_message("기말 시험이랑 과제 성적 때문에 너무 힘들어"))

    response = result["response"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]
    assert "압박" in response or "힘들" in response or "막막" in response
    assert all(marker in response for marker in ("공부량", "성적", "마감"))
    assert "지쳐서" in response or "피로" in response or "버티기 어려운" in response
    assert followup and followup in response
    assert "10분" in response or "첫 단계" in response


def test_positive_academic_relief_does_not_trigger_pressure_response():
    result = asyncio.run(_run_message("드디어 시험이 1개 남아서 기뻐\n숨통이 풀리기 시작한다."))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "academic_relief"
    assert "input_academic_relief_signal" in cause["reason_codes"]
    assert "공부나 시험, 과제 부담" not in response
    assert "압박이 꽤 크게 쌓인 상태" not in response
    assert "지금 부담은 공부량" not in response
    assert any(marker in response for marker in ("숨이 조금 트", "기쁜", "후련", "끝이 보", "많이 버텼"))
    assert "남은 하나" in response or "남은 시험" in response
    assert "쉬" in response and ("3가지" in response or "시험 범위" in response)


def test_mixed_academic_concern_with_coping_confidence_preserves_context():
    _, followup = asyncio.run(
        _run_messages(
            [
                "드디어 시험이 1개 남아서 기뻐\n숨통이 풀리기 시작한다.",
                "근데 인공지능 과목이라 조금은 어려워 암기를 많이 해야할 것 같은데 주말에 시간이 있으니 괜찮을 것 같아",
            ]
        )
    )

    response = followup["response"]
    cause = followup["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "manageable_academic_concern"
    assert "input_manageable_academic_concern_signal" in cause["reason_codes"]
    assert "previous_academic_context_preserved" in cause["reason_codes"]
    assert all(
        phrase not in response
        for phrase in (
            "지금 느끼는 부담이 꽤 컸을 것 같아요",
            "많이 버텨온 마음이 보내는 신호",
            "오늘 할 수 있는 가장 작은 한 가지",
            "숨을 고르고",
        )
    )
    assert any(marker in response for marker in ("인공지능", "암기", "주말에 시간"))
    assert "완전히 막막" in response or "마무리할 수" in response
    assert all(marker in response for marker in ("개념 암기", "이해", "문제 풀이"))
    assert "개념 3개" in response and "체크리스트" in response


def test_specific_academic_burden_after_relief_uses_narrow_response():
    _, followup = asyncio.run(
        _run_messages(
            [
                "다음 주 시험이지만 이제 종강해서 기뻐",
                "부담은 크기 왜냐면 인공지능 과목이라 암기할 것이 많아",
            ]
        )
    )

    response = followup["response"]
    agents = followup["pipeline_details"]["agents"]
    assert agents["cause_exploration"]["selected_cause"] == "specific_academic_burden_after_relief"
    assert "인공지능" in response and "암기" in response
    assert "개념 정의 암기, 알고리즘 흐름 이해, 용어 구분 중 뭐가 제일 부담인가요?" in response
    assert "꼭 외워야 할 개념 5개" in response
    assert "아는 것과 모르는 것" in response
    assert all(
        phrase not in response
        for phrase in (
            "공부나 시험, 과제 부담을 말할 정도면",
            "공부량 자체, 성적 걱정, 마감 압박",
            "지금 부담은 공부량, 성적 걱정, 마감 압박",
            "타이머를 10분만 맞추고",
            "가장 작은 첫 단계 하나만",
        )
    )


def test_specific_academic_burden_without_history_uses_narrow_response():
    result = asyncio.run(
        _run_message("인공지능 시험 때문에 암기할 게 너무 많아서 막막해")
    )

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "specific_academic_burden"
    assert "인공지능 시험" in response and "암기" in response
    assert "개념 정의 암기, 알고리즘 흐름 이해, 용어 구분 중 뭐가 제일 막막한가요?" in response
    assert "꼭 외워야 할 개념 5개" in response
    assert "아는 것과 모르는 것" in response
    assert all(
        phrase not in response
        for phrase in (
            "공부나 시험, 과제 부담을 말할 정도면",
            "공부량 자체, 성적 걱정, 마감 압박",
            "지금 부담은 공부량, 성적 걱정, 마감 압박",
            "범위를 줄이는 게 먼저 필요할까요, 아니면 시작할 첫 단계",
            "과제 하나나 시험 범위 하나를 골라 첫 10분",
        )
    )


def test_crisis_overrides_specific_academic_burden_without_history():
    result = asyncio.run(
        _run_message("인공지능 시험 암기가 너무 많고 막막해서 죽고 싶어요.")
    )

    assert result["requires_crisis_response"] is True
    assert "개념 정의 암기" not in result["response"]
    assert "꼭 외워야 할 개념 5개" not in result["response"]


def test_crisis_still_overrides_specific_academic_burden_after_relief():
    _, crisis = asyncio.run(
        _run_messages(
            [
                "다음 주 시험이지만 이제 종강해서 기뻐",
                "인공지능 암기가 너무 많아서 죽고 싶어요.",
            ]
        )
    )

    assert crisis["requires_crisis_response"] is True
    assert "개념 정의 암기" not in crisis["response"]
    assert "꼭 외워야 할 개념 5개" not in crisis["response"]


def test_action_checkin_partial_progress_is_acknowledged_and_narrowed():
    results, reflection = asyncio.run(
        _run_messages(
            [
                "인공지능 시험 때문에 암기할 게 너무 많아서 막막해",
                "3개 정도 정리했어",
            ],
            include_reflection=True,
        )
    )
    first, checkin = results

    agents = checkin["pipeline_details"]["agents"]
    assert first["pipeline_details"]["agents"]["small_action"]["has_action"] is True
    assert agents["action_checkin"]["status"] == "partial"
    assert "조금이라도 해본 건" in checkin["response"]
    assert "가장 헷갈리거나 막히는 건 뭐였어?" in checkin["response"]
    assert "물음표 하나" in agents["small_action"]["action_text"]
    assert reflection[-1].action_status == "partial"
    assert "물음표 하나" in reflection[-1].last_small_action


def test_action_checkin_completed_is_reinforced_with_specific_next_step():
    results, reflection = asyncio.run(
        _run_messages(
            [
                "인공지능 시험 때문에 암기할 게 너무 많아서 막막해",
                "개념 5개 정리했어",
            ],
            include_reflection=True,
        )
    )
    _, checkin = results

    agents = checkin["pipeline_details"]["agents"]
    assert agents["action_checkin"]["status"] == "completed"
    assert "직접 해봤다는 게 중요해" in checkin["response"]
    assert "정의와 예시를 한 줄씩" in agents["small_action"]["action_text"]
    assert reflection[-1].action_status == "completed"
    assert "정의와 예시를 한 줄씩" in reflection[-1].last_small_action


def test_action_checkin_numeric_progress_compares_with_previous_target():
    completed = classify_action_checkin(
        "항목 6개 정리했어",
        has_previous_action=True,
        previous_action_text="항목 5개를 정리하기",
    )
    partial = classify_action_checkin(
        "항목 3개 정리했어",
        has_previous_action=True,
        previous_action_text="항목 5개를 정리하기",
    )

    assert completed.status == "completed"
    assert partial.status == "partial"


def test_action_checkin_failure_is_non_blaming_and_shrinks_action():
    _, checkin = asyncio.run(
        _run_messages(
            [
                "인공지능 시험 때문에 암기할 게 너무 많아서 막막해",
                "못 했어",
            ]
        )
    )

    agents = checkin["pipeline_details"]["agents"]
    assert agents["action_checkin"]["status"] == "not_completed"
    assert "네가 약해서가 아니라" in checkin["response"]
    assert "첫 항목 제목만" in agents["small_action"]["action_text"]


def test_crisis_overrides_action_checkin_after_previous_action():
    _, crisis = asyncio.run(
        _run_messages(
            [
                "인공지능 시험 때문에 암기할 게 너무 많아서 막막해",
                "못 했어. 죽고 싶어",
            ]
        )
    )

    agents = crisis["pipeline_details"].get("agents", {})
    assert crisis["requires_crisis_response"] is True
    assert "action_checkin" not in agents
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "네가 약해서가 아니라" not in crisis["response"]


def test_recovery_response_includes_empathy_and_explores_what_helped():
    result = asyncio.run(_run_message("어제보다 좀 나아졌어"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]
    assert cause["selected_cause"] == "recovery_improvement"
    assert "나아졌" in response or "괜찮아지" in response or "좋아진" in response
    assert "중요한 변화" in response or "회복" in response
    assert any(marker in response for marker in ("도움", "휴식", "거리두기", "누군가"))
    assert followup and followup in response
    assert "내일" in response or "5분" in response


def test_scolding_criticism_response_includes_empathy_and_content_tone_evaluation_exploration():
    result = asyncio.run(_run_message("교수한테 혼나서 너무 창피하고 화나"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    followup = result["pipeline_details"]["agents"]["followup"]["question"]
    assert cause["selected_cause"] == "authority_criticism"
    assert "창피" in response or "화" in response
    assert all(marker in response for marker in ("내용", "말투", "평가"))
    assert followup and followup in response
    assert "세 칸" in response or "한 단어" in response


def test_academic_overload_followup_uses_narrowed_response_and_new_action():
    first, followup = asyncio.run(
        _run_messages(
            [
                "공부하느라 너무 힘들어",
                "과제 마감 기말고사 시험 준비 둘 다",
            ]
        )
    )

    first_action = first["pipeline_details"]["agents"]["small_action"]["action_text"]
    followup_response = followup["response"]
    followup_agents = followup["pipeline_details"]["agents"]
    followup_action = followup_agents["small_action"]["action_text"]

    assert followup_agents["cause_exploration"]["selected_cause"] == "academic_deadline_exam_overload"
    assert "과제 마감과 기말 준비가 동시에 겹친 상황이군요." in followup_response
    assert "공부나 시험, 과제 부담이 겹치면" not in followup_response
    assert "과제는 제출 시간이 더 급한가요, 아니면 기말고사 범위가 더 막막한가요?" in followup_response
    assert (
        "오늘은 과제 1개와 시험 범위 1개만 적고, "
        "각각 10분 안에 시작할 첫 행동을 하나씩 정해보세요."
    ) in followup_response
    assert followup_action != first_action
    assert "오늘 할 수 있는 가장 작은 한 가지" not in followup_action


def test_repeated_academic_turns_do_not_use_identical_action_step():
    first, second = asyncio.run(
        _run_messages(
            [
                "공부하느라 너무 힘들어",
                "시험 때문에 계속 부담돼",
            ]
        )
    )

    first_action = first["pipeline_details"]["agents"]["small_action"]["action_text"]
    second_action = second["pipeline_details"]["agents"]["small_action"]["action_text"]
    assert first_action
    assert second_action
    assert second_action != first_action


def test_crisis_input_still_overrides_academic_followup_branch():
    _, crisis = asyncio.run(
        _run_messages(
            [
                "공부하느라 너무 힘들어",
                "과제 마감 기말고사 시험 준비 둘 다인데 죽고 싶어요.",
            ]
        )
    )

    agents = crisis["pipeline_details"].get("agents", {})
    assert crisis["requires_crisis_response"] is True
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "과제 마감과 기말 준비가 동시에 겹친 상황이군요." not in crisis["response"]


def test_crisis_input_still_overrides_all_empathy_cause_branches():
    result = asyncio.run(_run_message("교수한테 혼나서 속상하고 죽고 싶어요."))

    agents = result["pipeline_details"].get("agents", {})
    assert result["requires_crisis_response"] is True
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "지적 내용" not in result["response"]
    assert "말투" not in result["response"]


def test_self_blame_response_is_specific():
    result = asyncio.run(_run_message("내가 너무 한심해"))

    response = result["response"]
    core_response = response.split("\n\n이 AI는 의료 진단이나 치료", 1)[0]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "self_blame"
    assert len([part for part in core_response.split("\n\n") if part.strip()]) == 4
    assert any(marker in core_response for marker in ("생각이 곧 당신 전체", "사실", "자기평가", "느낀 감정"))
    assert "구체적인 일이 있었나요" in core_response or "여러 일이 쌓여서" in core_response
    assert "?" in core_response
    assert "실제로 일어난 사실 하나" in core_response
    assert "느낀 감정 하나" in core_response
    assert "공부나 시험, 과제 부담" not in core_response
    assert "오늘 할 수 있는 가장 작은 한 가지" not in core_response


def test_self_blame_strategy_rejects_one_sentence_naturalization():
    agent = PsychologistAgent(
        config=AgentConfig(enable_rag=False, enable_audit_logging=False),
        mock_mode=True,
    )

    assert not agent._is_valid_response_for_strategy(
        "스스로가 너무 한심하다고 느껴지시는군요.",
        "self_blame",
    )
    assert agent._is_valid_response_for_strategy(
        "마음이 많이 지친 것 같아요. 생각이 곧 당신 전체를 설명하지는 않아요. "
        "그 생각이 든 구체적인 일이 있었나요? 사실과 느낀 감정을 나눠 적어보세요.",
        "self_blame",
    )


def test_anger_frustration_response_is_specific():
    result = asyncio.run(_run_message("너무 화나고 억울해서 답답해"))

    response = result["response"]
    cause = result["pipeline_details"]["agents"]["cause_exploration"]
    assert cause["selected_cause"] == "anger_frustration"
    assert any(marker in response for marker in ("화", "억울", "답답"))
    assert "숨" in response or "한 문장" in response


def test_small_action_is_actual_action_sentence():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    small_action = result["pipeline_details"]["agents"]["small_action"]
    action_text = small_action["action_text"]
    assert small_action["has_action"] is True
    assert any(marker in action_text for marker in ACTION_MARKERS)
    assert "상담 참고" not in action_text
    assert "공감 참고" not in action_text


def test_dataset_hint_labels_are_not_exposed():
    result = asyncio.run(
        _run_message(
            "요즘 잠을 못 자고 불안해요",
            counseling_hint=RAW_DATASET_TEXT,
            empathy_hint="웰니스 참고: 그대로 노출하지 마세요",
        )
    )

    assert "상담 참고" not in result["response"]
    assert "공감 참고" not in result["response"]
    assert "웰니스 참고" not in result["response"]
    assert "intervention_hint" not in result["response"]
    assert "empathy_style_hint" not in result["response"]
    assert "support_hint" not in result["response"]
    assert "therapeutic_guidance" not in result["response"]


def test_disclaimer_is_not_duplicated():
    result = asyncio.run(_run_message("요즘 잠을 못 자고 불안해요"))

    assert result["response"].count(SAFETY_NOTICE_MARKER) == 1


def test_crisis_response_has_no_general_followup_or_small_action():
    result = asyncio.run(_run_message("죽고 싶어요. 지금 혼자라서 너무 위험해요."))

    agents = result["pipeline_details"].get("agents", {})
    assert result["requires_crisis_response"] is True
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "잠드는 데 오래 걸리는 편인가요" not in result["response"]
    assert "오늘의 작은 행동" not in result["response"]


def test_crisis_response_overrides_emotional_distress_templates():
    result = asyncio.run(_run_message("가슴이 답답하고 속상한데 죽고 싶어요."))

    agents = result["pipeline_details"].get("agents", {})
    assert result["requires_crisis_response"] is True
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "몸의 긴장처럼 느껴지나요" not in result["response"]
    assert "어깨를 한번 아래로" not in result["response"]


def test_response_strategy_layer_selects_context_specific_paths():
    cases = (
        ("공부하느라 너무 힘들어", "academic_pressure"),
        ("인공지능 시험 때문에 암기할 게 너무 많아서 막막해", "specific_academic_burden"),
        ("교수님한테 지적받아서 짜증났어", "criticism_scolding"),
        ("자다가 자주 깨고 숙면이 안 돼", "sleep_problem"),
        ("내가 너무 한심해", "self_blame"),
        ("요즘 괜찮아졌어", "recovery_improvement"),
    )

    for message, expected_strategy in cases:
        result = asyncio.run(_run_message(message))
        strategy = result["pipeline_details"]["agents"]["response_strategy"]
        assert strategy["name"] == expected_strategy
        assert result["pipeline_details"]["agents"]["followup"]["has_question"] is True
        assert result["pipeline_details"]["agents"]["small_action"]["has_action"] is True


def test_specific_strategies_do_not_use_generic_academic_language_or_actions():
    messages = (
        "인공지능 시험 때문에 암기할 게 너무 많아서 막막해",
        "교수님한테 지적받아서 짜증났어",
        "자다가 자주 깨고 숙면이 안 돼",
        "내가 너무 한심해",
        "요즘 괜찮아졌어",
    )
    generic_phrases = (
        "공부나 시험, 과제 부담을 말할 정도면",
        "공부량 자체, 성적 걱정, 마감 압박",
        "오늘 할 수 있는 가장 작은 한 가지",
        "타이머를 10분만 맞추고",
        "과제나 공부에서 가장 작은 첫 단계 하나만",
        "숨을 고르고",
    )

    for message in messages:
        result = asyncio.run(_run_message(message))
        response = result["response"]
        assert all(phrase not in response for phrase in generic_phrases)


def test_crisis_safety_still_bypasses_response_strategy_composition():
    result = asyncio.run(_run_message("죽고 싶어요"))

    agents = result["pipeline_details"].get("agents", {})
    assert result["requires_crisis_response"] is True
    assert not agents.get("followup", {}).get("has_question", False)
    assert not agents.get("small_action", {}).get("has_action", False)
    assert "오늘의 작은 행동" not in result["response"]


def test_raw_inputs_dataset_text_and_memory_transcript_are_not_exposed():
    result = asyncio.run(
        _run_message(
            RAW_USER_INPUT,
            counseling_hint=RAW_DATASET_TEXT,
            empathy_hint=RAW_MEMORY_TEXT,
        )
    )

    exposed = "\n".join([result["response"], *list(_flatten_strings(result["pipeline_details"]))])
    assert RAW_USER_INPUT not in exposed
    assert RAW_DATASET_TEXT not in exposed
    assert RAW_MEMORY_TEXT not in exposed
