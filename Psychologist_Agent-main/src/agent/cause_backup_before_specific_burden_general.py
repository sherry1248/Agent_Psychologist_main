"""Rule-based cause exploration helper for counseling-style mock responses.

The helper uses structured intent labels and dataset metadata only. It does not
store raw user turns or raw dataset text in its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.models import IntentAgentResult, IntentLabel


SLEEP_MAINTENANCE_MARKERS = (
    "자주깨",
    "자주 깨",
    "깨는 편",
    "중간에 깨",
    "계속 깨",
    "새벽에 깨",
    "자다가 깨",
    "눈이 떠져",
    "자꾸 일어",
    "숙면이 안",
)

SLEEP_WAKE_THOUGHT_MARKERS = ("걱정", "생각", "불안", "떠올", "머리")
SLEEP_WAKE_NO_THOUGHT_MARKERS = ("특별한 생각 없이", "생각 없이", "이유 없이", "그냥")
SLEEP_ONSET_MARKERS = ("잠들기", "잠이 안", "잠 안", "뒤척", "오래 걸")
ACADEMIC_RELIEF_MARKERS = (
    "드디어",
    "기뻐",
    "기쁘",
    "숨통이 풀",
    "후련",
    "다행",
    "끝이 보",
    "괜찮아졌",
    "나아졌",
    "좋아졌",
    "풀리기 시작",
    "이제 좀",
    "한숨 돌",
)
MANAGEABLE_CONCERN_MARKERS = (
    "어렵",
    "암기",
    "부담",
    "걱정",
    "해야할 것 같",
    "해야 할 것 같",
    "빡세",
    "막막",
)
COPING_CONFIDENCE_MARKERS = (
    "괜찮을 것 같",
    "시간이 있",
    "주말에 시간",
    "할 수 있",
    "그래도",
    "다행",
    "괜찮아",
    "될 것 같",
)
SPECIFIC_ACADEMIC_SUBJECT_MARKERS = (
    "인공지능",
    "AI",
    "ai",
    "머신러닝",
    "딥러닝",
    "알고리즘",
    "자료구조",
    "전공",
    "시험",
)
SPECIFIC_ACADEMIC_CONTENT_MARKERS = (
    "암기",
    "외울",
    "개념",
    "용어",
    "공식",
    "정리",
    "이해",
    "문제풀이",
    "문제 풀이",
)
SPECIFIC_ACADEMIC_DIFFICULTY_MARKERS = (
    "어렵",
    "부담",
    "많아",
    "많으",
    "많이",
    "헷갈려",
    "헷갈리",
    "막막",
    "너무 많",
    "힘들",
    "걱정",
    "불안",
)
ACADEMIC_RELIEF_FOLLOWUP_MARKERS = (
    "남은 하나",
    "남은 시험",
    "마무리만 잘하면",
)
RECOVERY_IMPROVEMENT_MARKERS = (
    "괜찮아졌어",
    "괜찮아졌",
    "나아졌어",
    "나아졌",
    "좋아졌어",
    "좋아졌",
    "회복",
)
CHEST_TIGHTNESS_MARKERS = ("가슴이 답답", "가슴 답답", "가슴답답", "답답")
SADNESS_MARKERS = ("속상", "슬프", "마음이 아", "서럽")
CRYING_URGE_MARKERS = ("울고 싶", "울고싶", "눈물이", "눈물 나", "눈물나", "울 것 같")
AUTHORITY_CRITICISM_MARKERS = (
    "교수님",
    "교수한테 혼",
    "선생님",
    "선생님한테 혼",
    "상사",
    "상사한테 혼",
    "팀장",
    "부장",
    "비판",
    "혼났",
    "혼나",
    "지적",
    "지적받",
    "꾸중",
    "싫은 소리",
    "욕먹",
)
EXAM_ASSIGNMENT_MARKERS = (
    "시험",
    "과제",
    "레포트",
    "리포트",
    "중간",
    "중간고사",
    "기말",
    "마감",
    "공부",
    "성적",
    "암기",
)
ACADEMIC_CONTEXT_MARKERS = (
    *EXAM_ASSIGNMENT_MARKERS,
    "과목",
    "인공지능",
    "AI",
    "개념",
    "문제 풀이",
    "문제풀이",
    "암기",
)
ACADEMIC_FOLLOWUP_CONTEXT_MARKERS = (
    "공부",
    "시험",
    "과제",
    "성적",
    "중간",
    "기말",
    "마감",
    "압박",
    "범위",
    "첫 단계",
)
ACADEMIC_OVERLOAD_ASSIGNMENT_MARKERS = ("과제",)
ACADEMIC_OVERLOAD_DEADLINE_MARKERS = ("마감",)
ACADEMIC_OVERLOAD_EXAM_MARKERS = ("기말", "시험")
ACADEMIC_OVERLOAD_BOTH_MARKERS = ("둘 다", "둘다")
SELF_BLAME_MARKERS = (
    "내 탓",
    "내탓",
    "내가 문제",
    "나 때문",
    "자책",
    "나는 왜",
    "못난",
    "한심",
)
ANGER_FRUSTRATION_MARKERS = (
    "화나",
    "화가",
    "짜증",
    "열받",
    "억울",
    "분해",
    "빡쳐",
    "답답해서 미치",
)


BASE_CAUSES: Dict[IntentLabel, List[str]] = {
    IntentLabel.SLEEP_PROBLEM: [
        "worry_or_anxiety",
        "sleep_maintenance",
        "lifestyle_rhythm",
        "physical_fatigue",
    ],
    IntentLabel.ANXIETY_SUPPORT: [
        "task_pressure",
        "relationship_stress",
        "future_uncertainty",
        "accumulated_fatigue",
    ],
    IntentLabel.LOW_MOOD_SUPPORT: [
        "exhaustion",
        "isolation",
        "low_self_evaluation",
        "repeated_failure_experience",
    ],
    IntentLabel.STRESS_SUPPORT: [
        "overload",
        "unclear_starting_point",
        "pressure_to_finish",
        "fear_of_failure",
    ],
    IntentLabel.WORK_OR_STUDY_STRESS: [
        "overload",
        "unclear_starting_point",
        "pressure_to_finish",
        "fear_of_failure",
    ],
    IntentLabel.RELATIONSHIP_STRESS: [
        "communication_gap",
        "fear_of_rejection",
        "loneliness_in_relationship",
        "boundary_pressure",
    ],
}


CAUSE_QUESTIONS = {
    "worry_or_anxiety": "잠들기 전 걱정이 많아지는 편인가요, 아니면 잠들어도 중간에 자주 깨는 편인가요?",
    "sleep_maintenance": "깨고 난 뒤 걱정이 떠올라 다시 잠들기 어려운 편인가요, 아니면 특별한 생각 없이 자주 깨는 편인가요?",
    "lifestyle_rhythm": "최근 잠드는 시간이나 화면을 보는 시간이 조금씩 밀리고 있는지도 같이 살펴볼 수 있을까요?",
    "physical_fatigue": "몸은 피곤한데 긴장이 풀리지 않는 느낌에 가까운지도 확인해볼 수 있을까요?",
    "task_pressure": "불안이 해야 할 일을 떠올릴 때 커지는 편인가요, 아니면 특별한 이유 없이 올라오는 편인가요?",
    "relationship_stress": "불안이나 긴장이 특정 사람과의 관계를 떠올릴 때 더 커지는지도 같이 살펴볼 수 있을까요?",
    "future_uncertainty": "앞으로 어떻게 될지 모른다는 생각이 불안을 키우는 쪽에 가까울까요?",
    "accumulated_fatigue": "최근 피로가 쌓이면서 불안을 견디는 힘도 같이 줄어든 느낌이 있을까요?",
    "exhaustion": "지금의 무기력은 쉬어도 회복이 잘 안 되는 소진감에 가까울까요?",
    "isolation": "혼자 감당하고 있다는 느낌이 기분을 더 가라앉히는지도 살펴볼 수 있을까요?",
    "low_self_evaluation": "스스로를 낮게 평가하는 생각이 반복되면서 기분이 더 무거워지는 편일까요?",
    "repeated_failure_experience": "최근 반복된 실망이나 실패감이 마음에 남아 있는지도 같이 볼 수 있을까요?",
    "overload": "부담이 일의 양이 많은 데서 오는지, 감당해야 한다는 압박에서 오는지 같이 좁혀볼 수 있을까요?",
    "unclear_starting_point": "어디서부터 시작해야 할지 모르는 막막함이 스트레스를 더 키우는 편인가요?",
    "pressure_to_finish": "끝내야 한다는 압박이 몸의 긴장까지 올리는 쪽에 가까울까요?",
    "fear_of_failure": "잘 못하면 어떡하지 하는 걱정이 시작을 어렵게 만드는지도 살펴볼 수 있을까요?",
    "communication_gap": "말이 잘 통하지 않는 느낌이 가장 힘든 지점인지 같이 확인해볼 수 있을까요?",
    "fear_of_rejection": "거절당하거나 멀어질까 봐 조심하게 되는 마음도 영향을 주고 있을까요?",
    "loneliness_in_relationship": "관계 안에서도 혼자 감당하는 느낌이 있는지 살펴볼 수 있을까요?",
    "boundary_pressure": "상대에게 맞추느라 내 경계가 흐려지는 느낌이 있는지도 확인해볼 수 있을까요?",
    "recovery_improvement": "조금 나아지는 데 도움이 된 건 휴식, 거리두기, 누군가의 말 중 어느 쪽에 가까웠나요?",
    "academic_relief": "남은 하나는 부담이 큰 시험인가요, 아니면 마무리만 잘하면 될 것 같은 시험인가요?",
    "specific_academic_burden_after_relief": "개념 정의 암기, 알고리즘 흐름 이해, 용어 구분 중 뭐가 제일 부담인가요?",
    "specific_academic_burden": "개념 정의 암기, 알고리즘 흐름 이해, 용어 구분 중 뭐가 제일 막막한가요?",
    "manageable_academic_concern": "남은 시험에서 가장 부담되는 건 개념 암기인가요, 이해가 필요한 부분인가요, 아니면 문제 풀이인가요?",
    "authority_criticism": "가장 아팠던 부분은 지적 내용 자체였나요, 말투였나요, 아니면 평가받는 느낌이었나요?",
    "exam_assignment_pressure": "지금 부담은 공부량, 성적 걱정, 마감 압박, 지쳐서 버티기 어려운 느낌 중 어디에 가장 가까울까요?",
    "academic_deadline_exam_overload": "과제는 제출 시간이 더 급한가요, 아니면 기말고사 범위가 더 막막한가요?",
    "self_blame": "그 일이 전부 내 탓처럼 느껴지는지, 아니면 책임을 나눠볼 여지가 조금 있는지도 같이 볼 수 있을까요?",
    "anger_frustration": "화의 중심은 억울함, 지친 상태, 아니면 계속 밀리는 압박 중 어디에 가까울까요?",
    "physical_tension_chest": "답답함이 주로 몸의 긴장처럼 느껴지나요, 아니면 어떤 일이 떠올라서 속상함이 커지는 쪽인가요?",
    "sadness_burden": "속상함이 특정한 일에 묶여 있는 느낌인가요, 아니면 피로와 감정이 누적된 쪽에 가까울까요?",
    "crying_urge": "눈물이 특정한 일 때문에 올라오는 느낌인가요, 아니면 쌓인 피로와 감정이 한꺼번에 올라온 쪽인가요?",
}

ALTERNATIVE_CAUSE_QUESTIONS = {
    "authority_criticism": "그 말에서 실제로 참고할 부분과 마음에 상처로 남은 부분을 나눠보고 싶으세요?",
    "exam_assignment_pressure": "지금은 범위를 줄이는 게 먼저 필요할까요, 아니면 시작할 첫 단계가 필요한 쪽일까요?",
    "self_blame": "내가 책임질 수 있는 부분과 그렇지 않은 부분을 구분해보는 것부터 해볼까요?",
    "anger_frustration": "지금은 화를 가라앉히는 게 먼저일까요, 아니면 억울했던 지점을 말로 정리하는 게 먼저일까요?",
    "physical_tension_chest": "몸의 답답함을 먼저 낮추고 싶은지, 마음에 걸린 일을 조금 말해보고 싶은지 어느 쪽에 가까울까요?",
    "sadness_burden": "지금 속상함을 말로 정리하고 싶은 쪽인가요, 아니면 잠깐 감정을 가라앉히는 게 먼저 필요할까요?",
    "crying_urge": "지금은 울음을 참기보다 안전한 곳에서 감정을 지나가게 두는 게 더 필요할까요?",
}

DEEPER_CAUSE_QUESTIONS = {
    "sleep_maintenance_thought": "깨고 난 뒤 가장 먼저 떠오르는 생각은 해야 할 일 쪽인가요, 아니면 막연한 걱정 쪽인가요?",
    "sleep_maintenance_no_thought": "특별한 생각 없이 깨는 날에는 몸의 긴장이나 불편감이 먼저 느껴지는 편인가요?",
    "worry_or_anxiety_bedtime": "잠들기 전 걱정은 오늘 있었던 일에 가까운가요, 아니면 앞으로의 일이 떠오르는 쪽에 가까운가요?",
}


@dataclass
class CauseExplorationResult:
    cause_candidates: List[str] = field(default_factory=list)
    selected_cause: str = ""
    exploration_question: str = ""
    reason_codes: List[str] = field(default_factory=list)
    dataset_signals: Dict[str, str] = field(default_factory=dict)

    def to_pipeline_dict(self) -> Dict[str, Any]:
        return {
            "cause_candidates": list(self.cause_candidates),
            "selected_cause": self.selected_cause,
            "reason_codes": list(self.reason_codes),
            "dataset_signals": dict(self.dataset_signals),
        }


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value or "")).upper()


def _primary_intent(intent_result: Optional[IntentAgentResult]) -> IntentLabel:
    if intent_result and intent_result.primary_intent:
        return intent_result.primary_intent
    return IntentLabel.OTHER_CONCERN


def _dataset_signals(
    counseling_recommendation: Any,
    empathy_recommendation: Any,
    wellness_recommendation: Any,
) -> Dict[str, str]:
    return {
        "counseling_category": str(getattr(counseling_recommendation, "category", "") or ""),
        "empathy_emotion": str(getattr(empathy_recommendation, "emotion_label", "") or ""),
        "wellness_topic": str(getattr(wellness_recommendation, "matched_topic", "") or ""),
    }


def _sleep_followup_answer_type(user_input: str, previous_followup: str) -> str:
    if not (previous_followup or "").strip():
        return ""

    previous = previous_followup or ""
    if not any(marker in previous for marker in ("잠", "수면", "깨", "걱정")):
        return ""

    if any(marker in user_input for marker in SLEEP_MAINTENANCE_MARKERS):
        return "sleep_maintenance"
    if any(marker in user_input for marker in SLEEP_WAKE_NO_THOUGHT_MARKERS):
        return "sleep_maintenance_no_thought"
    if any(marker in user_input for marker in SLEEP_ONSET_MARKERS):
        return "worry_or_anxiety"
    if any(marker in user_input for marker in SLEEP_WAKE_THOUGHT_MARKERS):
        return "sleep_maintenance_thought"
    return ""


def is_academic_relief(user_input: str) -> bool:
    current = user_input or ""
    return (
        any(marker in current for marker in EXAM_ASSIGNMENT_MARKERS)
        and any(marker in current for marker in ACADEMIC_RELIEF_MARKERS)
    )


def _has_previous_academic_context(previous_followup: str) -> bool:
    previous = previous_followup or ""
    return any(marker in previous for marker in ACADEMIC_CONTEXT_MARKERS)


def is_manageable_academic_concern(user_input: str, previous_followup: str = "") -> bool:
    current = user_input or ""
    has_mixed_emotion = (
        any(marker in current for marker in MANAGEABLE_CONCERN_MARKERS)
        and any(marker in current for marker in COPING_CONFIDENCE_MARKERS)
    )
    has_academic_context = (
        any(marker in current for marker in ACADEMIC_CONTEXT_MARKERS)
        or (
            _has_previous_academic_context(previous_followup)
            and any(marker in current for marker in (*MANAGEABLE_CONCERN_MARKERS, *COPING_CONFIDENCE_MARKERS))
        )
    )
    return has_mixed_emotion and has_academic_context


def is_specific_academic_burden_after_relief(
    user_input: str,
    previous_followup: str = "",
) -> bool:
    """Detect a concrete burden disclosed after an academic-relief turn."""
    current = user_input or ""
    previous = previous_followup or ""
    previous_was_academic_relief = any(
        marker in previous for marker in ACADEMIC_RELIEF_FOLLOWUP_MARKERS
    )
    has_specific_content = any(
        marker in current for marker in SPECIFIC_ACADEMIC_CONTENT_MARKERS
    )
    has_specific_subject = any(
        marker in current for marker in SPECIFIC_ACADEMIC_SUBJECT_MARKERS
    )
    has_difficulty = any(
        marker in current for marker in SPECIFIC_ACADEMIC_DIFFICULTY_MARKERS
    )
    has_coping_confidence = any(
        marker in current for marker in COPING_CONFIDENCE_MARKERS
    )
    return (
        previous_was_academic_relief
        and has_specific_content
        and (has_specific_subject or has_difficulty)
        and has_difficulty
        and not has_coping_confidence
    )


def is_specific_academic_burden(user_input: str) -> bool:
    """Detect a concrete academic target, burden type, and distress signal."""
    current = user_input or ""
    return (
        any(marker in current for marker in SPECIFIC_ACADEMIC_SUBJECT_MARKERS)
        and any(marker in current for marker in SPECIFIC_ACADEMIC_CONTENT_MARKERS)
        and any(marker in current for marker in SPECIFIC_ACADEMIC_DIFFICULTY_MARKERS)
    )


def is_narrowed_academic_overload_followup(user_input: str, previous_followup: str) -> bool:
    """Detect a reply that narrows the previous study-pressure question."""
    previous = (previous_followup or "").strip()
    current = (user_input or "").strip()
    if not previous or not current:
        return False

    if not any(marker in previous for marker in ACADEMIC_FOLLOWUP_CONTEXT_MARKERS):
        return False

    return (
        any(marker in current for marker in ACADEMIC_OVERLOAD_ASSIGNMENT_MARKERS)
        and any(marker in current for marker in ACADEMIC_OVERLOAD_DEADLINE_MARKERS)
        and any(marker in current for marker in ACADEMIC_OVERLOAD_EXAM_MARKERS)
        and any(marker in current for marker in ACADEMIC_OVERLOAD_BOTH_MARKERS)
    )


def _deeper_question(selected: str, answer_type: str, previous_followup: str) -> str:
    previous = (previous_followup or "").strip()
    if selected == "sleep_maintenance" and "깨고 난 뒤" in previous:
        if answer_type == "sleep_maintenance_no_thought":
            return DEEPER_CAUSE_QUESTIONS["sleep_maintenance_no_thought"]
        return DEEPER_CAUSE_QUESTIONS["sleep_maintenance_thought"]
    if selected == "worry_or_anxiety" and previous:
        return DEEPER_CAUSE_QUESTIONS["worry_or_anxiety_bedtime"]
    return ""


def _score_candidates(
    candidates: List[str],
    *,
    user_input: str,
    signals: Dict[str, str],
    previous_followup: str,
) -> tuple[str, List[str]]:
    scores = {candidate: 1 for candidate in candidates}
    reason_codes: List[str] = ["intent_candidate_set"]
    lowered = " ".join([user_input, *signals.values()]).lower()
    followup_answer_type = _sleep_followup_answer_type(user_input, previous_followup)
    academic_overload_followup = is_narrowed_academic_overload_followup(user_input, previous_followup)
    academic_relief = is_academic_relief(user_input)
    manageable_academic_concern = is_manageable_academic_concern(user_input, previous_followup)
    specific_academic_burden = is_specific_academic_burden_after_relief(
        user_input,
        previous_followup,
    )
    general_specific_academic_burden = (
        is_specific_academic_burden(user_input)
        and not specific_academic_burden
        and not academic_relief
        and not manageable_academic_concern
    )

    if academic_overload_followup:
        scores["academic_deadline_exam_overload"] = scores.get("academic_deadline_exam_overload", 0) + 30
        reason_codes.append("previous_followup_narrows_academic_deadline_exam_overload")
    if academic_relief:
        if "academic_relief" in scores:
            scores["academic_relief"] += 60
        if "recovery_improvement" in scores:
            scores["recovery_improvement"] += 36
        reason_codes.append("input_academic_relief_signal")
    if manageable_academic_concern:
        if "manageable_academic_concern" in scores:
            scores["manageable_academic_concern"] += 58
        reason_codes.append("input_manageable_academic_concern_signal")
        if _has_previous_academic_context(previous_followup):
            reason_codes.append("previous_academic_context_preserved")
    if specific_academic_burden:
        if "specific_academic_burden_after_relief" in scores:
            scores["specific_academic_burden_after_relief"] += 64
        reason_codes.extend(
            [
                "input_specific_academic_burden_signal",
                "previous_academic_relief_context_preserved",
            ]
        )
    if general_specific_academic_burden:
        if "specific_academic_burden" in scores:
            scores["specific_academic_burden"] += 62
        reason_codes.append("input_specific_academic_burden_signal")

    if any(marker in user_input for marker in SLEEP_MAINTENANCE_MARKERS):
        scores["sleep_maintenance"] = scores.get("sleep_maintenance", 0) + 20
        reason_codes.append("followup_answer_sleep_maintenance")
    if any(marker in user_input for marker in RECOVERY_IMPROVEMENT_MARKERS):
        if "recovery_improvement" in scores:
            scores["recovery_improvement"] += 24
        reason_codes.append("input_recovery_improvement_signal")
    if followup_answer_type:
        reason_codes.append("previous_followup_answer_detected")
    if followup_answer_type in {"sleep_maintenance", "sleep_maintenance_thought", "sleep_maintenance_no_thought"}:
        if "sleep_maintenance" in scores:
            scores["sleep_maintenance"] += 20
        reason_codes.append("previous_followup_narrows_sleep_maintenance")
    elif followup_answer_type == "worry_or_anxiety":
        if "worry_or_anxiety" in scores:
            scores["worry_or_anxiety"] += 20
        reason_codes.append("previous_followup_narrows_worry_or_anxiety")
    if any(marker in lowered for marker in ("불안", "걱정", "anxiety", "worry")):
        for cause in ("worry_or_anxiety", "task_pressure", "future_uncertainty"):
            if cause in scores:
                scores[cause] += 3
        reason_codes.append("dataset_or_input_anxiety_signal")
    if any(marker in lowered for marker in ("sleep", "수면", "불면", "잠")):
        for cause in ("worry_or_anxiety", "lifestyle_rhythm"):
            if cause in scores:
                scores[cause] += 2
        reason_codes.append("dataset_sleep_signal")
    if any(marker in lowered for marker in ("depression", "우울", "슬픔")):
        for cause in ("exhaustion", "isolation", "low_self_evaluation"):
            if cause in scores:
                scores[cause] += 1
        reason_codes.append("dataset_low_mood_signal")
    if any(marker in lowered for marker in ("관계", "relationship", "친구", "연애")):
        for cause in ("relationship_stress", "communication_gap"):
            if cause in scores:
                scores[cause] += 2
        reason_codes.append("dataset_relationship_signal")
    if any(marker in lowered for marker in ("업무", "공부", "시험", "work", "study", "overload")):
        for cause in ("overload", "pressure_to_finish", "unclear_starting_point"):
            if cause in scores:
                scores[cause] += 2
        reason_codes.append("dataset_task_pressure_signal")
    if any(marker in user_input for marker in CHEST_TIGHTNESS_MARKERS):
        if "physical_tension_chest" in scores:
            scores["physical_tension_chest"] += 20
        reason_codes.append("input_chest_tightness_signal")
    if any(marker in user_input for marker in AUTHORITY_CRITICISM_MARKERS):
        if "authority_criticism" in scores:
            scores["authority_criticism"] += 28
        reason_codes.append("input_authority_criticism_signal")
    if (
        any(marker in user_input for marker in EXAM_ASSIGNMENT_MARKERS)
        and not academic_relief
        and not manageable_academic_concern
        and not specific_academic_burden
        and not general_specific_academic_burden
    ):
        if "exam_assignment_pressure" in scores:
            scores["exam_assignment_pressure"] += 22
        reason_codes.append("input_exam_assignment_signal")
    if any(marker in user_input for marker in SELF_BLAME_MARKERS):
        if "self_blame" in scores:
            scores["self_blame"] += 22
        reason_codes.append("input_self_blame_signal")
    if any(marker in user_input for marker in ANGER_FRUSTRATION_MARKERS):
        if "anger_frustration" in scores:
            scores["anger_frustration"] += 22
        reason_codes.append("input_anger_frustration_signal")
    if any(marker in user_input for marker in SADNESS_MARKERS):
        if "sadness_burden" in scores:
            scores["sadness_burden"] += 12
        reason_codes.append("input_sadness_burden_signal")
    if any(marker in user_input for marker in CRYING_URGE_MARKERS):
        if "crying_urge" in scores:
            scores["crying_urge"] += 24
        reason_codes.append("input_crying_urge_signal")

    selected = max(candidates, key=lambda cause: scores.get(cause, 0)) if candidates else ""
    return selected, reason_codes


def _expression_specific_candidates(user_input: str, previous_followup: str = "") -> List[str]:
    candidates: List[str] = []
    academic_relief = is_academic_relief(user_input)
    manageable_academic_concern = is_manageable_academic_concern(user_input, previous_followup)
    specific_academic_burden = is_specific_academic_burden_after_relief(
        user_input,
        previous_followup,
    )
    general_specific_academic_burden = (
        is_specific_academic_burden(user_input)
        and not specific_academic_burden
        and not academic_relief
        and not manageable_academic_concern
    )
    if academic_relief:
        candidates.append("academic_relief")
        candidates.append("recovery_improvement")
    if manageable_academic_concern:
        candidates.append("manageable_academic_concern")
    if specific_academic_burden:
        candidates.append("specific_academic_burden_after_relief")
    if general_specific_academic_burden:
        candidates.append("specific_academic_burden")
    if any(marker in user_input for marker in RECOVERY_IMPROVEMENT_MARKERS):
        candidates.append("recovery_improvement")
    if any(marker in user_input for marker in AUTHORITY_CRITICISM_MARKERS):
        candidates.append("authority_criticism")
    if (
        any(marker in user_input for marker in EXAM_ASSIGNMENT_MARKERS)
        and not academic_relief
        and not manageable_academic_concern
        and not specific_academic_burden
        and not general_specific_academic_burden
    ):
        candidates.append("exam_assignment_pressure")
    if any(marker in user_input for marker in SELF_BLAME_MARKERS):
        candidates.append("self_blame")
    if any(marker in user_input for marker in ANGER_FRUSTRATION_MARKERS):
        candidates.append("anger_frustration")
    if any(marker in user_input for marker in CHEST_TIGHTNESS_MARKERS):
        candidates.append("physical_tension_chest")
    if any(marker in user_input for marker in SADNESS_MARKERS):
        candidates.append("sadness_burden")
    if any(marker in user_input for marker in CRYING_URGE_MARKERS):
        candidates.append("crying_urge")
    return candidates


def explore_causes(
    *,
    user_input: str,
    intent_result: Optional[IntentAgentResult],
    counseling_recommendation: Any,
    empathy_recommendation: Any,
    wellness_recommendation: Any,
    proactive_recall: Any = None,
    previous_followup: str = "",
) -> CauseExplorationResult:
    del proactive_recall
    intent = _primary_intent(intent_result)
    candidates = list(BASE_CAUSES.get(intent, []))
    expression_candidates = _expression_specific_candidates(user_input or "", previous_followup or "")
    if is_narrowed_academic_overload_followup(user_input or "", previous_followup or ""):
        expression_candidates.insert(0, "academic_deadline_exam_overload")
    for candidate in expression_candidates:
        if candidate not in candidates:
            candidates.insert(0, candidate)
    if not candidates:
        return CauseExplorationResult(
            dataset_signals=_dataset_signals(
                counseling_recommendation,
                empathy_recommendation,
                wellness_recommendation,
            )
        )

    signals = _dataset_signals(
        counseling_recommendation,
        empathy_recommendation,
        wellness_recommendation,
    )
    selected, reason_codes = _score_candidates(
        candidates,
        user_input=user_input or "",
        signals=signals,
        previous_followup=previous_followup or "",
    )
    answer_type = _sleep_followup_answer_type(user_input or "", previous_followup or "")
    question = _deeper_question(selected, answer_type, previous_followup or "")
    if question:
        reason_codes.append("deeper_exploration_question_selected")
    else:
        question = CAUSE_QUESTIONS.get(selected, "")
    if question and question == (previous_followup or "").strip():
        alternative = ALTERNATIVE_CAUSE_QUESTIONS.get(selected, "")
        if alternative and alternative != (previous_followup or "").strip():
            reason_codes.append("duplicate_previous_question_alternative_selected")
            question = alternative
        else:
            reason_codes.append("duplicate_previous_question_omitted")
            question = ""

    return CauseExplorationResult(
        cause_candidates=candidates,
        selected_cause=selected,
        exploration_question=question,
        reason_codes=reason_codes,
        dataset_signals=signals,
    )


class CauseExplorationAgent:
    """Agent that explores underlying causes of counseling concerns,
    particularly focused on sleep problems and emotional/wellness triggers.
    """

    def __init__(self) -> None:
        pass

    def explore(
        self,
        user_input: str,
        intent_result: Any,
        counseling_rec: Any,
        empathy_rec: Any,
        wellness_rec: Any,
        proactive_recall: Any,
    ) -> Dict[str, Any]:
        # 1. Determine primary intent and initial candidates
        primary_intent = None
        if intent_result:
            if hasattr(intent_result, "primary_intent"):
                primary_intent = intent_result.primary_intent
            elif isinstance(intent_result, dict):
                primary_intent = intent_result.get("primary_intent")

        intent_str = ""
        if primary_intent is not None:
            if hasattr(primary_intent, "value"):
                intent_str = str(primary_intent.value)
            elif hasattr(primary_intent, "name"):
                intent_str = str(primary_intent.name)
            else:
                intent_str = str(primary_intent)

        is_sleep_problem = False
        if intent_str.lower() in ("sleep_problem", "intentlabel.sleep_problem"):
            is_sleep_problem = True

        candidates = []
        if is_sleep_problem:
            candidates = ["worry_or_anxiety", "sleep_maintenance", "lifestyle_rhythm", "physical_fatigue"]
        else:
            # General fallback to other BASE_CAUSES if applicable
            for key, val in BASE_CAUSES.items():
                key_name = getattr(key, "name", str(key)).lower()
                key_value = getattr(key, "value", str(key)).lower()
                if intent_str.lower() in (key_name, key_value):
                    candidates = list(val)
                    break

        # 2. Extract empathy and wellness fields
        emotion_label = ""
        if empathy_rec:
            if hasattr(empathy_rec, "emotion_label"):
                emotion_label = getattr(empathy_rec, "emotion_label") or ""
            elif isinstance(empathy_rec, dict):
                emotion_label = empathy_rec.get("emotion_label") or ""

        matched_topic = ""
        if wellness_rec:
            if hasattr(wellness_rec, "matched_topic"):
                matched_topic = getattr(wellness_rec, "matched_topic") or ""
            elif isinstance(wellness_rec, dict):
                matched_topic = wellness_rec.get("matched_topic") or ""

        # 3. Handle empathy/wellness prioritization and reason_codes
        reason_codes = []
        if emotion_label == "불안" or matched_topic == "불면":
            if "worry_or_anxiety" in candidates:
                candidates.remove("worry_or_anxiety")
                candidates.insert(0, "worry_or_anxiety")
            reason_codes.append("EMOTION_WELLNESS_MATCH")

        # 4. Check proactive recall force matching rule
        has_previous_followup = False
        if proactive_recall:
            if hasattr(proactive_recall, "previous_followup"):
                has_previous_followup = True
            elif isinstance(proactive_recall, dict) and "previous_followup" in proactive_recall:
                has_previous_followup = True

        contains_keyword = False
        if user_input:
            contains_keyword = any(word in user_input for word in ("깨", "중간에", "자주"))

        selected_cause = ""
        if has_previous_followup and contains_keyword:
            selected_cause = "sleep_maintenance"
        else:
            selected_cause = candidates[0] if candidates else ""

        # 5. Build dataset signals
        dataset_signals = {
            "counseling_category": "",
            "empathy_emotion": "",
            "wellness_topic": "",
        }
        if counseling_rec:
            if hasattr(counseling_rec, "category"):
                dataset_signals["counseling_category"] = str(getattr(counseling_rec, "category") or "")
            elif isinstance(counseling_rec, dict):
                dataset_signals["counseling_category"] = str(counseling_rec.get("category") or "")

        if empathy_rec:
            if hasattr(empathy_rec, "emotion_label"):
                dataset_signals["empathy_emotion"] = str(getattr(empathy_rec, "emotion_label") or "")
            elif isinstance(empathy_rec, dict):
                dataset_signals["empathy_emotion"] = str(empathy_rec.get("emotion_label") or "")

        if wellness_rec:
            if hasattr(wellness_rec, "matched_topic"):
                dataset_signals["wellness_topic"] = str(getattr(wellness_rec, "matched_topic") or "")
            elif isinstance(wellness_rec, dict):
                dataset_signals["wellness_topic"] = str(wellness_rec.get("matched_topic") or "")

        return {
            "cause_candidates": candidates,
            "selected_cause": selected_cause,
            "reason_codes": reason_codes,
            "dataset_signals": dataset_signals,
        }

    def __call__(
        self,
        user_input: str,
        intent_result: Any,
        counseling_rec: Any,
        empathy_rec: Any,
        wellness_rec: Any,
        proactive_recall: Any,
    ) -> Dict[str, Any]:
        return self.explore(
            user_input=user_input,
            intent_result=intent_result,
            counseling_rec=counseling_rec,
            empathy_rec=empathy_rec,
            wellness_rec=wellness_rec,
            proactive_recall=proactive_recall,
        )
