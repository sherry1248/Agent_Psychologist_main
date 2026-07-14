"""Gradio demo for the Psychologist Agent."""

from __future__ import annotations

import asyncio
import html
import json
import os
import socket
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

os.environ.setdefault("LLM_TYPE", "MOCK")

from src.agent.reflection_report import (
    REPORT_FALLBACK_MESSAGE,
    build_reflection_report,
    sanitize_reflection_records,
)
from src.memory.store import UserSettingsStore
from src.utils.logging_config import setup_logging

if TYPE_CHECKING:
    from src.main import PsychologistAgent

logger = setup_logging("demo_app")

agent: Optional["PsychologistAgent"] = None
current_session_id: Optional[str] = None
last_agent_result: Optional[Dict[str, Any]] = None
consented_user_sessions: Dict[str, str] = {}
user_settings_store = UserSettingsStore()

ChatMessage = Dict[str, str]
INITIAL_ASSISTANT_MESSAGE = (
    "안녕하세요. 오늘 마음 상태는 어떤가요? 편하게 한 문장으로 이야기해도 괜찮아요."
)
INITIAL_CHAT_HISTORY: List[ChatMessage] = [
    {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE}
]
EMPTY_REPORT_MESSAGE = (
    "아직 장기 흐름을 만들 만큼 기록이 충분하지 않아요. "
    "상담을 몇 번 더 진행하면 변화 흐름을 볼 수 있어요."
)
EMPTY_REPORT_FAST_MESSAGE = (
    "아직 마음 리포트를 만들 만큼 상담 기록이 충분하지 않아요. "
    "상담을 몇 번 더 진행하면 변화 흐름을 볼 수 있어요."
)
REPORT_LOAD_ERROR_MESSAGE = (
    "리포트를 불러오는 중 문제가 생겼어요. 상담을 조금 더 진행한 뒤 다시 시도해 주세요."
)
REPORT_TIMEOUT_SECONDS = 0.8
REPORT_MEMORY_TIMEOUT_SECONDS = 0.5
TREND_SNAPSHOT_LIMIT = 10
TREND_INSUFFICIENT_MESSAGE = "아직 변화 방향을 판단하기에는 기록이 충분하지 않아요."
LOGIN_RESTORE_TIMEOUT_SECONDS = 0.5
RESTORED_REFLECTION_FIELDS = {
    "user_id",
    "session_id",
    "intent_label",
    "main_issue",
    "emotion_hint",
    "emotional_trend",
    "last_small_action",
    "action_status",
    "next_follow_up",
    "repeated_themes",
    "risk_stage",
    "created_at",
}
DEMO_REPORT_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "demo_user_report_cache.json"
)
DEMO_REPORT_SNAPSHOT_FIELDS = {
    "risk_stage",
    "intent_label",
    "selected_cause",
    "action_status",
    "small_action",
    "anxiety",
    "stress",
    "sleep",
    "energy",
    "created_at",
}

RISK_KEYWORDS = (
    "죽고 싶",
    "자해",
    "죽어버리",
    "사라지고 싶",
)

RAW_LOOKING_KEYS = {
    "raw_text",
    "raw_input",
    "user_input",
    "assistant_response",
    "conversation",
    "content",
    "transcript",
    "message",
    "source_conversation",
}

INTERNAL_HINT_LABELS = (
    "상담 참고",
    "공감 참고",
    "웰니스 참고",
    "심리상담 데이터 기반 힌트",
    "공감형 대화 기반 힌트",
    "웰니스 기반 힌트",
)

AGENT_SECTION_TITLES = (
    "Safety Agent",
    "Emotion Agent",
    "Intent Agent",
    "Dataset Strategy Agent",
    "Memory Agent / Proactive Recall",
    "Emotional State Agent",
    "Decision Agent",
    "Response Agent",
)

INTENT_LABEL_KO = {
    "SLEEP_PROBLEM": "수면 문제",
    "ANXIETY_SUPPORT": "불안",
    "STRESS_SUPPORT": "스트레스",
    "LOW_MOOD_SUPPORT": "무기력",
    "NEED_EMPATHY": "공감 필요",
    "NEED_ADVICE": "조언 필요",
    "LONELINESS_SUPPORT": "고립감",
    "CRISIS_RISK": "위험 신호",
    "CRISIS_SIGNAL": "위험 신호",
    "RELATIONSHIP_STRESS": "관계 스트레스",
    "WORK_OR_STUDY_STRESS": "학업/업무 부담",
    "FAMILY_CONFLICT": "가족 갈등",
    "LOW_SELF_ESTEEM": "자존감 저하",
    "SUBSTANCE_OR_ADDICTION": "중독 관련 고민",
    "GRIEF_SUPPORT": "상실/애도",
    "SUPPORT_REQUEST": "정서 지원 요청",
    "EMOTIONAL_DISCLOSURE": "감정 표현",
    "SAFETY_CONCERN": "안전 우려",
    "PRACTICAL_HELP": "실질 도움 요청",
    "REFLECTION": "자기 성찰",
    "MEMORY_UPDATE": "상담 내용 업데이트",
    "SMALL_ACTION": "작은 실천",
    "CLARIFICATION": "상황 확인",
    "OTHER_CONCERN": "기타 고민",
    "ACADEMIC_PRESSURE": "학업 부담",
    "SPECIFIC_ACADEMIC_BURDEN": "구체적인 학업 부담",
    "ACADEMIC_RELIEF": "학업 부담 완화/안도",
    "MANAGEABLE_CONCERN": "감당 가능한 걱정",
    "CRITICISM_SCOLDING": "지적/평가 스트레스",
    "SLEEP_PROBLEM": "수면 문제",
    "SADNESS_CRYING": "슬픔/눈물",
    "ANGER_FRUSTRATION": "분노/답답함",
    "SELF_BLAME": "자기비난/자책",
    "RECOVERY_IMPROVEMENT": "회복/호전",
    "CRISIS_SAFETY": "위기 신호",
}

CAUSE_LABEL_KO = {
    "sleep_maintenance": "수면 중 자주 깸",
    "worry_or_anxiety": "걱정이나 불안",
    "lifestyle_rhythm": "생활 리듬",
    "physical_fatigue": "몸의 피로",
    "task_pressure": "해야 할 일의 압박",
    "relationship_stress": "관계 스트레스",
    "future_uncertainty": "미래에 대한 불확실성",
    "accumulated_fatigue": "누적된 피로",
    "exhaustion": "소진감",
    "isolation": "고립감",
    "low_self_evaluation": "자기 평가 저하",
    "repeated_failure_experience": "반복된 실패감",
    "overload": "과부하",
    "unclear_starting_point": "시작점이 불명확함",
    "pressure_to_finish": "끝내야 한다는 압박",
    "fear_of_failure": "실패에 대한 걱정",
    "communication_gap": "소통의 어긋남",
    "fear_of_rejection": "거절에 대한 두려움",
    "loneliness_in_relationship": "관계 안의 외로움",
    "boundary_pressure": "관계 경계 부담",
    "exam_assignment_pressure": "시험·과제 부담",
    "academic_pressure": "학업 부담",
    "study_overload": "공부량 과부하",
}


async def get_agent() -> Optional["PsychologistAgent"]:
    """Return the cached agent without constructing it during lightweight UI actions."""
    return agent


async def get_or_create_agent() -> "PsychologistAgent":
    """Construct the cached counseling agent lazily for the first chat action."""
    global agent
    if agent is None:
        from src.main import PsychologistAgent

        agent = PsychologistAgent()
        logger.info("Agent container created for demo; initialization remains lazy")
    return agent


async def get_session_id(active_agent: "PsychologistAgent") -> str:
    global current_session_id
    if current_session_id is None:
        session = await active_agent.session_manager.create_session()
        current_session_id = session.session_id
    return current_session_id


def has_risk_keyword(message: str) -> bool:
    return any(keyword in message for keyword in RISK_KEYWORDS)


def _contains_crisis_phrase(text: str) -> bool:
    crisis_keywords = [
        "죽고싶",
        "죽고 싶",
        "죽고싶어",
        "죽고 싶어",
        "죽고싶어요",
        "죽고 싶어요",
        "자살",
        "극단적 선택",
        "사라지고 싶",
        "사라지고싶",
        "죽어버리고 싶",
        "죽어버리고싶",
        "살기 싫",
        "살기싫",
        "목숨",
    ]
    normalized = str(text or "").replace(" ", "").lower()
    return any(
        keyword.replace(" ", "").lower() in normalized for keyword in crisis_keywords
    )


def build_korean_crisis_response() -> str:
    return (
        "지금 정말 많이 힘든 상태로 보여요. 혼자 버티려고 하지 말고, "
        "바로 도움을 받을 수 있는 사람이나 기관에 연결되는 게 중요해요.\n\n"
        "지금 당장 스스로를 해칠 위험이 있거나 혼자 있기 어렵다면, "
        "아래 연락처로 바로 도움을 요청해 주세요.\n\n"
        "- 자살예방 상담전화: 109\n"
        "- 청소년 상담: 1388\n"
        "- 긴급 신고: 112\n"
        "- 응급 상황: 119\n\n"
        "아래의 **전문가 상담 연결 탭으로 이동** 버튼을 눌러 "
        "도움 받을 수 있는 기관도 확인할 수 있어요."
    )


def _normalized_text(text: str) -> str:
    return str(text or "").replace(" ", "").lower()


def _contains_any_phrase(text: str, phrases: List[str]) -> bool:
    normalized = _normalized_text(text)
    return any(_normalized_text(phrase) in normalized for phrase in phrases)


def _contains_positive_completion_phrase(text: str) -> bool:
    phrases = [
        "기뻐",
        "좋아",
        "다끝났다",
        "다 끝났다",
        "끝났다",
        "끝났어",
        "해냈어",
        "성공",
        "다행",
        "후련",
        "마지막이라",
        "마무리",
        "드디어",
        "완료",
        "끝냄",
        "끝냈다",
        "끝냈어",
    ]
    return _contains_any_phrase(text, phrases)


def _contains_greeting_or_thanks(text: str) -> bool:
    phrases = [
        "고마워",
        "고맙",
        "감사",
        "안녕",
        "하이",
        "좋은아침",
        "수고했어",
        "괜찮아졌어",
        "괜찮아진 것 같아",
    ]
    return _contains_any_phrase(text, phrases)


def _looks_overly_heavy_response(text: str) -> bool:
    heavy_markers = [
        "부담",
        "힘든",
        "힘들",
        "불안",
        "스트레스",
        "버텨온",
        "위태",
        "걱정",
        "압박",
    ]
    return _contains_any_phrase(text, heavy_markers)


def build_positive_completion_response() -> str:
    return (
        "오, 드디어 끝났구나. 정말 수고 많았어.\n\n"
        "마지막까지 해낸 것도, 끝났다는 걸 스스로 느끼는 것도 꽤 큰 일이야. "
        "지금은 다음 걱정으로 바로 넘어가기보다, ‘끝냈다’는 감각을 잠깐 느껴도 괜찮아.\n\n"
        "오늘은 작은 보상처럼 잠깐 쉬는 시간을 가져보자."
    )


def build_light_support_response() -> str:
    return (
        "말해줘서 고마워. 지금처럼 짧게라도 마음을 표현하는 것도 충분히 의미 있어.\n\n"
        "오늘은 너무 깊게 파고들기보다, 지금 기분을 편하게 이어서 이야기해도 좋아."
    )


def polish_chat_response_for_user_message(
    user_message: str,
    response_text: str,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    del summary
    user_message_text = str(user_message or "")
    candidate = str(response_text or "").strip()
    if _contains_crisis_phrase(user_message_text):
        return build_korean_crisis_response()
    if _contains_positive_completion_phrase(user_message_text):
        return build_positive_completion_response()
    if _contains_greeting_or_thanks(user_message_text) and _looks_overly_heavy_response(
        candidate
    ):
        return build_light_support_response()
    return candidate


def _should_show_expert_button(user_message: str, agent_result: Any) -> bool:
    del agent_result
    try:
        return _contains_crisis_phrase(user_message)
    except Exception:
        return False


def go_expert_tab() -> Any:
    return {"selected": "expert", "__type__": "update"}


def update_expert_button_visibility(
    chat_history: Optional[List[Any]],
    summary: Any,
) -> Any:
    user_message = ""

    for item in reversed(chat_history or []):
        # 1) dict 형태: {"role": "user", "content": "..."}
        if isinstance(item, dict):
            if item.get("role") == "user":
                user_message = str(item.get("content") or "")
                break

        # 2) ChatMessage 같은 객체 형태: item.role, item.content
        elif hasattr(item, "role") and hasattr(item, "content"):
            if getattr(item, "role", None) == "user":
                user_message = str(getattr(item, "content", "") or "")
                break

        # 3) 예전 Gradio tuple/list 형태: (user_message, assistant_message)
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            possible_user = item[0]
            if isinstance(possible_user, str) and possible_user.strip():
                user_message = possible_user
                break

    return _ui_update(visible=_should_show_expert_button(user_message, summary))


def escape_text(value: Any) -> str:
    return html.escape(str(value))


def safe_body_text(value: Any) -> str:
    return escape_text(value).replace("\n", "<br>")


def wrap_card(title: str, body_md: str, crisis: bool = False) -> str:
    card_class = "output-card crisis" if crisis else "output-card"
    return (
        f"<div class='{card_class}'>\n\n## {escape_text(title)}\n\n{body_md}\n\n</div>"
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_key_value(key: str, value: Any) -> Optional[str]:
    if key in RAW_LOOKING_KEYS:
        return None
    if isinstance(value, bool):
        return f"{key}: {value}"
    if isinstance(value, (int, float)):
        return f"{key}: {value}"
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 80 or "\n" in cleaned:
            return None
        if any(label in cleaned for label in INTERNAL_HINT_LABELS):
            return None
        return f"{key}: {escape_text(cleaned)}"
    return None


def _safe_list(values: Any, *, max_items: int = 6) -> List[str]:
    if not isinstance(values, list):
        return []

    safe_values: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 80 or "\n" in cleaned:
            continue
        if cleaned in RAW_LOOKING_KEYS:
            continue
        safe_values.append(escape_text(cleaned))
        if len(safe_values) >= max_items:
            break
    return safe_values


def _bool_from_presence(value: Any) -> bool:
    return bool(value) if not isinstance(value, dict) else bool(value.keys())


def _agent_details(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    details = _as_dict((result or {}).get("pipeline_details", {}))
    return _as_dict(details.get("agents", {}))


def _pipeline_details(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _as_dict((result or {}).get("pipeline_details", {}))


def _section(title: str, lines: List[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines if line) or "- not available"
    return f"### {title}\n{body}"


def _extract_labels(agent_data: Dict[str, Any]) -> List[str]:
    labels = _safe_list(agent_data.get("labels"))
    if labels:
        return labels

    candidates = agent_data.get("candidates")
    if isinstance(candidates, list):
        extracted = []
        for candidate in candidates:
            candidate_dict = _as_dict(candidate)
            label = candidate_dict.get("label")
            if isinstance(label, str):
                extracted.append(label)
        return _safe_list(extracted)

    secondary = _safe_list(agent_data.get("secondary_labels"))
    primary = agent_data.get("primary_label") or agent_data.get("primary_intent")
    if isinstance(primary, str):
        return _safe_list([primary] + secondary)
    return secondary


def _korean_intent_labels(labels: List[str]) -> List[str]:
    translated: List[str] = []
    seen = set()
    for label in labels:
        cleaned = str(label or "").strip()
        if not cleaned:
            continue
        display = INTENT_LABEL_KO.get(cleaned.upper(), cleaned)
        if display not in seen:
            translated.append(display)
            seen.add(display)
    return translated


def _expert_guidance_for_stage(risk_stage: str) -> str:
    if risk_stage == "위험":
        return "즉시 109, 119, 112에 연락하고, 가까운 믿을 수 있는 사람에게 알리세요. 즉각적인 위험이 있으면 가까운 응급실이나 지역 정신건강복지센터로 가세요."
    return "필요하면 가까운 사람이나 상담센터에 도움을 요청할 수 있어요."


def _dataset_lines(
    summary: Dict[str, Any], result: Optional[Dict[str, Any]]
) -> List[str]:
    details = _pipeline_details(result)
    hint_keys = []
    for key in ("counseling_hint", "empathy_style_hint", "wellness_hint"):
        if summary.get(key) or (result or {}).get(key):
            hint_keys.append(key)

    lines = [f"hint_keys: {', '.join(hint_keys) if hint_keys else 'none'}"]
    for source_key in ("counseling", "empathy", "wellness"):
        source = _as_dict(details.get(source_key))
        category = source.get("category") or source.get("matched_category")
        score = source.get("score")
        if score is None:
            score = source.get("similarity_score")
        if score is None:
            score = source.get("confidence")
        matched_record_id = source.get("matched_record_id")
        safe_parts = []
        if isinstance(category, str) and len(category) <= 80:
            safe_parts.append(f"category={escape_text(category)}")
        if isinstance(score, (int, float)):
            safe_parts.append(f"score={score}")
            if score <= 0:
                safe_parts.append("low_confidence_match=True")
        if isinstance(matched_record_id, str) and matched_record_id:
            safe_parts.append(f"record_id_present=True")
        if safe_parts:
            lines.append(f"{source_key}: " + ", ".join(safe_parts))
    return lines


def _final_safety_summary(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    safety: Dict[str, Any],
) -> Dict[str, Any]:
    final_stage = result.get("risk_stage") or summary.get("risk_stage", "관심")
    final_level = result.get("risk_level") or summary.get("risk_level", "none")
    final_crisis = result.get(
        "requires_crisis_response",
        summary.get("requires_crisis_response", False),
    )

    merged = dict(safety)
    merged.update(
        {
            "risk_stage": final_stage,
            "risk_level": final_level,
            "requires_crisis_response": final_crisis,
        }
    )
    return merged


DEFAULT_SAFETY_NOTICE_START = (
    "이 AI는 의료 진단이나 치료를 하지 않으며 전문 상담사를 대체하지 않습니다."
)


def _strip_default_safety_notice(response_text: str, *, risk_stage: str) -> str:
    if risk_stage == "위험":
        return response_text

    marker = "\n\n" + DEFAULT_SAFETY_NOTICE_START
    if marker in response_text:
        return response_text.split(marker, 1)[0].rstrip()
    if response_text.startswith(DEFAULT_SAFETY_NOTICE_START):
        return ""
    return response_text


def build_agent_pipeline_markdown(
    summary: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a raw-text-safe Agent Pipeline View markdown block."""
    summary = summary or {}
    result = result or {}
    details = _pipeline_details(result)
    agents = _agent_details(result)
    timing = _as_dict(details.get("timing"))

    safety = _final_safety_summary(
        summary,
        result,
        _as_dict(agents.get("safety")) or _as_dict(details.get("safety")),
    )
    emotion = _as_dict(agents.get("emotion"))
    intent = _as_dict(agents.get("intent"))
    memory = (
        _as_dict(agents.get("recall"))
        or _as_dict(agents.get("memory"))
        or _as_dict(details.get("memory_context"))
    )
    memory = _as_dict(agents.get("memory_recall")) or memory
    state = _as_dict(agents.get("state")) or _as_dict(agents.get("emotional_state"))
    decision = _as_dict(agents.get("decision"))

    safety_lines = [
        _safe_key_value(
            "risk_stage", safety.get("risk_stage", summary.get("risk_stage", "관심"))
        ),
        _safe_key_value(
            "requires_crisis_response",
            safety.get(
                "requires_crisis_response",
                summary.get("requires_crisis_response", False),
            ),
        ),
        _safe_key_value(
            "risk_level", safety.get("risk_level", summary.get("risk_level", "none"))
        ),
    ]

    emotion_labels = _extract_labels(emotion)
    emotion_lines = [
        _safe_key_value(
            "dominant_label",
            emotion.get("dominant_label") or emotion.get("primary_label"),
        ),
        f"labels: {', '.join(emotion_labels) if emotion_labels else 'none'}",
        _safe_key_value("labels_count", len(emotion_labels)),
        _safe_key_value("intensity", emotion.get("intensity")),
        _safe_key_value("confidence", emotion.get("confidence")),
    ]

    intent_labels = _extract_labels(intent)
    intent_lines = [
        _safe_key_value("primary_intent", intent.get("primary_intent")),
        _safe_key_value("s2_suspected", intent.get("s2_suspected")),
        _safe_key_value("s3_sos", intent.get("s3_sos")),
        f"labels: {', '.join(intent_labels) if intent_labels else 'none'}",
    ]

    memory_counts = []
    for key in ("recent_summaries", "facts", "directives", "emotional_trend"):
        value = memory.get(key)
        if isinstance(value, int):
            memory_counts.append(f"{key}={value}")
        elif isinstance(value, list):
            memory_counts.append(f"{key}={len(value)}")
    recalled_keys = _safe_list(memory.get("recalled_keys"))
    repeated_concerns = _safe_list(memory.get("repeated_concerns"))
    memory_lines = [
        f"memory_context_count: {', '.join(memory_counts) if memory_counts else 'none'}",
        f"recalled_keys: {', '.join(recalled_keys) if recalled_keys else 'none'}",
        f"repeated_concerns: {', '.join(repeated_concerns) if repeated_concerns else 'none'}",
        _safe_key_value(
            "last_small_action_present",
            _bool_from_presence(memory.get("last_small_action")),
        ),
        _safe_key_value(
            "next_follow_up_present", _bool_from_presence(memory.get("next_follow_up"))
        ),
    ]

    state_summary = _safe_list(state.get("state_summary"))
    state_lines = [
        f"state_summary: {', '.join(state_summary) if state_summary else 'none'}",
    ]
    for key in ("mood", "anxiety", "stress", "sleep", "energy", "safety", "rapport"):
        state_lines.append(_safe_key_value(key, state.get(key)))

    secondary_actions = _safe_list(decision.get("secondary_actions"))
    reason_codes = _safe_list(decision.get("reason_codes"))
    constraints = _as_dict(decision.get("response_constraints"))
    constraint_lines = []
    for key in sorted(constraints.keys()):
        rendered = _safe_key_value(key, constraints.get(key))
        if rendered:
            constraint_lines.append(rendered)
    decision_lines = [
        _safe_key_value(
            "primary_action", decision.get("primary_action") or decision.get("action")
        ),
        f"secondary_actions: {', '.join(secondary_actions) if secondary_actions else 'none'}",
        f"reason_codes: {', '.join(reason_codes) if reason_codes else 'none'}",
        f"response_constraints: {', '.join(constraint_lines) if constraint_lines else 'none'}",
    ]

    response_lines = [
        _safe_key_value(
            "response_generated",
            bool(result.get("response") or summary.get("response_preview")),
        ),
        _safe_key_value(
            "safety_notice_added", bool(summary.get("requires_crisis_response"))
        ),
        _safe_key_value(
            "mode", result.get("response_source") or summary.get("response_source")
        ),
    ]
    timing_lines = []
    for key in (
        "initialize",
        "safety",
        "dataset_retrieval",
        "memory_context",
        "agent_pipeline",
        "response_generation",
        "total",
    ):
        rendered = _safe_key_value(key, timing.get(key))
        if rendered:
            timing_lines.append(rendered)

    sections = [
        _section("Safety Agent", safety_lines),
        _section("Emotion Agent", emotion_lines),
        _section("Intent Agent", intent_lines),
        _section("Dataset Strategy Agent", _dataset_lines(summary, result)),
        _section("Memory Agent / Proactive Recall", memory_lines),
        _section("Emotional State Agent", state_lines),
        _section("Decision Agent", decision_lines),
        _section("Response Agent", response_lines),
        _section("Timing", timing_lines),
    ]
    return wrap_card("Agent Pipeline View", "\n\n".join(sections))


def build_crisis_markdown() -> str:
    return wrap_card(
        "위기 안내 카드",
        "\n".join(
            [
                "- 위험 단계: 위험",
                "- 지금은 안전이 가장 중요해요. 혼자 버티지 말고 즉시 도움을 요청하세요.",
                "- 109(자살예방), 119(응급), 112(경찰) 중 하나로 바로 연락하세요.",
                "- 가까이에 믿을 수 있는 사람에게 지금 연락해서 혼자 있지 않도록 해주세요.",
                "- 즉각적인 위험이 있으면 가장 가까운 응급실이나 지역 정신건강복지센터로 가세요.",
            ]
        ),
        crisis=True,
    )


def infer_risk_stage(wellness_checkin: Dict[str, int]) -> str:
    concern_signals = [
        wellness_checkin.get("mood_score", 3) <= 2,
        wellness_checkin.get("anxiety_score", 3) >= 4,
        wellness_checkin.get("loneliness_score", 3) >= 4,
        wellness_checkin.get("sleep_quality", 3) <= 2,
        wellness_checkin.get("meal_status", 3) <= 2,
        wellness_checkin.get("energy_score", 3) <= 2,
        wellness_checkin.get("stress_score", 3) >= 4,
    ]
    return "주의" if any(concern_signals) else "관심"


def build_counseling_hint(message: str, wellness_checkin: Dict[str, int]) -> str:
    if "공부" in message:
        return "할 일을 아주 작게 쪼개서 10분만 시작해도 부담이 줄어요."
    if wellness_checkin.get("stress_score", 3) >= 4:
        return "스트레스가 높을 때는 우선순위를 하나만 정하고 나머지는 잠시 미뤄보세요."
    if wellness_checkin.get("anxiety_score", 3) >= 4:
        return "불안이 올라올 때는 호흡을 천천히 맞추며 현재 감각에 집중해보세요."
    return "감정을 설명하기 어려우면, 지금 가장 힘든 점 하나만 먼저 적어도 충분해요."


def build_empathy_hint(message: str, wellness_checkin: Dict[str, int]) -> str:
    if "외로" in message:
        return (
            "지금 많이 혼자 버티고 있다는 느낌을 먼저 알아봐 주는 반응이 도움이 됩니다."
        )
    if wellness_checkin.get("loneliness_score", 3) >= 4:
        return "외로움이 높을수록 판단보다 공감과 동행 메시지가 먼저 필요해요."
    return "반응은 조언보다 감정 확인으로 시작하면 더 안전하고 자연스럽습니다."


def build_wellness_hint(wellness_checkin: Dict[str, int]) -> str:
    hints = []
    if wellness_checkin.get("sleep_quality", 3) <= 2:
        hints.append("수면 회복을 우선해 보세요.")
    if wellness_checkin.get("meal_status", 3) <= 2:
        hints.append("따뜻한 물이나 간단한 식사부터 챙겨보세요.")
    if wellness_checkin.get("energy_score", 3) <= 2:
        hints.append("오늘은 짧게 쉬는 시간을 자주 두는 편이 좋아요.")
    if wellness_checkin.get("stress_score", 3) >= 4:
        hints.append("스트레스가 높으면 해야 할 일을 한 줄로만 적어보세요.")
    return " ".join(hints) or "기본 리듬을 지키는 것만으로도 회복에 도움이 됩니다."


def build_mock_summary(
    message: str,
    wellness_checkin: Dict[str, int],
    response_text: str,
    *,
    risk_stage: Optional[str] = None,
    source: str = "mock",
) -> Dict[str, Any]:
    stage = risk_stage or infer_risk_stage(wellness_checkin)
    risk_level = (
        "attention" if stage == "관심" else "high" if stage == "위험" else "moderate"
    )
    return {
        "session_id": "",
        "risk_level": risk_level,
        "risk_stage": stage,
        "requires_crisis_response": False,
        "counseling_hint": build_counseling_hint(message, wellness_checkin),
        "empathy_style_hint": build_empathy_hint(message, wellness_checkin),
        "wellness_hint": build_wellness_hint(wellness_checkin),
        "response_source": source,
        "response_preview": response_text,
        "wellness_checkin": wellness_checkin,
    }


def build_general_markdown(
    summary: Dict[str, Any],
    response_text: str,
    result: Optional[Dict[str, Any]] = None,
) -> str:
    risk_stage = summary.get("risk_stage", "관심")
    visible_response = _strip_default_safety_notice(
        response_text, risk_stage=risk_stage
    )
    return "\n\n".join(
        [
            wrap_card("상담 응답 카드", f"- 위험 단계: {escape_text(risk_stage)}"),
            wrap_card("AI 상담 응답", safe_body_text(visible_response)),
        ]
    )


def build_chat_response_text(
    summary: Dict[str, Any],
    response_text: str,
) -> str:
    """Return only user-facing chat text, without debug pipeline or duplicate notice."""
    risk_stage = (summary or {}).get("risk_stage", "관심")
    return _strip_default_safety_notice(response_text, risk_stage=risk_stage).strip()


def initial_chat_messages() -> List[ChatMessage]:
    """Return a fresh copy of the initial chat messages."""
    return [dict(message) for message in INITIAL_CHAT_HISTORY]


def structured_memory_chat_messages(report: Dict[str, Any]) -> List[ChatMessage]:
    """Build a chat-safe account summary from whitelisted structured fields."""
    metrics = _as_dict(report.get("metrics"))
    issues = (
        ", ".join(_strings_for_display(report.get("main_issue")))
        or "확인된 주요 고민 없음"
    )
    status = str(report.get("action_status") or "확인되지 않음")
    follow_up = str(
        report.get("next_follow_up") or "이어서 이야기하며 함께 정해볼게요."
    )
    lines = [
        "## 지난 상담 요약",
        "이전 상담 흐름을 바탕으로 이어서 이야기할 수 있어요.",
        f"- 주요 고민: {issues}",
        f"- 최근 행동 상태: {status}",
        f"- 다음 질문: {follow_up}",
    ]
    if metrics.get("recent_risk_signal"):
        lines.append("- 최근 위험 신호: 있음")
    content = "\n".join(lines)
    return [{"role": "assistant", "content": content}]


def _structured_record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def structured_records_chat_messages(records: List[Any]) -> List[ChatMessage]:
    """Build a short login summary directly from at most three structured records."""
    recent_records = list(records[-3:])
    issues: List[str] = []
    action_status = "확인되지 않음"
    next_follow_up = "이어서 이야기하며 함께 정해볼게요."
    risk_signal = False
    status_labels = {
        "suggested": "제안됨",
        "partial": "일부 진행됨",
        "in_progress": "일부 진행됨",
        "completed": "완료함",
        "done": "완료함",
        "not_completed": "진행하지 못함",
        "failed": "진행하지 못함",
    }
    for record in reversed(recent_records):
        if not issues:
            labels = _strings_for_display(
                _structured_record_value(record, "main_issue")
            )
            if not labels:
                labels = _strings_for_display(
                    _structured_record_value(record, "intent_label")
                )
            issues = [
                label
                for label in _korean_intent_labels(labels)
                if label not in {"기타 고민", "OTHER_CONCERN", "other_concern"}
            ][:3]
        if action_status == "확인되지 않음":
            status = (
                str(_structured_record_value(record, "action_status") or "")
                .strip()
                .lower()
            )
            if status:
                action_status = status_labels.get(status, "확인되지 않음")
        if next_follow_up.startswith("이어서"):
            follow_up = str(
                _structured_record_value(record, "next_follow_up") or ""
            ).strip()
            if follow_up:
                next_follow_up = follow_up
        risk_signal = (
            risk_signal or _structured_record_value(record, "risk_stage") == "위험"
        )

    report = {
        "main_issue": issues or ["이전 상담 주제"],
        "action_status": action_status,
        "next_follow_up": next_follow_up,
        "metrics": {"recent_risk_signal": risk_signal},
    }
    return structured_memory_chat_messages(report)


def _strings_for_display(value: Any) -> List[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        str(item).strip() for item in values if isinstance(item, str) and item.strip()
    ]


def initial_chat_history() -> List[ChatMessage]:
    """Backward-compatible alias for the initial chat messages."""
    return initial_chat_messages()


def reset_chat_history() -> Tuple[List[ChatMessage], List[ChatMessage]]:
    """Reset chat display and state to the initial assistant greeting."""
    history = initial_chat_messages()
    return history, [dict(message) for message in history]


def _bounded_score(value: Any, default: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return max(1, min(5, score))


def _short_note(note: str) -> str:
    compact = " ".join((note or "").split())
    if not compact:
        return ""
    if any(label in compact for label in INTERNAL_HINT_LABELS):
        return ""
    return compact[:80]


def _diary_entries(diary_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(diary_state, dict) or not diary_state:
        return []
    entries = diary_state.get("entries")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]

    if "mood_score" in diary_state:
        legacy_entry = {
            "timestamp": diary_state.get("timestamp", ""),
            "emotion_label": diary_state.get("emotion_label", "선택 안 함"),
            "mood_score": diary_state.get("mood_score"),
            "anxiety_score": diary_state.get("anxiety_score"),
            "loneliness_score": diary_state.get("loneliness_score"),
            "sleep_quality": diary_state.get(
                "sleep_quality", diary_state.get("sleep_score")
            ),
            "meal_status": diary_state.get("meal_status", 3),
            "energy_score": diary_state.get("energy_score", 3),
            "stress_score": diary_state.get("stress_score", 3),
            "risk_stage": diary_state.get("risk_stage", "관심"),
            "note": diary_state.get("note", ""),
        }
        return [legacy_entry]
    return []


def _score_label(value: Any) -> str:
    return {
        1: "매우 낮음",
        2: "낮음",
        3: "보통",
        4: "높음",
        5: "매우 높음",
    }[_bounded_score(value)]


def _burden_label(value: Any) -> str:
    return {
        1: "낮음",
        2: "약간 있음",
        3: "보통",
        4: "높음",
        5: "매우 높음",
    }[_bounded_score(value)]


def _sleep_label(value: Any) -> str:
    return {
        1: "매우 부족",
        2: "부족",
        3: "보통",
        4: "양호",
        5: "매우 양호",
    }[_bounded_score(value)]


def _energy_label(value: Any) -> str:
    return {
        1: "매우 낮음",
        2: "낮음",
        3: "보통",
        4: "양호",
        5: "매우 높음",
    }[_bounded_score(value)]


def _normalize_agent_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _primary_intent_label(intent: Dict[str, Any]) -> str:
    candidates = []
    primary = intent.get("primary_intent")
    if isinstance(primary, str) and primary:
        candidates.append(primary)
    candidates.extend(_extract_labels(intent))

    for candidate in candidates:
        normalized = _normalize_agent_key(candidate)
        if normalized:
            return INTENT_LABEL_KO.get(normalized, normalized)
    return "아직 판단 전"


def _state_level(value: Any, *, reverse: bool = False) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return ""

    if reverse:
        if score < 0.4:
            return "낮음"
        if score < 0.65:
            return "보통"
        return "양호"

    if score >= 0.65:
        return "높음"
    if score >= 0.4:
        return "보통"
    return "낮음"


def _diary_burden_level(value: Any) -> str:
    score = _bounded_score(value)
    if score >= 4:
        return "높음"
    if score == 3:
        return "보통"
    return "낮음"


def _diary_recovery_level(value: Any) -> str:
    score = _bounded_score(value)
    if score <= 2:
        return "낮음"
    if score == 3:
        return "보통"
    return "양호"


def _emotional_state_labels(
    state: Dict[str, Any],
    latest_diary: Dict[str, Any],
    wellness_checkin: Dict[str, Any],
) -> List[str]:
    labels = []

    anxiety = _state_level(state.get("anxiety"))
    stress = _state_level(state.get("stress"))
    sleep = _state_level(state.get("sleep"), reverse=True)

    if not anxiety and (latest_diary or wellness_checkin):
        anxiety = _diary_burden_level(
            latest_diary.get("anxiety_score", wellness_checkin.get("anxiety_score", 3))
        )
    if not stress and (latest_diary or wellness_checkin):
        stress = _diary_burden_level(
            latest_diary.get("stress_score", wellness_checkin.get("stress_score", 3))
        )
    if not sleep and (latest_diary or wellness_checkin):
        sleep = _diary_recovery_level(
            latest_diary.get("sleep_quality", wellness_checkin.get("sleep_quality", 3))
        )

    if anxiety:
        labels.append(f"불안 {anxiety}")
    if stress:
        labels.append(f"스트레스 {stress}")
    if sleep:
        labels.append(f"수면 회복감 {sleep}")
    return labels or ["상담 또는 감정일기 저장 후 표시"]


def _is_high_anxiety_or_stress(
    state: Dict[str, Any],
    latest_diary: Dict[str, Any],
    wellness_checkin: Dict[str, Any],
) -> bool:
    for key in ("anxiety", "stress"):
        try:
            if float(state.get(key, 0.0)) >= 0.65:
                return True
        except (TypeError, ValueError):
            pass

    anxiety_score = latest_diary.get(
        "anxiety_score", wellness_checkin.get("anxiety_score", 3)
    )
    stress_score = latest_diary.get(
        "stress_score", wellness_checkin.get("stress_score", 3)
    )
    return _bounded_score(anxiety_score) >= 4 or _bounded_score(stress_score) >= 4


def _selected_cause_label(cause: Dict[str, Any]) -> Tuple[str, str]:
    selected = (
        cause.get("selected_cause")
        if isinstance(cause.get("selected_cause"), str)
        else ""
    )
    if not selected:
        candidates = cause.get("cause_candidates")
        if isinstance(candidates, list):
            selected = next(
                (item for item in candidates if isinstance(item, str) and item), ""
            )
    return selected, CAUSE_LABEL_KO.get(selected, selected) if selected else ""


def _decision_action_summary(
    decision: Dict[str, Any], small_action: Dict[str, Any]
) -> str:
    primary = _normalize_agent_key(
        decision.get("primary_action") or decision.get("action")
    )
    secondary = {
        _normalize_agent_key(action)
        for action in _safe_list(decision.get("secondary_actions"), max_items=8)
    }

    parts = []
    if primary == "ESCALATE_SAFETY":
        return "즉시 안전 안내"
    if primary == "ASK_FOLLOW_UP":
        parts.append("후속 질문")
    elif primary == "RESPOND_SUPPORTIVELY":
        parts.append("공감 응답")
    elif primary == "SUMMARIZE_STATE":
        parts.append("상태 요약")

    if "SUGGEST_SMALL_ACTION" in secondary or bool(small_action.get("has_action")):
        parts.append("작은 실천 행동 제안")
    if "UPDATE_MEMORY" in secondary:
        parts.append("상담 흐름 기억")
    return " + ".join(parts) if parts else "공감 응답"


def _next_counseling_plan(
    *,
    risk_stage: str,
    selected_cause: str,
    high_anxiety_or_stress: bool,
) -> List[str]:
    if risk_stage == "위험":
        return [
            "지금은 상담 계획보다 즉각적인 안전 확보가 우선입니다.",
            "109, 119, 112 중 하나로 바로 연락하고, 가까이에 믿을 수 있는 사람에게 지금 알려 혼자 있지 않도록 합니다.",
        ]

    plan = []
    if selected_cause == "sleep_maintenance":
        plan.append("중간에 깨는 원인이 걱정 때문인지, 신체 긴장 때문인지 확인합니다.")
    elif selected_cause == "worry_or_anxiety":
        plan.append("잠들기 전 커지는 걱정의 주제를 함께 좁혀봅니다.")
    else:
        plan.append(
            "현재 가장 부담이 큰 감정과 상황을 한 가지로 좁혀 다음 대화를 이어갑니다."
        )

    if high_anxiety_or_stress:
        plan.append("불안과 스트레스 변화를 감정일기로 추적합니다.")
    return plan


def _safe_report_sentence(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return fallback
    if len(text) > 120:
        return fallback
    if any(label in text for label in INTERNAL_HINT_LABELS):
        return fallback
    if any(raw_key in text for raw_key in RAW_LOOKING_KEYS):
        return fallback
    return escape_text(text)


def _recent_status_labels(
    state: Dict[str, Any],
    latest_diary: Dict[str, Any],
    wellness_checkin: Dict[str, Any],
) -> Dict[str, str]:
    if latest_diary:
        return {
            "불안": _burden_label(latest_diary.get("anxiety_score")),
            "스트레스": _burden_label(latest_diary.get("stress_score")),
            "수면 회복감": _diary_recovery_level(latest_diary.get("sleep_quality")),
            "활력": _diary_recovery_level(latest_diary.get("energy_score")),
        }

    if wellness_checkin:
        return {
            "불안": _burden_label(wellness_checkin.get("anxiety_score")),
            "스트레스": _burden_label(wellness_checkin.get("stress_score")),
            "수면 회복감": _diary_recovery_level(wellness_checkin.get("sleep_quality")),
            "활력": _diary_recovery_level(wellness_checkin.get("energy_score")),
        }

    labels: Dict[str, str] = {}
    anxiety = _state_level(state.get("anxiety"))
    stress = _state_level(state.get("stress"))
    sleep = _state_level(state.get("sleep"), reverse=True)
    energy = _state_level(state.get("energy"))
    if anxiety:
        labels["불안"] = anxiety
    if stress:
        labels["스트레스"] = stress
    if sleep:
        labels["수면 회복감"] = sleep
    if energy:
        labels["활력"] = energy
    return labels


def _recent_status_summary(status_labels: Dict[str, str]) -> str:
    if not status_labels:
        return "상담 또는 감정일기를 저장하면 최근 상태를 요약해드릴게요."

    high_burdens = [
        label
        for label in ("불안", "스트레스")
        if status_labels.get(label) in {"높음", "매우 높음"}
    ]
    low_recovery = [
        label
        for label in ("수면 회복감", "활력")
        if status_labels.get(label) in {"낮음", "매우 낮음", "부족", "매우 부족"}
    ]

    parts = []
    if high_burdens:
        parts.append(f"{'과 '.join(high_burdens)}가 높고")
    if low_recovery:
        parts.append(f"{'과 '.join(low_recovery)}이 낮은")

    if parts:
        return "현재는 " + ", ".join(parts) + " 상태로 보입니다."

    stable = [
        label
        for label, value in status_labels.items()
        if value in {"보통", "양호", "매우 양호", "높음", "매우 높음"}
    ]
    if stable:
        return (
            f"현재는 {', '.join(stable)} 상태가 비교적 유지되고 있는 것으로 보입니다."
        )
    return "현재 마음 상태는 추가 기록을 통해 더 분명하게 확인할 수 있습니다."


def _change_word(first: Any, last: Any) -> str:
    delta = _bounded_score(last) - _bounded_score(first)
    if delta > 0:
        return "높아지고"
    if delta < 0:
        return "낮아지고"
    return "비슷하게 유지되고"


def _natural_trend_summary(first: Dict[str, Any], last: Dict[str, Any]) -> str:
    changes = {
        "기분": _change_word(first.get("mood_score"), last.get("mood_score")),
        "불안": _change_word(first.get("anxiety_score"), last.get("anxiety_score")),
        "스트레스": _change_word(first.get("stress_score"), last.get("stress_score")),
        "수면 회복감": _change_word(
            first.get("sleep_quality"), last.get("sleep_quality")
        ),
        "활력": _change_word(first.get("energy_score"), last.get("energy_score")),
    }

    groups: Dict[str, List[str]] = {}
    for label, change in changes.items():
        groups.setdefault(change, []).append(label)

    parts = [f"{', '.join(labels)}: {change}" for change, labels in groups.items()]
    return "최근 기록과 비교했을 때 " + ", ".join(parts) + " 있는 흐름입니다."


def build_emotional_trend_markdown(diary_state: Optional[Dict[str, Any]]) -> str:
    entries = _diary_entries(diary_state)
    if len(entries) < 2:
        if entries:
            return "- 첫 기준선이 기록됐어요. 감정일기를 한 번 더 저장하면 변화 방향을 함께 보여드릴게요."
        return "- 아직 감정 변화 기록이 없습니다."

    first = entries[0]
    last = entries[-1]
    return _natural_trend_summary(first, last)


def _trend_snapshots(context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    values = context.get("trend_snapshots", []) if isinstance(context, dict) else []
    if not isinstance(values, list):
        return []
    return [
        dict(item) for item in values[-TREND_SNAPSHOT_LIMIT:] if isinstance(item, dict)
    ]


def _sanitize_demo_report_snapshot(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    snapshot: Dict[str, Any] = {}
    intent = value.get("intent_label") or value.get("primary_intent")
    text_fields = {
        "risk_stage": value.get("risk_stage"),
        "intent_label": intent,
        "selected_cause": value.get("selected_cause"),
        "action_status": value.get("action_status"),
        "small_action": value.get("small_action"),
        "created_at": value.get("created_at"),
    }
    for key, item in text_fields.items():
        if isinstance(item, str) and item.strip():
            snapshot[key] = item.strip()[:200]
    for key in ("anxiety", "stress", "sleep", "energy"):
        number = _numeric_trend_value(value.get(key))
        if number is not None:
            snapshot[key] = number
    snapshot.setdefault("risk_stage", "관심")
    snapshot.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    return snapshot


def load_demo_report_cache() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    try:
        if not DEMO_REPORT_CACHE_PATH.exists():
            return {}
        with DEMO_REPORT_CACHE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return {}
        cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for user_id, value in payload.items():
            if not isinstance(user_id, str) or not isinstance(value, dict):
                continue
            snapshots = value.get("snapshots")
            if not isinstance(snapshots, list):
                continue
            safe = [
                snapshot
                for item in snapshots[-TREND_SNAPSHOT_LIMIT:]
                if (snapshot := _sanitize_demo_report_snapshot(item))
            ]
            if safe:
                cache[user_id] = {"snapshots": safe}
        return cache
    except (OSError, ValueError, TypeError):
        return {}


def save_demo_report_cache(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    safe_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for user_id, value in data.items():
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(value, dict)
        ):
            continue
        snapshots = value.get("snapshots")
        if not isinstance(snapshots, list):
            continue
        safe = [
            snapshot
            for item in snapshots[-TREND_SNAPSHOT_LIMIT:]
            if (snapshot := _sanitize_demo_report_snapshot(item))
        ]
        if safe:
            safe_data[user_id.strip()] = {"snapshots": safe}
    temporary_path = DEMO_REPORT_CACHE_PATH.with_suffix(
        f"{DEMO_REPORT_CACHE_PATH.suffix}.tmp"
    )
    try:
        DEMO_REPORT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(safe_data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, DEMO_REPORT_CACHE_PATH)
        return True
    except OSError:
        return False


def append_demo_report_snapshot(user_id: str, snapshot: Any) -> None:
    normalized_user_id = _normalized_user_id(user_id)
    safe_snapshot = _sanitize_demo_report_snapshot(snapshot)
    if not normalized_user_id or not safe_snapshot:
        return
    cache = load_demo_report_cache()
    snapshots = cache.get(normalized_user_id, {}).get("snapshots", [])
    cache[normalized_user_id] = {
        "snapshots": [*snapshots, safe_snapshot][-TREND_SNAPSHOT_LIMIT:]
    }
    save_demo_report_cache(cache)


def get_demo_report_snapshots(user_id: str) -> List[Dict[str, Any]]:
    normalized_user_id = _normalized_user_id(user_id)
    if not normalized_user_id:
        return []
    return list(
        load_demo_report_cache().get(normalized_user_id, {}).get("snapshots", [])
    )


def build_previous_summary_bubble(snapshots: Any) -> str:
    safe = [
        snapshot
        for item in (
            snapshots[-TREND_SNAPSHOT_LIMIT:] if isinstance(snapshots, list) else []
        )
        if (snapshot := _sanitize_demo_report_snapshot(item))
    ]
    intent = "이전 상담"
    for snapshot in reversed(safe):
        candidate = str(snapshot.get("intent_label") or "").strip()
        if candidate and not _is_generic_report_intent(candidate):
            translated = _korean_intent_labels([candidate])
            intent = translated[0] if translated else candidate
            break
    return f"지난 상담에서는 {intent} 흐름이 있었어요. 요즘은 어떤가요?"


def _cache_snapshots_as_trend(snapshots: Any) -> List[Dict[str, Any]]:
    trend = []
    for item in (
        snapshots[-TREND_SNAPSHOT_LIMIT:] if isinstance(snapshots, list) else []
    ):
        snapshot = _sanitize_demo_report_snapshot(item)
        if not snapshot:
            continue
        converted = dict(snapshot)
        converted["primary_intent"] = converted.pop("intent_label", "")
        trend.append(converted)
    return trend


def build_trend_from_snapshots(snapshots: Any) -> str:
    trend = _cache_snapshots_as_trend(snapshots)
    if len(trend) < 2:
        return TREND_INSUFFICIENT_MESSAGE
    return build_recent_trend_markdown(trend)


def _restored_reflection_records(
    context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values = (
        context.get("restored_reflection_records", [])
        if isinstance(context, dict)
        else []
    )
    if not isinstance(values, list):
        return []
    records = []
    for value in values[-30:]:
        if isinstance(value, dict):
            records.append(
                {
                    key: value.get(key)
                    for key in RESTORED_REFLECTION_FIELDS
                    if key in value
                }
            )
    return records


def _restored_records_as_trend_snapshots(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    snapshots = []
    for record in records:
        issues = record.get("main_issue")
        issue = issues[-1] if isinstance(issues, list) and issues else ""
        snapshot: Dict[str, Any] = {
            "risk_stage": str(record.get("risk_stage") or "관심"),
            "primary_intent": str(issue or record.get("intent_label") or ""),
            "action_status": str(record.get("action_status") or ""),
        }
        action = record.get("last_small_action")
        if isinstance(action, str) and action.strip():
            snapshot["small_action"] = action.strip()
        emotion = str(record.get("emotion_hint") or "")
        if "불안" in emotion:
            snapshot["anxiety"] = 0.8
        if "스트레스" in emotion:
            snapshot["stress"] = 0.8
        if "수면" in emotion:
            snapshot["sleep"] = 0.2
        if "피로" in emotion:
            snapshot["energy"] = 0.2
        snapshots.append(snapshot)
    return snapshots


def _numeric_trend_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_trend_snapshot(
    summary: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    safe_summary = summary if isinstance(summary, dict) else {}
    agents = _agent_details(result)
    intent = _as_dict(agents.get("intent"))
    cause = _as_dict(agents.get("cause_exploration"))
    action_checkin = _as_dict(agents.get("action_checkin"))
    small_action = _as_dict(agents.get("small_action"))
    emotional_state = _as_dict(agents.get("emotional_state"))
    wellness = _as_dict(safe_summary.get("wellness_checkin"))
    snapshot: Dict[str, Any] = {
        "risk_stage": str(
            safe_summary.get("risk_stage") or (result or {}).get("risk_stage") or "관심"
        ),
        "primary_intent": str(intent.get("primary_intent") or ""),
        "selected_cause": str(cause.get("selected_cause") or ""),
        "action_status": str(action_checkin.get("status") or ""),
    }
    action_text = small_action.get("action_text")
    if isinstance(action_text, str) and action_text.strip():
        snapshot["small_action"] = action_text.strip()
    for key in ("anxiety", "stress", "sleep", "energy"):
        value = _numeric_trend_value(emotional_state.get(key))
        if value is None:
            wellness_key = (
                f"{key}_score" if key in {"anxiety", "stress"} else f"{key}_quality"
            )
            if key == "energy":
                wellness_key = "energy_score"
            value = _numeric_trend_value(wellness.get(wellness_key))
        if value is not None:
            snapshot[key] = value
    return snapshot


def append_trend_snapshot(
    context: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    updated = dict(context) if isinstance(context, dict) else {}
    snapshots = _trend_snapshots(updated)
    snapshots.append(_build_trend_snapshot(summary, result))
    updated["trend_snapshots"] = snapshots[-TREND_SNAPSHOT_LIMIT:]
    return updated


def _is_generic_report_intent(value: Any) -> bool:
    normalized = "_".join(str(value or "").strip().lower().split())
    return not normalized or normalized in {"기타_고민", "other_concern"}


def _latest_snapshot_text(snapshots: Any, key: str) -> str:
    if not isinstance(snapshots, list):
        return ""
    for snapshot in reversed(snapshots[-TREND_SNAPSHOT_LIMIT:]):
        if not isinstance(snapshot, dict):
            continue
        value = snapshot.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _low_recovery_burden(value: Any) -> float:
    score = _numeric_trend_value(value)
    if score is None:
        return 0.0
    if 0.0 <= score <= 1.0:
        return 1.0 - score
    return max(0.0, min(1.0, (5.0 - score) / 4.0))


def _trend_burden_score(snapshot: Dict[str, Any]) -> float:
    risk_score = {"위험": 3.0, "주의": 2.0, "관심": 1.0}.get(
        str(snapshot.get("risk_stage") or "관심"), 1.0
    )
    burden = risk_score
    for key in ("anxiety", "stress"):
        value = _numeric_trend_value(snapshot.get(key))
        if value is not None:
            burden += value
    burden += _low_recovery_burden(snapshot.get("sleep"))
    burden += _low_recovery_burden(snapshot.get("energy"))
    action_status = str(snapshot.get("action_status") or "")
    if action_status in {"completed", "done"}:
        burden -= 0.5
    elif action_status in {"partial", "in_progress"}:
        burden -= 0.25
    elif action_status in {"not_done", "not_completed", "failed"}:
        burden += 0.5
    return burden


def build_recent_trend_markdown(snapshots: Any) -> str:
    safe_snapshots = (
        [item for item in snapshots[-TREND_SNAPSHOT_LIMIT:] if isinstance(item, dict)]
        if isinstance(snapshots, list)
        else []
    )
    if len(safe_snapshots) < 2:
        return TREND_INSUFFICIENT_MESSAGE
    delta = _trend_burden_score(safe_snapshots[-1]) - _trend_burden_score(
        safe_snapshots[0]
    )
    if delta < -0.25:
        return "초기보다 부담 신호가 낮아져 회복 방향으로 움직이는 흐름이 보입니다."
    if delta > 0.25:
        return "최근 부담 신호가 높아지는 흐름이 있어 안정화와 도움 요청을 우선하는 것이 좋습니다."
    return "최근 부담 수준은 비슷하게 유지되고 있으며, 작은 행동을 이어가며 변화를 살펴볼 수 있습니다."


def diary_graph_message(diary_state: Optional[Dict[str, Any]]) -> str:
    guidance = (
        "그래프는 여러 번 기록했을 때 장기적인 변화 흐름을 보기 위한 참고 자료입니다. "
        "높을수록 더 안정적이고 회복된 상태를 의미합니다."
    )
    entries = _diary_entries(diary_state)
    if len(entries) < 2:
        return (
            f"기록이 2개 이상이면 변화 흐름을 더 분명하게 볼 수 있습니다.\n\n{guidance}"
        )
    return guidance


def diary_trend_dataframe(diary_state: Optional[Dict[str, Any]]) -> Any:
    rows = []
    for index, entry in enumerate(_diary_entries(diary_state), start=1):
        record_label = f"{index}번째 기록"
        display_scores = (
            ("기분", _bounded_score(entry.get("mood_score"))),
            ("안정감", 6 - _bounded_score(entry.get("anxiety_score"))),
            ("여유감", 6 - _bounded_score(entry.get("stress_score"))),
            ("수면 회복감", _bounded_score(entry.get("sleep_quality"))),
            ("활력", _bounded_score(entry.get("energy_score"))),
        )
        for label, score in display_scores:
            rows.append(
                {
                    "기록": record_label,
                    "항목": label,
                    "마음 회복 수준": score,
                }
            )

    try:
        import pandas as pd

        frame = pd.DataFrame(rows, columns=["기록", "항목", "마음 회복 수준"])
        return frame
    except Exception:
        return rows


def save_emotion_diary(
    diary_state: Optional[Dict[str, Any]],
    emotion_label: str,
    mood_score: int,
    anxiety_score: int,
    loneliness_score: int,
    sleep_quality: int,
    meal_status: int,
    energy_score: int,
    stress_score: int,
    diary_line: str,
    save_consent: bool,
) -> Tuple[Dict[str, Any], str]:
    """Store timestamped structured diary values for the current demo session."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion_label": str(emotion_label or "선택 안 함"),
        "mood_score": _bounded_score(mood_score),
        "anxiety_score": _bounded_score(anxiety_score),
        "loneliness_score": _bounded_score(loneliness_score),
        "sleep_quality": _bounded_score(sleep_quality),
        "meal_status": _bounded_score(meal_status),
        "energy_score": _bounded_score(energy_score),
        "stress_score": _bounded_score(stress_score),
        "risk_stage": infer_risk_stage(
            {
                "mood_score": _bounded_score(mood_score),
                "anxiety_score": _bounded_score(anxiety_score),
                "loneliness_score": _bounded_score(loneliness_score),
                "sleep_quality": _bounded_score(sleep_quality),
                "meal_status": _bounded_score(meal_status),
                "energy_score": _bounded_score(energy_score),
                "stress_score": _bounded_score(stress_score),
            }
        ),
        "note": _short_note(diary_line),
        "save_consent": bool(save_consent),
    }
    entries = _diary_entries(diary_state)
    entries.append(entry)
    entries = entries[-20:]
    updated_state = {"entries": entries, "latest_entry": entry}

    summary_lines = [
        "- 감정일기가 저장되었습니다. 마음정리 보고서에서 감정 변화 그래프를 확인할 수 있어요.",
    ]
    if not save_consent:
        summary_lines.append(
            "- 기록 저장 동의가 꺼져 있어 현재 데모 세션에서만 임시로 보여줍니다."
        )

    return updated_state, wrap_card("감정일기 저장 완료", "\n".join(summary_lines))


def build_service_report(
    summary: Optional[Dict[str, Any]],
    diary_state: Optional[Dict[str, Any]],
    trend_snapshots: Any = None,
    result_override: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a user-facing structured report without raw text."""
    summary = summary or {}
    diary_state = diary_state or {}
    result = (
        result_override if result_override is not None else (last_agent_result or {})
    )
    if not summary and not diary_state and not result and not trend_snapshots:
        return wrap_card("마음정리 보고서", "- 아직 상담 기록이 없습니다.")

    agents = _agent_details(result)
    intent = _as_dict(agents.get("intent"))
    state = _as_dict(agents.get("emotional_state"))
    decision = _as_dict(agents.get("decision"))
    small_action = _as_dict(agents.get("small_action"))
    cause = _as_dict(agents.get("cause_exploration"))
    diary_entries = _diary_entries(diary_state)
    latest_diary = diary_entries[-1] if diary_entries else {}
    wellness_checkin = _as_dict(summary.get("wellness_checkin"))

    risk_stage = summary.get("risk_stage") or result.get("risk_stage") or "관심"
    intent_labels = _extract_labels(intent)
    primary_intent = intent.get("primary_intent")
    action_text = (
        small_action.get("action_text")
        if isinstance(small_action.get("action_text"), str)
        else ""
    )
    if _is_generic_report_intent(primary_intent):
        for snapshot in reversed(
            trend_snapshots[-TREND_SNAPSHOT_LIMIT:]
            if isinstance(trend_snapshots, list)
            else []
        ):
            candidate = (
                snapshot.get("primary_intent") if isinstance(snapshot, dict) else ""
            )
            if not _is_generic_report_intent(candidate):
                primary_intent = str(candidate).strip()
                intent = dict(intent)
                intent["primary_intent"] = primary_intent
                break
    if not str(cause.get("selected_cause") or "").strip():
        previous_cause = _latest_snapshot_text(trend_snapshots, "selected_cause")
        if previous_cause:
            cause = dict(cause)
            cause["selected_cause"] = previous_cause
    if not action_text.strip():
        action_text = _latest_snapshot_text(trend_snapshots, "small_action")
    if not action_text.strip() and last_agent_result:
        previous_action = _as_dict(
            _agent_details(last_agent_result).get("small_action")
        ).get("action_text")
        if isinstance(previous_action, str):
            action_text = previous_action.strip()

    concern_keywords = []
    if isinstance(primary_intent, str) and primary_intent:
        concern_keywords.append(primary_intent)
    concern_keywords.extend(intent_labels)
    concern_keywords = _korean_intent_labels(_safe_list(concern_keywords, max_items=8))[
        :4
    ]
    primary_intent_ko = _primary_intent_label(intent)
    secondary_concerns = [
        label for label in concern_keywords if label != primary_intent_ko
    ]
    selected_cause, selected_cause_label = _selected_cause_label(cause)
    main_concern = (
        escape_text(selected_cause_label)
        if selected_cause_label
        else (", ".join(secondary_concerns) if secondary_concerns else "아직 없음")
    )
    action_summary = _decision_action_summary(decision, small_action)
    high_anxiety_or_stress = _is_high_anxiety_or_stress(
        state, latest_diary, wellness_checkin
    )
    next_plan = _next_counseling_plan(
        risk_stage=str(risk_stage),
        selected_cause=selected_cause,
        high_anxiety_or_stress=high_anxiety_or_stress,
    )

    recent_status = _recent_status_labels(state, latest_diary, wellness_checkin)
    recent_state_lines = [
        f"- {label}: {escape_text(value)}" for label, value in recent_status.items()
    ]
    if not recent_state_lines:
        recent_state_lines = ["- 상담 또는 감정일기 저장 후 표시됩니다."]
    recent_state_lines.append(
        f"- 최근 상태 요약: {escape_text(_recent_status_summary(recent_status))}"
    )

    risk_lines = [f"- 현재 단계: {escape_text(risk_stage)}"]
    if risk_stage == "위험":
        risk_lines.extend(
            [
                "- 지금은 안전 확보가 가장 중요합니다.",
                "- 109, 119, 112 중 하나로 바로 연락하고 가까이에 믿을 수 있는 사람에게 알려주세요.",
            ]
        )

    agent_summary_lines = [
        f"- 의도 판단: {escape_text(primary_intent_ko)}",
        f"- 원인 탐색: {escape_text(selected_cause_label or main_concern or '아직 탐색 전')}",
        f"- 다음 상담 방향: {escape_text(' '.join(next_plan))}",
    ]
    next_plan_lines = [f"- {escape_text(item)}" for item in next_plan]
    small_action_text = _safe_report_sentence(action_text, "상담 채팅 후 표시됩니다.")
    small_action_lines = [
        f"- {small_action_text}",
        f"- 방향: {escape_text(action_summary)}",
    ]

    sections = []
    sections.append(wrap_card("최근 마음 상태", "\n".join(recent_state_lines)))
    sections.append(
        wrap_card("현재 위험 단계", "\n".join(risk_lines), crisis=risk_stage == "위험")
    )
    sections.append(wrap_card("Agent 판단 요약", "\n".join(agent_summary_lines)))
    sections.append(wrap_card("오늘의 작은 실천", "\n".join(small_action_lines)))
    sections.append(
        wrap_card("최근 변화 방향", f"- {build_trend_from_snapshots(trend_snapshots)}")
    )
    sections.append(
        wrap_card("최근 마음 회복 흐름", build_emotional_trend_markdown(diary_state))
    )
    return "\n\n".join(sections)


def build_report_outputs(
    summary: Optional[Dict[str, Any]],
    diary_state: Optional[Dict[str, Any]],
) -> Tuple[str, Any, str]:
    return (
        build_service_report(summary, diary_state),
        diary_trend_dataframe(diary_state),
        diary_graph_message(diary_state),
    )


def render_reflection_report(report: Dict[str, Any]) -> str:
    """Render a whitelist-built reflection report as a user-facing card."""
    metrics = _as_dict(report.get("metrics"))
    top_themes = _korean_intent_labels(
        _safe_list(metrics.get("top_repeated_themes"), max_items=3)
    )
    metric_card = wrap_card(
        "최근 상담 요약",
        "\n".join(
            [
                f"- 최근 상담 수: {int(metrics.get('recent_counseling_count', 0) or 0)}",
                f"- 감지된 주요 주제 수: {int(metrics.get('main_topic_count', 0) or 0)}",
                f"- 최근 위험 신호 여부: {'있음' if metrics.get('recent_risk_signal') else '없음'}",
                f"- 최근 작은 행동 진행 상태: {escape_text(metrics.get('action_status') or '확인되지 않음')}",
                "- 반복 주제 top 3: "
                + escape_text(", ".join(top_themes) or "아직 없음"),
            ]
        ),
    )
    if not report.get("has_history"):
        message_card = wrap_card(
            "마음 리포트", f"- {escape_text(report.get('message', ''))}"
        )
        return f"{metric_card}\n\n{message_card}"

    issues = ", ".join(
        _korean_intent_labels(_safe_list(report.get("main_issue"), max_items=5))
    )
    themes = ", ".join(
        _korean_intent_labels(_safe_list(report.get("repeated_themes"), max_items=5))
    )
    risk_stage = str(report.get("risk_stage") or "관심")
    cards = [
        wrap_card(
            "현재 마음 상태",
            f"- {escape_text(report.get('current_emotional_state') or '조금 더 살펴보고 있어요.')}"
            f"\n- 주요 고민: {escape_text(issues or '조금 더 살펴보고 있어요.')}",
        ),
        wrap_card(
            "반복 주제", f"- {escape_text(themes or '아직 뚜렷한 반복 주제가 없어요.')}"
        ),
        wrap_card(
            "최근 작은 행동",
            f"- {escape_text(report.get('last_small_action') or '아직 제안된 행동이 없어요.')}",
        ),
        wrap_card(
            "진행 상태",
            f"- {escape_text(report.get('action_status') or '확인되지 않음')}",
        ),
        wrap_card(
            "다음 질문",
            f"- {escape_text(report.get('next_follow_up') or '다음 상담에서 함께 정해볼게요.')}",
        ),
        wrap_card("장기 흐름", f"- {escape_text(report.get('long_term_trend') or '')}"),
    ]
    risk_lines = [
        "- 최근 위험 신호가 확인되었습니다. 즉시 안전 안내를 확인해주세요."
        if metrics.get("recent_risk_signal") or risk_stage == "위험"
        else "- 최근 확인된 위험 신호가 없습니다."
    ]
    if risk_stage == "위험":
        risk_lines.insert(
            0,
            "- **안전 상태: 위험. 지금은 안전 확보가 우선이며 109, 119, 112 또는 가까운 사람에게 즉시 도움을 요청하세요.**",
        )
    cards.append(
        wrap_card("위험 신호", "\n".join(risk_lines), crisis=risk_stage == "위험")
    )
    return "\n\n".join([metric_card, *cards])


async def _build_chat_reflection_report(
    summary: Optional[Dict[str, Any]] = None,
    service_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Read only safe structured memory and build the chat-side report."""
    try:
        risk_stage = str((summary or {}).get("risk_stage") or "관심")
        history: List[Any] = []
        report_session_id = str(
            (service_context or {}).get("session_id") or current_session_id or ""
        )
        if agent is not None and report_session_id:
            store = getattr(agent, "memory_store", None)
            restored_user_id = str(
                (service_context or {}).get("restored_user_id") or ""
            )
            if restored_user_id:
                getter = getattr(store, "get_user_reflection_history", None)
                getter_key = restored_user_id
            else:
                getter = getattr(store, "get_reflection_history", None)
                getter_key = report_session_id
            if callable(getter):
                history = await asyncio.wait_for(
                    getter(getter_key, limit=20),
                    timeout=REPORT_MEMORY_TIMEOUT_SECONDS,
                )

        records, had_items = sanitize_reflection_records(history)
        if had_items and not records:
            return wrap_card("마음 리포트", f"- {REPORT_FALLBACK_MESSAGE}")
        report = build_reflection_report(records)
        if risk_stage == "위험":
            report = dict(report)
            report["risk_stage"] = "위험"
            if not report.get("has_history"):
                safety = wrap_card(
                    "현재 안전 상태",
                    "- **위험 신호가 확인되었습니다. 기존 위기 안전 안내에 따라 109, 119, 112 또는 가까운 사람에게 즉시 도움을 요청하세요.**",
                    crisis=True,
                )
                return f"{safety}\n\n{render_reflection_report(report)}"
        return render_reflection_report(report)
    except Exception:
        logger.exception("Structured reflection report loading failed")
        return wrap_card("마음 리포트", f"- {REPORT_LOAD_ERROR_MESSAGE}")


async def build_chat_reflection_report(
    summary: Optional[Dict[str, Any]] = None,
    service_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a bounded report without initializing or invoking the counseling agent."""
    try:
        return await asyncio.wait_for(
            _build_chat_reflection_report(summary, service_context),
            timeout=REPORT_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Reflection report loading failed or timed out")
        return wrap_card("마음 리포트", f"- {REPORT_LOAD_ERROR_MESSAGE}")


def build_error_markdown(error_text: str) -> str:
    return wrap_card("오류", f"오류가 발생했습니다: {safe_body_text(error_text)}")


def build_wellness_checkin(
    mood_score: int,
    anxiety_score: int,
    loneliness_score: int,
    sleep_quality: int,
    meal_status: int,
    energy_score: int,
    stress_score: int,
) -> Dict[str, int]:
    return {
        "mood_score": int(mood_score),
        "anxiety_score": int(anxiety_score),
        "loneliness_score": int(loneliness_score),
        "sleep_quality": int(sleep_quality),
        "meal_status": int(meal_status),
        "energy_score": int(energy_score),
        "stress_score": int(stress_score),
    }


def build_summary(
    result: Dict[str, Any], wellness_checkin: Dict[str, int]
) -> Dict[str, Any]:
    details = result.get("pipeline_details", {})
    wellness = details.get("wellness", {}) if isinstance(details, dict) else {}
    safety = details.get("safety", {}) if isinstance(details, dict) else {}

    return {
        "session_id": result.get("session_id", ""),
        "risk_level": result.get("risk_level", "none"),
        "risk_stage": result.get("risk_stage", "관심"),
        "requires_crisis_response": result.get("requires_crisis_response", False),
        "counseling_hint": "present" if result.get("counseling_hint") else "",
        "empathy_style_hint": "present" if result.get("empathy_style_hint") else "",
        "wellness_hint": "present"
        if (result.get("wellness_hint") or wellness.get("support_hint"))
        else "",
        "counseling_record_id": details.get("counseling", {}).get(
            "matched_record_id", ""
        )
        if isinstance(details, dict)
        else "",
        "empathy_record_id": details.get("empathy", {}).get("matched_record_id", "")
        if isinstance(details, dict)
        else "",
        "wellness_record_id": wellness.get("matched_record_id", ""),
        "safety_action": safety.get("action", ""),
        "wellness_checkin": wellness_checkin,
        "response_preview": result.get("response", ""),
    }


async def handle_chat(
    message: str,
    mood_score: int,
    anxiety_score: int,
    loneliness_score: int,
    sleep_quality: int,
    meal_status: int,
    energy_score: int,
    stress_score: int,
    session_id_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    global last_agent_result
    wellness_checkin = build_wellness_checkin(
        mood_score=mood_score,
        anxiety_score=anxiety_score,
        loneliness_score=loneliness_score,
        sleep_quality=sleep_quality,
        meal_status=meal_status,
        energy_score=energy_score,
        stress_score=stress_score,
    )
    session_id = ""

    try:
        message_text = (message or "").strip()
        if not message_text:
            return "오늘 어떤 일이 있었는지 한 문장만 적어주세요.", {
                "session_id": "",
                "risk_stage": "관심",
                "wellness_checkin": wellness_checkin,
                "empty_message": True,
            }

        if has_risk_keyword(message_text):
            summary = build_mock_summary(
                message_text,
                wellness_checkin,
                "위기 신호가 감지되어 안전 안내를 우선합니다. 지금은 109, 119, 112 중 하나로 바로 연락하고, 가까이에 믿을 수 있는 사람에게 알려 혼자 있지 않도록 해주세요. 즉각적인 위험이 있으면 가장 가까운 응급실이나 지역 정신건강복지센터로 가세요.",
                risk_stage="위험",
                source="crisis-fallback",
            )
            last_agent_result = None
            return build_crisis_markdown(), summary

        active_agent = await get_or_create_agent()
        session_id = session_id_override or await get_session_id(active_agent)
        result = await active_agent.process_message(
            user_input=message_text,
            session_id=session_id,
            wellness_checkin=wellness_checkin,
        )
        if not isinstance(result, dict):
            raise TypeError("Agent response must be a dictionary.")
        last_agent_result = result

        response_text = str(result.get("response", "") or "응답이 비어 있습니다.")
        summary = build_summary(result, wellness_checkin)
        if summary.get("requires_crisis_response"):
            return build_crisis_markdown(), summary

        return build_general_markdown(summary, response_text, result), summary

    except Exception as exc:
        logger.exception("Agent path failed, using mock fallback")
        try:
            fallback_response = "지금은 한 번에 다 해결하려 하지 말고, 가장 부담이 작은 한 가지부터 시작해 보세요."
            summary = build_mock_summary(
                (message or "").strip(),
                wellness_checkin,
                fallback_response,
                source="mock-fallback",
            )
            last_agent_result = None
            return build_general_markdown(summary, fallback_response), summary
        except Exception as fallback_exc:
            logger.exception("Fallback generation failed")
            return build_error_markdown(f"{exc} / fallback: {fallback_exc}"), {
                "session_id": session_id,
                "error": str(exc),
                "fallback_error": str(fallback_exc),
                "wellness_checkin": wellness_checkin,
            }


async def handle_chat_ui(
    message: str,
    chat_history: Optional[List[Any]],
    mood_score: int,
    anxiety_score: int,
    loneliness_score: int,
    sleep_quality: int,
    meal_status: int,
    energy_score: int,
    stress_score: int,
    service_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChatMessage], List[ChatMessage], str, str, Dict[str, Any], str]:
    """Handle one chat turn for the user-facing Gradio chat UI."""
    history = normalize_chat_history(chat_history)
    if not history:
        history = initial_chat_messages()
    message_text = (message or "").strip()
    if not message_text:
        return (
            history,
            [dict(message) for message in history],
            "",
            "",
            {"empty_message": True},
            "",
        )

    markdown, summary = await handle_chat(
        message=message_text,
        mood_score=mood_score,
        anxiety_score=anxiety_score,
        loneliness_score=loneliness_score,
        sleep_quality=sleep_quality,
        meal_status=meal_status,
        energy_score=energy_score,
        stress_score=stress_score,
    )

    response_text = markdown
    pipeline_markdown = ""

    if not summary.get("empty_message"):
        pipeline_markdown = build_agent_pipeline_markdown(summary, last_agent_result)
        response_text = build_chat_response_text(
            summary,
            str(summary.get("response_preview") or markdown),
        )

    history.extend(
        [
            {"role": "user", "content": message_text},
            {"role": "assistant", "content": response_text},
        ]
    )
    return (
        history,
        [dict(message) for message in history],
        "",
        pipeline_markdown,
        summary,
        "응답이 준비됐어요.",
    )


def normalize_chat_history(chat_history: Optional[List[Any]]) -> List[ChatMessage]:
    """Return Gradio Chatbot messages format, accepting older tuple history safely."""
    normalized: List[ChatMessage] = []
    for item in chat_history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(item, (tuple, list)) and len(item) == 2:
            user_message, assistant_message = item
            if isinstance(user_message, str) and user_message:
                normalized.append({"role": "user", "content": user_message})
            if isinstance(assistant_message, str) and assistant_message:
                normalized.append({"role": "assistant", "content": assistant_message})

    return normalized


def toggle_nickname_input(anonymous_enabled: bool) -> Dict[str, Any]:
    """Enable nickname only when anonymous mode is off."""
    try:
        import gradio as gr

        return gr.update(
            interactive=not bool(anonymous_enabled),
            value="" if anonymous_enabled else None,
        )
    except Exception:
        return {"interactive": not bool(anonymous_enabled)}


def show_status_checkin_panel() -> Tuple[Dict[str, Any], str]:
    """Reveal optional status check controls from the next-step button."""
    message = "상태 체크하기 영역에서 기분, 불안, 외로움, 수면 상태를 선택해 상담에 반영할 수 있어요."
    try:
        import gradio as gr

        return gr.update(visible=True), message
    except Exception:
        return {"visible": True}, message


def _ui_update(**kwargs: Any) -> Any:
    try:
        import gradio as gr

        return gr.update(**kwargs)
    except Exception:
        return kwargs


def _password_textbox(gr: Any) -> Any:
    """Build a masked, non-persistent credential input outside create_demo."""
    return gr.Textbox(label="비밀번호", type="password")


def _normalized_user_id(value: str) -> str:
    return "".join(
        character
        for character in (value or "").strip()
        if character.isalnum() or character in "-_."
    )[:64]


def account_status_markdown(context: Optional[Dict[str, Any]]) -> str:
    context = context or {}
    if context.get("mode") == "anonymous":
        return wrap_card(
            "현재 이용 상태",
            "- 익명 이용 중\n- 상담 및 리포트 기록은 현재 세션에서만 임시로 유지됩니다.",
        )
    user_id = escape_text(context.get("user_id") or "확인되지 않음")
    return wrap_card(
        "현재 이용 상태",
        f"- 로그인 이용 중\n- 사용자 ID: {user_id}\n- 구조화된 기록은 이 사용자 세션에만 연결됩니다.",
    )


def service_header_markdown(context: Optional[Dict[str, Any]]) -> str:
    context = context or {}
    if context.get("mode") == "anonymous":
        badge = "익명 모드"
    else:
        identity = context.get("nickname") or context.get("user_id") or "사용자"
        badge = f"로그인: {escape_text(identity)}"
    return (
        "<div class='main-service-header'>"
        "<div><div class='app-title'>Psychologist AI Agent</div>"
        "<div class='app-sub'>오늘의 마음을 이야기하고, 다음 행동을 함께 정리해요.</div></div>"
        f"<span class='identity-badge'>{badge}</span></div>"
    )


def privacy_settings_markdown(
    context: Optional[Dict[str, Any]],
    save_consent: bool,
) -> str:
    context = context or {}
    if context.get("mode") == "anonymous":
        return "기록 저장 설정: 익명 모드에서는 임시 세션에서만 구조화된 상담 기억을 유지합니다."
    elif save_consent:
        return "기록 저장이 켜졌어요. 이 계정의 마음 리포트는 구조화된 상담 기록만 사용합니다."
    return "기록 저장이 꺼졌어요. 장기 리포트에는 저장되지 않습니다."


def update_record_saving_setting(
    context: Optional[Dict[str, Any]],
    save_consent: bool,
) -> str:
    """Persist logged-in consent by user ID; anonymous consent stays session-only."""
    context = context or {}
    if context.get("mode") == "logged_in":
        user_id = str(context.get("user_id") or "").strip()
        if user_id:
            saved = user_settings_store.set_record_saving_enabled(
                user_id, bool(save_consent)
            )
            if not saved:
                return "기록 저장 설정을 저장하지 못했어요. 잠시 후 다시 시도해 주세요."
    return privacy_settings_markdown(context, save_consent)


def build_expert_contacts_markdown() -> str:
    contacts = (
        (
            "자살예방 상담전화",
            "마음이 위태롭거나 자살 생각이 들 때 24시간 도움을 요청할 수 있습니다.",
            "109",
        ),
        ("청소년 상담", "청소년과 보호자가 상담 지원을 받을 수 있습니다.", "1388"),
        (
            "긴급 신고",
            "즉각적인 신변 위험이나 긴급 보호가 필요할 때 연락하세요.",
            "112",
        ),
        (
            "응급 상황",
            "생명이 위급하거나 응급 의료 지원이 필요할 때 연락하세요.",
            "119",
        ),
    )
    return "\n\n".join(
        wrap_card(
            title, f"- {description}\n- **{number}**", crisis=number in {"112", "119"}
        )
        for title, description, number in contacts
    )


async def start_service_session(
    mode: str,
    user_id: str = "",
    nickname: str = "",
) -> Dict[str, Any]:
    """Create lightweight UI session state; the agent session is created lazily."""
    normalized_mode = "anonymous" if mode == "anonymous" else "logged_in"
    normalized_user_id = _normalized_user_id(user_id)
    if normalized_mode == "logged_in" and not normalized_user_id:
        raise ValueError("로그인 또는 가입을 위해 사용자 ID를 입력해주세요.")

    active_agent = await get_agent()
    if active_agent is not None:
        session = await active_agent.session_manager.create_session(
            user_id=normalized_user_id if normalized_mode == "logged_in" else None,
            metadata={"access_mode": normalized_mode, "persistence_scope": "session"},
        )
        session_id = session.session_id
        if normalized_mode == "logged_in":
            await active_agent.memory_store.set_session_metadata(
                session_id, "user_id", normalized_user_id
            )
        else:
            await active_agent.memory_store.set_session_metadata(
                session_id, "anonymous_session_id", session_id
            )
        await active_agent.memory_store.set_session_metadata(
            session_id, "persistence_scope", "session"
        )
    else:
        session_id = f"pending-{uuid.uuid4()}"
    saved_setting = (
        user_settings_store.get_record_saving_enabled(normalized_user_id)
        if normalized_mode == "logged_in"
        else False
    )
    return {
        "mode": normalized_mode,
        "user_id": normalized_user_id if normalized_mode == "logged_in" else "",
        "session_id": session_id,
        "nickname": (nickname or "").strip()[:40],
        "persistence_scope": "session",
        "agent_session_ready": active_agent is not None,
        "record_saving_enabled": saved_setting,
    }


async def restore_logged_in_structured_memory(
    context: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[ChatMessage], str, bool]:
    """Restore the small demo report cache without accessing agent memory."""
    if context.get("mode") != "logged_in" or not context.get("record_saving_enabled"):
        history = initial_chat_messages()
        return context, history, EMPTY_REPORT_MESSAGE, False

    user_id = str(context.get("user_id") or "")
    snapshots = get_demo_report_snapshots(user_id)
    if not snapshots:
        history = initial_chat_messages()
        return context, history, EMPTY_REPORT_MESSAGE, False

    restored_context = dict(context)
    restored_context.update(
        {
            "restored_report_snapshots": snapshots,
        }
    )
    chat_messages = [
        {
            "role": "assistant",
            "content": build_previous_summary_bubble(snapshots),
        }
    ]
    try:
        restored_trend = _cache_snapshots_as_trend(snapshots)
        if not restored_trend:
            return restored_context, chat_messages, EMPTY_REPORT_MESSAGE, True

        latest = restored_trend[-1]
        risk_stage = str(latest.get("risk_stage") or "관심")
        snapshot_result = {
            "risk_stage": risk_stage,
            "pipeline_details": {
                "agents": {
                    "intent": {"primary_intent": latest.get("primary_intent", "")},
                    "cause_exploration": {
                        "selected_cause": latest.get("selected_cause", "")
                    },
                    "emotional_state": {
                        key: latest[key]
                        for key in ("anxiety", "stress", "sleep", "energy")
                        if key in latest
                    },
                }
            },
        }
        restored_report = build_service_report(
            {"risk_stage": risk_stage},
            {},
            restored_trend,
            snapshot_result,
        )

        raw_intent = str(latest.get("primary_intent") or "").strip()
        translated_intents = _korean_intent_labels([raw_intent]) if raw_intent else []
        display_intent = (
            translated_intents[0] if translated_intents else "이전 상담 주제"
        )
        raw_status = str(latest.get("action_status") or "").strip().lower()
        status_map = {
            "completed": "완료",
            "done": "완료",
            "partial": "일부 진행",
            "in_progress": "일부 진행",
            "suggested": "제안됨",
            "not_done": "아직 못함",
            "not_completed": "아직 못함",
            "failed": "아직 못함",
            "none": "확인 중",
            "unknown": "확인 중",
            "not_available": "확인 중",
        }
        display_status = status_map.get(raw_status, raw_status or "확인 중")
        previous_lines = [
            f"- 주요 흐름: {escape_text(display_intent)}",
            f"- 행동 상태: {escape_text(display_status)}",
        ]
        previous_action = latest.get("small_action")
        if isinstance(previous_action, str) and previous_action:
            previous_lines.append(f"- 작은 실천: {escape_text(previous_action)}")
        restored_report = (
            f"{restored_report}\n\n"
            f"{wrap_card('이전 상담 흐름', chr(10).join(previous_lines))}"
        )
        return restored_context, chat_messages, restored_report, True
    except Exception:
        return restored_context, chat_messages, EMPTY_REPORT_MESSAGE, True


async def _call_maybe_async_with_timeout(
    func: Any,
    *args: Any,
    timeout: float = 0.1,
) -> None:
    try:
        result = func(*args)
        if hasattr(result, "__await__"):
            await asyncio.wait_for(result, timeout=timeout)
    except Exception:
        pass


async def ensure_service_memory_scope(
    service_context: Optional[Dict[str, Any]],
    save_consent: bool,
) -> Dict[str, Any]:
    """Prepare lightweight demo session scope without touching agent memory."""
    context = dict(service_context or {})
    mode = str(context.get("mode") or "")

    if mode not in {"anonymous", "logged_in"}:
        return context

    desired_scope = "user" if mode == "logged_in" and save_consent else "session"

    user_id = str(context.get("user_id") or "").strip()

    if desired_scope == "user" and user_id:
        session_id = consented_user_sessions.get(user_id, "")
        if not session_id:
            session_id = str(context.get("session_id") or "")
        if not session_id:
            session_id = f"user-{user_id}-{uuid.uuid4().hex[:12]}"
        if user_id not in consented_user_sessions:
            consented_user_sessions[user_id] = session_id
    else:
        session_id = str(context.get("session_id") or "")
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:12]}"

    context["session_id"] = session_id
    context["persistence_scope"] = desired_scope

    if mode == "logged_in":
        context["record_saving_enabled"] = bool(save_consent)

    try:
        active_agent = await asyncio.wait_for(get_agent(), timeout=0.1)
        context["agent_session_ready"] = active_agent is not None
        store = getattr(active_agent, "memory_store", None) if active_agent else None
        setter = getattr(store, "set_session_metadata", None)

        if callable(setter):
            if mode == "logged_in":
                await _call_maybe_async_with_timeout(
                    setter,
                    session_id,
                    "user_id",
                    user_id,
                )
            else:
                await _call_maybe_async_with_timeout(
                    setter,
                    session_id,
                    "anonymous_session_id",
                    session_id,
                )

            await _call_maybe_async_with_timeout(
                setter,
                session_id,
                "persistence_scope",
                desired_scope,
            )
    except Exception:
        context["agent_session_ready"] = False

    return context


async def handle_chat_ui_for_service(
    message: str,
    chat_history: Optional[List[Any]],
    mood_score: int,
    anxiety_score: int,
    loneliness_score: int,
    sleep_quality: int,
    meal_status: int,
    energy_score: int,
    stress_score: int,
    service_context: Optional[Dict[str, Any]],
    save_consent: bool,
) -> Tuple[
    List[ChatMessage],
    List[ChatMessage],
    str,
    str,
    Dict[str, Any],
    str,
    Dict[str, Any],
]:
    context = await ensure_service_memory_scope(service_context, save_consent)
    if context.get("mode") == "logged_in":
        context["record_saving_enabled"] = bool(save_consent)
    outputs = await handle_chat_ui(
        message,
        chat_history,
        mood_score,
        anxiety_score,
        loneliness_score,
        sleep_quality,
        meal_status,
        energy_score,
        stress_score,
        context,
    )
    context = append_trend_snapshot(context, outputs[4], last_agent_result)

    snapshots = context.get("trend_snapshots", [])
    latest_snapshot = (
        snapshots[-1] if isinstance(snapshots, list) and snapshots else None
    )

    if (
        latest_snapshot
        and context.get("mode") == "logged_in"
        and context.get("user_id")
        and context.get("record_saving_enabled")
    ):
        append_demo_report_snapshot(
            str(context["user_id"]),
            latest_snapshot,
        )

    return (*outputs, context)


async def enter_service(
    mode: str,
    user_id: str = "",
    password: str = "",
    nickname: str = "",
) -> Tuple[Any, ...]:
    """Move from the entry screen to the app; password is intentionally discarded."""
    del password
    try:
        context = await start_service_session(mode, user_id, nickname)
    except ValueError as exc:
        history = initial_chat_messages()
        return (
            _ui_update(visible=True),
            _ui_update(visible=False),
            {},
            str(exc),
            "",
            False,
            _ui_update(visible=False),
            service_header_markdown({}),
            privacy_settings_markdown({}, False),
            history,
            [dict(item) for item in history],
            EMPTY_REPORT_MESSAGE,
            False,
        )
    history = initial_chat_messages()
    report_markdown = EMPTY_REPORT_MESSAGE
    restored = False
    if mode == "login":
        (
            context,
            history,
            report_markdown,
            restored,
        ) = await restore_logged_in_structured_memory(context)
    saved_setting = bool(context.get("record_saving_enabled", False))
    return (
        _ui_update(visible=False),
        _ui_update(visible=True),
        context,
        "상담을 시작할 준비가 됐어요.",
        account_status_markdown(context),
        context.get("mode") == "anonymous",
        _ui_update(visible=context.get("mode") == "logged_in"),
        service_header_markdown(context),
        privacy_settings_markdown(context, saved_setting),
        history,
        [dict(item) for item in history],
        report_markdown,
        saved_setting,
    )


async def enter_login(user_id: str, password: str) -> Tuple[Any, ...]:
    return await enter_service("login", user_id, password)


async def enter_signup(
    user_id: str,
    password: str,
    nickname: str,
) -> Tuple[Any, ...]:
    return await enter_service("signup", user_id, password, nickname)


async def enter_anonymous() -> Tuple[Any, ...]:
    return await enter_service("anonymous")


async def open_reflection_report(
    summary: Optional[Dict[str, Any]],
    service_context: Optional[Dict[str, Any]],
) -> Tuple[str, Any]:
    report = await build_chat_reflection_report(summary, service_context)
    return report, {"selected": "report", "__type__": "update"}


def open_reflection_report_for_service(
    summary: Optional[Dict[str, Any]],
    service_context: Optional[Dict[str, Any]],
    save_consent: bool,
) -> Tuple[str, Any, Dict[str, Any]]:
    import gradio as gr

    context = dict(service_context or {})

    if context.get("mode") == "logged_in":
        context["record_saving_enabled"] = bool(save_consent)

    latest_summary = summary if isinstance(summary, dict) else {}

    risk_stage = str(latest_summary.get("risk_stage") or "관심")

    pipeline_details = latest_summary.get("pipeline_details", {})
    agents = (
        pipeline_details.get("agents", {}) if isinstance(pipeline_details, dict) else {}
    )

    intent_info = agents.get("intent", {}) if isinstance(agents, dict) else {}
    cause_info = agents.get("cause_exploration", {}) if isinstance(agents, dict) else {}

    primary_intent = str(intent_info.get("primary_intent") or "학업 부담")
    selected_cause = str(cause_info.get("selected_cause") or "공부 부담")

    report = (
        "## 마음정리 보고서\n\n"
        "### 1. 현재 상태 요약\n"
        f"- 위험 단계: {risk_stage}\n"
        f"- 주요 상담 의도: {primary_intent}\n"
        f"- 추정 원인: {selected_cause}\n\n"
        "### 2. 상담 흐름 정리\n"
        "- 사용자는 현재 부담감이나 막막함을 표현함\n"
        "- Agent는 상담 의도, 감정 상태, 원인 후보를 바탕으로 응답 방향을 구성함\n"
        "- 데이터셋은 답변을 그대로 출력하는 용도가 아니라 판단 힌트로 사용됨\n\n"
        "### 3. 작은 실천 제안\n"
        "- 오늘 해야 할 일을 전부 해결하려고 하기보다, 10분 안에 할 수 있는 가장 작은 일 하나를 정함\n"
        "- 부담을 낮춘 뒤 다시 시작할 수 있도록 작은 단위로 나누는 전략이 적절함\n\n"
        "### 4. 다음 상담 방향\n"
        "- 학업 부담이 계속되는지 확인함\n"
        "- 수면, 스트레스, 에너지 상태 변화를 함께 관찰함\n"
    )

    return report, gr.update(selected="report"), context


def nearby_resource_placeholder(region: str, location_consent: bool) -> str:
    del region, location_consent
    return wrap_card(
        "주변 기관 찾기",
        "- 위치 기반 추천은 사용자가 명시적으로 동의한 경우에만 사용할 수 있으며, "
        "현재 데모에서는 지역 입력 기반 안내를 제공합니다.",
    )


def nearby_resource_links(
    region: str,
    location_consent: bool,
) -> str:
    """Build one-time typed-region map links without retaining location data."""
    if not location_consent:
        return wrap_card(
            "주변 기관 찾기",
            "- 위치 기반 추천은 사용자가 명시적으로 동의한 경우에만 사용할 수 있어요.",
        )
    region_text = str(region or "").strip()
    if not region_text:
        return wrap_card(
            "주변 기관 찾기",
            "- 지역명을 입력하면 근처 병원·상담센터 검색 링크를 안내할 수 있어요.",
        )

    links = []
    for label in ("정신건강복지센터", "상담센터", "응급실", "정신건강의학과"):
        query = urllib.parse.quote_plus(f"{region_text} {label}")
        links.append(
            f"- [{label} 지도에서 찾기](https://www.google.com/maps/search/?api=1&query={query})"
        )
    links.append(
        "- 입력한 지역 정보는 저장하지 않고 지도 검색 링크 생성에만 일회성으로 사용합니다."
    )
    return wrap_card("주변 기관 찾기", "\n".join(links))


async def reset_counseling_memory(
    service_context: Optional[Dict[str, Any]],
) -> Tuple[List[ChatMessage], List[ChatMessage], str, Dict[str, Any]]:
    context = dict(service_context or {})
    session_id = str(context.get("session_id") or "")
    if agent is not None and session_id:
        await agent.memory_store.clear_session(session_id)
    history = initial_chat_messages()
    return history, [dict(item) for item in history], "상담 기억을 초기화했습니다.", {}


async def reset_report_memory(
    service_context: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    context = service_context or {}
    session_id = str(context.get("session_id") or "")
    if agent is not None and session_id:
        await agent.memory_store.clear_reflection_memory(session_id)
    return render_reflection_report(
        build_reflection_report([])
    ), "리포트 기억을 초기화했습니다."


def logout_service() -> Tuple[
    Any,
    Any,
    Dict[str, Any],
    List[ChatMessage],
    List[ChatMessage],
    Dict[str, Any],
    str,
    str,
]:
    history = initial_chat_messages()
    return (
        _ui_update(visible=True),
        _ui_update(visible=False),
        {},
        history,
        [dict(item) for item in history],
        {},
        EMPTY_REPORT_MESSAGE,
        "",
    )


def create_demo():
    """Create and return the Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the demo. Install with: pip install gradio"
        ) from exc
    custom_css = """
    :root { --accent:#e8792e; --accent-dark:#c85e1d; --ink:#253047; --muted:#697386; --line:#e7e9ee; }
    body, .gradio-container, .main, .wrap {
      background:
        radial-gradient(circle at top, rgba(255,255,255,.92) 0, rgba(255,255,255,.92) 18%, rgba(246,242,236,.92) 55%, rgba(241,246,252,.95) 100%) !important;
    }
    .chat-shell { max-width:1040px; margin:24px auto; padding:0 16px; }
    .start-screen {
      padding: 32px 16px;
    }
    .start-card {
      width: 100%;
      max-width: 720px;
      margin: 0 auto !important;
      padding: 32px !important;
      background: rgba(255,255,255,.97) !important;
      border: 1px solid rgba(231,233,238,.98) !important;
      border-radius: 30px !important;
      box-shadow: 0 24px 70px rgba(37,48,71,.10);
      backdrop-filter: blur(8px);
    }
    .start-hero { text-align:center; padding:4px 8px 10px; }
    .app-title { font-size:28px; font-weight:800; letter-spacing:-0.02em; color:var(--ink); margin-bottom:8px; }
    .app-sub { max-width:620px; margin:0 auto; font-size:14px; color:var(--muted); line-height:1.7; }
    .privacy-card {
      background:#fcfaf7;
      border:1px solid #eee3d5;
      border-radius:18px;
      padding:16px 18px;
      color:#5f6673;
      font-size:13px;
      line-height:1.65;
      margin:4px 0 2px;
    }
    .main-service-header { display:flex; justify-content:space-between; align-items:center; gap:18px; padding:20px 22px;
      background:#fff; border:1px solid var(--line); border-radius:18px; margin-bottom:14px; box-shadow:0 8px 28px rgba(37,48,71,.05); }
    .identity-badge { display:inline-flex; padding:7px 12px; border-radius:999px; background:#fff2e8; color:#a64b16;
      font-size:12px; font-weight:700; white-space:nowrap; }
    .service-card { background:#fff !important; border:1px solid var(--line) !important; border-radius:18px !important;
      padding:16px !important; box-shadow:0 8px 26px rgba(37,48,71,.045); }
    .chatbot { border-radius:14px !important; height:520px; max-height:520px; overflow-y:auto; border:0 !important; }
    .input-row textarea { border-radius:12px !important; background:#fbfbfc !important; }
    button.primary, .orange-primary { background:var(--accent) !important; border-color:var(--accent) !important;
      color:#fff !important; border-radius:12px !important; font-weight:700 !important; }
    button.primary:hover, .orange-primary:hover { background:var(--accent-dark) !important; }
    .start-card .orange-primary { width:100%; min-height:44px; margin-top:4px; box-shadow:0 10px 22px rgba(232,121,46,.16); }
    .secondary-action { border:1px solid #efb68f !important; color:#a64b16 !important; background:#fff9f5 !important;
      border-radius:12px !important; font-weight:700 !important; }
    .start-tabs { margin-top:6px; }
    .start-tabs .tab-nav {
      background:#f5f7fb;
      border:1px solid #e4e8ef;
      border-radius:999px;
      padding:6px;
      gap:8px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.8);
    }
    .start-tabs .tab-nav button {
      border:0 !important;
      border-radius:999px !important;
      background:transparent !important;
      color:#5f6673 !important;
      font-weight:700 !important;
      padding:10px 16px !important;
      box-shadow:none !important;
    }
    .start-tabs .tab-nav button[aria-selected="true"] {
      background:#fff !important;
      color:var(--accent-dark) !important;
      box-shadow:0 8px 20px rgba(37,48,71,.08);
    }
    .start-tabs .tabitem, .start-tabs .tabitem > div { padding-top:14px; }
    .start-card input, .start-card textarea {
      border-radius:14px !important;
      background:#fbfbfc !important;
      border:1px solid #dfe4ea !important;
      box-shadow:inset 0 1px 2px rgba(37,48,71,.04);
    }
    .start-card input:focus, .start-card textarea:focus {
      border-color:#efb68f !important;
      box-shadow:0 0 0 3px rgba(232,121,46,.12);
    }
    .status-line { min-height:28px; color:#4c6f8f; font-size:13px; }
    .small-note { font-size:12px; color:#6f7378; text-align:center; margin-top:8px; }
    .helper-note { color:var(--muted); font-size:12px; line-height:1.55; margin:4px 2px 12px; }
    .output-card { background:#fff; border-radius:16px; padding:16px 18px; margin-top:10px; border:1px solid var(--line);
      box-shadow:0 5px 18px rgba(37,48,71,.035); }
    .crisis { background:#fff1f1; border-left:4px solid #d93025; }
    """

    with gr.Blocks(title="Psychologist AI Agent", css=custom_css) as demo:
        with gr.Column(elem_classes="chat-shell"):
            service_context = gr.State({})
            chat_history_state = gr.State(initial_chat_messages())
            summary_output = gr.JSON({}, visible=False)

            with gr.Column(
                visible=True,
                elem_classes=["start-screen", "start-card"],
            ) as start_container:
                gr.Markdown(
                    "<div class='start-hero'><div class='app-title'>Psychologist AI Agent</div>"
                    "<div class='app-sub'>상담 대화를 중심으로 마음의 변화와 다음 행동을 이어가는 정서 지원 서비스입니다.</div></div>"
                )
                gr.Markdown(
                    "<div class='privacy-card'><strong>상담 기록은 동의한 경우에만 저장됩니다.</strong><br>"
                    "익명 시작 시 현재 세션에서만 임시로 사용됩니다.</div>"
                )
                gr.Markdown(
                    "<div class='helper-note'>로그인하거나 익명으로 시작할 수 있어요. 기록 저장 동의가 켜진 경우에만 상담 리포트가 계정에 연결됩니다.</div>"
                )
                entry_status = gr.Markdown("", elem_classes="status-line")
                with gr.Tabs(elem_classes="start-tabs"):
                    with gr.TabItem("로그인"):
                        login_user_id = gr.Textbox(
                            label="사용자 ID", placeholder="사용자 ID"
                        )
                        login_password = _password_textbox(gr)
                        gr.Markdown(
                            "<div class='helper-note'>기록 저장에 동의한 계정은 마음 리포트를 이어서 볼 수 있어요.</div>"
                        )
                        login_button = gr.Button(
                            "로그인하고 상담 시작",
                            variant="primary",
                            elem_classes="orange-primary",
                        )
                    with gr.TabItem("회원가입"):
                        signup_user_id = gr.Textbox(
                            label="사용자 ID", placeholder="사용자 ID"
                        )
                        signup_password = _password_textbox(gr)
                        signup_nickname = gr.Textbox(
                            label="닉네임", placeholder="선택 입력"
                        )
                        gr.Markdown(
                            "<div class='helper-note'>닉네임은 상담 화면과 내 정보에서만 표시됩니다.</div>"
                        )
                        signup_button = gr.Button(
                            "가입하고 상담 시작",
                            variant="primary",
                            elem_classes="orange-primary",
                        )
                    with gr.TabItem("익명 시작"):
                        gr.Markdown(
                            "<div class='helper-note'>가입 없이 바로 상담할 수 있어요. 브라우저 세션이 끝나면 이어서 불러오기 어려울 수 있어요.</div>"
                        )
                        anonymous_start = gr.Button(
                            "익명으로 시작하기",
                            variant="primary",
                            interactive=True,
                            elem_classes="orange-primary",
                        )

            with gr.Column(visible=False) as main_app_container:
                main_header = gr.Markdown(service_header_markdown({}))
                status_output = gr.Markdown("", elem_classes="status-line")
                with gr.Tabs(selected="chat", elem_classes="main-tabs") as main_tabs:
                    with gr.TabItem("상담 채팅", id="chat"):
                        with gr.Group(elem_classes="service-card"):
                            chatbot = gr.Chatbot(
                                label="상담 채팅",
                                elem_classes="chatbot",
                                height=460,
                                value=initial_chat_messages(),
                            )
                            with gr.Row(elem_classes="input-row"):
                                message = gr.Textbox(
                                    label="메시지",
                                    placeholder="오늘 이야기하고 싶은 마음이나 상황을 적어주세요.",
                                    lines=2,
                                    scale=5,
                                )
                                submit = gr.Button(
                                    "보내기",
                                    variant="primary",
                                    elem_classes="orange-primary",
                                    scale=1,
                                )
                            reflection_button = gr.Button(
                                "마음 리포트 보기",
                                variant="secondary",
                                elem_classes="secondary-action",
                            )
                            expert_tab_button = gr.Button(
                                "전문가 상담 연결 탭으로 이동",
                                visible=True,
                                variant="primary",
                                elem_classes="orange-primary",
                            )
                        with gr.Group(visible=False):
                            mood_score = gr.Slider(
                                1, 5, value=3, step=1, label="오늘 기분"
                            )
                            anxiety_score = gr.Slider(
                                1, 5, value=3, step=1, label="불안감"
                            )
                            loneliness_score = gr.Slider(
                                1, 5, value=3, step=1, label="외로움"
                            )
                            sleep_quality = gr.Slider(
                                1, 5, value=3, step=1, label="수면 상태"
                            )
                            meal_status = gr.Slider(
                                1, 5, value=3, step=1, label="식사 상태"
                            )
                            energy_score = gr.Slider(
                                1, 5, value=3, step=1, label="에너지"
                            )
                            stress_score = gr.Slider(
                                1, 5, value=3, step=1, label="스트레스"
                            )
                        with gr.Accordion("Agent Pipeline Details", open=False):
                            pipeline_output = gr.Markdown("")

                    with gr.TabItem("마음정리 보고서", id="report"):
                        with gr.Group(elem_classes="service-card"):
                            report_output = gr.Markdown(EMPTY_REPORT_MESSAGE)

                    with gr.TabItem("전문가 상담 연결", id="expert"):
                        with gr.Group(elem_classes="service-card"):
                            gr.Markdown(build_expert_contacts_markdown())
                            region = gr.Textbox(
                                label="지역",
                                placeholder="지역을 입력하세요. 예: 서울시 강남구",
                            )
                            location_consent = gr.Checkbox(
                                value=False,
                                label="위치 사용 동의",
                                info="입력한 지역은 주변 기관 검색 링크 생성에만 사용됩니다.",
                            )
                            nearby_button = gr.Button(
                                "위치 동의 후 주변 기관 찾기",
                                elem_classes="secondary-action",
                            )
                            nearby_output = gr.Markdown("")
                            gr.Markdown(
                                "<div class='helper-note'>입력한 지역 정보는 저장하지 않고 지도 검색 링크 생성에만 일회성으로 사용합니다.</div>"
                            )

                    with gr.TabItem("내 정보", id="profile"):
                        with gr.Group(elem_classes="service-card"):
                            account_status = gr.Markdown(account_status_markdown({}))
                            privacy_guidance = gr.Markdown(
                                privacy_settings_markdown({}, False)
                            )
                            nickname = gr.Textbox(
                                label="닉네임",
                                placeholder="선택 입력",
                                interactive=True,
                            )
                            save_consent = gr.Checkbox(
                                value=False, label="기록 저장 동의", interactive=True
                            )
                            anonymous_status = gr.Checkbox(
                                value=False, label="익명 모드", interactive=True
                            )
                            gr.Markdown(
                                wrap_card(
                                    "기록 초기화",
                                    "- 상담 기억과 리포트 기억을 각각 초기화할 수 있습니다.",
                                )
                            )
                            with gr.Row():
                                reset_counseling_button = gr.Button(
                                    "상담 기억 초기화", variant="stop"
                                )
                                reset_report_button = gr.Button(
                                    "리포트 기억 초기화", variant="secondary"
                                )
                            reset_status = gr.Markdown("")
                            logout_button = gr.Button(
                                "로그아웃",
                                visible=False,
                                elem_classes="secondary-action",
                            )

                gr.Markdown(
                    "<div class='small-note'>본 서비스는 전문 상담사나 의료진을 대체하지 않습니다. "
                    "위기 상황에서는 즉시 109, 119, 112 또는 가까운 사람에게 도움을 요청하세요.</div>"
                )

        entry_outputs = [
            start_container,
            main_app_container,
            service_context,
            entry_status,
            account_status,
            anonymous_status,
            logout_button,
            main_header,
            privacy_guidance,
            chatbot,
            chat_history_state,
            report_output,
            save_consent,
        ]
        login_button.click(
            enter_login,
            inputs=[login_user_id, login_password],
            outputs=entry_outputs,
        )
        signup_button.click(
            enter_signup,
            inputs=[signup_user_id, signup_password, signup_nickname],
            outputs=entry_outputs,
        ).then(lambda value: value, inputs=signup_nickname, outputs=nickname)
        anonymous_start.click(enter_anonymous, outputs=entry_outputs)

        reflection_button.click(
            open_reflection_report_for_service,
            inputs=[summary_output, service_context, save_consent],
            outputs=[report_output, main_tabs, service_context],
            queue=False,
        )

        submit.click(
            handle_chat_ui_for_service,
            inputs=[
                message,
                chat_history_state,
                mood_score,
                anxiety_score,
                loneliness_score,
                sleep_quality,
                meal_status,
                energy_score,
                stress_score,
                service_context,
                save_consent,
            ],
            outputs=[
                chatbot,
                chat_history_state,
                message,
                pipeline_output,
                summary_output,
                status_output,
                service_context,
            ],
            queue=False,
        )
        expert_tab_button.click(
            go_expert_tab,
            inputs=[],
            outputs=[main_tabs],
            queue=False,
        )

        nearby_button.click(
            nearby_resource_links,
            inputs=[region, location_consent],
            outputs=nearby_output,
            queue=False,
        )
        anonymous_status.change(
            toggle_nickname_input, inputs=anonymous_status, outputs=nickname
        )
        save_consent.change(
            update_record_saving_setting,
            inputs=[service_context, save_consent],
            outputs=privacy_guidance,
        )
        reset_counseling_button.click(
            reset_counseling_memory,
            inputs=service_context,
            outputs=[chatbot, chat_history_state, reset_status, summary_output],
        )
        reset_report_button.click(
            reset_report_memory,
            inputs=service_context,
            outputs=[report_output, reset_status],
            queue=False,
        )
        logout_button.click(
            logout_service,
            outputs=[
                start_container,
                main_app_container,
                service_context,
                chatbot,
                chat_history_state,
                summary_output,
                report_output,
                status_output,
            ],
        )

    return demo


def find_available_port(preferred_port: int = 7860, max_tries: int = 50) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


def main():
    """Run the demo."""
    demo = create_demo()
    launch_port = find_available_port(int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=launch_port,
        share=False,
    )


if __name__ == "__main__":
    main()
