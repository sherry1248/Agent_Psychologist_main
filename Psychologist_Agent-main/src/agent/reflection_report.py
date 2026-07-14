"""Deterministic reflection reports built only from safe structured memory."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List


INSUFFICIENT_HISTORY_MESSAGE = (
    "아직 장기 흐름을 만들 만큼 기록이 충분하지 않아요. "
    "상담을 몇 번 더 진행하면 변화 흐름을 볼 수 있어요."
)
REPORT_FALLBACK_MESSAGE = (
    "리포트를 불러오는 중 문제가 생겼어요. "
    "상담을 조금 더 진행한 뒤 다시 시도해 주세요."
)
MAX_REPORT_RECORDS = 20
MAX_FIELD_ITEMS = 10
MAX_FIELD_LENGTH = 300

SAFE_MEMORY_FIELDS = {
    "user_id",
    "anonymous_session_id",
    "session_id",
    "intent_label",
    "main_issue",
    "emotion_hint",
    "emotional_trend",
    "last_small_action",
    "action_status",
    "next_follow_up",
    "repeated_themes",
    "risk_flag",
    "risk_stage",
    "created_at",
    "timestamp",
}

ACTION_STATUS_KO = {
    "suggested": "제안됨",
    "not_started": "아직 실행 전",
    "in_progress": "일부 진행됨",
    "partial": "일부 진행됨",
    "completed": "완료됨",
    "done": "완료됨",
    "paused": "잠시 멈춤",
    "not_completed": "아직 실행 전",
    "not_done": "아직 실행 전",
    "failed": "아직 실행 전",
    "unknown": "확인 중",
}

ACTION_CHECKIN_STATUSES = {
    "completed", "done", "partial", "in_progress", "not_completed", "not_done", "failed",
}
VALID_ACTION_STATUSES = set(ACTION_STATUS_KO) - {"unknown"}
ACTION_RESPONSE_STATUSES = {
    "completed", "done", "partial", "in_progress",
    "not_completed", "not_done", "failed",
}
GENERIC_ISSUES = {
    "other_concern", "기타_고민",
    "최근_작은_행동_점검", "action_checkin", "small_action_checkin",
    "memory_update", "small_action", "clarification",
    "need_empathy", "support_request", "emotional_disclosure",
}
ISSUE_LABELS_KO = {
    "work_or_study": "학업/업무 부담",
    "work_or_study_stress": "학업/업무 부담",
    "academic_pressure": "학업 부담",
    "specific_academic_burden": "구체적인 학업 부담",
    "sleep": "수면 유지 문제",
    "sleep_problem": "수면 유지 문제",
    "criticism_scolding": "지적/평가 스트레스",
    "self_blame": "자기비난/자책",
    "recovery_improvement": "회복/호전 흐름",
    "relationship": "관계 스트레스",
    "relationship_stress": "관계 스트레스",
    "family_conflict": "가족 갈등",
    "anxiety": "불안",
    "anxiety_support": "불안",
    "stress": "스트레스",
    "stress_support": "스트레스",
    "low_mood": "무기력",
    "low_mood_support": "무기력",
    "low_self_esteem": "자존감 저하",
    "crisis_signal": "위험 신호",
    "crisis_safety": "위험 신호",
}
THEME_RULES = {
    "학업/업무_부담": ("academic", 0, True),
    "학업_부담": ("academic", 1, True),
    "구체적인_학업_부담": ("academic", 2, False),
    "시험_준비": ("academic", 2, False),
    "시험_준비_부담": ("academic", 3, False),
    "과제_부담": ("academic", 2, False),
    "암기_부담": ("academic", 3, False),
    "암기량_부담": ("academic", 3, False),
    "수면_문제": ("sleep", 0, True),
    "수면_유지_문제": ("sleep", 2, False),
    "숙면_어려움": ("sleep", 2, False),
    "지적/평가_스트레스": ("evaluation", 0, True),
    "교수/상사_피드백_부담": ("evaluation", 2, False),
    "자기비난/자책": ("self_view", 2, False),
    "자신감_저하": ("self_view", 1, False),
}
VAGUE_EMOTIONS = {"", "비교적_안정적", "neutral", "현재_감정_흐름을_더_살펴보고_있어요"}
EMOTION_LABELS_KO = {
    "anxiety": "불안", "stress": "스트레스", "sadness": "슬픔",
    "anger": "분노", "loneliness": "외로움", "hopelessness": "막막함",
    "fatigue": "피로", "relief": "안도감", "calm": "차분함", "neutral": "",
}


def _safe_record(value: Any) -> Dict[str, Any]:
    """Read only whitelisted fields without recursively copying object graphs."""
    try:
        if is_dataclass(value) and not isinstance(value, type):
            field_names = {field.name for field in fields(value)}
            return {
                key: getattr(value, key)
                for key in SAFE_MEMORY_FIELDS.intersection(field_names)
            }
        if isinstance(value, dict):
            return {key: value.get(key) for key in SAFE_MEMORY_FIELDS if key in value}
    except Exception:
        return {}
    return {}


def _strings(value: Any) -> List[str]:
    values = value[-MAX_FIELD_ITEMS:] if isinstance(value, (list, tuple)) else [value]
    return [
        item.strip()[:MAX_FIELD_LENGTH]
        for item in values
        if isinstance(item, str) and item.strip()
    ]


def _latest_text(record: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        values = _strings(record.get(key))
        if values:
            return values[-1]
    return ""


def _normalized(value: str) -> str:
    return "_".join(value.strip().lower().split())


def _main_issue_labels(record: Dict[str, Any]) -> List[str]:
    return [
        issue for issue in _strings(record.get("main_issue"))
        if _normalized(issue) not in GENERIC_ISSUES
    ]


def _intent_labels(record: Dict[str, Any]) -> List[str]:
    return [
        issue for issue in _strings(record.get("intent_label"))
        if _normalized(issue) not in GENERIC_ISSUES
    ]


def _previous_specific_issues(records: List[Dict[str, Any]]) -> List[str]:
    broad_fallback: List[str] = []
    for record in reversed(records[:-1]):
        issues = _main_issue_labels(record) or _intent_labels(record)
        specific_issues = _specific_issues(issues)
        if specific_issues:
            return specific_issues
        if issues and not broad_fallback:
            broad_fallback = issues
    return broad_fallback


def _specific_issues(labels: List[str]) -> List[str]:
    return [
        label for label in labels
        if (display := _user_facing_label(label)) and not _theme_rule(display)[2]
    ]


def _user_facing_label(value: str) -> str:
    key = _normalized(value)
    if key in GENERIC_ISSUES:
        return ""
    if key in ISSUE_LABELS_KO:
        return ISSUE_LABELS_KO[key]
    if any("a" <= character <= "z" for character in key):
        return ""
    return value.strip()


def _theme_rule(value: str) -> tuple[str, int, bool]:
    return THEME_RULES.get(_normalized(value), (_normalized(value), 1, False))


def _issue_summary(labels: List[str]) -> str:
    displays = [display for label in labels if (display := _user_facing_label(label))]
    if not displays:
        return ""
    return max(displays, key=lambda display: _theme_rule(display)[1])


def _clean_repeated_themes(themes: List[str], limit: int = 5) -> List[str]:
    counts = Counter(themes)
    first_seen = {theme: index for index, theme in enumerate(themes)}
    families_with_specific = {
        family
        for theme in counts
        for family, _, broad in [_theme_rule(theme)]
        if not broad
    }
    candidates = [
        theme
        for theme in counts
        if not (_theme_rule(theme)[2] and _theme_rule(theme)[0] in families_with_specific)
    ]
    candidates.sort(
        key=lambda theme: (
            -counts[theme],
            -_theme_rule(theme)[1],
            first_seen[theme],
        )
    )
    return candidates[:limit]


def _emotion_text(record: Dict[str, Any]) -> str:
    value = _latest_text(record, "emotion_hint", "emotional_trend")
    key = _normalized(value)
    for suffix in ("_high", "_moderate", "_low"):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
            break
    return EMOTION_LABELS_KO.get(key, value)


def _is_vague_emotion(value: str) -> bool:
    return _normalized(value) in VAGUE_EMOTIONS


def _same_topic(current: List[str], previous: List[str]) -> bool:
    current_families = {
        _theme_rule(display)[0]
        for value in current
        if (display := _user_facing_label(value))
    }
    previous_families = {
        _theme_rule(display)[0]
        for value in previous
        if (display := _user_facing_label(value))
    }
    return bool(current_families.intersection(previous_families))


def _action_checkin_context(
    status: str,
    current_issues: List[str],
    previous_issues: List[str],
) -> bool:
    if status not in ACTION_CHECKIN_STATUSES or not previous_issues:
        return False
    return not current_issues or _same_topic(current_issues, previous_issues)


def _latest_or_previous_text(records: List[Dict[str, Any]], key: str) -> str:
    for record in reversed(records):
        value = _latest_text(record, key)
        if value:
            return value
    return ""


def _selected_action_status(records: List[Dict[str, Any]]) -> str:
    for record in reversed(records):
        candidate = _latest_text(record, "action_status").lower()
        if candidate in ACTION_RESPONSE_STATUSES:
            return candidate
    for record in reversed(records):
        candidate = _latest_text(record, "action_status").lower()
        if candidate in VALID_ACTION_STATUSES:
            return candidate
    return ""


def _academic_issue_from_context(
    labels: List[str],
    intent_labels: List[str],
    last_small_action: str,
    next_follow_up: str,
) -> str:
    displays = [
        display
        for label in [*labels, *intent_labels]
        if (display := _user_facing_label(label))
    ]
    if not any(_theme_rule(display)[0] == "academic" for display in displays):
        return ""
    context = f"{last_small_action} {next_follow_up}"
    if "시험" in context:
        return "시험 준비 부담"
    if any(keyword in context for keyword in ("암기", "외우")):
        return "암기량 부담"
    if any(
        keyword in context
        for keyword in ("공부", "학습", "과목", "개념", "정의", "알고리즘")
    ):
        return "학업 부담"
    return ""


def _progress_trend(
    issue: str,
    status: str,
    current_emotion: str,
    last_small_action: str,
    next_follow_up: str,
) -> str:
    if status in {"partial", "in_progress"}:
        return (
            f"처음에는 {issue} 관련 부담이 있었고, 이후 일부 행동을 시도하며 "
            "진행이 시작된 흐름이 보입니다."
        )
    if status in {"completed", "done"}:
        return (
            f"처음에는 {issue} 관련 부담이 있었고, 이후 제안된 행동을 실행하며 "
            "다음 단계로 이어갈 수 있는 흐름이 보입니다."
        )
    if status in {"not_completed", "not_done", "failed"}:
        return f"{issue} 관련 어려움이 확인되었고, 아직 행동을 실행하기 전인 상태입니다."
    if status == "suggested":
        return f"{issue} 관련 어려움이 확인되었고, 작은 행동을 제안한 상태입니다."
    if last_small_action:
        return f"{issue} 관련 어려움이 확인되었고, 작은 행동을 제안한 상태입니다."
    if next_follow_up:
        return f"{issue} 관련 어려움이 확인되었고, 다음 질문으로 흐름을 이어갈 수 있습니다."
    return f"{issue} 관련 어려움이 확인되어 상담 흐름을 살펴보고 있습니다."


def _empty_report(message: str = INSUFFICIENT_HISTORY_MESSAGE) -> Dict[str, Any]:
    return {
        "has_history": False,
        "message": message,
        "metrics": {
            "recent_counseling_count": 0,
            "main_topic_count": 0,
            "recent_risk_signal": False,
            "action_status": "확인 중",
            "top_repeated_themes": [],
        },
    }


def _recent_history(history: Iterable[Any] | None) -> List[Any]:
    if history is None:
        return []
    if isinstance(history, (list, tuple)):
        return list(history[-MAX_REPORT_RECORDS:])
    return list(islice(iter(history), MAX_REPORT_RECORDS))


def sanitize_reflection_records(
    history: Iterable[Any] | None,
) -> tuple[List[Dict[str, Any]], bool]:
    """Bound and shallowly sanitize structured records before report analysis."""
    recent_items = _recent_history(history)
    records: List[Dict[str, Any]] = []
    for item in recent_items:
        record = _safe_record(item)
        if record:
            records.append(record)
    return records, bool(recent_items)


def build_reflection_report(history: Iterable[Any] | None) -> Dict[str, Any]:
    """Build a stable report without accepting or returning raw conversation text."""
    try:
        return _build_reflection_report(history)
    except Exception:
        return _empty_report(REPORT_FALLBACK_MESSAGE)


def _build_reflection_report(history: Iterable[Any] | None) -> Dict[str, Any]:
    records, had_items = sanitize_reflection_records(history)
    if not had_items:
        return _empty_report()
    if not records:
        return _empty_report(REPORT_FALLBACK_MESSAGE)

    latest = records[-1]
    current_issues = _main_issue_labels(latest)
    current_specific_issues = _specific_issues(current_issues)
    previous_issues = _previous_specific_issues(records)
    normalized_status = _selected_action_status(records)
    is_action_checkin = _action_checkin_context(
        normalized_status,
        current_issues,
        previous_issues,
    )
    last_small_action = _latest_or_previous_text(records, "last_small_action")
    next_follow_up = _latest_text(latest, "next_follow_up")
    if is_action_checkin and not next_follow_up:
        next_follow_up = _latest_or_previous_text(records[:-1], "next_follow_up")
    inferred_academic_issue = _academic_issue_from_context(
        current_issues or previous_issues,
        _intent_labels(latest),
        last_small_action,
        next_follow_up,
    )
    if current_specific_issues:
        selected_issues = current_specific_issues
    elif is_action_checkin:
        selected_issues = previous_issues
    elif inferred_academic_issue:
        selected_issues = [inferred_academic_issue]
    else:
        selected_issues = _intent_labels(latest) or current_issues or previous_issues

    issue_summary = _issue_summary(selected_issues)
    if not issue_summary:
        inferred_labels = _intent_labels(latest) or previous_issues
        issue_summary = _issue_summary(inferred_labels)

    if not issue_summary and next_follow_up:
        issue_summary = "이전 상담 후속 주제"

    themes: List[str] = []
    for record in records:
        labels = _main_issue_labels(record) or _intent_labels(record)
        themes.extend(display for label in labels if (display := _user_facing_label(label)))
    repeated_themes = _clean_repeated_themes(themes)

    first_emotion = _emotion_text(records[0])
    current_emotion = _emotion_text(latest)
    if is_action_checkin and _is_vague_emotion(current_emotion):
        for record in reversed(records[:-1]):
            inherited_emotion = _emotion_text(record)
            if not _is_vague_emotion(inherited_emotion):
                current_emotion = inherited_emotion
                break

    translated_status = ACTION_STATUS_KO.get(normalized_status, "확인 중")
    recent_risk_signal = any(
        _latest_text(record, "risk_stage") == "위험" for record in records[-3:]
    )
    if recent_risk_signal:
        long_term_trend = "최근 위험 신호가 있어 안전 확보와 즉각적인 도움 연결을 우선해야 합니다."
    elif issue_summary:
        long_term_trend = _progress_trend(
            issue_summary,
            normalized_status,
            current_emotion,
            last_small_action,
            next_follow_up,
        )
    elif len(records) < 2:
        long_term_trend = INSUFFICIENT_HISTORY_MESSAGE
    elif first_emotion and current_emotion and first_emotion != current_emotion:
        long_term_trend = f"처음에는 {first_emotion} 흐름이었고, 최근에는 {current_emotion} 흐름이 나타납니다."
    elif current_emotion:
        long_term_trend = f"최근 상담에서 {current_emotion} 흐름이 이어지고 있습니다."
    else:
        long_term_trend = INSUFFICIENT_HISTORY_MESSAGE

    unique_themes = list(dict.fromkeys(themes))
    current_emotional_state = current_emotion
    if not current_emotional_state and issue_summary:
        current_emotional_state = f"{issue_summary}과 관련된 부담을 살펴보고 있어요."
    return {
        "has_history": True,
        "current_emotional_state": current_emotional_state or "현재 감정 흐름을 더 살펴보고 있어요.",
        "main_issue": [issue_summary] if issue_summary else [],
        "repeated_themes": repeated_themes,
        "last_small_action": last_small_action,
        "action_status": translated_status,
        "next_follow_up": next_follow_up,
        "risk_stage": _latest_text(latest, "risk_stage") or "관심",
        "long_term_trend": long_term_trend,
        "metrics": {
            "recent_counseling_count": len(records),
            "main_topic_count": len(unique_themes),
            "recent_risk_signal": recent_risk_signal,
            "action_status": translated_status,
            "top_repeated_themes": repeated_themes[:3],
        },
    }
