"""
Rule-based deterministic proactive recall builder.

The recall agent extracts structured labels and keys from memory context. It
does not render raw summaries or stringify unknown records.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from src.agent.models import ProactiveRecallResult, SmallActionPlan


CONCERN_CATEGORIES = {"concern", "issue", "topic"}
PREFERRED_STYLE_KINDS = {"prefer_style", "preferred_style", "response_style"}
AVOID_TOPIC_KINDS = {"avoid_topic", "avoid_topics"}
RAW_LOOKING_KEYS = {
    "raw_text",
    "raw_input",
    "user_input",
    "assistant_response",
    "conversation",
    "content",
}


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _append_unique(items: List[str], value: Any) -> None:
    if not isinstance(value, str):
        return

    cleaned = value.strip()
    if cleaned and cleaned not in items:
        items.append(cleaned)


def _matches_avoid_topics(text: str, avoid_topics: Sequence[str]) -> bool:
    return any(topic and topic in text for topic in avoid_topics)


def _extract_repeated_concerns(memory_context: Any) -> List[str]:
    if memory_context is None:
        return []

    scored: List[Tuple[int, str]] = []
    facts = _get_value(memory_context, "facts", []) or []
    for fact in facts:
        category = _get_value(fact, "category", "")
        if category not in CONCERN_CATEGORIES:
            continue

        normalized_value = _get_value(fact, "normalized_value", "")
        label = _get_value(fact, "label", "")
        concern = normalized_value if isinstance(normalized_value, str) and normalized_value else label
        if not isinstance(concern, str) or not concern.strip():
            continue

        evidence_count = _get_value(fact, "evidence_count", 1)
        try:
            score = int(evidence_count)
        except (TypeError, ValueError):
            score = 1
        scored.append((score, concern.strip()))

    recent_summaries = _get_value(memory_context, "recent_summaries", []) or []
    for summary in recent_summaries:
        for topic in _get_value(summary, "key_topics", []) or []:
            if isinstance(topic, str) and topic.strip():
                scored.append((1, topic.strip()))

    ordered: List[str] = []
    for _, concern in sorted(scored, key=lambda item: (-item[0], item[1])):
        _append_unique(ordered, concern)

    return ordered


def _extract_emotional_trend_summary(memory_context: Any) -> str:
    if memory_context is None:
        return ""

    labels: List[str] = []
    for item in _get_value(memory_context, "emotional_trend", []) or []:
        _append_unique(labels, _get_value(item, "label", ""))

    if not labels:
        return ""

    if labels == ["anxiety", "sleep_low"]:
        return "recent anxiety and poor sleep observed"

    readable = {
        "anxiety": "anxiety",
        "sleep_low": "poor sleep",
        "low_sleep": "poor sleep",
        "stress": "stress",
        "low_mood": "low mood",
        "sadness": "sadness",
        "fatigue": "fatigue",
    }
    parts = [readable.get(label, label) for label in labels[:3]]
    if len(parts) == 1:
        return f"recent {parts[0]} observed"
    return "recent " + " and ".join(parts) + " observed"


def _extract_directives(memory_context: Any) -> Tuple[List[str], List[str]]:
    preferred: List[str] = []
    avoided: List[str] = []
    if memory_context is None:
        return preferred, avoided

    for directive in _get_value(memory_context, "directives", []) or []:
        if _get_value(directive, "active", True) is False:
            continue

        kind = _get_value(directive, "kind", "")
        term = _get_value(directive, "term", "")
        if kind in PREFERRED_STYLE_KINDS:
            _append_unique(preferred, term)
        elif kind in AVOID_TOPIC_KINDS:
            _append_unique(avoided, term)

    return preferred, avoided


def _small_action_status_key(last_small_action: SmallActionPlan) -> str:
    status = getattr(last_small_action, "status", "")
    if status in {"suggested", "done", "skipped"}:
        return f"last_small_action_status_{status}"
    return "last_small_action"


def build_proactive_recall(
    memory_context: Any = None,
    last_small_action: Optional[SmallActionPlan] = None,
    next_followup: str = "",
    preferred_response_style: Optional[List[str]] = None,
    avoid_topics: Optional[List[str]] = None,
) -> ProactiveRecallResult:
    """Build structured recall information for response generation."""
    recalled_keys: List[str] = []
    memory_types: List[str] = []
    relevance_scores = {}

    repeated_concerns = _extract_repeated_concerns(memory_context)
    if repeated_concerns:
        recalled_keys.append("repeated_concerns")
        memory_types.append("facts")
        relevance_scores["repeated_concerns"] = 1.0

    emotional_trend_summary = _extract_emotional_trend_summary(memory_context)
    if emotional_trend_summary:
        recalled_keys.append("emotional_trend")
        memory_types.append("emotional_trend")
        relevance_scores["emotional_trend"] = 0.9

    directive_styles, directive_avoid_topics = _extract_directives(memory_context)
    final_preferred_style = list(preferred_response_style or directive_styles)
    final_avoid_topics = list(avoid_topics or directive_avoid_topics)

    if final_preferred_style:
        recalled_keys.append("preferred_response_style")
        memory_types.append("directives")
        relevance_scores["preferred_response_style"] = 0.8
    if final_avoid_topics:
        recalled_keys.append("avoid_topics")
        if "directives" not in memory_types:
            memory_types.append("directives")
        relevance_scores["avoid_topics"] = 0.8

    last_action_text = ""
    if last_small_action is not None:
        candidate = getattr(last_small_action, "action_text", "")
        if isinstance(candidate, str):
            last_action_text = candidate.strip()
        if last_action_text:
            recalled_keys.append("last_small_action")
            recalled_keys.append(_small_action_status_key(last_small_action))
            memory_types.append("small_action")
            relevance_scores["last_small_action"] = 0.85

    next_follow_up = (next_followup or "").strip()
    if next_follow_up and _matches_avoid_topics(next_follow_up, final_avoid_topics):
        next_follow_up = ""
    if next_follow_up:
        recalled_keys.append("next_follow_up")
        memory_types.append("follow_up")
        relevance_scores["next_follow_up"] = 0.75

    deduped_memory_types: List[str] = []
    for memory_type in memory_types:
        _append_unique(deduped_memory_types, memory_type)

    return ProactiveRecallResult(
        recall_needed=bool(recalled_keys),
        memory_types=deduped_memory_types,
        relevance_scores=relevance_scores,
        reason_tags=list(recalled_keys),
        repeated_concerns=repeated_concerns,
        emotional_trend_summary=emotional_trend_summary,
        last_small_action=last_action_text,
        preferred_response_style=final_preferred_style,
        avoid_topics=final_avoid_topics,
        next_follow_up=next_follow_up,
        recalled_keys=recalled_keys,
        stale=False,
    )


@dataclass
class ProactiveRecallAgent:
    """Small wrapper class for callers that prefer an agent object."""

    def build(
        self,
        memory_context: Any = None,
        last_small_action: Optional[SmallActionPlan] = None,
        next_followup: str = "",
        preferred_response_style: Optional[List[str]] = None,
        avoid_topics: Optional[List[str]] = None,
    ) -> ProactiveRecallResult:
        return build_proactive_recall(
            memory_context=memory_context,
            last_small_action=last_small_action,
            next_followup=next_followup,
            preferred_response_style=preferred_response_style,
            avoid_topics=avoid_topics,
        )
