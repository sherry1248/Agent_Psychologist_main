"""
Rule-based extractors for structured memory candidates.

All functions accept already-masked text and return structured data only.
They do not call LLMs, store raw conversation text, or require external
packages.
"""

from datetime import datetime, timedelta
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.memory.models import (
    EmotionalStateEntry,
    FactMemoryEntry,
    RecentMemoryEntry,
    UserDirective,
    utc_now_iso,
)


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n。！？]+")


TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "work": ("work", "job", "office", "직장", "회사", "업무", "일이", "상사"),
    "study": ("study", "school", "exam", "공부", "학교", "시험", "과제"),
    "sleep": ("sleep", "insomnia", "잠", "수면", "불면", "잠을"),
    "relationship": ("relationship", "family", "friend", "가족", "친구", "관계", "연인"),
    "loneliness": ("lonely", "alone", "isolated", "외로", "혼자", "고립"),
    "health": ("health", "body", "pain", "건강", "몸", "통증", "아파"),
}


EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "crisis_signal": ("자살", "자해", "죽고", "죽고 싶", "harm myself", "suicide", "kill myself"),
    "anxiety": ("불안", "걱정", "초조", "긴장", "anxious", "anxiety", "worried", "panic"),
    "sadness": ("우울", "슬퍼", "눈물", "힘들", "depressed", "sad", "hopeless", "down"),
    "loneliness": ("외로", "혼자", "고립", "lonely", "alone", "isolated"),
    "anger": ("화나", "짜증", "분노", "열받", "angry", "annoyed", "furious"),
    "overwhelm": ("벅차", "압도", "감당", "지쳤", "overwhelmed", "burned out", "exhausted"),
    "numbness": ("무감각", "아무것도", "공허", "numb", "empty"),
    "hope": ("괜찮아질", "해볼", "희망", "hope", "try", "better"),
}


DIRECTIVE_TOPICS: Dict[str, Tuple[str, ...]] = {
    "family": ("가족", "부모", "엄마", "아빠", "family", "parents"),
    "work": ("직장", "회사", "업무", "work", "job"),
    "relationship": ("친구", "연인", "관계", "friend", "partner", "relationship"),
    "advice": ("조언", "해결책", "advice", "solution"),
}


PREFERENCE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("listening", "들어"),
    ("listening", "공감"),
    ("listening", "listen"),
    ("practical_steps", "구체"),
    ("practical_steps", "practical"),
    ("short_response", "짧게"),
    ("short_response", "brief"),
)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _segments(text: str) -> List[str]:
    return [_normalize(part) for part in _SENTENCE_SPLIT_RE.split(text) if _normalize(part)]


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def extract_key_topics(masked_text: str, limit: int = 5) -> List[str]:
    """Extract coarse topic labels from masked text."""
    text = _normalize(masked_text)
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if _contains_any(text, keywords)
    ]
    return topics[:limit]


def extract_emotion_labels(masked_text: str, limit: int = 3) -> List[Tuple[str, float, float]]:
    """
    Return emotion label candidates as (label, intensity, confidence).

    The score is keyword based and intentionally coarse. Crisis language is
    surfaced as a label, but safety components remain responsible for action.
    """
    text = _normalize(masked_text)
    matches: List[Tuple[str, float, float, int]] = []
    lowered = text.lower()

    for label, keywords in EMOTION_KEYWORDS.items():
        hit_count = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if hit_count:
            intensity = _clamp(0.45 + 0.15 * hit_count)
            confidence = _clamp(0.5 + 0.12 * hit_count)
            matches.append((label, intensity, confidence, hit_count))

    matches.sort(key=lambda item: (item[3], item[2]), reverse=True)
    return [(label, intensity, confidence) for label, intensity, confidence, _ in matches[:limit]]


def extract_emotional_states(
    masked_text: str,
    session_id: str,
    risk_stage: str = "관심",
    source: str = "message",
    created_at: Optional[str] = None,
) -> List[EmotionalStateEntry]:
    """Build emotional state entries from masked text."""
    timestamp = created_at or utc_now_iso()
    return [
        EmotionalStateEntry(
            session_id=session_id,
            label=label,
            intensity=intensity,
            confidence=confidence,
            source=source,
            risk_stage=risk_stage,
            created_at=timestamp,
        )
        for label, intensity, confidence in extract_emotion_labels(masked_text)
    ]


def extract_user_directives(
    masked_text: str,
    session_id: str,
    ttl_days: int = 3,
    created_at: Optional[str] = None,
) -> List[UserDirective]:
    """
    Extract explicit user boundaries or response preferences.

    Only normalized terms such as "family" or "listening" are stored, never
    the source sentence.
    """
    text = _normalize(masked_text)
    lowered = text.lower()
    timestamp = created_at or utc_now_iso()
    expires_at = (
        datetime.utcnow().replace(microsecond=0) + timedelta(days=ttl_days)
    ).isoformat() + "Z"
    directives: List[UserDirective] = []

    avoid_markers = (
        "하기 싫",
        "얘기하기 싫",
        "얘기는 하기 싫",
        "말하기 싫",
        "꺼내지 마",
        "하지 마",
        "피하고 싶",
        "don't mention",
        "do not mention",
        "avoid talking",
        "don't talk about",
    )
    avoid_segments = [
        segment
        for segment in _segments(text)
        if any(marker in segment.lower() for marker in avoid_markers)
    ]
    if avoid_segments:
        avoid_text = " ".join(avoid_segments)
        matched_terms = [
            topic
            for topic, keywords in DIRECTIVE_TOPICS.items()
            if _contains_any(avoid_text, keywords)
        ] or ["unspecified_topic"]
        for term in matched_terms:
            directives.append(
                UserDirective(
                    directive_id=_stable_id("directive", session_id, "avoid_topic", term),
                    session_id=session_id,
                    kind="avoid_topic",
                    term=term,
                    created_at=timestamp,
                    expires_at=expires_at,
                )
            )

    preference_markers = ("줘", "해줘", "좋겠", "원해", "prefer", "please", "would like")
    if any(marker in lowered for marker in preference_markers):
        for term, marker in PREFERENCE_PATTERNS:
            if marker in lowered:
                directives.append(
                    UserDirective(
                        directive_id=_stable_id("directive", session_id, "prefer_style", term),
                        session_id=session_id,
                        kind="prefer_style",
                        term=term,
                        created_at=timestamp,
                        expires_at=expires_at,
                    )
                )

    return _dedupe_directives(directives)


def _dedupe_directives(directives: Sequence[UserDirective]) -> List[UserDirective]:
    seen = set()
    unique: List[UserDirective] = []
    for directive in directives:
        key = (directive.kind, directive.term)
        if key in seen:
            continue
        seen.add(key)
        unique.append(directive)
    return unique


def extract_fact_candidates(
    masked_text: str,
    session_id: str,
    created_at: Optional[str] = None,
) -> List[FactMemoryEntry]:
    """Extract normalized fact candidates from masked text."""
    text = _normalize(masked_text)
    timestamp = created_at or utc_now_iso()
    facts: List[FactMemoryEntry] = []

    for topic in extract_key_topics(text):
        facts.append(
            FactMemoryEntry(
                fact_id=_stable_id("fact", session_id, "concern", topic),
                session_id=session_id,
                category="concern",
                label=topic,
                normalized_value=topic,
                confidence=0.6,
                evidence_count=1,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
            )
        )

    if _contains_any(text, ("들어", "공감", "listen")):
        facts.append(
            FactMemoryEntry(
                fact_id=_stable_id("fact", session_id, "support_style", "listening"),
                session_id=session_id,
                category="support_style",
                label="listening_preferred",
                normalized_value="listening",
                confidence=0.65,
                evidence_count=1,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
            )
        )

    return facts


def build_recent_memory_entry(
    masked_text: str,
    session_id: str,
    risk_stage: str = "관심",
    created_at: Optional[str] = None,
) -> RecentMemoryEntry:
    """Create a recent-memory summary without storing raw text."""
    topics = extract_key_topics(masked_text)
    emotions = [label for label, _, _ in extract_emotion_labels(masked_text)]
    topic_text = ", ".join(topics) if topics else "general"
    emotion_text = ", ".join(emotions) if emotions else "neutral"
    summary = f"masked_turn_summary: topics={topic_text}; emotions={emotion_text}; risk_stage={risk_stage}"

    return RecentMemoryEntry(
        session_id=session_id,
        summary=summary,
        key_topics=topics,
        emotional_themes=emotions,
        risk_stage=risk_stage,
        created_at=created_at or utc_now_iso(),
    )
