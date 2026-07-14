"""
Memory store for conversation history management.

This module provides the MemoryStore class for storing and retrieving
conversation history, supporting both in-memory and persistent storage.
Also provides UserProfile for long-term clinical context.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from src.memory.extractors import (
    build_recent_memory_entry,
    extract_emotional_states,
    extract_fact_candidates,
    extract_user_directives,
)
from src.memory.models import (
    ConversationContinuityMemory,
    EmotionalStateEntry,
    FactMemoryEntry,
    MemoryContext,
    RecentMemoryEntry,
    SessionStructuredSummary,
    ReflectionMemoryEntry,
    StructuredSmallActionMemory,
    UserDirective,
    utc_now_iso,
)
from src.privacy.pii_redactor import PIIRedactor
from src.utils.logging_config import setup_logging

logger = setup_logging("memory_store")

DEFAULT_USER_SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "user_settings.json"
)
DEFAULT_STRUCTURED_REFLECTION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "structured_reflection_memory.json"
)
PERSISTED_REFLECTION_FIELDS = (
    "user_id", "session_id", "intent_label", "main_issue", "emotion_hint",
    "emotional_trend", "last_small_action", "action_status", "next_follow_up",
    "repeated_themes", "risk_stage", "created_at",
)


class UserSettingsStore:
    """Small persistent store for account settings; never stores conversation text."""

    def __init__(self, path: Optional[str] = None):
        configured_path = path or os.getenv("USER_SETTINGS_PATH", "").strip()
        self.path = Path(configured_path) if configured_path else DEFAULT_USER_SETTINGS_PATH

    def _load(self) -> Dict[str, Dict[str, bool]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return {}
            return {
                str(user_id): {"record_saving_enabled": bool(settings.get("record_saving_enabled", False))}
                for user_id, settings in data.items()
                if isinstance(user_id, str) and isinstance(settings, dict)
            }
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not load user settings from %s: %s", self.path, exc)
            return {}

    def get_record_saving_enabled(self, user_id: str) -> bool:
        normalized_user_id = (user_id or "").strip()
        if not normalized_user_id:
            return False
        return bool(
            self._load().get(normalized_user_id, {}).get("record_saving_enabled", False)
        )

    def set_record_saving_enabled(self, user_id: str, enabled: bool) -> bool:
        normalized_user_id = (user_id or "").strip()
        if not normalized_user_id:
            return False
        settings = self._load()
        settings[normalized_user_id] = {"record_saving_enabled": bool(enabled)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            return True
        except OSError as exc:
            logger.error("Could not save user settings to %s: %s", self.path, exc)
            return False


def _emotion_hint_from_vector(vector: Dict[str, float]) -> str:
    """Convert normalized state scores into one coarse, report-safe label."""
    burdens = {
        "불안": float(vector.get("anxiety", 0.0)),
        "스트레스": float(vector.get("stress", 0.0)),
        "가라앉은 기분": 1.0 - float(vector.get("mood", 0.5)),
        "수면 부담": 1.0 - float(vector.get("sleep", 0.5)),
        "피로": 1.0 - float(vector.get("energy", 0.5)),
    }
    label, strength = max(burdens.items(), key=lambda item: item[1])
    return label if strength >= 0.55 else "비교적 안정적"


@dataclass
class UserProfile:
    """Long-term user profile for clinical context."""
    user_id: str = ""
    diagnosis_history: List[str] = field(default_factory=list)
    chronic_stressors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    therapeutic_preferences: Dict[str, Any] = field(default_factory=dict)
    previous_concerns: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from JSON dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            diagnosis_history=data.get("diagnosis_history", []),
            chronic_stressors=data.get("chronic_stressors", []),
            risk_factors=data.get("risk_factors", []),
            therapeutic_preferences=data.get("therapeutic_preferences", {}),
            previous_concerns=data.get("previous_concerns", []),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation for context compression."""
    summary: str
    key_topics: List[str]
    emotional_themes: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MemoryConfig:
    """Configuration for memory store."""
    max_messages: int = 50
    summary_threshold: int = 20
    persist_path: Optional[str] = None
    auto_persist: bool = True
    structured_reflection_path: Optional[str] = None


class MemoryStore:
    """
    Store for conversation history and context.

    Maintains per-session conversation history with support for
    summarization and persistence.

    Example:
        store = MemoryStore()
        await store.add("session_123", "I'm feeling anxious", "That sounds difficult...")
        history = await store.get_history("session_123")
    """

    STRUCTURED_RECENT_MAX_ITEMS = 20
    STRUCTURED_FACT_MAX_ITEMS = 50
    STRUCTURED_DIRECTIVE_MAX_ITEMS = 20
    STRUCTURED_EMOTION_MAX_ITEMS = 30
    REFLECTION_MAX_ITEMS = 30

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory store.

        Args:
            config: Memory store configuration
        """
        self.config = config or MemoryConfig()
        self._sessions: Dict[str, List[Message]] = defaultdict(list)
        self._summaries: Dict[str, List[ConversationSummary]] = defaultdict(list)
        self._metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._profiles: Dict[str, UserProfile] = {}
        self._recent_entries: Dict[str, List[RecentMemoryEntry]] = defaultdict(list)
        self._fact_entries: Dict[str, List[FactMemoryEntry]] = defaultdict(list)
        self._directives: Dict[str, List[UserDirective]] = defaultdict(list)
        self._emotional_states: Dict[str, List[EmotionalStateEntry]] = defaultdict(list)
        self._continuity: Dict[str, ConversationContinuityMemory] = {}
        self._reflection_entries: Dict[str, List[ReflectionMemoryEntry]] = defaultdict(list)
        self._user_structured_sessions: Dict[str, List[str]] = defaultdict(list)
        configured_reflection_path = (
            self.config.structured_reflection_path
            or os.getenv("STRUCTURED_REFLECTION_MEMORY_PATH", "").strip()
        )
        self._structured_reflection_path = (
            Path(configured_reflection_path)
            if configured_reflection_path
            else DEFAULT_STRUCTURED_REFLECTION_PATH
        )
        self._persistent_reflections: Dict[str, List[ReflectionMemoryEntry]] = defaultdict(list)
        self._memory_redactor = PIIRedactor(mock_mode=True, use_presidio=False)

        self._load_structured_reflections()

        if self.config.persist_path:
            self._load_persisted()

        logger.info("MemoryStore initialized")

    async def add(
        self,
        session_id: str,
        user_input: str,
        response: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to history.

        Args:
            session_id: Session identifier
            user_input: User's message
            response: Assistant's response
            user_metadata: Metadata for user message
            response_metadata: Metadata for response
        """
        sanitized_user_input = self._sanitize_for_memory(user_input)
        sanitized_response = self._sanitize_for_memory(response)
        summary_text, key_topics, emotional_themes = self._build_turn_summary(
            sanitized_user_input,
            sanitized_response
        )

        user_msg = Message(
            role="user",
            content=self._format_user_summary(key_topics, emotional_themes),
            metadata={**(user_metadata or {}), "summary": True}
        )
        assistant_msg = Message(
            role="assistant",
            content=f"응답 요약: {self._compact_text(sanitized_response, 140)}",
            metadata={**(response_metadata or {}), "summary": True}
        )

        self._sessions[session_id].append(user_msg)
        self._sessions[session_id].append(assistant_msg)
        self._summaries[session_id].append(
            ConversationSummary(
                summary=summary_text,
                key_topics=key_topics,
                emotional_themes=emotional_themes,
            )
        )

        # Truncate if exceeding max
        if len(self._sessions[session_id]) > self.config.max_messages * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.config.max_messages * 2:]

        if self.config.auto_persist and self.config.persist_path:
            await self._persist_session(session_id)

        logger.debug(f"Added conversation turn to session {session_id}")

    async def add_structured_memory(
        self,
        session_id: str,
        masked_text: str,
        risk_stage: str = "관심",
        source: str = "message",
    ) -> None:
        """
        Add structured memory candidates from already-masked text.

        This is an opt-in API. It is intentionally not called by add(), so the
        existing conversation-history behavior remains unchanged.
        """
        if not masked_text:
            return

        recent_entry = build_recent_memory_entry(
            masked_text=masked_text,
            session_id=session_id,
            risk_stage=risk_stage,
        )
        fact_entries = extract_fact_candidates(
            masked_text=masked_text,
            session_id=session_id,
        )
        directives = extract_user_directives(
            masked_text=masked_text,
            session_id=session_id,
        )
        emotional_states = extract_emotional_states(
            masked_text=masked_text,
            session_id=session_id,
            risk_stage=risk_stage,
            source=source,
        )

        self._recent_entries[session_id].append(recent_entry)
        self._merge_fact_entries(session_id, fact_entries)
        self._merge_directives(session_id, directives)
        self._emotional_states[session_id].extend(emotional_states)
        self._trim_structured_memory(session_id)

    async def get_memory_context(
        self,
        session_id: str,
        max_recent: int = 5,
        max_facts: int = 8,
        max_directives: int = 5,
        max_emotions: int = 5,
    ) -> MemoryContext:
        """Return structured memory layers for optional prompt use."""
        return MemoryContext(
            recent_summaries=await self.get_recent_memory(session_id, limit=max_recent),
            facts=await self.get_fact_memory(session_id, limit=max_facts),
            directives=await self.get_user_directives(
                session_id,
                active_only=True,
                limit=max_directives,
            ),
            emotional_trend=await self.get_emotional_trend(session_id, limit=max_emotions),
        )

    async def get_recent_memory(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[RecentMemoryEntry]:
        """Get recent structured summaries for a session."""
        entries = self._recent_entries.get(session_id, [])
        return list(entries[-limit:]) if limit else list(entries)

    async def get_fact_memory(
        self,
        session_id: str,
        limit: int = 8,
        categories: Optional[List[str]] = None,
    ) -> List[FactMemoryEntry]:
        """Get structured fact candidates for a session."""
        entries = self._fact_entries.get(session_id, [])
        if categories:
            category_set = set(categories)
            entries = [entry for entry in entries if entry.category in category_set]
        return list(entries[-limit:]) if limit else list(entries)

    async def get_user_directives(
        self,
        session_id: str,
        active_only: bool = True,
        limit: int = 5,
    ) -> List[UserDirective]:
        """Get user directives for a session."""
        entries = self._directives.get(session_id, [])
        if active_only:
            entries = [entry for entry in entries if entry.active]
        return list(entries[-limit:]) if limit else list(entries)

    async def get_emotional_trend(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[EmotionalStateEntry]:
        """Get recent emotional state observations for a session."""
        entries = self._emotional_states.get(session_id, [])
        return list(entries[-limit:]) if limit else list(entries)

    async def clear_structured_memory(self, session_id: str) -> None:
        """Clear only the structured memory layers for a session."""
        self._recent_entries.pop(session_id, None)
        self._fact_entries.pop(session_id, None)
        self._directives.pop(session_id, None)
        self._emotional_states.pop(session_id, None)
        self._continuity.pop(session_id, None)
        self._reflection_entries.pop(session_id, None)

    async def get_reflection_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> List[ReflectionMemoryEntry]:
        """Return whitelist-only snapshots for the reflection report."""
        entries = self._reflection_entries.get(session_id, [])
        return list(entries[-limit:]) if limit else list(entries)

    async def get_user_reflection_history(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[ReflectionMemoryEntry]:
        """Return recent structured report snapshots for one logged-in user."""
        normalized_user_id = (user_id or "").strip()
        if not normalized_user_id:
            return []
        requested_limit = max(1, min(int(limit or 20), self.REFLECTION_MAX_ITEMS))
        persisted = self._persistent_reflections.get(normalized_user_id, [])
        if persisted:
            return list(persisted[-requested_limit:])
        entries: List[ReflectionMemoryEntry] = []
        session_ids = self._user_structured_sessions.get(normalized_user_id, [])[-20:]
        for session_id in reversed(session_ids):
            if self._metadata.get(session_id, {}).get("persistence_scope") != "user":
                continue
            remaining = requested_limit - len(entries)
            if remaining <= 0:
                break
            entries.extend(self._reflection_entries.get(session_id, [])[-remaining:])
        entries.sort(key=lambda entry: entry.created_at)
        return list(entries[-requested_limit:])

    async def get_latest_structured_session_id(self, user_id: str) -> str:
        """Return the newest session containing structured memory for a user."""
        normalized_user_id = (user_id or "").strip()
        persisted = self._persistent_reflections.get(normalized_user_id, [])
        if persisted:
            return persisted[-1].session_id
        session_ids = self._user_structured_sessions.get(normalized_user_id, [])[-20:]
        for session_id in reversed(session_ids):
            if (
                self._metadata.get(session_id, {}).get("persistence_scope") == "user"
                and self._reflection_entries.get(session_id)
            ):
                return session_id
        return ""

    async def clear_reflection_memory(self, session_id: str) -> None:
        """Clear report snapshots without changing counseling continuity."""
        self._reflection_entries.pop(session_id, None)
        if self.config.auto_persist and self.config.persist_path:
            await self._persist_session(session_id)

    async def update_last_action_status(self, session_id: str, status: str) -> None:
        """Update only the previous structured action status after a check-in."""
        normalized = (status or "").strip()
        continuity = self._continuity.get(session_id)
        if continuity and continuity.last_small_action and normalized:
            continuity.last_small_action.status = normalized
        if self._reflection_entries.get(session_id) and normalized:
            self._reflection_entries[session_id][-1].action_status = normalized
        user_id = str(self._metadata.get(session_id, {}).get("user_id") or "").strip()
        if user_id and normalized:
            for entry in reversed(self._persistent_reflections.get(user_id, [])):
                if entry.session_id == session_id:
                    entry.action_status = normalized
                    self._save_structured_reflections()
                    break
        if self.config.auto_persist and self.config.persist_path:
            await self._persist_session(session_id)

    async def get_conversation_continuity(
        self,
        session_id: str,
    ) -> ConversationContinuityMemory:
        """Return the structured continuity snapshot for a session."""
        return self._continuity.get(
            session_id,
            ConversationContinuityMemory(session_id=session_id),
        )

    async def update_conversation_continuity(
        self,
        session_id: str,
        last_small_action: Optional[Any] = None,
        next_follow_up: str = "",
        emotional_state_vector: Optional[Dict[str, float]] = None,
        risk_stage: str = "관심",
        intent_label: str = "",
        latest_structured_summary: Optional[Any] = None,
        action_status_override: Optional[str] = None,
    ) -> ConversationContinuityMemory:
        """Store privacy-preserving continuity data for the next turn."""
        previous = await self.get_conversation_continuity(session_id)
        turn_count = previous.turn_count + 1
        risk_stage_start = previous.risk_stage_start or risk_stage or "관심"

        small_action_memory = previous.last_small_action
        action_text = getattr(last_small_action, "action_text", "") if last_small_action else ""
        if isinstance(action_text, str) and action_text.strip():
            small_action_memory = StructuredSmallActionMemory(
                action_id=str(getattr(last_small_action, "action_id", "")),
                intent_label=str(getattr(last_small_action, "intent_label", "") or intent_label),
                action_text=action_text.strip(),
                status=str(getattr(last_small_action, "status", "suggested")),
                created_at=str(getattr(last_small_action, "created_at", "") or utc_now_iso()),
            )

        summary_memory = previous.latest_structured_summary
        if latest_structured_summary is not None:
            summary_memory = SessionStructuredSummary(
                summary_id=str(getattr(latest_structured_summary, "summary_id", "")),
                main_issue=[
                    str(item)
                    for item in getattr(latest_structured_summary, "main_issue", []) or []
                ],
                emotional_trend=[
                    str(item)
                    for item in getattr(latest_structured_summary, "emotional_trend", []) or []
                ],
                risk_stage_start=str(getattr(latest_structured_summary, "risk_stage_start", risk_stage_start)),
                risk_stage_end=str(getattr(latest_structured_summary, "risk_stage_end", risk_stage)),
                last_small_action=str(getattr(latest_structured_summary, "last_small_action", "")),
                next_follow_up=str(getattr(latest_structured_summary, "next_follow_up", "")),
                created_at=str(getattr(latest_structured_summary, "created_at", "") or utc_now_iso()),
            )

        memory = ConversationContinuityMemory(
            session_id=session_id,
            last_small_action=small_action_memory,
            next_follow_up=(next_follow_up or "").strip(),
            emotional_state_vector=dict(emotional_state_vector or previous.emotional_state_vector or {}),
            latest_structured_summary=summary_memory,
            turn_count=turn_count,
            risk_stage_start=risk_stage_start,
            risk_stage_end=risk_stage or previous.risk_stage_end or "관심",
            updated_at=utc_now_iso(),
        )
        self._continuity[session_id] = memory
        summary_issues = list(getattr(summary_memory, "main_issue", []) or [])
        summary_trend = list(getattr(summary_memory, "emotional_trend", []) or [])
        emotion_vector = dict(emotional_state_vector or previous.emotional_state_vector or {})
        emotion_hint = _emotion_hint_from_vector(emotion_vector) if emotion_vector else ""
        reflection_action_status = (action_status_override or "").strip()
        if not reflection_action_status:
            reflection_action_status = getattr(small_action_memory, "status", "") or ""
        reflection_entry = ReflectionMemoryEntry(
                user_id=str(self._metadata.get(session_id, {}).get("user_id") or ""),
                anonymous_session_id=str(
                    self._metadata.get(session_id, {}).get("anonymous_session_id") or ""
                ),
                session_id=session_id,
                intent_label=(intent_label or "").strip(),
                main_issue=[str(item) for item in summary_issues if str(item).strip()],
                emotion_hint=emotion_hint,
                emotional_trend=[str(item) for item in summary_trend if str(item).strip()],
                last_small_action=getattr(small_action_memory, "action_text", "") or "",
                action_status=reflection_action_status,
                next_follow_up=(next_follow_up or "").strip(),
                risk_stage=risk_stage or "관심",
            )
        self._reflection_entries[session_id].append(reflection_entry)
        self._reflection_entries[session_id] = self._reflection_entries[session_id][
            -self.REFLECTION_MAX_ITEMS:
        ]
        if (
            self._metadata.get(session_id, {}).get("persistence_scope") == "user"
            and reflection_entry.user_id
        ):
            self._persistent_reflections[reflection_entry.user_id].append(reflection_entry)
            self._persistent_reflections[reflection_entry.user_id] = (
                self._persistent_reflections[reflection_entry.user_id][
                    -self.REFLECTION_MAX_ITEMS:
                ]
            )
            self._save_structured_reflections()

        if self.config.auto_persist and self.config.persist_path:
            await self._persist_session(session_id)

        return memory

    @staticmethod
    def _persistent_reflection_dict(entry: ReflectionMemoryEntry) -> Dict[str, Any]:
        data = entry.to_dict()
        return {field: data.get(field) for field in PERSISTED_REFLECTION_FIELDS}

    def _load_structured_reflections(self) -> None:
        if not self._structured_reflection_path.exists():
            return
        try:
            with self._structured_reflection_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return
            for user_id, values in payload.items():
                if not isinstance(user_id, str) or not user_id.strip() or not isinstance(values, list):
                    continue
                records = []
                for value in values[-self.REFLECTION_MAX_ITEMS:]:
                    if not isinstance(value, dict):
                        continue
                    safe = {
                        field: value.get(field)
                        for field in PERSISTED_REFLECTION_FIELDS
                        if field in value and value.get(field) is not None
                    }
                    safe["user_id"] = user_id.strip()
                    records.append(ReflectionMemoryEntry(**safe))
                self._persistent_reflections[user_id.strip()] = records
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not load structured reflection memory: %s", exc)

    def _save_structured_reflections(self) -> None:
        payload = {
            user_id: [self._persistent_reflection_dict(entry) for entry in entries[-self.REFLECTION_MAX_ITEMS:]]
            for user_id, entries in self._persistent_reflections.items()
            if user_id and entries
        }
        temporary_path = self._structured_reflection_path.with_suffix(
            f"{self._structured_reflection_path.suffix}.tmp"
        )
        try:
            self._structured_reflection_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._structured_reflection_path)
        except OSError as exc:
            logger.error("Could not persist structured reflection memory: %s", exc)

    def _merge_fact_entries(
        self,
        session_id: str,
        new_facts: List[FactMemoryEntry],
    ) -> None:
        """Merge fact candidates by category and normalized value."""
        entries = self._fact_entries[session_id]
        by_key = {
            (entry.category, entry.normalized_value): entry
            for entry in entries
        }

        for fact in new_facts:
            key = (fact.category, fact.normalized_value)
            existing = by_key.get(key)
            if existing:
                existing.evidence_count += fact.evidence_count
                existing.last_seen_at = fact.last_seen_at
                existing.confidence = max(existing.confidence, fact.confidence)
            else:
                entries.append(fact)
                by_key[key] = fact

    def _merge_directives(
        self,
        session_id: str,
        new_directives: List[UserDirective],
    ) -> None:
        """Merge user directives by kind and normalized term."""
        entries = self._directives[session_id]
        by_key = {
            (entry.kind, entry.term): entry
            for entry in entries
        }

        for directive in new_directives:
            key = (directive.kind, directive.term)
            existing = by_key.get(key)
            if existing:
                existing.hit_count += directive.hit_count
                existing.active = existing.active or directive.active
                if directive.expires_at:
                    existing.expires_at = directive.expires_at
            else:
                entries.append(directive)
                by_key[key] = directive

    def _trim_structured_memory(self, session_id: str) -> None:
        """Keep structured memory layers bounded in memory."""
        self._recent_entries[session_id] = self._recent_entries[session_id][
            -self.STRUCTURED_RECENT_MAX_ITEMS:
        ]
        self._fact_entries[session_id] = self._fact_entries[session_id][
            -self.STRUCTURED_FACT_MAX_ITEMS:
        ]
        self._directives[session_id] = self._directives[session_id][
            -self.STRUCTURED_DIRECTIVE_MAX_ITEMS:
        ]
        self._emotional_states[session_id] = self._emotional_states[session_id][
            -self.STRUCTURED_EMOTION_MAX_ITEMS:
        ]

    def _sanitize_for_memory(self, text: str) -> str:
        """Redact PII before storing text in memory."""
        if not text:
            return ""
        return self._memory_redactor.redact(text).redacted_text

    def _compact_text(self, text: str, max_length: int) -> str:
        """Collapse whitespace and trim to a stable preview length."""
        compacted = " ".join(text.split()).strip()
        if len(compacted) <= max_length:
            return compacted
        return compacted[: max_length - 1].rstrip() + "…"

    def _build_turn_summary(self, user_input: str, response: str) -> Tuple[str, List[str], List[str]]:
        """Build a masked summary for one conversation turn."""
        combined = f"{user_input} {response}".lower()
        key_topics = self._extract_topics(combined)
        emotional_themes = self._extract_emotional_themes(combined)

        summary = self._format_user_summary(key_topics, emotional_themes)
        return summary, key_topics, emotional_themes

    def _format_user_summary(self, key_topics: List[str], emotional_themes: List[str]) -> str:
        """Build a privacy-preserving user summary without raw transcript text."""
        topic_text = ", ".join(key_topics) if key_topics else "general"
        theme_text = ", ".join(emotional_themes) if emotional_themes else "neutral"
        return f"사용자 요약: 토픽={topic_text} | 정서={theme_text}"

    def _extract_topics(self, text: str) -> List[str]:
        """Extract coarse topics for summary memory."""
        topic_map = {
            "anxiety": ["anxious", "anxiety", "불안", "걱정"],
            "depression": ["depressed", "depression", "우울", "무기력"],
            "sleep": ["sleep", "insomnia", "잠", "수면"],
            "work": ["work", "job", "직장", "업무"],
            "relationships": ["family", "friend", "partner", "관계", "가족", "친구"],
            "safety": ["suicide", "self-harm", "hurt myself", "자살", "자해", "위험"],
        }
        topics = []
        for topic, keywords in topic_map.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        return topics[:5]

    def _extract_emotional_themes(self, text: str) -> List[str]:
        """Extract emotional themes for the summary memory."""
        theme_map = {
            "불안": ["anxious", "불안", "걱정", "worried"],
            "우울": ["depressed", "우울", "down", "hopeless"],
            "스트레스": ["stressed", "stress", "압박", "overwhelmed"],
            "고립": ["alone", "lonely", "혼자", "isolated"],
            "위기": ["suicide", "self-harm", "죽고", "자살", "자해", "위험"],
        }
        themes = []
        for theme, keywords in theme_map.items():
            if any(keyword in text for keyword in keywords):
                themes.append(theme)
        return themes[:5]

    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Message]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            include_metadata: Whether to include message metadata

        Returns:
            List of Message objects
        """
        messages = self._sessions.get(session_id, [])

        if limit:
            messages = messages[-limit:]

        if not include_metadata:
            # Create copies without metadata
            messages = [
                Message(role=m.role, content=m.content, timestamp=m.timestamp)
                for m in messages
            ]

        return messages

    async def get_formatted_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        format_type: str = "simple"
    ) -> str:
        """
        Get formatted conversation history as string.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages
            format_type: Format type ("simple", "detailed", "chat")

        Returns:
            Formatted history string
        """
        messages = await self.get_history(session_id, limit)

        if not messages:
            return ""

        if format_type == "simple":
            parts = []
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            return "\n".join(parts)

        elif format_type == "detailed":
            parts = []
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"[{msg.timestamp}] {role}:\n{msg.content}")
            return "\n\n".join(parts)

        elif format_type == "chat":
            parts = []
            for msg in messages:
                parts.append({"role": msg.role, "content": msg.content})
            return json.dumps(parts, indent=2)

        return ""

    async def get_last_n_turns(
        self,
        session_id: str,
        n: int = 3
    ) -> List[Dict[str, str]]:
        """
        Get last N conversation turns.

        Args:
            session_id: Session identifier
            n: Number of turns (pairs of user/assistant messages)

        Returns:
            List of message dictionaries
        """
        messages = await self.get_history(session_id, limit=n * 2)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def get_profile(self, session_id: str) -> UserProfile:
        """
        Get user profile for session.

        Args:
            session_id: Session identifier

        Returns:
            UserProfile for the session (creates new if doesn't exist)
        """
        if session_id not in self._profiles:
            self._profiles[session_id] = UserProfile(user_id=session_id)
            logger.debug(f"Created new profile for session {session_id}")
        return self._profiles[session_id]

    async def update_profile(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update user profile from DeepSeek analysis.

        Args:
            session_id: Session identifier
            updates: Dictionary of profile updates
        """
        profile = await self.get_profile(session_id)

        # Update list fields by appending unique items
        list_fields = ["diagnosis_history", "chronic_stressors", "risk_factors", "previous_concerns"]
        for field_name in list_fields:
            if field_name in updates:
                current_list = getattr(profile, field_name)
                new_items = updates[field_name]
                if isinstance(new_items, list):
                    for item in new_items:
                        if item and item not in current_list:
                            current_list.append(item)
                elif new_items and new_items not in current_list:
                    current_list.append(new_items)

        # Update dict fields by merging
        if "therapeutic_preferences" in updates:
            profile.therapeutic_preferences.update(updates["therapeutic_preferences"])

        # Update timestamp
        profile.last_updated = datetime.now().isoformat()

        logger.debug(f"Updated profile for session {session_id}")

    async def get_cloud_context(
        self,
        session_id: str
    ) -> Tuple[List[Dict[str, str]], UserProfile]:
        """
        Get context for cloud API (10 turns + profile).

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (history list, user profile)
        """
        history = await self.get_last_n_turns(session_id, n=10)
        profile = await self.get_profile(session_id)
        return history, profile

    async def get_local_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get context for local model (3 turns only).

        Args:
            session_id: Session identifier

        Returns:
            List of recent message dictionaries
        """
        return await self.get_last_n_turns(session_id, n=3)

    async def set_session_metadata(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Set session-level metadata."""
        self._metadata[session_id][key] = value
        if key == "user_id" and isinstance(value, str) and value.strip():
            sessions = self._user_structured_sessions[value.strip()]
            if session_id not in sessions:
                sessions.append(session_id)

    async def get_session_metadata(
        self,
        session_id: str,
        key: Optional[str] = None
    ) -> Any:
        """Get session-level metadata."""
        if key:
            return self._metadata[session_id].get(key)
        return dict(self._metadata[session_id])

    async def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._summaries:
            del self._summaries[session_id]
        if session_id in self._metadata:
            del self._metadata[session_id]
        if session_id in self._profiles:
            del self._profiles[session_id]
        if session_id in self._continuity:
            del self._continuity[session_id]
        for session_ids in self._user_structured_sessions.values():
            if session_id in session_ids:
                session_ids.remove(session_id)
        await self.clear_structured_memory(session_id)

        logger.info(f"Cleared session {session_id}")

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        messages = self._sessions.get(session_id, [])
        return {
            "message_count": len(messages),
            "turn_count": len(messages) // 2,
            "first_message": messages[0].timestamp if messages else None,
            "last_message": messages[-1].timestamp if messages else None,
            "has_summaries": len(self._summaries.get(session_id, [])) > 0
        }

    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs."""
        return list(self._sessions.keys())

    def _load_persisted(self) -> None:
        """Load persisted sessions from disk."""
        if not self.config.persist_path or not os.path.exists(self.config.persist_path):
            return

        try:
            persist_dir = Path(self.config.persist_path)
            for session_file in persist_dir.glob("*.json"):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                session_id = session_file.stem
                self._sessions[session_id] = [
                    Message.from_dict(m) for m in data.get("messages", [])
                ]
                self._metadata[session_id] = data.get("metadata", {})
                user_id = str(self._metadata[session_id].get("user_id") or "").strip()
                if user_id and session_id not in self._user_structured_sessions[user_id]:
                    self._user_structured_sessions[user_id].append(session_id)
                continuity_data = data.get("continuity")
                if isinstance(continuity_data, dict):
                    small_action_data = continuity_data.get("last_small_action")
                    summary_data = continuity_data.get("latest_structured_summary")
                    self._continuity[session_id] = ConversationContinuityMemory(
                        session_id=session_id,
                        last_small_action=StructuredSmallActionMemory(**small_action_data)
                        if isinstance(small_action_data, dict)
                        else None,
                        next_follow_up=continuity_data.get("next_follow_up", ""),
                        emotional_state_vector=continuity_data.get("emotional_state_vector", {}),
                        latest_structured_summary=SessionStructuredSummary(**summary_data)
                        if isinstance(summary_data, dict)
                        else None,
                        turn_count=int(continuity_data.get("turn_count", 0)),
                        risk_stage_start=continuity_data.get("risk_stage_start", "관심"),
                        risk_stage_end=continuity_data.get("risk_stage_end", "관심"),
                        updated_at=continuity_data.get("updated_at", utc_now_iso()),
                    )
                self._reflection_entries[session_id] = [
                    ReflectionMemoryEntry(**entry)
                    for entry in data.get("reflection_history", [])
                    if isinstance(entry, dict)
                ][-self.REFLECTION_MAX_ITEMS:]

            logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.error(f"Error loading persisted sessions: {e}")

    async def _persist_session(self, session_id: str) -> None:
        """Persist a session to disk."""
        if not self.config.persist_path:
            return

        try:
            persist_dir = Path(self.config.persist_path)
            persist_dir.mkdir(parents=True, exist_ok=True)

            session_file = persist_dir / f"{session_id}.json"
            data = {
                "messages": [m.to_dict() for m in self._sessions[session_id]],
                "metadata": self._metadata.get(session_id, {}),
                "continuity": self._continuity[session_id].to_dict()
                if session_id in self._continuity
                else None,
                "reflection_history": [
                    entry.to_dict() for entry in self._reflection_entries.get(session_id, [])
                ],
                "updated_at": datetime.now().isoformat()
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error persisting session {session_id}: {e}")


class ConversationContext:
    """Helper for building conversation context from memory."""

    def __init__(self, memory_store: MemoryStore):
        self.store = memory_store

    async def build_context(
        self,
        session_id: str,
        max_turns: int = 5,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Build conversation context for model input.

        Args:
            session_id: Session identifier
            max_turns: Maximum conversation turns to include
            include_summary: Whether to include session summary

        Returns:
            Context dictionary
        """
        history = await self.store.get_last_n_turns(session_id, max_turns)

        context = {
            "conversation_history": history,
            "turn_count": len(history) // 2,
            "session_id": session_id
        }

        # Add session metadata
        metadata = await self.store.get_session_metadata(session_id)
        if metadata:
            context["session_metadata"] = metadata

        return context
