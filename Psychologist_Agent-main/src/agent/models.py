"""
Structured result models for the agent pipeline.

These schemas carry labels, scores, summaries, and decisions only. They
intentionally do not define fields for storing raw user or assistant text.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Type


RAW_TEXT_FIELD_NAMES = {
    "raw_text",
    "raw_input",
    "user_input",
    "assistant_response",
    "conversation",
    "content",
    "transcript",
    "message",
}


def _clamp01(value: float) -> float:
    """Clamp a numeric score into the 0.0 to 1.0 range."""
    return max(0.0, min(1.0, float(value)))


def validate_no_raw_fields(schema_classes: Iterable[Type[Any]]) -> bool:
    """Validate that dataclass schemas do not define raw text field names."""
    for schema_class in schema_classes:
        if not is_dataclass(schema_class):
            raise TypeError(f"{schema_class!r} is not a dataclass")

        blocked = RAW_TEXT_FIELD_NAMES.intersection(
            item.name for item in fields(schema_class)
        )
        if blocked:
            names = ", ".join(sorted(blocked))
            raise ValueError(f"{schema_class.__name__} cannot store raw text fields: {names}")

    return True


class EmotionLabel(str, Enum):
    """Coarse emotional labels used by the emotion agent."""

    NEUTRAL = "neutral"
    CALM = "calm"
    SADNESS = "sadness"
    ANXIETY = "anxiety"
    ANGER = "anger"
    STRESS = "stress"
    LONELINESS = "loneliness"
    HOPELESSNESS = "hopelessness"
    FATIGUE = "fatigue"
    RELIEF = "relief"


class IntentSeverity(str, Enum):
    """Severity levels for detected user intent."""

    S1_CONCERN = "s1_concern"
    S2_SUSPECTED_CONDITION = "s2_suspected_condition"
    S3_SOS = "s3_sos"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class IntentLabel(str, Enum):
    """High-level intent categories for pipeline routing."""

    SLEEP_PROBLEM = "sleep_problem"
    ANXIETY_SUPPORT = "anxiety_support"
    LOW_MOOD_SUPPORT = "low_mood_support"
    STRESS_SUPPORT = "stress_support"
    RELATIONSHIP_STRESS = "relationship_stress"
    WORK_OR_STUDY_STRESS = "work_or_study_stress"
    FAMILY_CONFLICT = "family_conflict"
    LOW_SELF_ESTEEM = "low_self_esteem"
    NEED_EMPATHY = "need_empathy"
    NEED_ADVICE = "need_advice"
    CRISIS_SIGNAL = "crisis_signal"
    SUBSTANCE_OR_ADDICTION = "substance_or_addiction"
    GRIEF_SUPPORT = "grief_support"
    OTHER_CONCERN = "other_concern"
    SUPPORT_REQUEST = "support_request"
    EMOTIONAL_DISCLOSURE = "emotional_disclosure"
    SAFETY_CONCERN = "safety_concern"
    PRACTICAL_HELP = "practical_help"
    REFLECTION = "reflection"
    MEMORY_UPDATE = "memory_update"
    SMALL_ACTION = "small_action"
    CLARIFICATION = "clarification"


class DecisionAction(str, Enum):
    """Allowed actions from the decision agent."""

    RESPOND_SUPPORTIVELY = "respond_supportively"
    ASK_FOLLOW_UP = "ask_follow_up"
    SUGGEST_SMALL_ACTION = "suggest_small_action"
    SUMMARIZE_STATE = "summarize_state"
    UPDATE_MEMORY = "update_memory"
    ESCALATE_SAFETY = "escalate_safety"


@dataclass
class EmotionalStateVector:
    """Normalized emotional state scores for a session turn."""

    mood: float = 0.5
    anxiety: float = 0.3
    stress: float = 0.3
    sleep: float = 0.5
    energy: float = 0.5
    safety: float = 0.9
    rapport: float = 0.2

    def __post_init__(self) -> None:
        self.mood = _clamp01(self.mood)
        self.anxiety = _clamp01(self.anxiety)
        self.stress = _clamp01(self.stress)
        self.sleep = _clamp01(self.sleep)
        self.energy = _clamp01(self.energy)
        self.safety = _clamp01(self.safety)
        self.rapport = _clamp01(self.rapport)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyAgentResult:
    """Structured safety assessment for routing."""

    is_safe: bool = True
    risk_stage: str = "관심"
    requires_escalation: bool = False
    safety_topics: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionAgentResult:
    """Emotion agent labels and normalized state vector."""

    primary_label: EmotionLabel = EmotionLabel.NEUTRAL
    secondary_labels: List[EmotionLabel] = field(default_factory=list)
    intensity: float = 0.0
    confidence: float = 1.0
    state_vector: EmotionalStateVector = field(default_factory=EmotionalStateVector)

    def __post_init__(self) -> None:
        self.intensity = _clamp01(self.intensity)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentCandidate:
    """Single intent candidate emitted by the intent agent."""

    label: IntentLabel
    severity: IntentSeverity = IntentSeverity.LOW
    confidence: float = 1.0
    rationale_tags: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentAgentResult:
    """Ranked intent result for downstream routing."""

    primary_intent: IntentLabel = IntentLabel.SUPPORT_REQUEST
    severity: IntentSeverity = IntentSeverity.LOW
    candidates: List[IntentCandidate] = field(default_factory=list)
    needs_follow_up: bool = False
    confidence: float = 1.0
    s2_suspected: bool = False
    s3_sos: bool = False
    chat_label_hint: Dict[str, bool] = field(
        default_factory=lambda: {
            "question": False,
            "knowledge": False,
            "negative": False,
        }
    )

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetStrategyResult:
    """Dataset and retrieval strategy chosen for a response turn."""

    strategy_id: str = "supportive_default"
    selected_sources: List[str] = field(default_factory=list)
    retrieval_tags: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        self.source_weights = {
            source: _clamp01(weight) for source, weight in self.source_weights.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProactiveRecallResult:
    """Memory recall routing without storing raw prior turns."""

    recall_needed: bool = False
    memory_types: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    reason_tags: List[str] = field(default_factory=list)
    repeated_concerns: List[str] = field(default_factory=list)
    emotional_trend_summary: str = ""
    last_small_action: str = ""
    preferred_response_style: List[str] = field(default_factory=list)
    avoid_topics: List[str] = field(default_factory=list)
    next_follow_up: str = ""
    recalled_keys: List[str] = field(default_factory=list)
    stale: bool = False

    def __post_init__(self) -> None:
        self.relevance_scores = {
            key: _clamp01(score) for key, score in self.relevance_scores.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionAgentResult:
    """Decision agent output for the next pipeline action."""

    action: DecisionAction = DecisionAction.RESPOND_SUPPORTIVELY
    primary_action: DecisionAction = DecisionAction.RESPOND_SUPPORTIVELY
    secondary_actions: List[DecisionAction] = field(default_factory=list)
    response_constraints: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    confidence: float = 1.0
    rationale_tags: List[str] = field(default_factory=list)
    state_vector: EmotionalStateVector = field(default_factory=EmotionalStateVector)
    safety_escalation: Optional[SafetyAgentResult] = None

    def __post_init__(self) -> None:
        if self.primary_action == DecisionAction.RESPOND_SUPPORTIVELY:
            self.primary_action = self.action
        self.action = self.primary_action
        if not self.rationale_tags:
            self.rationale_tags = list(self.reason_codes)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SmallActionPlan:
    """A small behavioral step proposed by the agent."""

    action_id: str
    title: str
    session_id: str = ""
    intent_label: str = ""
    action_text: str = ""
    rationale_label: str = ""
    status: str = "suggested"
    created_at: str = ""
    check_after_turns: int = 1
    steps: List[str] = field(default_factory=list)
    estimated_minutes: int = 5
    difficulty: str = "easy"
    rationale_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionDreamSummary:
    """Compact session-level synthesis for future recall."""

    session_id: str
    summary_id: str
    main_issue: List[str] = field(default_factory=list)
    emotional_trend: List[str] = field(default_factory=list)
    risk_stage_start: str = "관심"
    risk_stage_end: str = "관심"
    last_small_action: str = ""
    next_follow_up: str = ""
    important_user_directives: List[str] = field(default_factory=list)
    created_at: str = ""
    emotional_arc: List[EmotionLabel] = field(default_factory=list)
    recurring_themes: List[str] = field(default_factory=list)
    memory_updates: List[str] = field(default_factory=list)
    unresolved_needs: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    next_session_focus: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ALL_AGENT_SCHEMAS = (
    SafetyAgentResult,
    EmotionAgentResult,
    IntentCandidate,
    IntentAgentResult,
    DatasetStrategyResult,
    ProactiveRecallResult,
    EmotionalStateVector,
    DecisionAgentResult,
    SmallActionPlan,
    SessionDreamSummary,
)


validate_no_raw_fields(ALL_AGENT_SCHEMAS)
