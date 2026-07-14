"""
Prompt Generator for cloud and local model inference.

This module provides the PromptGenerator class that creates prompts
for both the cloud analysis (Deepseek-V3) and local generation (GGUF) stages.
"""

import os
import json
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field

from src.prompt.templates import TemplateLoader, PromptTemplate, DEFAULT_TEMPLATES
from src.utils.logging_config import setup_logging

logger = setup_logging("prompt_generator")

if TYPE_CHECKING:
    from src.memory.models import MemoryContext


@dataclass
class CloudPrompt:
    """Prompt for cloud (Deepseek-V3) analysis."""
    system_message: str
    user_message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for API."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message}
        ]


@dataclass
class LocalPrompt:
    """Prompt for local (GGUF) generation."""
    system_message: str
    user_message: str
    full_prompt: str  # Formatted for single-prompt models
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to messages list for chat completion API."""
        if self.messages:
            return self.messages
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message}
        ]


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    max_history_turns: int = 5
    max_rag_context_length: int = 1500
    include_timestamps: bool = False
    therapeutic_approach: str = "integrative"


class PromptGenerator:
    """
    Generator for cloud and local model prompts.

    Creates structured prompts for the two-stage inference pipeline:
    1. Cloud prompt for Deepseek-V3 analysis
    2. Local prompt for GGUF response generation

    Example:
        generator = PromptGenerator()
        cloud_prompt = generator.gen_cloud_prompt(
            sanitized_input="I'm feeling anxious",
            rag_context="CBT techniques for anxiety...",
            history=[{"role": "user", "content": "Hi"}]
        )
    """

    def __init__(
        self,
        config: Optional[PromptConfig] = None,
        template_path: Optional[str] = None
    ):
        """
        Initialize prompt generator.

        Args:
            config: Prompt generation configuration
            template_path: Path to custom template file
        """
        self.config = config or PromptConfig()
        self.template_loader = TemplateLoader(template_path)
        self._templates = self.template_loader.load()

        logger.info("PromptGenerator initialized")

    def gen_cloud_prompt(
        self,
        sanitized_input: str,
        rag_context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        memory_context: Optional["MemoryContext"] = None
    ) -> CloudPrompt:
        """
        Generate prompt for cloud (Deepseek-V3) analysis.

        Args:
            sanitized_input: User input after PII redaction
            rag_context: Retrieved knowledge context
            history: Conversation history (10 turns for cloud)
            user_profile: Long-term user profile for clinical context
            additional_context: Additional context variables
            memory_context: Optional structured memory layers

        Returns:
            CloudPrompt ready for API call
        """
        template = self._templates.get("cloud_analysis", DEFAULT_TEMPLATES["cloud_analysis"])

        # Format conversation history
        history_str = self._format_history(history or [], max_turns=10)

        # Truncate RAG context if needed
        rag_context = self._truncate_context(rag_context, self.config.max_rag_context_length)

        # Format user profile
        profile_str = json.dumps(user_profile, indent=2) if user_profile else "{}"

        memory_context_str = self._format_memory_context(memory_context)

        rendered = template.format(
            user_input=sanitized_input,
            conversation_history=history_str or "(No prior conversation)",
            rag_context=rag_context or "(No additional context)",
            user_profile=profile_str,
            memory_context=memory_context_str,
            safety_notice=(
                "이 AI는 의료 진단이나 치료를 하지 않으며, 전문 상담사를 대체하지 않습니다. "
                "위험 신호가 있으면 109, 119, 112 또는 가까운 응급실 연결을 우선 안내하세요."
            ),
            output_schema=(
                '{"risk_stage":"관심|주의|위험","risk_level":"low|moderate|high|critical",'
                '"primary_concern":"...","risk_reasoning":"...",'
                '"guidance_for_local_model":"...","suggested_technique":"...",'
                '"updated_user_profile":{}}'
            )
        )

        system_message = rendered["system"]
        user_message = rendered["user"]

        return CloudPrompt(
            system_message=system_message,
            user_message=user_message,
            metadata={
                "template": "cloud_analysis_supervisor",
                "input_length": len(sanitized_input),
                "history_turns": len(history or []) // 2,
                "has_rag_context": bool(rag_context),
                "has_user_profile": bool(user_profile),
                "has_memory_context": bool(memory_context_str)
            }
        )

    def gen_local_prompt(
        self,
        user_input: str,
        cloud_analysis: Union[str, Dict[str, Any]],
        rag_context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        therapeutic_guidance: str = "",
        additional_context: Optional[Dict[str, Any]] = None,
        memory_context: Optional["MemoryContext"] = None,
        agent_context: Optional[Dict[str, Any]] = None
    ) -> LocalPrompt:
        """
        Generate prompt for local (GGUF) response generation.

        Args:
            user_input: Original user input
            cloud_analysis: Analysis from cloud API (Dict or string)
            rag_context: Retrieved knowledge context
            history: Conversation history (3 turns for local)
            therapeutic_guidance: Additional therapeutic guidance
            additional_context: Additional context variables
            memory_context: Optional structured memory layers
            agent_context: Optional allowlisted agent decision context

        Returns:
            LocalPrompt ready for local inference
        """
        template = self._templates.get("local_generation", DEFAULT_TEMPLATES["local_generation"])

        # Handle both Dict and string formats
        if isinstance(cloud_analysis, dict):
            analysis_dict = cloud_analysis
        else:
            # Legacy string format
            analysis_dict = {
                "primary_concern": "",
                "suggested_technique": therapeutic_guidance or self._get_default_guidance(),
                "guidance_for_local_model": cloud_analysis
            }

        # Format conversation history (3 turns only for local)
        history_str = self._format_history(history or [], max_turns=3)

        # Truncate RAG context if needed
        rag_context = self._truncate_context(rag_context, self.config.max_rag_context_length)

        memory_context_str = self._format_memory_context(memory_context)
        dataset_hints, dataset_hint_keys = self._format_dataset_hints(additional_context)
        agent_context_str = self._format_agent_context(agent_context)

        rendered = template.format(
            user_input=user_input,
            cloud_analysis=json.dumps(analysis_dict, ensure_ascii=False, indent=2),
            rag_context=rag_context or "(No additional context)",
            conversation_history=history_str or "(No prior conversation)",
            therapeutic_guidance=analysis_dict.get("guidance_for_local_model") or self._get_default_guidance(),
            agent_context=agent_context_str or "(No agent decision context)",
            dataset_hints=dataset_hints or "(No processed dataset hints)",
            memory_context=memory_context_str,
            safety_notice=(
                "이 응답은 의료 진단이나 치료가 아니며, 사용자 감정 정리와 안전 안내를 우선한다."
            ),
            expert_referral=(
                "위험 신호가 있으면 109, 119, 112 또는 가까운 응급실/지역 정신건강복지센터 연결을 제안한다."
            )
        )

        system_content = rendered["system"]
        user_content = rendered["user"]

        # Build messages list for chat completion API
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Create full prompt for single-prompt models (backwards compatibility)
        full_prompt = self._build_full_prompt(system_content, user_content)

        return LocalPrompt(
            system_message=system_content,
            user_message=user_content,
            full_prompt=full_prompt,
            messages=messages,
            metadata={
                "template": "local_generation_agent",
                "input_length": len(user_input),
                "analysis_length": len(str(cloud_analysis)),
                "history_turns": len(history or []) // 2,
                "has_memory_context": bool(memory_context_str),
                "has_agent_context": bool(agent_context_str),
                "has_dataset_hints": bool(dataset_hints),
                "dataset_hint_keys": dataset_hint_keys
            }
        )

    def gen_crisis_prompt(
        self,
        user_input: str,
        risk_level: str,
        matched_pattern: str = "",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LocalPrompt:
        """
        Generate prompt for crisis response.

        Args:
            user_input: User's crisis message
            risk_level: Detected risk level
            matched_pattern: Pattern that triggered crisis detection
            additional_context: Additional context

        Returns:
            LocalPrompt for crisis response
        """
        template = self._templates.get("crisis_response", DEFAULT_TEMPLATES["crisis_response"])

        rendered = template.format(
            user_input=user_input,
            risk_level=risk_level,
            matched_pattern=matched_pattern or "(General crisis indicators)",
            emergency_numbers="109, 119, 112",
            safety_notice=(
                "이 AI는 의료 진단이나 치료를 하지 않으며, 즉시 안전 확보와 전문기관 연결을 우선한다."
            ),
            **(additional_context or {})
        )

        system_message = rendered["system"]
        user_message = rendered["user"]

        full_prompt = self._build_full_prompt(system_message, user_message)

        return LocalPrompt(
            system_message=system_message,
            user_message=user_message,
            full_prompt=full_prompt,
            metadata={
                "template": "crisis_response",
                "risk_level": risk_level
            }
        )

    def _format_history(
        self,
        history: List[Dict[str, str]],
        max_turns: Optional[int] = None
    ) -> str:
        """Format conversation history as string."""
        if not history:
            return ""

        max_turns = max_turns or self.config.max_history_turns
        # Take last N messages (each turn is 2 messages)
        recent = history[-(max_turns * 2):]

        parts = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _format_memory_context(
        self,
        memory_context: Optional["MemoryContext"]
    ) -> str:
        """Format privacy-preserving structured memory for prompt use."""
        if not memory_context or memory_context.is_empty():
            return ""

        sections = ["[Memory - Structured Context]"]

        if memory_context.recent_summaries:
            sections.append("[Recent Summaries]")
            for item in memory_context.recent_summaries:
                topics = ", ".join(item.key_topics) if item.key_topics else "none"
                emotions = ", ".join(item.emotional_themes) if item.emotional_themes else "neutral"
                sections.append(
                    f"- {item.summary} "
                    f"(topics: {topics}; observed emotions: {emotions}; risk stage: {item.risk_stage})"
                )

        if memory_context.facts:
            sections.append("[Facts]")
            for item in memory_context.facts:
                sections.append(
                    f"- {item.category}: {item.label}={item.normalized_value} "
                    f"(confidence: {item.confidence:.2f}; evidence_count: {item.evidence_count})"
                )

        active_directives = [
            item for item in memory_context.directives
            if getattr(item, "active", True)
        ]
        if active_directives:
            sections.append("[User Directives]")
            for item in active_directives:
                sections.append(f"- {item.kind}: {item.term}")

        if memory_context.emotional_trend:
            sections.append("[Emotional Trend - Observed, Not Diagnostic]")
            for item in memory_context.emotional_trend:
                sections.append(
                    f"- observed {item.label} trend "
                    f"(intensity: {item.intensity:.2f}; confidence: {item.confidence:.2f}; "
                    f"risk stage: {item.risk_stage}; source: {item.source})"
                )

        return "\n".join(sections)

    def _format_dataset_hints(
        self,
        additional_context: Optional[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        """Format only allowlisted processed dataset hints for local prompts."""
        if not additional_context:
            return "", []

        allowed_hints = [
            ("counseling_hint", "Counseling intervention"),
            ("empathy_style_hint", "Empathy style"),
            ("wellness_hint", "Wellness support"),
        ]

        sections = ["[Processed Dataset Hints]"]
        included_keys = []

        for key, label in allowed_hints:
            value = additional_context.get(key)
            if not isinstance(value, str):
                continue

            value = value.strip()
            if not value:
                continue

            sections.append(f"- {label}: {value}")
            included_keys.append(key)

        if not included_keys:
            return "", []

        return "\n".join(sections), included_keys

    def _format_agent_context(
        self,
        agent_context: Optional[Dict[str, Any]]
    ) -> str:
        """Format only allowlisted agent context for local prompts."""
        if not isinstance(agent_context, dict):
            return ""

        def _as_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                data = to_dict()
                return data if isinstance(data, dict) else {}
            return {}

        def _stringify(value: Any) -> str:
            value = getattr(value, "name", value)
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value)
            return ""

        def _string_list(value: Any) -> List[str]:
            if not isinstance(value, (list, tuple, set)):
                return []
            formatted = []
            for item in value:
                item_str = _stringify(item)
                if item_str:
                    formatted.append(item_str)
            return formatted

        sections = ["[Agent Decision Context]"]

        decision = _as_dict(agent_context.get("decision"))
        primary_action = _stringify(decision.get("primary_action"))
        if primary_action:
            sections.append(f"- Primary action: {primary_action}")

        secondary_actions = _string_list(decision.get("secondary_actions"))
        if secondary_actions:
            sections.append(f"- Secondary actions: {', '.join(secondary_actions)}")

        constraints = _as_dict(decision.get("response_constraints"))
        allowed_constraints = {
            "must_include_followup",
            "must_include_small_action",
            "max_questions",
            "avoid_topics",
        }
        rendered_constraints = []
        for key in sorted(allowed_constraints):
            value = constraints.get(key)
            if key == "avoid_topics":
                topics = _string_list(value)
                if topics:
                    rendered_constraints.append(f"avoid_topics={', '.join(topics)}")
                continue
            if isinstance(value, (bool, int, float, str)):
                rendered_constraints.append(f"{key}={value}")
        if rendered_constraints:
            sections.append(f"- Response constraints: {'; '.join(rendered_constraints)}")

        if constraints.get("must_include_followup") is True:
            sections.append("- Constraint instruction: include at most one follow-up question.")
        if constraints.get("must_include_small_action") is True:
            sections.append("- Constraint instruction: include one small action.")
        if constraints.get("max_questions") == 1:
            sections.append("- Constraint instruction: ask no more than one question.")

        emotional_state = _as_dict(agent_context.get("emotional_state"))
        state_summary = _stringify(emotional_state.get("state_summary"))
        if state_summary:
            sections.append(f"- Emotional state summary: {state_summary}")

        numeric_keys = ["mood", "anxiety", "stress", "sleep", "energy", "safety", "rapport"]
        numeric_parts = []
        for key in numeric_keys:
            value = emotional_state.get(key)
            if isinstance(value, (int, float)):
                numeric_parts.append(f"{key}={float(value):.2f}")
        if numeric_parts:
            sections.append(f"- Emotional state scores: {', '.join(numeric_parts)}")

        proactive_recall = _as_dict(agent_context.get("proactive_recall"))
        recalled_keys = _string_list(proactive_recall.get("recalled_keys"))
        if recalled_keys:
            sections.append(f"- Proactive recall keys: {', '.join(recalled_keys)}")

        repeated_concerns = _string_list(proactive_recall.get("repeated_concerns"))
        if repeated_concerns:
            sections.append(f"- Repeated concerns: {', '.join(repeated_concerns)}")

        preferred_response_style = _string_list(proactive_recall.get("preferred_response_style"))
        if preferred_response_style:
            sections.append(f"- Preferred response style: {', '.join(preferred_response_style)}")

        avoid_topics = _string_list(proactive_recall.get("avoid_topics"))
        if avoid_topics:
            sections.append(f"- Avoid topics: {', '.join(avoid_topics)}")
            sections.append("- Constraint instruction: avoid the listed topics unless safety requires addressing them.")

        followup = _as_dict(agent_context.get("followup"))
        followup_question = _stringify(followup.get("question"))
        if followup_question:
            sections.append(f"- Follow-up question to consider: {followup_question}")

        small_action = _as_dict(agent_context.get("small_action"))
        action_text = _stringify(small_action.get("action_text"))
        intent_label = _stringify(small_action.get("intent_label"))
        if action_text:
            sections.append(f"- Small action: {action_text}")
        if intent_label:
            sections.append(f"- Small action intent: {intent_label}")

        if primary_action == "ESCALATE_SAFETY":
            sections.append("- Safety priority: follow the existing safety/crisis flow before normal local generation.")

        return "\n".join(sections) if len(sections) > 1 else ""

    def _truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context to maximum length."""
        if len(context) <= max_length:
            return context
        return context[:max_length] + "..."

    def _build_full_prompt(self, system: str, user: str) -> str:
        """Build full prompt for single-prompt models (llama-cpp-python)."""
        return f"""<|system|>
{system}
<|user|>
{user}
<|assistant|>
"""

    def _get_default_guidance(self) -> str:
        """Get default therapeutic guidance based on config."""
        guidance_map = {
            "cbt": "Focus on cognitive restructuring and behavioral activation.",
            "dbt": "Emphasize distress tolerance and emotional regulation skills.",
            "supportive": "Provide validation, empathy, and emotional support.",
            "integrative": "Combine elements from various therapeutic approaches as appropriate."
        }
        return guidance_map.get(
            self.config.therapeutic_approach,
            guidance_map["integrative"]
        )

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self._templates.keys())
