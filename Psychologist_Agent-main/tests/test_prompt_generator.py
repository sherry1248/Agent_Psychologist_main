"""
Tests for Prompt Generator module.

All tests use MOCK mode.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.prompt.generator import (
    PromptGenerator, PromptConfig,
    CloudPrompt, LocalPrompt
)
from src.prompt.templates import TemplateLoader, PromptTemplate, DEFAULT_TEMPLATES
from src.memory.models import (
    EmotionalStateEntry,
    FactMemoryEntry,
    MemoryContext,
    RecentMemoryEntry,
    UserDirective,
)


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_template_format(self):
        """Test template formatting."""
        template = PromptTemplate(
            name="test",
            system_message="You are {role}",
            user_template="User says: {message}",
            variables=["role", "message"]
        )

        formatted = template.format(role="assistant", message="hello")
        assert formatted["system"] == "You are assistant"
        assert formatted["user"] == "User says: hello"

    def test_template_format_no_vars(self):
        """Test template formatting without variables."""
        template = PromptTemplate(
            name="test",
            system_message="Static system",
            user_template="Static user"
        )

        formatted = template.format()
        assert formatted["system"] == "Static system"


class TestTemplateLoader:
    """Tests for TemplateLoader class."""

    def test_load_default_templates(self):
        """Test loading default templates."""
        loader = TemplateLoader()
        templates = loader.load()

        assert "cloud_analysis" in templates
        assert "local_generation" in templates
        assert "crisis_response" in templates

    def test_get_template(self):
        """Test getting a specific template."""
        loader = TemplateLoader()
        template = loader.get("cloud_analysis")

        assert template is not None
        assert template.name == "cloud_analysis"

    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist."""
        loader = TemplateLoader()
        template = loader.get("nonexistent_template")

        assert template is None

    def test_get_all_templates(self):
        """Test getting all templates."""
        loader = TemplateLoader()
        templates = loader.get_all()

        assert len(templates) >= 4  # At least the defaults


class TestPromptGenerator:
    """Tests for PromptGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a prompt generator."""
        return PromptGenerator()

    def test_gen_cloud_prompt(self, generator):
        """Test cloud prompt generation."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="I'm feeling anxious",
            rag_context="CBT techniques for anxiety",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        )

        assert isinstance(prompt, CloudPrompt)
        assert len(prompt.system_message) > 0
        assert "I'm feeling anxious" in prompt.user_message
        # RAG context is now in user_message for supervisor prompt
        assert "CBT" in prompt.user_message

    def test_gen_cloud_prompt_no_context(self, generator):
        """Test cloud prompt without context."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Hello"
        )

        assert isinstance(prompt, CloudPrompt)
        assert "Hello" in prompt.user_message

    def test_gen_local_prompt(self, generator):
        """Test local prompt generation."""
        # New format: cloud_analysis is a dict
        prompt = generator.gen_local_prompt(
            user_input="I'm feeling anxious",
            cloud_analysis={
                "risk_level": "low",
                "primary_concern": "anxiety",
                "guidance_for_local_model": "Validate feelings",
                "suggested_technique": "DBT grounding"
            },
            rag_context="DBT skills for anxiety",
            history=[]
        )

        assert isinstance(prompt, LocalPrompt)
        assert len(prompt.full_prompt) > 0
        assert "I'm feeling anxious" in prompt.user_message
        # Analysis info is now in system_message as Supervisor Guidance
        assert "anxiety" in prompt.system_message
        # Check messages list is populated
        assert len(prompt.messages) == 2
        assert prompt.messages[0]["role"] == "system"

    def test_gen_local_prompt_without_agent_context_preserves_existing_call(self, generator):
        """Test local prompt generation still works without agent context."""
        prompt = generator.gen_local_prompt(
            user_input="요즘 불안해요.",
            cloud_analysis={"primary_concern": "anxiety"},
            history=[],
        )

        assert isinstance(prompt, LocalPrompt)
        assert prompt.metadata["has_agent_context"] is False
        assert "(No agent decision context)" in prompt.system_message

    def test_gen_local_prompt_includes_allowlisted_agent_context(self, generator):
        """Test local prompt includes allowlisted agent decision fields."""
        prompt = generator.gen_local_prompt(
            user_input="요즘 불안해요.",
            cloud_analysis={"primary_concern": "anxiety"},
            agent_context={
                "decision": {
                    "primary_action": "ASK_FOLLOW_UP",
                    "secondary_actions": ["SUGGEST_SMALL_ACTION"],
                    "response_constraints": {
                        "must_include_followup": True,
                        "must_include_small_action": True,
                        "max_questions": 1,
                        "avoid_topics": ["family_conflict"],
                        "tone": "calm",
                    },
                    "reason_codes": ["not_allowed"],
                },
                "emotional_state": {
                    "state_summary": "불안과 스트레스가 높고 수면이 낮음",
                    "mood": 0.4,
                    "anxiety": 0.8,
                    "stress": 0.7,
                    "sleep": 0.2,
                    "energy": 0.3,
                    "safety": 0.9,
                    "rapport": 0.2,
                },
                "proactive_recall": {
                    "recalled_keys": ["sleep_pattern"],
                    "repeated_concerns": ["insomnia"],
                    "preferred_response_style": ["concise"],
                    "avoid_topics": ["family_conflict"],
                },
                "followup": {"question": "잠드는 데 특히 어려운 시간이 있나요?"},
                "small_action": {
                    "action_text": "물을 한 모금 마시고 어깨 힘을 10초만 풀어보세요.",
                    "intent_label": "SLEEP_PROBLEM",
                },
            },
        )

        combined_prompt = prompt.full_prompt
        assert "ASK_FOLLOW_UP" in combined_prompt
        assert "불안과 스트레스가 높고 수면이 낮음" in combined_prompt
        assert "잠드는 데 특히 어려운 시간이 있나요?" in combined_prompt
        assert "물을 한 모금 마시고 어깨 힘을 10초만 풀어보세요." in combined_prompt
        assert "max_questions=1" in combined_prompt
        assert "family_conflict" in combined_prompt
        assert "tone" not in combined_prompt
        assert "reason_codes" not in combined_prompt
        assert prompt.metadata["has_agent_context"] is True

    def test_gen_local_prompt_excludes_raw_looking_agent_context_keys(self, generator):
        """Test raw-looking agent context keys and values are not rendered."""
        prompt = generator.gen_local_prompt(
            user_input="도움이 필요해요.",
            cloud_analysis={"primary_concern": "support"},
            agent_context={
                "decision": {
                    "primary_action": "RESPOND_SUPPORTIVELY",
                    "user_input": "RAW_USER_INPUT_SHOULD_NOT_APPEAR",
                    "raw_text": "RAW_TEXT_SHOULD_NOT_APPEAR",
                    "conversation": "CONVERSATION_SHOULD_NOT_APPEAR",
                    "content": "CONTENT_SHOULD_NOT_APPEAR",
                    "assistant_response": "ASSISTANT_RESPONSE_SHOULD_NOT_APPEAR",
                },
                "raw_text": "TOP_LEVEL_RAW_SHOULD_NOT_APPEAR",
            },
        )

        combined_prompt = prompt.full_prompt
        for forbidden in (
            "RAW_USER_INPUT_SHOULD_NOT_APPEAR",
            "RAW_TEXT_SHOULD_NOT_APPEAR",
            "CONVERSATION_SHOULD_NOT_APPEAR",
            "CONTENT_SHOULD_NOT_APPEAR",
            "ASSISTANT_RESPONSE_SHOULD_NOT_APPEAR",
            "TOP_LEVEL_RAW_SHOULD_NOT_APPEAR",
            "user_input",
            "raw_text",
            "assistant_response",
        ):
            assert forbidden not in combined_prompt
        assert "conversation=" not in combined_prompt
        assert "content=" not in combined_prompt

    def test_gen_local_prompt_includes_processed_dataset_hints(self, generator):
        """Test local prompt includes allowlisted processed dataset hints."""
        prompt = generator.gen_local_prompt(
            user_input="요즘 너무 지쳤어요.",
            cloud_analysis={
                "risk_level": "low",
                "primary_concern": "stress",
                "guidance_for_local_model": "Validate stress and offer one small step",
            },
            additional_context={
                "counseling_hint": "오늘 할 일을 작게 나누어 보세요.",
                "empathy_style_hint": "지친 마음을 먼저 인정하세요.",
                "wellness_hint": "수면과 식사를 점검하고 짧게 쉬도록 제안하세요.",
            },
        )

        assert "오늘 할 일을 작게 나누어 보세요." in prompt.system_message
        assert "지친 마음을 먼저 인정하세요." in prompt.system_message
        assert "수면과 식사를 점검하고 짧게 쉬도록 제안하세요." in prompt.system_message
        assert prompt.metadata["has_dataset_hints"] is True
        assert prompt.metadata["dataset_hint_keys"] == [
            "counseling_hint",
            "empathy_style_hint",
            "wellness_hint",
        ]

    def test_gen_local_prompt_excludes_unknown_and_raw_context_keys(self, generator):
        """Test raw-looking additional context keys are not rendered."""
        prompt = generator.gen_local_prompt(
            user_input="요즘 너무 지쳤어요.",
            cloud_analysis={"primary_concern": "stress"},
            additional_context={
                "counseling_hint": "작은 실행 단계를 제안하세요.",
                "raw_text": "RAW_DATA_SHOULD_NOT_APPEAR",
                "matched_source_record": "MATCHED_RECORD_SHOULD_NOT_APPEAR",
                "raw_id": "RAW_ID_SHOULD_NOT_APPEAR",
                "filename": "SOURCE_FILE_SHOULD_NOT_APPEAR.jsonl",
                "timestamp": "2026-06-09T00:00:00Z",
                "wellness_risk_stage": "주의",
            },
        )

        combined_prompt = prompt.full_prompt
        assert "작은 실행 단계를 제안하세요." in combined_prompt
        assert "RAW_DATA_SHOULD_NOT_APPEAR" not in combined_prompt
        assert "MATCHED_RECORD_SHOULD_NOT_APPEAR" not in combined_prompt
        assert "RAW_ID_SHOULD_NOT_APPEAR" not in combined_prompt
        assert "SOURCE_FILE_SHOULD_NOT_APPEAR.jsonl" not in combined_prompt
        assert "2026-06-09T00:00:00Z" not in combined_prompt
        assert "wellness_risk_stage" not in combined_prompt

    def test_memory_context_optional_preserves_existing_calls(self, generator):
        """Test prompt generation still works without memory context."""
        cloud_prompt = generator.gen_cloud_prompt(
            sanitized_input="Hello",
            rag_context="Context",
            history=[]
        )
        local_prompt = generator.gen_local_prompt(
            user_input="Hello",
            cloud_analysis={"primary_concern": "general support"},
            rag_context="Context",
            history=[]
        )

        assert isinstance(cloud_prompt, CloudPrompt)
        assert isinstance(local_prompt, LocalPrompt)
        assert "[Memory - Structured Context]" not in cloud_prompt.user_message
        assert "[Memory - Structured Context]" not in local_prompt.user_message
        assert cloud_prompt.metadata["has_memory_context"] is False
        assert local_prompt.metadata["has_memory_context"] is False
        assert local_prompt.metadata["has_dataset_hints"] is False
        assert local_prompt.metadata["dataset_hint_keys"] == []

    def test_memory_context_included_in_cloud_and_local_prompts(self, generator):
        """Test structured memory layers are inserted into prompts."""
        memory_context = MemoryContext(
            recent_summaries=[
                RecentMemoryEntry(
                    session_id="session-1",
                    summary="User reported recurring work stress.",
                    key_topics=["work", "sleep"],
                    emotional_themes=["anxiety"],
                    risk_stage="주의",
                )
            ],
            facts=[
                FactMemoryEntry(
                    fact_id="fact-1",
                    session_id="session-1",
                    category="preference",
                    label="communication_style",
                    normalized_value="short concrete steps",
                    confidence=0.9,
                    evidence_count=2,
                    first_seen_at="2026-06-08T00:00:00Z",
                    last_seen_at="2026-06-08T00:00:00Z",
                )
            ],
            directives=[
                UserDirective(
                    directive_id="directive-1",
                    session_id="session-1",
                    kind="preference",
                    term="answer in Korean",
                    active=True,
                )
            ],
            emotional_trend=[
                EmotionalStateEntry(
                    session_id="session-1",
                    label="anxiety",
                    intensity=0.75,
                    confidence=0.8,
                    source="structured_memory",
                    risk_stage="주의",
                )
            ],
        )

        cloud_prompt = generator.gen_cloud_prompt(
            sanitized_input="I am stressed again",
            memory_context=memory_context,
        )
        local_prompt = generator.gen_local_prompt(
            user_input="I am stressed again",
            cloud_analysis={"primary_concern": "work stress"},
            memory_context=memory_context,
        )

        for prompt_text in (cloud_prompt.user_message, local_prompt.user_message):
            assert "[Memory - Structured Context]" in prompt_text
            assert "[Recent Summaries]" in prompt_text
            assert "User reported recurring work stress." in prompt_text
            assert "[Facts]" in prompt_text
            assert "communication_style=short concrete steps" in prompt_text
            assert "confidence: 0.90" in prompt_text
            assert "evidence_count: 2" in prompt_text
            assert "[User Directives]" in prompt_text
            assert "answer in Korean" in prompt_text
            assert "[Emotional Trend - Observed, Not Diagnostic]" in prompt_text
            assert "observed anxiety trend" in prompt_text

        assert cloud_prompt.metadata["has_memory_context"] is True
        assert local_prompt.metadata["has_memory_context"] is True

    def test_inactive_directives_are_excluded_from_memory_context(self, generator):
        """Test inactive directives are not rendered into memory prompts."""
        memory_context = MemoryContext(
            directives=[
                UserDirective(
                    directive_id="directive-active",
                    session_id="session-1",
                    kind="preference",
                    term="use concise replies",
                    active=True,
                ),
                UserDirective(
                    directive_id="directive-inactive",
                    session_id="session-1",
                    kind="boundary",
                    term="avoid grounding exercises",
                    active=False,
                ),
            ]
        )

        prompt = generator.gen_local_prompt(
            user_input="Can we talk?",
            cloud_analysis={"primary_concern": "support"},
            memory_context=memory_context,
        )

        assert "use concise replies" in prompt.user_message
        assert "avoid grounding exercises" not in prompt.user_message

    def test_gen_crisis_prompt(self, generator):
        """Test crisis prompt generation."""
        prompt = generator.gen_crisis_prompt(
            user_input="I want to hurt myself",
            risk_level="high",
            matched_pattern="self_harm"
        )

        assert isinstance(prompt, LocalPrompt)
        assert "988" in prompt.system_message  # Crisis line
        assert "high" in prompt.user_message

    def test_cloud_prompt_to_messages(self, generator):
        """Test CloudPrompt.to_messages()."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test message"
        )

        messages = prompt.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_metadata(self, generator):
        """Test prompt metadata."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test",
            rag_context="Context",
            history=[{"role": "user", "content": "Hi"}]
        )

        assert "template" in prompt.metadata
        assert "input_length" in prompt.metadata
        assert prompt.metadata["has_rag_context"] is True

    def test_history_formatting(self, generator):
        """Test conversation history formatting."""
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"}
        ]

        prompt = generator.gen_cloud_prompt(
            sanitized_input="Message 3",
            history=history
        )

        assert "Message 1" in prompt.user_message
        assert "Response 1" in prompt.user_message

    def test_history_truncation(self, generator):
        """Test that long history is truncated for local prompt."""
        # Cloud prompt uses 10 turns, local uses 3 turns
        # Test local prompt truncation
        config = PromptConfig(max_history_turns=2)
        generator = PromptGenerator(config=config)

        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]

        prompt = generator.gen_local_prompt(
            user_input="Test",
            cloud_analysis={"primary_concern": "test"},
            history=history
        )

        # Should only include last few messages (3 turns = 6 messages for local)
        assert "Message 0" not in prompt.user_message

    def test_rag_context_truncation(self, generator):
        """Test that long RAG context is truncated."""
        config = PromptConfig(max_rag_context_length=100)
        generator = PromptGenerator(config=config)

        long_context = "A" * 500

        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test",
            rag_context=long_context
        )

        # RAG context is now in user_message for supervisor prompt
        # Should be truncated with ellipsis
        assert "..." in prompt.user_message

    def test_available_templates(self, generator):
        """Test getting available templates."""
        templates = generator.get_available_templates()

        assert "cloud_analysis" in templates
        assert "local_generation" in templates
        assert len(templates) >= 4


class TestPromptConfig:
    """Tests for PromptConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()

        assert config.max_history_turns == 5
        assert config.max_rag_context_length == 1500
        assert config.therapeutic_approach == "integrative"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PromptConfig(
            max_history_turns=10,
            therapeutic_approach="cbt"
        )

        assert config.max_history_turns == 10
        assert config.therapeutic_approach == "cbt"


class TestPromptIntegration:
    """Integration tests for prompt generation."""

    def test_full_pipeline_prompts(self):
        """Test generating prompts for full pipeline."""
        generator = PromptGenerator()

        # Step 1: Cloud analysis prompt
        cloud_prompt = generator.gen_cloud_prompt(
            sanitized_input="I've been feeling really anxious about work",
            rag_context="CBT techniques: cognitive restructuring, exposure therapy",
            history=[]
        )

        # Simulate cloud analysis result (new dict format)
        cloud_analysis = {
            "risk_level": "low",
            "primary_concern": "work-related anxiety",
            "suggested_technique": "CBT cognitive restructuring",
            "guidance_for_local_model": "Validate feelings of work stress and explore triggers",
            "risk_reasoning": "No crisis indicators",
            "updated_user_profile": {}
        }

        # Step 2: Local generation prompt
        local_prompt = generator.gen_local_prompt(
            user_input="I've been feeling really anxious about work",
            cloud_analysis=cloud_analysis,
            rag_context="CBT techniques: cognitive restructuring",
            history=[]
        )

        assert "anxious" in cloud_prompt.user_message
        # Analysis info is in system_message as Supervisor Guidance
        assert "work-related anxiety" in local_prompt.system_message
        assert "CBT" in local_prompt.system_message
        # Verify messages list is correctly built
        assert len(local_prompt.to_messages()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
