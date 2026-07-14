# PromptGenerator MemoryContext Implementation Log

## Changed Files

- `src/prompt/generator.py`
- `prompts/prompt_templates.yaml`
- `tests/test_prompt_generator.py`

## Implementation

- Added optional `memory_context=None` argument to `gen_cloud_prompt()`.
- Added optional `memory_context=None` argument to `gen_local_prompt()`.
- Added `PromptGenerator._format_memory_context(memory_context)` helper.
- Preserved existing prompt generation behavior when `memory_context` is `None` or empty.
- Inserted structured memory only when memory context has content.
- Rendered recent summaries, facts, active user directives, and emotional trend into a structured prompt section.
- Excluded inactive user directives from prompt output.
- Included `confidence` and `evidence_count` for facts.
- Labeled emotional trend as observed and non-diagnostic.
- Avoided raw conversation text in the structured memory section.

## Template Updates

- Added `{memory_context}` to `cloud_analysis` user template after long-term profile.
- Added `{memory_context}` to `local_generation` user template after current user message.
- Added `memory_context` to both template variable lists.

## Tests Added

- Existing cloud/local prompt calls work without `memory_context`.
- Structured memory is included when `MemoryContext` has content.
- Inactive directives are excluded from prompt output.

## Guardrails Checked

- `src/main.py` was not modified.
- `src/memory/store.py` was not modified.
- `requirements.txt` was not modified.
- No external package was added.

## Verification

- `python -m py_compile src/prompt/generator.py tests/test_prompt_generator.py` passed.
- `pytest tests/test_prompt_generator.py -q` could not complete because the current virtualenv is missing `PyYAML`, which is imported by the existing `src/prompt/templates.py`.
