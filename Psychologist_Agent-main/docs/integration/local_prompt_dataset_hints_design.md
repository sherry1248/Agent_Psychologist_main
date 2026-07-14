# Local Prompt Dataset Hint Integration Design

## Goal

Reflect processed dataset hints in the non-mock local generation prompt:

- `counseling_hint`
- `empathy_style_hint`
- `wellness_hint`

The prompt must use only normalized/processed hints, never raw dataset source text. Crisis and safety flows must continue to return safety responses before dataset hints can affect generation.

## Current Structure

### `src/main.py`

`PsychologistAgent.process_message()` initializes result fields for all three hints:

- `counseling_hint`
- `empathy_style_hint`
- `wellness_hint`

The first safety gateway check can return immediately when `safety_result.is_safe` is false. The early risk audit before dataset hint retrieval can also return immediately when `risk_assessment.requires_crisis_response` is true. This means immediate crisis flow currently takes priority before counseling, empathy, or wellness hints are retrieved.

After the early safety/risk gates, main retrieves:

- `self.counseling_retriever.recommend(user_input)`
- `self.empathy_retriever.recommend(user_input)`
- `self._get_wellness_recommendation(wellness_checkin)`

It stores processed hints in both top-level result fields and `pipeline_details`.

In mock mode, `_compose_mock_response()` receives all three hints:

```python
self._compose_mock_response(
    counseling_recommendation.intervention_hint,
    empathy_recommendation.empathy_style_hint,
    wellness_recommendation.support_hint if wellness_recommendation else "",
)
```

In non-mock mode, the local prompt call currently passes only wellness-derived guidance into `gen_local_prompt()`:

```python
therapeutic_guidance=wellness_recommendation.support_hint if wellness_recommendation else "",
additional_context={
    "wellness_support_hint": wellness_recommendation.support_hint if wellness_recommendation else "",
    "wellness_risk_stage": wellness_recommendation.risk_stage if wellness_recommendation else "",
} if wellness_recommendation else None,
```

The later risk audit after cloud analysis can still return a crisis handler response before local generation. That keeps the cloud-informed crisis path ahead of the local prompt.

### `src/prompt/generator.py`

`PromptGenerator.gen_local_prompt()` already accepts:

```python
therapeutic_guidance: str = "",
additional_context: Optional[Dict[str, Any]] = None,
```

However, the current implementation does not consume `additional_context` for local generation. It formats the local template with fixed variables only.

For dict `cloud_analysis`, the method sets:

```python
therapeutic_guidance=analysis_dict.get("guidance_for_local_model") or self._get_default_guidance()
```

This means the explicit `therapeutic_guidance` argument is ignored whenever `cloud_analysis` is a dict, which is the non-mock path in `src/main.py`. As a result, even the current wellness hint may not actually appear in the formatted local prompt through `therapeutic_guidance`.

The prompt templates in `prompts/prompt_templates.yaml` and `DEFAULT_TEMPLATES` both include `{therapeutic_guidance}` in the local system message, but neither includes a dedicated dataset-hints section yet.

## Safest Design

Use the existing `additional_context` parameter in `PromptGenerator.gen_local_prompt()` and add a small internal formatter for processed dataset hints. This preserves the public method signature and does not break existing callers.

### 1. Main passes all processed hints through `additional_context`

In `src/main.py`, build an `additional_context` dict before calling `gen_local_prompt()`:

```python
local_additional_context = {
    "counseling_hint": counseling_recommendation.intervention_hint,
    "empathy_style_hint": empathy_recommendation.empathy_style_hint,
    "wellness_hint": wellness_recommendation.support_hint if wellness_recommendation else "",
    "wellness_risk_stage": wellness_recommendation.risk_stage if wellness_recommendation else "",
}
```

Then pass it to `gen_local_prompt(additional_context=local_additional_context, ...)`.

Keep `therapeutic_guidance` behavior unchanged for now, or pass an empty/default value. The new dataset hint section should not depend on `therapeutic_guidance`, because current dict analysis logic overrides it.

### 2. Generator formats only allowlisted processed hints

In `src/prompt/generator.py`, add a private helper, for example:

```python
def _format_dataset_hints(self, additional_context: Optional[Dict[str, Any]]) -> str:
    ...
```

The helper should only read allowlisted keys:

- `counseling_hint`
- `empathy_style_hint`
- `wellness_hint`

It should ignore every other key for prompt text. This prevents accidental raw dataset fields from entering the prompt if future callers add broader metadata to `additional_context`.

Suggested output format:

```text
[Processed Dataset Hints]
- Counseling intervention: ...
- Empathy style: ...
- Wellness support: ...
```

If all values are empty, return an empty string.

### 3. Inject the section as a new template variable

Add a local variable in `gen_local_prompt()`:

```python
dataset_hints = self._format_dataset_hints(additional_context)
```

Pass it into `template.format(...)`:

```python
dataset_hints=dataset_hints,
```

Update the local generation template to include:

```text
Processed dataset hints:
{dataset_hints}
```

Best placement is in the system message after `Therapeutic guidance`, because these are response-shaping instructions rather than user content. The local model can then use them as style/intervention constraints without treating them as user-provided facts.

### 4. Backward-compatible template handling

There is one compatibility risk: custom templates loaded from YAML may not contain `{dataset_hints}`. Passing extra kwargs to `PromptTemplate.format()` is safe for Python string formatting. Therefore adding `dataset_hints=...` will not break templates that do not reference it.

Templates that do reference `{dataset_hints}` will work once the YAML/default template is updated. Existing callers that omit `additional_context` will get an empty dataset-hints section.

### 5. Metadata for traceability

Add non-sensitive metadata flags to the returned `LocalPrompt.metadata`:

```python
"has_dataset_hints": bool(dataset_hints),
"dataset_hint_keys": [...]
```

Do not store raw hint text in prompt metadata unless already exposed elsewhere. The result already exposes top-level processed hints and `pipeline_details`, so prompt metadata only needs presence/keys for debugging.

## Crisis Priority

No crisis behavior needs to move.

The design keeps hint retrieval and prompt generation after:

1. Safety gateway immediate return.
2. Early risk audit immediate return.
3. Cloud-informed risk audit immediate return.

Since dataset hints are only injected at Step 7 local generation, any crisis path that returns before Step 7 continues to bypass dataset hints completely. If local generation later receives non-crisis risk guidance, the prompt's existing safety and expert referral instructions still remain in the template.

## Raw Data Safety

Only these normalized recommendation fields should enter the local prompt:

- `CounselingRecommendation.intervention_hint`
- `EmpathyRecommendation.empathy_style_hint`
- `WellnessRecommendation.support_hint`

Do not pass source records, matched examples, raw dialogue, text previews, or unfiltered `to_dict()` payloads into the prompt section.

The formatter should explicitly ignore unknown keys in `additional_context` rather than rendering the whole dict.

## Recommended Implementation Order

1. Add `_format_dataset_hints()` to `PromptGenerator`.
2. Call it inside `gen_local_prompt()` using `additional_context`.
3. Pass `dataset_hints` into `template.format()`.
4. Add `{dataset_hints}` to the YAML and default local templates.
5. Update `src/main.py` to pass all three processed hint strings in `additional_context`.
6. Add focused tests for:
   - all three hints appear in `LocalPrompt.system_message` or `full_prompt`;
   - unknown/raw keys in `additional_context` do not appear;
   - calling `gen_local_prompt()` without `additional_context` still works;
   - crisis returns happen before local prompt generation in `process_message()`.

## Alternative Considered

Appending all dataset hints into `therapeutic_guidance` from `src/main.py` would be smaller, but it is less safe:

- `gen_local_prompt()` currently ignores `therapeutic_guidance` when `cloud_analysis` is a dict and `guidance_for_local_model` exists.
- It mixes cloud guidance, wellness support, counseling intervention, and empathy style into one undifferentiated string.
- It does not create an allowlist boundary against accidental raw fields.

Using `additional_context` with an allowlisted formatter is therefore the safer API-compatible path.
