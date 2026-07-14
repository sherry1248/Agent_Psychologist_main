# Processed Dataset Main Flow Trace

## Scope

This document traces whether the generated processed datasets are used by the current `main.py` response flow.

Checked files:

- `data/processed/counseling_processed.jsonl`
- `data/processed/empathy_processed.jsonl`
- `data/processed/wellness_processed.jsonl`

No code or raw data files were modified for this trace.

## Processed File Availability

The processed files exist and contain:

| File | Records |
| --- | ---: |
| `data/processed/counseling_processed.jsonl` | 204,962 |
| `data/processed/empathy_processed.jsonl` | 204,356 |
| `data/processed/wellness_processed.jsonl` | 1,216 |

## Loader Default Path Compatibility

The existing loaders are configured to look for these exact processed paths before falling back to sample files.

### Counseling

`src/counseling/dataset_loader.py` default candidates:

```text
data/processed/counseling_processed.jsonl
data/raw/counseling_sample.jsonl
```

Relevant behavior:

- Loads JSONL.
- Normalizes records into `id`, `user_input`, `counselor_response`, `category`, `intervention_hint`.
- Keeps only records where `user_input` and `intervention_hint` are present.

Result: `counseling_processed.jsonl` is directly compatible.

### Empathy

`src/empathy/dataset_loader.py` default candidates:

```text
data/processed/empathy_processed.jsonl
data/raw/empathy_sample.jsonl
```

Relevant behavior:

- Loads JSONL.
- Validates `emotion_label` against `기쁨`, `당황`, `분노`, `불안`, `상처`, `슬픔`.
- Validates `empathy_label` against `동조`, `조언`, `위로`, `격려`.
- Keeps only records where `user_input`, `emotion_label`, and `empathy_label` are present.

Result: `empathy_processed.jsonl` is directly compatible.

### Wellness

`src/wellness/dataset_loader.py` default candidates:

```text
data/processed/wellness_processed.jsonl
data/raw/wellness_sample.jsonl
```

Relevant behavior:

- Loads JSONL through the existing data preparation helper.
- Normalizes `question`, `answer`, `topic`, `risk_stage`, `support_hint`.
- Coerces or derives numeric wellness fields:
  - `mood_score`
  - `anxiety_score`
  - `loneliness_score`
  - `sleep_quality`
  - `meal_status`
  - `energy_score`
  - `stress_score`

Result: `wellness_processed.jsonl` is directly compatible.

## Main Agent Initialization

`PsychologistAgent.__init__()` creates all three data-backed components by default:

```python
self.counseling_retriever = CounselingRetriever()
self.empathy_retriever = EmpathyRetriever()
self.wellness_recommender = WellnessRecommender()
```

Because no custom dataset path is passed, each component uses its loader's default processed path.

## `process_message()` Flow

For non-crisis input, `src/main.py` calls the retrievers before response generation:

```python
counseling_recommendation = self.counseling_retriever.recommend(user_input)
empathy_recommendation = self.empathy_retriever.recommend(user_input)
result["counseling_hint"] = counseling_recommendation.intervention_hint
result["empathy_style_hint"] = empathy_recommendation.empathy_style_hint

wellness_recommendation = self._get_wellness_recommendation(wellness_checkin)
if wellness_recommendation:
    result["wellness_hint"] = wellness_recommendation.support_hint
    result["pipeline_details"]["wellness"] = wellness_recommendation.to_dict()

result["pipeline_details"]["counseling"] = counseling_recommendation.to_dict()
result["pipeline_details"]["empathy"] = empathy_recommendation.to_dict()
```

This means:

- Counseling processed data affects `result["counseling_hint"]`.
- Empathy processed data affects `result["empathy_style_hint"]`.
- Wellness processed data affects `result["wellness_hint"]` when `wellness_checkin` is provided.
- All three matches are visible in `result["pipeline_details"]`.

## Actual Runtime Trace

Executed a mock-mode `PsychologistAgent.process_message()` call using the project `.venv`.

The test input was:

```text
요즘 너무 불안하고 잠도 잘 못 자서 힘들어요.
```

The wellness check-in was:

```json
{
  "mood_score": 4,
  "anxiety_score": 8,
  "loneliness_score": 5,
  "sleep_quality": 3,
  "meal_status": 5,
  "energy_score": 4,
  "stress_score": 7
}
```

Observed loader logs:

```text
Loaded 204962 counseling records from data/processed/counseling_processed.jsonl
Loaded 204356 empathy records from data/processed/empathy_processed.jsonl
Loaded 1216 wellness records from data/processed/wellness_processed.jsonl
```

Observed `pipeline_details` keys:

```text
counseling
empathy
wellness
```

Observed matches:

```json
{
  "counseling": {
    "matched_record_id": "counseling_100287",
    "category": "ANXIETY",
    "score": 5.0
  },
  "empathy": {
    "emotion_label": "불안",
    "empathy_label": "위로",
    "matched_record_id": "empathy_101418",
    "score": 9.0
  },
  "wellness": {
    "matched_record_id": "wellness_000204",
    "matched_topic": "감정/걱정/불면",
    "risk_stage": "주의",
    "distance": 0.14285714285714285
  }
}
```

Observed hints:

```json
{
  "counseling_hint": "감정을 먼저 확인하고, 작은 실행 단계를 하나만 제안하세요.",
  "empathy_style_hint": "불안을 가볍게 여기지 않고 차분하게 인정하세요. 지금의 고됨을 인정하며 위로하세요.",
  "wellness_hint": "걱정을 하는 시간을 정해놓는 것도 좋은 방법이라고 들었어요."
}
```

Observed mock response included the wellness hint:

```text
걱정을 하는 시간을 정해놓는 것도 좋은 방법이라고 들었어요.
```

## Response Generation Impact

### Mock Mode

In mock mode, `main.py` calls:

```python
response_text = self._compose_mock_response(
    counseling_recommendation.intervention_hint,
    empathy_recommendation.empathy_style_hint,
    wellness_recommendation.support_hint if wellness_recommendation else "",
)
```

Inside `_compose_mock_response()`:

- `empathy_style_hint` changes the second response segment to a validation-style sentence.
- `wellness_hint` is preferred over `counseling_hint` for the action step.
- `counseling_hint` is used as fallback only if no usable wellness hint exists.

Conclusion for mock mode: all three data-backed components are called; empathy and wellness directly shape the response text, and counseling can shape the action step when wellness is absent or unusable.

### Non-Mock / Local Generation Flow

In the non-mock path, `main.py` currently passes wellness guidance into the local prompt:

```python
local_prompt = self.prompt_generator.gen_local_prompt(
    user_input=user_input,
    cloud_analysis=cloud_analysis.to_dict(),
    rag_context=rag_context,
    history=local_history,
    therapeutic_guidance=wellness_recommendation.support_hint if wellness_recommendation else "",
    additional_context={
        "wellness_support_hint": wellness_recommendation.support_hint if wellness_recommendation else "",
        "wellness_risk_stage": wellness_recommendation.risk_stage if wellness_recommendation else "",
    } if wellness_recommendation else None,
    memory_context=memory_context,
)
```

Current implication:

- Wellness processed data is passed into the non-mock local generation prompt through `therapeutic_guidance` and `additional_context`.
- Counseling processed data is loaded and exposed in `result["counseling_hint"]` and `pipeline_details["counseling"]`, but it is not currently passed into `gen_local_prompt()`.
- Empathy processed data is loaded and exposed in `result["empathy_style_hint"]` and `pipeline_details["empathy"]`, but it is not currently passed into `gen_local_prompt()`.

## Crisis Flow Caveat

If Safety Gateway or risk audit requires a crisis response, `process_message()` returns early before dataset retrieval.

Current behavior:

- Crisis/safety response takes priority.
- Counseling, empathy, and wellness recommendations are skipped for immediate crisis cases.

This matches the project policy that crisis handling should override dataset-based recommendations.

## Summary

The processed JSONL files are wired into the default loader paths and are successfully loaded by the actual retriever/recommender instances used by `PsychologistAgent`.

Current end-to-end status:

| Dataset | Loaded by default? | Appears in `pipeline_details`? | Directly affects mock response? | Passed to non-mock local prompt? |
| --- | --- | --- | --- | --- |
| Counseling | Yes | Yes | Fallback action hint | No |
| Empathy | Yes | Yes | Yes, response validation style | No |
| Wellness | Yes, when `wellness_checkin` is provided | Yes | Yes, preferred action hint | Yes |

Main conclusion:

The three processed datasets are readable and active in the `main.py` pipeline. However, in the current non-mock generation path, only wellness guidance is passed into the local model prompt. Counseling and empathy recommendations are computed and returned as metadata, but they are not yet injected into the local generation prompt.
