# Raw Dataset Analysis and Processing Design

## Scope

This document summarizes the current raw data under `data/` and proposes a processed JSONL layout that can be used by the existing counseling, empathy, and wellness loaders.

No source data files were modified during this analysis.

## 1. Data Directory Structure

Current high-level structure:

```text
data/
  baseline/
    responses_augmented.jsonl
  counseling/
    3.개방데이터/1.데이터/
      Training/01.원천데이터/*.zip
      Training/02.라벨링데이터/*.zip
      Validation/01.원천데이터/*.zip
      Validation/02.라벨링데이터/*.zip
  empathy/
    01-1.정식개방데이터/
      Training/01.원천데이터/*.zip
      Training/02.라벨링데이터/*.zip
      Validation/01.원천데이터/*.zip
      Validation/02.라벨링데이터/*.zip
  wellness/
    02)웰니스_대화_스크립트_데이터셋.xlsx
    웰니스_대화_스크립트_데이터셋.xlsx
  raw/
  processed/
  crisis/
  knowledge/
  safety/
```

`data/raw` and `data/processed` currently do not contain the new usable dataset files expected by the project loaders.

## 2. Dataset Classification

| Path | Dataset type | Reason |
| --- | --- | --- |
| `data/counseling/...` | Counseling dataset | Contains counselor/client utterances, diagnostic classes, and intervention labels. |
| `data/empathy/...` | Empathy dialogue dataset | Contains emotion labels and listener empathy behavior labels. |
| `data/wellness/*.xlsx` | Wellness dataset | Contains user wellness utterances, chatbot responses, clinical keywords, and positive/negative response scripts. |

## 3. File Formats

| Dataset | Container format | Inner format |
| --- | --- | --- |
| Counseling | 150 ZIP files | 1501 TXT files and 1501 JSON files |
| Empathy | 168 ZIP files | 28638 TSV files and 28638 JSON files |
| Wellness | 2 XLSX files | Excel worksheets |

Wellness workbook structure:

| File | Sheets | Key columns |
| --- | --- | --- |
| `02)웰니스_대화_스크립트_데이터셋.xlsx` | `사용자 발화`, `작업 통계` | `핵심증상`, `intent`, `keyword(임상키워드)`, `utterance`, `utterance(2차)`, `response(공감)`, `임상질문그룹(연세의료원제공)`, `utterance(긍정)`, `utterance(부정)`, `긍정에 대한 챗봇 답변`, `부정에 대한 챗봇 답변` |
| `웰니스_대화_스크립트_데이터셋.xlsx` | `Sheet1` | `구분`, `유저`, `챗봇` |

## 4. Source Schema Summary

### Counseling Label JSON

Representative structure:

```json
{
  "filename": "...",
  "id": "...",
  "age": 48,
  "gender": "남",
  "depression": 0,
  "anxiety": 0,
  "addiction": 0,
  "class": "DEPRESSION",
  "summary": "...",
  "silence": 0.0,
  "total_time": 0,
  "paragraph": [
    {
      "start_point": 0,
      "end_point": 0,
      "paragraph_speaker": "상담사",
      "paragraph_text": "...",
      "depressive_mood": 0,
      "worthlessness": 0,
      "suicidal": 0,
      "acceptance_change": 0,
      "sympathy_support": 0,
      "clarification_reflection": 0,
      "cognitive_restructuring": 0,
      "information_provision": 0,
      "goal_setting": 0,
      "behavioral_intervention": 0,
      "task_assignment": 0,
      "training_of_coping_skills": 0
    }
  ]
}
```

Observed speaker labels include `상담사` and `내담자`.

Observed diagnostic class examples include:

- `DEPRESSION`
- `ANXIETY`
- `ADDICTION`
- general/normal group files inferred from filenames such as `일반군`

### Empathy Source TSV

Representative TSV header:

```text
id, utterance_id, utterance_type, utterance_text, terminate, regDate, updDate
```

### Empathy Label JSON

Representative structure:

```json
{
  "info": {
    "category": "...",
    "evaluation": {
      "avg_rating": 5.0,
      "grade": "우수"
    },
    "id": "...",
    "listener_behavior": ["동조", "조언", "위로", "격려"],
    "name": "...",
    "relation": "...",
    "situation": "...",
    "speaker_emotion": "기쁨",
    "speaker_relation": "...",
    "votes": [
      {
        "rating": null,
        "voter_id": null
      }
    ]
  },
  "utterances": [
    {
      "listener_empathy": null,
      "role": "speaker",
      "speaker_changeEmotion": null,
      "terminate": false,
      "text": "...",
      "utterance_id": "..."
    }
  ]
}
```

Observed emotion labels match the project allow-list:

- `기쁨`
- `당황`
- `분노`
- `불안`
- `상처`
- `슬픔`

Observed listener behavior labels match the project allow-list:

- `동조`
- `조언`
- `위로`
- `격려`

## 5. Privacy And Raw Text Risk

### Counseling

High-risk fields:

- `paragraph[].paragraph_text`: direct counseling transcript text.
- `summary`: can contain compressed personal context.
- `id`, `filename`: can link records back to source files.
- `age`, `gender`: low risk alone, but higher risk when combined with transcript and symptoms.
- `total_time`, `silence`: session metadata, not required for current retrieval.

### Empathy

High-risk fields:

- `utterances[].text`: direct dialogue text.
- TSV `utterance_text`: direct dialogue text.
- `info.situation`: may describe personal situations.
- `relation`, `speaker_relation`: relationship context.
- `regDate`, `updDate`: timestamp metadata, not needed for modeling.
- `votes[].voter_id`: evaluator identifier risk.

### Wellness

High-risk fields:

- `utterance`, `utterance(2차)`, `유저`: direct user utterances.
- `response(공감)`, `챗봇`, positive/negative chatbot answer columns: response text.
- `핵심증상`, `intent`, `keyword(임상키워드)`: sensitive mental-health metadata.

### Privacy Recommendation

Processed datasets should remove or transform:

- source filenames
- original IDs
- exact timestamps
- evaluator IDs
- raw demographic fields unless needed as coarse metadata
- full session transcripts

The processed records should keep only the minimal text span needed for retrieval or recommendation, after redaction.

## 6. Current Loader/Recommender Compatibility

The raw files cannot be read directly by the current project loaders.

### Counseling

`src/counseling/dataset_loader.py` reads only:

- `data/processed/counseling_processed.jsonl`
- `data/raw/counseling_sample.jsonl`

Supported formats are JSONL and JSON. It does not read ZIP, TXT, or nested source directories.

Required normalized fields:

- `id`
- `user_input`
- `counselor_response`
- `category`
- `intervention_hint`

### Empathy

`src/empathy/dataset_loader.py` reads only:

- `data/processed/empathy_processed.jsonl`
- `data/raw/empathy_sample.jsonl`

Supported formats are JSONL and JSON. It does not read ZIP or TSV.

Required normalized fields:

- `id`
- `user_input`
- `emotion_label`
- `empathy_label`
- `empathy_style_hint`

### Wellness

`src/wellness/dataset_loader.py` reads only:

- `data/processed/wellness_processed.jsonl`
- `data/raw/wellness_sample.jsonl`

Supported raw file formats are JSON, JSONL, and CSV. XLSX is not supported.

Required normalized fields:

- `id`
- `question`
- `answer`
- `topic`
- `risk_stage`
- `support_hint`
- numeric wellness fields used by the recommender:
  - `mood_score`
  - `anxiety_score`
  - `loneliness_score`
  - `sleep_quality`
  - `meal_status`
  - `energy_score`
  - `stress_score`

## 7. Proposed Processed Dataset Layout

Recommended output files:

```text
data/processed/
  counseling_processed.jsonl
  empathy_processed.jsonl
  wellness_processed.jsonl
```

These names match the existing default loader paths.

### `counseling_processed.jsonl`

Recommended schema:

```json
{
  "id": "counseling_000001",
  "user_input": "내담자 발화 일부 또는 정제된 문장",
  "counselor_response": "상담사 응답 일부 또는 정제된 문장",
  "category": "DEPRESSION",
  "intervention_hint": "감정을 반영하고 사고를 정리하도록 돕는 질문을 사용하세요.",
  "source": "counseling",
  "privacy_flags": ["redacted", "no_dates", "no_raw_id"]
}
```

Recommended mapping:

| Source | Processed field |
| --- | --- |
| `paragraph[].paragraph_speaker == "내담자"` | `user_input` |
| following or nearby `paragraph_speaker == "상담사"` | `counselor_response` |
| top-level `class` or filename group | `category` |
| intervention label fields such as `sympathy_support`, `clarification_reflection`, `cognitive_restructuring`, `goal_setting`, `behavioral_intervention` | `intervention_hint` |

Suggested intervention hint mapping:

| Positive source label | Hint fragment |
| --- | --- |
| `sympathy_support` | `감정을 먼저 인정하고 지지하세요.` |
| `clarification_reflection` | `내담자의 표현을 반영하고 핵심 감정을 명료화하세요.` |
| `cognitive_restructuring` | `자동사고를 점검하고 대안적 해석을 함께 탐색하세요.` |
| `information_provision` | `필요한 정보를 짧고 부담 없이 제공하세요.` |
| `goal_setting` | `작고 구체적인 다음 행동을 정하세요.` |
| `behavioral_intervention` | `실행 가능한 행동 실험이나 루틴을 제안하세요.` |
| `training_of_coping_skills` | `호흡, 기록, 거리두기 같은 대처 기술을 안내하세요.` |

### `empathy_processed.jsonl`

Recommended schema:

```json
{
  "id": "empathy_000001",
  "user_input": "speaker 발화 일부 또는 정제된 문장",
  "emotion_label": "불안",
  "empathy_label": "위로",
  "empathy_style_hint": "불안을 가볍게 여기지 않고 차분하게 인정한 뒤 위로하세요.",
  "source": "empathy",
  "privacy_flags": ["redacted", "no_dates", "no_voter_id"]
}
```

Recommended mapping:

| Source | Processed field |
| --- | --- |
| `utterances[].role == "speaker"` or TSV user-side utterance | `user_input` |
| `info.speaker_emotion` | `emotion_label` |
| `utterances[].listener_empathy` or `info.listener_behavior` | `empathy_label` |
| emotion + empathy label template | `empathy_style_hint` |

If multiple empathy labels exist for one listener turn, choose a primary label in this order:

1. `위로`
2. `격려`
3. `조언`
4. `동조`

This order favors emotionally supportive behavior for the current psychologist-agent use case.

### `wellness_processed.jsonl`

Recommended schema:

```json
{
  "id": "wellness_000001",
  "question": "사용자 웰니스 발화",
  "answer": "공감 또는 지원 응답",
  "topic": "우울감",
  "risk_stage": "주의",
  "support_hint": "지금의 감정을 인정하고 오늘 할 수 있는 작은 행동 하나를 제안하세요.",
  "mood_score": 4,
  "anxiety_score": 6,
  "loneliness_score": 5,
  "sleep_quality": 5,
  "meal_status": 5,
  "energy_score": 4,
  "stress_score": 6,
  "source": "wellness",
  "privacy_flags": ["redacted"]
}
```

Recommended mapping for `02)웰니스_대화_스크립트_데이터셋.xlsx`:

| Source column | Processed field |
| --- | --- |
| `utterance` or `utterance(2차)` | `question` |
| `response(공감)` | `answer` and default `support_hint` |
| `핵심증상` or `intent` | `topic` |
| `임상질문그룹(연세의료원제공)` | optional metadata or omit |
| `utterance(긍정)`, `utterance(부정)` | optional extra records only if paired with answer columns |
| `긍정에 대한 챗봇 답변`, `부정에 대한 챗봇 답변` | optional `answer` for positive/negative branch records |

Recommended mapping for `웰니스_대화_스크립트_데이터셋.xlsx`:

| Source column | Processed field |
| --- | --- |
| `구분` | `topic` |
| `유저` | `question` |
| `챗봇` | `answer` and default `support_hint` |

Because the wellness source does not provide numeric check-in scores, derive conservative scores from `topic` and `risk_stage`, or assign neutral defaults when uncertain.

Suggested score defaults:

| Risk stage | mood | anxiety | loneliness | sleep | meal | energy | stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `관심` | 7 | 3 | 3 | 7 | 7 | 6 | 3 |
| `주의` | 4 | 6 | 5 | 4 | 5 | 4 | 6 |
| `위험` | 2 | 8 | 7 | 2 | 3 | 2 | 8 |

Topic-specific overrides:

| Topic marker | Override |
| --- | --- |
| sleep / 수면 / 불면 | lower `sleep_quality`, raise `stress_score` |
| anxiety / 불안 / 걱정 | raise `anxiety_score` and `stress_score` |
| loneliness / 외로움 | raise `loneliness_score` |
| mood / 우울 / 슬픔 | lower `mood_score` and `energy_score` |
| stress / 업무 / 직장 | raise `stress_score` |

## 8. Processing Order

Recommended processing order:

1. Extract source files in memory or temporary workspace only.
2. Parse label files first because they contain structured labels.
3. Pair user and counselor/listener turns into small records.
4. Remove source IDs, filenames, timestamps, evaluator IDs, and unnecessary demographics.
5. Apply PII redaction to all text fields before writing processed JSONL.
6. Validate that processed JSONL can be loaded by:
   - `CounselingDatasetLoader`
   - `EmpathyDatasetLoader`
   - `WellnessDatasetLoader`
7. Keep raw data out of training and runtime paths unless explicitly needed for auditing.

## 9. Compatibility Summary

The raw data is useful but needs conversion.

Minimum required processed outputs:

- `data/processed/counseling_processed.jsonl`
- `data/processed/empathy_processed.jsonl`
- `data/processed/wellness_processed.jsonl`

After those files exist with the proposed schemas, the existing loader and recommender code can consume the datasets without changing runtime paths.
