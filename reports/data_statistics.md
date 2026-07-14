# Data Statistics Report

Generated: 2026-05-31 15:12:31

## Dataset Overview

| Metric | Count |
|--------|-------|
| Original Records | 6 |
| After Empty Removal | 6 |
| After Length Filter | 6 |
| After Deduplication | 6 |

## Data Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 4 | 80% |
| Eval | 0 | 10% |
| Test | 2 | 10% |
| **Total** | 6 | 100% |

## Question Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | 28 chars |
| Maximum | 41 chars |
| Average | 34.2 chars |

## Answer Length Statistics

| Metric | Value |
|--------|-------|
| Minimum | 111 chars |
| Maximum | 148 chars |
| Average | 133.5 chars |

## Topic Distribution

| Topic | Count | Percentage |
|-------|-------|------------|
| work_stress | 1 | 16.7% |
| loneliness | 1 | 16.7% |
| anxiety | 1 | 16.7% |
| relationships | 1 | 16.7% |
| mood | 1 | 16.7% |
| study | 1 | 16.7% |

## Cleaning Rules Applied

1. **Empty Removal**: Removed records with empty question or answer
2. **Length Filter**:
   - Minimum question length: 10 characters
   - Minimum answer length: 50 characters
3. **Deduplication**: Based on question text (case-insensitive)

## Output Files

- `data/processed/counsel_chat_train.jsonl` - Training data
- `data/processed/counsel_chat_eval.jsonl` - Evaluation data
- `data/processed/counsel_chat_test.jsonl` - Test data
- `data/processed/counsel_chat_cleaned.jsonl` - All cleaned data (combined)

## Record Format

```json
{
    "id": "counsel_00001",
    "question": "User's question text",
    "answer": "Counselor's response text",
    "topic": "Topic category"
}
```
