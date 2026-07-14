# Wellness Dataset Format

This project uses a canonical wellness counseling format that can be normalized by the existing preprocessing pipeline in `scripts/data_preparation.py`.

## Canonical Raw Fields

| Field            | Type          | Required | Description                                                                        |
| ---------------- | ------------- | -------: | ---------------------------------------------------------------------------------- |
| `id`             | string        |       No | Source record identifier. If omitted, preprocessing generates a new ID.            |
| `questionText`   | string        |      Yes | User-facing wellness question, concern, or prompt.                                 |
| `answerText`     | string        |      Yes | Counselor-style supportive response.                                               |
| `topic`          | string        |       No | High-level topic label such as `anxiety`, `sleep`, `work_stress`, `relationships`. |
| `language`       | string        |       No | Language code, for example `ko` or `en`.                                           |
| `wellness_stage` | string        |       No | Internal label such as `관심`, `주의`, `위험`.                                     |
| `source`         | string        |       No | Dataset source name or provenance.                                                 |
| `privacy_flags`  | array[string] |       No | Privacy annotations such as `no_pii` or `masked`.                                  |
| `metadata`       | object        |       No | Extra source-specific metadata.                                                    |

## Accepted Alias Fields

The current preprocessing code also accepts these aliases and maps them into the canonical output:

| Canonical Output | Accepted Raw Aliases                          |
| ---------------- | --------------------------------------------- |
| `question`       | `questionText`, `question`, `prompt`, `input` |
| `answer`         | `answerText`, `answer`, `response`, `output`  |
| `topic`          | `topic`, `category`, `label`                  |

This means a future dataset can keep its existing export format as long as one of the accepted aliases is present.

## Standard Output Format

After preprocessing, every record is written as JSONL with these fields:

```json
{
  "id": "counsel_00001",
  "question": "User's question text",
  "answer": "Counselor's response text",
  "topic": "Topic category"
}
```

## Sample Dataset

The repository includes a small synthetic sample dataset at:

- `data/raw/wellness_sample.jsonl`

It is designed to be low-risk, privacy-safe, and compatible with the existing preprocessing pipeline.

## Usage

Run preprocessing against any raw folder that contains JSON, JSONL, or CSV files using the same field aliases:

```bash
python scripts/data_preparation.py --raw-dir data/raw --output-dir data/processed
```

The script will load local raw files first and only fall back to HuggingFace if the raw folder is empty.
