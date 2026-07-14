# Technical Rationale: Data Processing and Agent Design

This project is intentionally designed as a safety-first counseling assistant, not a simple chatbot that sends a user message directly to a language model. The technical choices below create auditable boundaries around data quality, privacy, category balance, risk handling, and response behavior.

## Why Raw Counseling Data Is Not Used Directly

Raw counseling datasets often contain long conversational turns, inconsistent formatting, sensitive context, and speaker-specific details. Using that data directly would increase privacy risk, leak source-specific wording into responses, and make model behavior harder to test.

The project converts processed records into compact derived hints. The derived representation keeps the useful supervision signal while removing long raw counseling text:

- `intent_hint`: what kind of support the user is likely requesting
- `emotion_hint`: which emotional state should be reflected first
- `cause_hint`: which cause candidates should be explored without assuming certainty
- `action_hint`: what kind of low-burden next step is appropriate
- `safety_hint`: whether safety triage must be prioritized

This turns source examples into decision support for the agent pipeline rather than memorized counseling transcripts.

## Why Empty or Invalid Records Are Removed

Empty records, malformed JSON records, records without usable text, and records without a label/category/topic do not provide reliable supervision. Keeping them would distort dataset statistics and could teach the system that vague or content-free inputs are meaningful counseling patterns.

Removing these records makes downstream analysis honest: total record count, usable record count, and removed record count become explicit quality metrics.

## Why Duplicate-Like Records Are Removed

Counseling and wellness datasets often contain repeated phrases, boilerplate responses, near-identical short turns, and duplicated exports. Duplicate-like records can make a category look more important than it is and can make response behavior repetitive.

The analysis script normalizes text by removing whitespace, punctuation, and case differences, then counts duplicate-like records. This is a conservative heuristic: it does not claim semantic deduplication, but it catches the most obvious repeated records without adding external dependencies.

## Why Label, Category, and Topic Distribution Is Analyzed

Mental-health datasets are rarely balanced. Some labels such as depression, anxiety, support, or generic wellness topics can dominate. If the project ignores that distribution, the agent can overfit to common categories and under-serve lower-frequency concerns.

Distribution analysis gives CTO-level evidence for:

- dataset coverage
- overrepresented categories
- rare but important categories
- expected bias in response policy if balancing is not applied
- whether evaluation should include low-frequency topics

## Why Overrepresented Labels Are Downsampled and Low-Frequency Labels Are Preserved

The balancing strategy caps each label with `max_samples_per_label` while keeping labels below the cap intact. This reduces dominance from high-volume categories without deleting rare labels.

This approach is deliberately simple and auditable:

- overrepresented labels are downsampled to reduce response bias
- low-frequency labels are preserved because rare concerns can be clinically important
- output remains deterministic and easy to inspect
- raw and processed datasets are not modified

## Why Raw Text Is Converted Into Safe Summarized Hints

The derived safe hint file is designed for agent guidance, not transcript replay. Long source turns are replaced with short summaries and structured hints. This reduces the chance that the system copies sensitive source text into user-facing responses.

The safe hint fields have separate purposes:

- `intent_hint` supports intent detection for the current user message
- `emotion_hint` supports emotional state interpretation and tone selection
- `cause_hint` supports cause candidate selection before the system asks follow-up questions
- `action_hint` supports small action recommendation without overwhelming the user
- `safety_hint` keeps crisis-risk handling visible in the data artifact
- `short_summary` gives a compact evidence preview without retaining long raw counseling text

## Why Safety/Risk Runs Before Normal Counseling Flow

In a mental-health counseling product, safety triage is not an optional post-processing step. Risk detection must run before normal counseling because the correct response to crisis language is not a normal empathy-follow-up-small-action flow.

The Safety/Risk stage checks for crisis signals, self-harm indicators, and risk patterns first. If a risk condition is detected, the system must prioritize immediate safety guidance and help-seeking resources.

## Why Crisis Flow Overrides Follow-Up and Small-Action Responses

Normal counseling flow often asks reflective follow-up questions or recommends a small action. During a crisis, those behaviors can be inappropriate because they may delay urgent help.

The crisis path overrides normal response generation so the assistant can:

- acknowledge the risk directly
- encourage immediate contact with emergency or crisis resources
- avoid exploratory questions that keep the user in the chat
- avoid casual productivity-style action suggestions
- keep safety instructions concise and prominent

## Why Use an Agent Pipeline Instead of a Single Direct Chatbot Response

A single direct chatbot response is difficult to audit. It hides the difference between risk detection, intent interpretation, emotional state analysis, retrieval, memory, decision policy, and final wording.

The agent pipeline separates these responsibilities:

- Safety Agent: checks crisis and unsafe content before normal counseling
- Intent Agent: classifies the support request and likely user need
- Emotional State Agent: estimates mood, anxiety, stress, loneliness, and related signals
- Cause Exploration Agent: identifies likely cause candidates without assuming certainty
- Memory/RAG components: provide relevant context and safe support knowledge
- Decision Agent: chooses whether to ask, support, stabilize, or recommend a small action
- Response Agent: produces the final user-facing wording under the selected policy

This design is easier to test, debug, and explain. It also allows the UI to show pipeline evidence without exposing raw internal prompt text.

## How Dataset Hints Influence Agent Behavior

The balanced hint artifact is not a replacement for live reasoning. It acts as a compact supervision layer for the pipeline.

`intent_hint` informs whether the system should treat a message as anxiety support, low mood support, sleep concern, relationship stress, practical help, or a general emotional disclosure.

`emotion_hint` informs how the assistant should reflect feelings before offering direction. This helps avoid jumping too quickly into advice.

`cause_hint` informs cause candidate selection. The agent can check whether sleep, workload, relationships, accumulated fatigue, isolation, or uncertainty may be relevant.

`action_hint` informs small action recommendations. The system should suggest one low-burden step rather than a long list of tasks.

`safety_hint` informs the precedence rule: if risk is plausible, safety triage and crisis flow are evaluated before normal counseling behavior.

## How Tests Verify Expected Response Behavior

The test suite is part of the technical evidence. Existing tests cover safety gateway behavior, agent decisions, cause exploration, emotional state handling, small action planning, response quality, pipeline integration, and demo hint visibility.

The dataset balancing test adds coverage for the data side:

- the analyzer runs on temporary JSONL data
- balanced safe hints are written to a derived output path
- overrepresented labels are downsampled
- low-frequency labels are preserved
- the report includes before and after label distribution sections

Together, these tests verify that the project has both data-processing discipline and response-behavior discipline.

## Evidence Artifacts

- `scripts/analyze_and_balance_datasets.py`: stdlib-only analyzer and safe hint generator
- `data/derived/balanced_safe_hints.jsonl`: generated safe hint artifact
- `reports/dataset_processing_report.md`: generated quality, distribution, and balancing report
- `reports/technical_rationale.md`: architectural rationale and CTO review evidence
- `tests/test_dataset_balancing.py`: repeatable test for balancing behavior
