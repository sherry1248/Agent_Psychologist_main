# Cause Exploration Agent Log

## Scope

Added a lightweight rule-based cause exploration helper to improve MOCK counseling continuity without adding external dependencies.

## Changes

- Added `src/agent/cause.py` for structured cause candidates, selected cause, dataset signals, and one exploration question.
- Used counseling category, empathy emotion label, and wellness topic metadata as cause-selection signals.
- In MOCK mode, counseling/empathy/wellness recommendations now try the existing processed-dataset retrievers first and fall back to lightweight defaults only on failure.
- Added `pipeline_details["agents"]["cause_exploration"]` without raw user text or raw dataset text.
- Reworked MOCK response order toward emotion validation, cause exploration, one question, and one small action.
- Avoided repeating the same follow-up by using previous continuity and a more specific sleep-maintenance question after "자주 깨" style replies.

## Safety and Privacy

- Safety/crisis flow was not changed.
- Dataset hint text is used only as metadata context and is not appended verbatim to final responses.
- Raw user input is not stored in cause exploration pipeline details.
