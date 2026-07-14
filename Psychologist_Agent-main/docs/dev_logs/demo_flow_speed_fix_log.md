# Demo Flow and Speed Fix Log

## Scope

Focused on demo UX flow and response-start latency without adding new Agent features.

## Changes

- Kept `PsychologistAgent` caching through `get_agent()`.
- Reduced MOCK-mode dataset retrieval overhead in `src/main.py` by using lightweight deterministic counseling, empathy, and wellness recommendations instead of loading/scanning local datasets for every demo response.
- Preserved `pipeline_details["timing"]` for debug inspection only.
- Added an assistant-role initial greeting in Chatbot messages format.
- Made anonymous mode user-toggleable and added a nickname input plus record-save consent checkbox.
- Added chat action buttons for emotion diary, report, and expert counseling guidance.
- Improved report labels:
  - `SLEEP_PROBLEM` -> `수면 문제`
  - `ANXIETY_SUPPORT` -> `불안`
  - duplicate intent labels are removed.
- Softened expert guidance for `관심` and `주의`; emergency wording is reserved for `위험`.
- Kept Agent Pipeline Details in a collapsed accordion and separate from user chat.

## Safety and Privacy

- Safety/crisis flow was not changed.
- Internal hints and raw text remain hidden from the user-facing UI.
- Emotion diary stores only structured values in `gr.State`.

