# Demo UX Flow Refinement Log

## Scope

Refined the Gradio demo UX flow without adding Agent features.

## Changes

- Kept the default screen centered on the 상담 채팅 experience.
- Fixed the chat window height so previous messages can be reviewed by scrolling.
- Preserved assistant-role first greeting as an initial chat bubble.
- Made status check optional through the `상태 체크하기` next-step button and collapsed section.
- Kept emotion scores inside the 감정일기 flow.
- Added nickname UX behavior:
  - anonymous mode on: nickname input disabled
  - anonymous mode off: nickname input enabled
- Reframed post-chat actions as next steps:
  - 상태 체크하기
  - 감정일기 쓰기
  - 마음정리 보고서 보기
  - 전문가 상담 연결
- Kept next-step buttons hidden until a counseling response is ready.
- Revealed the optional status check panel only from the `상태 체크하기` button.
- Kept Agent Pipeline Details as a collapsed debug accordion.
- Added a debug timing notice when `dataset_retrieval` is 2000 ms or slower.

## Speed Check

- Checked one demo turn in `.venv`: `dataset_retrieval_ms=0.062`, `total_ms=13.136`.
- No extra caching/search optimization was needed for the current mock demo path.

## Safety and Privacy

- Safety/crisis flow was not changed.
- Internal guidance remains hidden.
- Raw user input, raw dataset text, and raw memory transcript remain excluded from the user-facing UI and debug panel.
