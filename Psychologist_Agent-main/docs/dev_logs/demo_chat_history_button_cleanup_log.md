# Demo Chat History Button Cleanup Log

## Scope

Minimal `demo/app.py` cleanup for chat history preservation and duplicate chat-area buttons.

## Changes

- Preserved the assistant first greeting when chat history is empty.
- Kept Gradio Chatbot messages format and appended new turns without resetting prior messages.
- Kept the chat window fixed at `height=460` with internal scrolling.
- Removed chat-area example buttons and duplicate next-step buttons.
- Removed the visible status check button while keeping check-in values available internally for later wiring.
- Translated `OTHER_CONCERN` to `기타 고민` in the user-facing report.

## Chat History Fix

- Split the chat display and chat state into explicit `chatbot` and `chat_history_state` outputs.
- Updated the send handler to return the same updated messages-format history to both outputs.
- Removed the delayed `chatbot -> state` sync chain that could leave state stale between sends.
- Added `initial_chat_messages()` and `reset_chat_history()` helpers so initialization is explicit and reusable.

## Safety and Privacy

- No Agent feature changes.
- Safety/crisis flow was not changed.
- Raw user input, raw dataset text, and internal guidance remain hidden.
