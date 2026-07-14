# UI Service Structure Log

## Purpose

Aligned the Gradio demo with the presentation service structure for the "Psychologist AI Agent" without adding new agent or LLM features.

## Updated Screen Structure

- Added a start area with:
  - `Psychologist AI Agent` title
  - brief service description for 2030 youth emotional support
  - anonymous mode checkbox
  - record-save consent checkbox
  - first AI greeting message
- Added service tabs:
  - `상담 채팅`
  - `감정일기`
  - `마음정리 보고서`
  - `전문가 상담 연결`
- Kept `Agent Pipeline Details` as a collapsed accordion for presentation/debug use.

## 상담 채팅

- Preserved the existing chat flow and safety pipeline.
- Chatbot remains limited to user/assistant messages.
- Pipeline markdown remains separate from chat history.
- Internal hints are not rendered on the user-facing screen.

## 감정일기

- Added structured diary inputs:
  - emotion label
  - mood score
  - sleep score
  - anxiety score
  - loneliness score
  - one-line diary input
- The one-line diary text is not stored as raw memory.
- The demo stores only structured values:
  - selected emotion label
  - numeric scores
  - whether diary text exists
  - text length bucket
  - save consent flag

## 마음정리 보고서

- Added a report view based on recent structured Agent Pipeline output and diary state.
- Displays:
  - main emotion
  - concern keywords
  - risk stage
  - recommended stabilizing action
  - next follow-up question
  - professional counseling guidance when relevant
- Raw user input, raw dataset text, and raw memory transcript are not shown.

## 전문가 상담 연결

- Added static service guidance for:
  - emergency contact: 109, 119, 112
  - mental health welfare center / counseling center connection
  - disclaimer that AI does not provide medical diagnosis or treatment

## Safety and Privacy Notes

- No safety/crisis logic was weakened.
- No new LLM feature was added.
- Existing Agent Pipeline logic remains unchanged.
- Debug information stays in a collapsed accordion.
- Raw text exposure protections are preserved.

## Final Minimal Adjustment

- Kept the service screen implementation centered in `demo/app.py`.
- Confirmed the start area, chat tab, emotion diary tab, report tab, expert connection tab, and collapsed Agent Pipeline accordion.
- Updated the report empty state to show `아직 상담 기록이 없습니다.`
