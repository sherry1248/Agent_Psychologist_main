# Final Agent Feature Integration Design

작성일: 2026-06-09

## 0. Scope

이 문서는 현재 `Psychologist_Agent-main`을 단순 상담 챗봇 느낌에서 역할 분리형 Agent 구조로 확장하기 위한 최종 통합 설계안이다.

제약은 다음과 같다.

- 실제 구현 파일은 아직 수정하지 않는다.
- 외부 참고 repo의 코드를 복사하지 않는다.
- Bitterbot의 Node/Gateway/P2P/Wallet/Marketplace/channel/browser/skill economy 계열 기능은 제외한다.
- Emotional First Aid Dataset 원본은 다운로드하거나 사용하지 않고, README의 라벨 구조만 참고한다.
- raw dataset text는 prompt나 memory에 직접 저장/주입하지 않는다.
- `data/raw`, `data/processed`, `requirements.txt`는 수정하지 않는다.
- 기존 테스트가 통과하는 방향으로 점진 통합한다.

참고한 핵심 아이디어는 다음이다.

- Psychologist AI Agent: Safety Gateway -> PII Redaction -> RAG -> Cloud Analysis -> Risk Audit -> Local Generation, Two-Prompt System, `LLM_TYPE`, Gradio Pipeline Details.
- Bitterbot: confidence/decay가 있는 구조화 memory, proactive recall, dream/session summary, emotional/hormonal state, identity/protocol files, tone modulation.
- Emotional First Aid Dataset README: S1 고민 유형, S2 심리질환 의심 수준, S3 SOS/위기 수준, chat label(`question`, `knowledge`, `negative`).
- Mental Health AI Pipeline: Emotion Agent와 Therapy/Response Agent의 strict pipeline 분리, 앞 Agent의 출력만 다음 Agent가 사용.
- GPT-J Emotional Conversational Agent: emotional state vector를 입력마다 조금씩 갱신하고 prompt에 반영.
- User-Oriented Multimodal Empathic Agent: 입력 처리 -> 감지/인식 -> 의도 분석 -> 통합 의사결정 -> 출력.

## 1. Current Code Mapping

| 현재 경로 | 현재 책임 | Agent 관점 매핑 | 비고 |
|---|---|---|---|
| `src/main.py` | 전체 inference orchestration, safety, risk, dataset hint, memory, prompt, generation 연결 | Agent Orchestrator / Pipeline Controller | 기존 단일 오케스트레이터를 유지하되 `agent_results`를 단계별로 쌓는 구조로 확장 |
| `src/safety/` | semantic safety gateway, risk pattern, safety response | Safety Agent 1차 관문 | 모든 Agent보다 우선. `is_safe=False`면 즉시 safety response 반환 |
| `src/audit/` | cloud analysis와 keyword 기반 risk audit, crisis handler, audit logger | Safety Agent 2차 검증 / Risk Audit | Safety Gateway 이후에도 only-escalate 정책으로 위험도를 낮추지 않음 |
| `src/memory/models.py` | raw text를 저장하지 않는 structured memory dataclass | Memory Agent schema 기반 | `MemoryContext` 확장 지점 |
| `src/memory/extractors.py` | masked text에서 topic/emotion/directive/fact 후보 추출 | Memory Agent extractor / Emotion observation 보조 | raw conversation 저장 금지 원칙과 맞음 |
| `src/memory/store.py` | session history, profile, structured memory 저장/조회 | Memory Agent store | proactive recall, small action, dream summary 저장 API 추가 후보 |
| `src/prompt/` | cloud/local two-prompt generation, memory/dataset hint formatting | Response Agent prompt builder | Decision 결과, emotional state vector, proactive recall을 local prompt에 allowlist 주입 |
| `src/counseling/` | processed counseling JSONL에서 intervention hint 추천 | Dataset Strategy Agent의 counseling branch | raw dataset text 대신 `intervention_hint`, record id, category만 사용 |
| `src/empathy/` | emotion/empathy label 기반 empathy style hint 추천 | Emotion Agent 보조 + Dataset Strategy Agent empathy branch | Emotion Agent의 label과 혼동하지 않게 역할 제한 필요 |
| `src/wellness/` | wellness check-in 기반 support hint 추천 | Dataset Strategy Agent wellness branch / Emotional State Agent 입력 | check-in 점수는 state update에 사용 가능 |
| `demo/app.py` | Gradio demo, mobile-like UI, summary JSON hidden output | Agent Pipeline View | 현재는 summary 중심. 단계별 Agent 결과 패널로 확장 |

현재 구조는 이미 기능적으로는 multi-stage pipeline이지만, 대부분의 판단이 `src/main.py` 안에 섞여 있어 사용자 입장에서는 "감정/위험도 판정 후 답변하는 챗봇"처럼 보인다. 목표 구조는 각 단계가 명시적 Agent result를 만들고, Decision Agent가 이번 턴의 행동을 선택한 뒤 Response Agent가 수행하도록 분리한다.

## 2. Final Agent Role Design

최종 온라인 턴 처리 순서:

```text
user_input
  -> Safety Agent
  -> PII Redaction
  -> Emotion Agent
  -> Intent Agent
  -> Dataset Strategy Agent
  -> Memory Agent / Proactive Recall
  -> Emotional State Agent
  -> Decision Agent
  -> Response Agent
  -> Memory Update / Small Action / Session Dream Summary
```

### 2.1 Safety Agent

책임:

- `src/safety/gateway.py`와 `src/audit/risk_checker.py`를 하나의 Safety Agent 결과로 묶는다.
- S3 위기 신호, 자해/자살/타해/즉시 위험 표현을 모든 Agent보다 먼저 처리한다.
- `risk_stage`는 `관심`, `주의`, `위험`을 유지한다.
- `risk_level`은 기존 enum 흐름을 유지한다.
- 위험도는 only-escalate: 뒤 단계가 Safety Agent 판단을 낮추지 않는다.

출력:

```python
SafetyAgentResult(
    is_safe: bool,
    risk_level: str,
    risk_stage: str,
    requires_crisis_response: bool,
    matched_category: str,
    action: str,
    resources: list[dict],
)
```

### 2.2 Emotion Agent

책임:

- 사용자 입력과 `src/empathy`의 emotion label, 기존 memory extractor emotion observation을 결합해 정서 label을 만든다.
- raw text를 저장하지 않고 label, intensity, confidence만 반환한다.
- Response Agent가 감정 분석을 반복하지 않게 한다.

출력 예:

```python
EmotionAgentResult(
    labels=[
        EmotionLabel(label="anxiety", intensity=0.72, confidence=0.82),
        EmotionLabel(label="sadness", intensity=0.51, confidence=0.64),
    ],
    dominant_label="anxiety",
)
```

### 2.3 Intent Agent

책임:

- Emotional First Aid Dataset README의 S1/S2/S3 구조를 참고해 상담 intent를 분류한다.
- 라벨은 진단이 아니라 "의도/의심 수준/위기 신호 후보"로만 사용한다.
- 수면, 스트레스, 관계, 조언 요청처럼 행동 선택에 필요한 intent를 만든다.
- S2/S3를 과분류하지 않도록 evidence rule을 둔다.

### 2.4 Dataset Strategy Agent

책임:

- `src/counseling`, `src/empathy`, `src/wellness` 결과를 단일 dataset strategy 결과로 묶는다.
- raw dataset text 대신 allowlist hint만 Decision Agent와 PromptGenerator에 전달한다.
- 상담 개입, 공감 스타일, 웰니스 제안을 서로 중복되지 않게 정리한다.

출력:

```python
DatasetStrategyResult(
    counseling_hint: str,
    empathy_style_hint: str,
    wellness_hint: str,
    matched_records: dict,
)
```

### 2.5 Memory Agent

책임:

- 기존 `MemoryContext` 조회.
- proactive recall 조회.
- 이번 턴 이후 structured memory, small action, follow-up, dream summary 업데이트.
- raw conversation text 저장 금지.

### 2.6 Emotional State Agent

책임:

- GPT-J Emotional Agent의 state vector 개념과 Bitterbot hormonal decay 개념을 상담 AI에 맞게 축소한다.
- 매 턴 state vector를 조금씩 업데이트한다.
- 현재 state를 Decision Agent와 Response Agent prompt에 전달한다.

### 2.7 Decision Agent

책임:

- 이전 단계 결과를 종합해 이번 턴 action을 선택한다.
- 직접 답변 내용을 생성하지 않는다.
- "공감만 할지", "질문할지", "작은 행동을 제안할지", "요약할지", "memory를 갱신할지", "safety escalation할지"를 결정한다.

### 2.8 Response Agent

책임:

- 기존 `PromptGenerator`와 `LocalGenerator`를 사용해 최종 응답을 생성한다.
- Decision Agent action을 따라 응답 구조를 제한한다.
- Safety Agent의 escalation 결과가 있으면 local generation을 우회한다.

## 3. Intent Agent Design

### 3.1 Intent Schema

Emotional First Aid Dataset README의 구조를 직접 복제하지 않고 다음처럼 프로젝트용 schema로 축소한다.

```python
class IntentSeverity(str, Enum):
    S1_CONCERN = "S1_CONCERN"
    S2_SUSPECTED_CONDITION = "S2_SUSPECTED_CONDITION"
    S3_SOS = "S3_SOS"

class IntentLabel(str, Enum):
    SLEEP_PROBLEM = "SLEEP_PROBLEM"
    ANXIETY_SUPPORT = "ANXIETY_SUPPORT"
    LOW_MOOD_SUPPORT = "LOW_MOOD_SUPPORT"
    STRESS_SUPPORT = "STRESS_SUPPORT"
    RELATIONSHIP_STRESS = "RELATIONSHIP_STRESS"
    WORK_OR_STUDY_STRESS = "WORK_OR_STUDY_STRESS"
    FAMILY_CONFLICT = "FAMILY_CONFLICT"
    LOW_SELF_ESTEEM = "LOW_SELF_ESTEEM"
    NEED_EMPATHY = "NEED_EMPATHY"
    NEED_ADVICE = "NEED_ADVICE"
    CRISIS_SIGNAL = "CRISIS_SIGNAL"
    SUBSTANCE_OR_ADDICTION = "SUBSTANCE_OR_ADDICTION"
    GRIEF_SUPPORT = "GRIEF_SUPPORT"
    OTHER_CONCERN = "OTHER_CONCERN"
```

### 3.2 Intent Result

```python
IntentAgentResult(
    labels=[
        IntentCandidate(
            label="SLEEP_PROBLEM",
            severity="S1_CONCERN",
            confidence=0.86,
            evidence=["sleep_keyword"],
        ),
        IntentCandidate(
            label="ANXIETY_SUPPORT",
            severity="S1_CONCERN",
            confidence=0.72,
            evidence=["anxiety_keyword"],
        ),
    ],
    primary_intent="SLEEP_PROBLEM",
    s2_suspected=False,
    s3_sos=False,
    chat_label_hint={"question": True, "knowledge": False, "negative": False},
)
```

### 3.3 Classification Rules

S1 concern:

- `잠`, `수면`, `불면`, `sleep`, `insomnia`, `can't sleep` -> `SLEEP_PROBLEM`
- `불안`, `초조`, `걱정`, `panic`, `anxious` -> `ANXIETY_SUPPORT`
- `우울`, `무기력`, `슬퍼`, `hopeless` -> `LOW_MOOD_SUPPORT`
- `스트레스`, `압박`, `번아웃`, `overwhelmed` -> `STRESS_SUPPORT`
- `가족`, `부모`, `엄마`, `아빠` + 갈등 표현 -> `FAMILY_CONFLICT`
- `직장`, `회사`, `업무`, `공부`, `시험`, `과제` -> `WORK_OR_STUDY_STRESS`
- `관계`, `친구`, `연인`, `이별` -> `RELATIONSHIP_STRESS`
- `자존감`, `내가 싫`, `쓸모없` but no self-harm -> `LOW_SELF_ESTEEM`
- `그냥 들어줘`, `공감`, `위로` -> `NEED_EMPATHY`
- `어떻게 해야`, `방법`, `조언` -> `NEED_ADVICE`

S2 suspected condition:

- 단일 키워드만으로 S2를 켜지 않는다.
- 기간, 기능 저하, 반복성 중 2개 이상이 함께 있을 때만 `s2_suspected=True`.
- 예: "몇 달째 불안해서 출근을 못 한다"는 S2 suspect 가능.
- 예: "요즘 잠을 못 자요"는 S1 `SLEEP_PROBLEM`만, S2/S3 아님.

S3 SOS:

- `죽고 싶`, `자살`, `자해`, `kill myself`, `harm myself`, 구체적 방법/시점/도구 표현 -> `CRISIS_SIGNAL`, `s3_sos=True`.
- 이 경우 Decision Agent는 `ESCALATE_SAFETY`만 선택 가능하다.

Chat label hint:

- 사용자가 정보 부족 상태로 감정/상태를 말하면 `question=True`.
- 사용자가 설명을 원하거나 psychoeducation이 필요한 intent면 `knowledge=True`.
- 사용자에게 죄책감/비난/단정이 포함될 수 있는 응답은 생성하지 않도록 `negative=False`를 항상 목표로 둔다.

## 4. Decision Agent Design

### 4.1 Actions

```python
class DecisionAction(str, Enum):
    RESPOND_SUPPORTIVELY = "RESPOND_SUPPORTIVELY"
    ASK_FOLLOW_UP = "ASK_FOLLOW_UP"
    SUGGEST_SMALL_ACTION = "SUGGEST_SMALL_ACTION"
    SUMMARIZE_STATE = "SUMMARIZE_STATE"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    ESCALATE_SAFETY = "ESCALATE_SAFETY"
```

한 턴에서 `primary_action`은 하나이고, `secondary_actions`는 여러 개 가능하다.

예:

```python
DecisionAgentResult(
    primary_action="ASK_FOLLOW_UP",
    secondary_actions=["RESPOND_SUPPORTIVELY", "SUGGEST_SMALL_ACTION", "UPDATE_MEMORY"],
    reason_codes=["sleep_problem_needs_clarification", "low_safety_risk", "actionable_wellness_hint"],
    response_constraints={
        "must_include_followup": True,
        "max_questions": 1,
        "must_include_small_action": True,
        "avoid_topics": ["family"],
    },
)
```

### 4.2 Required Inputs

Decision Agent는 반드시 다음 입력을 사용한다.

- `user_input`
- `risk_stage`
- emotion labels
- intent labels
- `counseling_hint`
- `empathy_style_hint`
- `wellness_hint`
- `MemoryContext`
- emotional state vector

### 4.3 Decision Rules

- `risk_stage == "위험"` 또는 `intent.s3_sos=True` 또는 Safety Agent `requires_crisis_response=True`:
  - `primary_action=ESCALATE_SAFETY`
  - local generation 우회
  - Memory에는 raw text 저장 금지, risk stage 변화 summary만 저장

- S1 intent가 있고 정보가 부족한 경우:
  - `primary_action=ASK_FOLLOW_UP`
  - `secondary_actions=[RESPOND_SUPPORTIVELY]`
  - 수면/불안/관계/가족/직장 스트레스는 최대 1개 follow-up question 생성

- wellness check-in 또는 emotional state가 명확히 나쁠 때:
  - `SUGGEST_SMALL_ACTION` 추가
  - 다음 턴에서 확인할 수 있게 small action 저장

- 사용자가 여러 고민을 나열하거나 최근 흐름이 누적된 경우:
  - `SUMMARIZE_STATE` 추가
  - 단, 사용자가 "짧게"를 선호하면 요약은 2문장 이하

- user directive나 반복 고민, small action, next follow-up이 새로 생긴 경우:
  - `UPDATE_MEMORY` 추가

## 5. Emotional State Vector Design

### 5.1 Vector

상담 AI용 상태 벡터는 7개 값으로 축소한다.

```python
EmotionalStateVector(
    mood: float,     # 0.0 low mood, 1.0 stable/positive mood
    anxiety: float,  # 0.0 calm, 1.0 high anxiety
    stress: float,   # 0.0 relaxed, 1.0 high stress
    sleep: float,    # 0.0 poor sleep, 1.0 good sleep
    energy: float,   # 0.0 exhausted, 1.0 energetic
    safety: float,   # 0.0 unsafe/crisis, 1.0 safe
    rapport: float,  # 0.0 no bond, 1.0 strong trust/continuity
)
```

초기값:

- `mood=0.5`
- `anxiety=0.3`
- `stress=0.3`
- `sleep=0.5`
- `energy=0.5`
- `safety=0.9`
- `rapport=0.2`

### 5.2 Update Formula

Bitterbot의 half-life decay 개념을 단순화한다.

```text
new_value = clamp(old_value * 0.85 + observed_value * 0.15, 0.0, 1.0)
```

세션 간 오랜 시간이 지났으면 baseline 쪽으로 완만히 회귀한다.

```text
decayed = baseline + (old_value - baseline) * decay_factor
```

권장 baseline:

- `mood=0.5`
- `anxiety=0.25`
- `stress=0.25`
- `sleep=0.5`
- `energy=0.5`
- `safety=0.9`
- `rapport`는 decay하지 않거나 매우 느리게 decay

### 5.3 Observation Rules

입력/label 기반:

- `anxiety` emotion label intensity 0.7 -> `anxiety` 관측값 0.7 이상
- `sadness`, `low_mood` -> `mood` 하향
- `overwhelm`, `stress` -> `stress` 상향, `energy` 하향
- `sleep`, `insomnia` intent -> `sleep` 하향
- `crisis_signal` 또는 `risk_stage="위험"` -> `safety` 0.0~0.2
- 사용자가 고마움/신뢰/선호를 표현 -> `rapport` 소폭 상승

wellness_checkin 기반:

- `mood_score`: 1~5를 0.0~1.0으로 변환해 `mood`에 반영
- `anxiety_score`: 1~5를 0.0~1.0으로 변환해 `anxiety`에 반영
- `stress_score`: 1~5를 0.0~1.0으로 변환해 `stress`에 반영
- `sleep_quality`: 1~5를 0.0~1.0으로 변환해 `sleep`에 반영
- `energy_score`: 1~5를 0.0~1.0으로 변환해 `energy`에 반영

risk_stage 기반:

- `관심`: `safety` 관측값 0.85~1.0
- `주의`: `safety` 관측값 0.45~0.75
- `위험`: `safety` 관측값 0.0~0.25

prompt 주입:

- Response Agent에는 숫자 그대로보다 짧은 설명도 함께 전달한다.
- 예: `state_summary="low sleep, elevated anxiety, moderate rapport"`
- 모델이 "상태 벡터" 자체를 사용자에게 말하지 않도록 지시한다.

## 6. Proactive Recall Design

Bitterbot의 proactive recall 아이디어를 상담 과제 범위에 맞게 축소한다. 응답 생성 전에 Memory Agent가 자동 조회한다.

조회 항목:

- 반복 고민: 최근 `FactMemoryEntry(category="concern")` 중 evidence_count가 높은 항목
- 최근 감정 흐름: `MemoryContext.emotional_trend`
- 지난번 제안한 small action: 새 `SmallActionMemory`
- 사용자 선호 응답 방식: `UserDirective(kind="prefer_style")`, `FactMemoryEntry(category="support_style")`
- 피하고 싶은 주제: `UserDirective(kind="avoid_topic")`
- 다음에 확인할 follow-up question: 새 `FollowUpMemory`

결과 schema:

```python
ProactiveRecallResult(
    repeated_concerns=["sleep", "work"],
    emotional_trend_summary="recent anxiety and poor sleep observed",
    last_small_action=SmallActionMemory(...),
    preferred_response_style=["listening", "short_response"],
    avoid_topics=["family"],
    next_follow_up="잠드는 데 오래 걸리는 편인가요, 자다가 자주 깨는 편인가요?",
)
```

Response Agent 사용 원칙:

- 사용자가 "기억에 따르면" 같은 표현을 듣지 않게 자연스럽게 반영한다.
- 오래된 recall은 `stale=True`로 표시하고 Decision Agent가 낮은 우선순위로 사용한다.
- avoid topic은 Response Agent constraint로 강제한다.

## 7. Session Dream Summary Design

Bitterbot의 Dream Engine을 과제 범위에 맞게 세션 종료 또는 일정 턴마다 실행되는 구조화 요약으로 축소한다.

트리거:

- 5턴마다
- 세션 종료 시
- 위험 단계 변화 발생 시
- small action/follow-up이 생성된 턴

저장 schema:

```python
SessionDreamSummary(
    session_id: str,
    main_issue: list[str],
    emotional_trend: list[str],
    risk_stage_start: str,
    risk_stage_end: str,
    last_small_action: str,
    next_follow_up: str,
    important_user_directives: list[str],
    created_at: str,
)
```

중요 원칙:

- raw conversation text는 저장하지 않는다.
- summary도 원문 문장 요약이 아니라 label, trend, action, directive 중심으로 저장한다.
- 예: `main_issue=["sleep", "anxiety"]`, `emotional_trend=["anxiety_high", "sleep_low"]`
- prompt 주입 시에도 structured summary만 사용한다.

## 8. Follow-up Question Generator Design

Follow-up Question Generator는 Intent Agent와 Emotional State Agent 결과를 받아 한 턴에 최대 1개 질문을 만든다.

입력:

- `primary_intent`
- emotion labels
- emotional state vector
- risk_stage
- proactive recall의 `next_follow_up`
- avoid topics

규칙:

- Safety escalation이면 follow-up 질문 생성하지 않는다. 즉시 안전 안내가 우선이다.
- 이미 미해결 follow-up이 있으면 새 질문보다 기존 질문을 우선한다.
- 질문은 사용자가 쉽게 답할 수 있는 선택형/구체형으로 만든다.
- 한 번에 여러 질문을 묻지 않는다.

예시:

| 입력 intent/state | 출력 질문 |
|---|---|
| `SLEEP_PROBLEM` + anxiety high | "잠드는 데 오래 걸리는 편인가요, 아니면 자다가 자주 깨는 편인가요?" |
| `ANXIETY_SUPPORT` | "불안이 가장 크게 올라오는 순간은 혼자 있을 때인가요, 아니면 해야 할 일을 마주할 때인가요?" |
| `LOW_MOOD_SUPPORT` | "오늘 가장 버겁게 느껴진 건 몸의 피곤함에 가까웠나요, 마음의 무거움에 가까웠나요?" |
| `WORK_OR_STUDY_STRESS` | "지금 제일 부담되는 건 시작하기 어려움인가요, 끝내야 한다는 압박인가요?" |
| `RELATIONSHIP_STRESS` | "그 관계에서 가장 힘든 부분은 말이 통하지 않는 느낌인가요, 혼자 감당하는 느낌인가요?" |

## 9. Small Action Plan Design

Small Action Plan은 오늘 할 수 있는 작은 행동 하나만 제안하고, 다음 대화에서 확인 가능하도록 저장한다.

입력:

- intent labels
- emotional state vector
- wellness_hint
- counseling_hint
- user directive
- risk_stage

생성 원칙:

- 5분 이내 또는 부담이 낮은 행동.
- 진단/치료처럼 보이지 않게 "오늘 해볼 수 있는 작은 행동"으로 표현.
- crisis 상황에서는 small action보다 safety action이 우선.
- 한 턴에 하나만 제안.

schema:

```python
SmallActionPlan(
    action_id: str,
    session_id: str,
    intent_label: str,
    action_text: str,
    rationale_label: str,
    status: "suggested|checked|done|skipped",
    created_at: str,
    check_after_turns: int = 1,
)
```

예:

- `SLEEP_PROBLEM`: "오늘 잠들기 전 10분만 화면을 내려놓고, 방 조명을 조금 낮춰보세요."
- `ANXIETY_SUPPORT`: "지금 자리에서 발바닥 감각을 30초만 느껴보세요."
- `STRESS_SUPPORT`: "해야 할 일을 한 줄로만 적고, 가장 작은 첫 단계에 동그라미를 쳐보세요."
- `LOW_MOOD_SUPPORT`: "물 한 잔을 마시고 창문 근처에서 1분만 서 있어보세요."

다음 대화 recall:

- Proactive Recall이 `last_small_action`을 가져온다.
- Decision Agent가 부담 없는 확인을 선택한다.
- 예: "지난번에 정했던 작은 행동은 해보지 못했어도 괜찮아요. 오늘은 수면 쪽이 더 힘든가요, 불안 쪽이 더 큰가요?"

## 10. Gradio Agent Pipeline View Design

`demo/app.py`는 현재 response markdown과 hidden JSON summary 중심이다. 새 구조에서는 사용자에게 Agent 처리 흐름을 볼 수 있는 "Agent Pipeline Details" 패널을 추가한다.

권장 UI:

- 상담 응답 카드: 최종 Response Agent 응답
- 상태 카드: risk_stage, primary_intent, dominant_emotion, primary_action
- Agent Pipeline Details accordion:
  - Safety Agent 결과
  - Emotion Agent 결과
  - Intent Agent 결과
  - Dataset Strategy Agent 결과
  - Memory Agent / Proactive Recall 결과
  - Emotional State Agent 결과
  - Decision Agent action
  - Response Agent final response metadata

표시 예:

```json
{
  "Safety Agent": {
    "risk_stage": "관심",
    "requires_crisis_response": false
  },
  "Emotion Agent": {
    "dominant_label": "anxiety",
    "labels": [{"label": "anxiety", "intensity": 0.72, "confidence": 0.82}]
  },
  "Intent Agent": {
    "primary_intent": "SLEEP_PROBLEM",
    "s2_suspected": false,
    "s3_sos": false
  },
  "Dataset Strategy Agent": {
    "hint_keys": ["counseling_hint", "empathy_style_hint", "wellness_hint"]
  },
  "Memory Agent": {
    "recalled": ["last_small_action", "preferred_response_style"]
  },
  "Decision Agent": {
    "primary_action": "ASK_FOLLOW_UP",
    "secondary_actions": ["SUGGEST_SMALL_ACTION", "UPDATE_MEMORY"]
  }
}
```

민감정보 정책:

- Gradio Pipeline View에도 raw user input, raw dataset text, raw memory transcript를 표시하지 않는다.
- record id, label, score, hint key, risk stage만 표시한다.

## 11. Implementation Priority

### 9-A. Agent schema/dataclass 설계

- `src/agent/models.py` 추가.
- 모든 Agent result dataclass와 enum 정의.
- 기존 pipeline에는 아직 최소 연결만 한다.
- 테스트는 schema serialization과 raw text field 부재 확인.

### 9-B. Intent Agent

- `src/agent/intent.py` 추가.
- keyword/rule 기반 deterministic classifier로 시작한다.
- Emotional First Aid Dataset 원본 없이 README 라벨 구조만 반영.

### 9-C. Emotional State Agent

- `src/agent/state.py` 추가.
- state vector, baseline, update rule, wellness check-in 반영.
- MemoryStore에는 structured state만 저장.

### 9-D. Decision Agent

- `src/agent/decision.py` 추가.
- Safety, Emotion, Intent, Dataset hints, MemoryContext, state vector를 받아 action 결정.
- local generation 전에 `DecisionAgentResult` 생성.

### 9-E. Follow-up Question Generator

- `src/agent/followup.py` 추가.
- intent별 template + avoid topic + previous follow-up 우선순위.

### 9-F. Small Action Plan

- `src/agent/planner.py` 추가.
- small action 생성, 저장 schema, 다음 턴 확인 규칙.

### 9-G. Proactive Recall

- `src/agent/recall.py` 추가.
- MemoryStore에서 repeated concern, emotional trend, last action, directive, follow-up 조회.

### 9-H. Session Dream Summary

- `src/agent/summary.py` 추가.
- 일정 턴마다 structured summary 생성.
- raw text 저장 금지 테스트를 먼저 둔다.

### 9-I. Gradio Agent Pipeline View

- `demo/app.py` 확장.
- `result["pipeline_details"]["agents"]`를 사람이 읽기 쉬운 accordion/JSON으로 표시.

## 12. File Plan

| 구분 | 파일 | 작업 | 비고 |
|---|---|---|---|
| 새 파일 | `src/agent/__init__.py` | Agent package export | requirements 변경 없음 |
| 새 파일 | `src/agent/models.py` | Agent result, enum, state, recall, plan schema | 9-A |
| 새 파일 | `src/agent/intent.py` | Intent Agent rule classifier | 9-B |
| 새 파일 | `src/agent/state.py` | Emotional State Agent update/decay | 9-C |
| 새 파일 | `src/agent/decision.py` | Decision Agent action selector | 9-D |
| 새 파일 | `src/agent/followup.py` | Follow-up question generator | 9-E |
| 새 파일 | `src/agent/planner.py` | Small action plan generator | 9-F |
| 새 파일 | `src/agent/recall.py` | Proactive recall aggregator | 9-G |
| 새 파일 | `src/agent/summary.py` | Session dream summary builder | 9-H |
| 수정 | `src/main.py` | 기존 순서 사이에 Agent 결과 생성 및 `pipeline_details["agents"]` 추가 | 기존 return schema 유지 |
| 수정 | `src/memory/models.py` | `SmallActionMemory`, `FollowUpMemory`, `SessionDreamSummary`, `EmotionalStateVectorEntry` 추가 | raw text 금지 validator 포함 |
| 수정 | `src/memory/store.py` | small action/follow-up/dream summary/state vector 저장/조회 API 추가 | 기존 `add()` behavior 유지 |
| 수정 | `src/prompt/generator.py` | Decision result, state summary, proactive recall allowlist formatting | raw text 주입 금지 |
| 수정 | `demo/app.py` | Agent Pipeline View 표시 | hidden JSON 유지 가능 |
| 테스트 | `tests/test_agent_intent.py` | intent classification, S1/S2/S3 overclassification 방지 | 9-B |
| 테스트 | `tests/test_emotional_state.py` | vector update, clamp, decay, wellness 반영 | 9-C |
| 테스트 | `tests/test_agent_decision.py` | action selection, safety precedence | 9-D |
| 테스트 | `tests/test_agent_followup.py` | intent별 질문 생성, max 1 question | 9-E |
| 테스트 | `tests/test_small_action_plan.py` | small action 생성/상태 저장 | 9-F |
| 테스트 | `tests/test_proactive_recall.py` | previous small action, directives, trend recall | 9-G |
| 테스트 | `tests/test_session_dream_summary.py` | raw text 없는 structured summary | 9-H |
| 테스트 수정 | `tests/test_demo_hint_visibility.py` | Pipeline View에 raw hints/text 노출 금지 확인 | 9-I |
| 테스트 수정 | `tests/test_main_memory_context.py` | 기존 MemoryContext 흐름 유지 확인 | regression |

## 13. Test Plan

### 13.1 Intent Agent

- `"요즘 잠을 못 자요"`는 `SLEEP_PROBLEM` intent로 분류한다.
- 수면 문제 단독은 `s2_suspected=False`, `s3_sos=False`, `risk_stage` 승격 없음.
- `"요즘 잠을 못 자고 불안해요"`는 `SLEEP_PROBLEM`과 `ANXIETY_SUPPORT`를 함께 반환하고 primary intent는 score 높은 쪽으로 결정한다.
- `"그냥 들어줬으면 좋겠어요"`는 `NEED_EMPATHY`.
- `"어떻게 해야 할지 모르겠어요"`는 `NEED_ADVICE`.
- 자해/죽고 싶다 표현은 `CRISIS_SIGNAL`, `s3_sos=True`.
- Intent Agent result에는 `raw_text`, `user_input`, `conversation`, `content` 필드가 없다.

### 13.2 Emotion Agent

- 불안 keyword는 anxiety label, 슬픔 keyword는 sadness/low mood label.
- confidence/intensity는 0.0~1.0 clamp.
- Emotion Agent가 raw text를 저장하지 않는다.
- Emotion Agent와 Intent Agent가 같은 일을 반복하지 않도록 Emotion Agent는 emotion label만 반환하고 intent label은 반환하지 않는다.

### 13.3 Emotional State Agent

- `wellness_checkin.sleep_quality=1`이면 `sleep` 값이 하향된다.
- `anxiety_score=5` 또는 anxiety emotion intensity high이면 `anxiety` 값이 상승한다.
- `risk_stage="위험"`이면 `safety`가 크게 하향된다.
- 모든 값은 0.0~1.0 범위를 벗어나지 않는다.
- 시간이 지난 state는 baseline 쪽으로 decay된다.

### 13.4 Decision Agent

- 자해/죽고 싶다 표현 또는 `s3_sos=True`는 `ESCALATE_SAFETY`.
- Safety escalation이 있으면 follow-up/small action이 primary가 될 수 없다.
- 수면 문제 + 낮은 risk는 `ASK_FOLLOW_UP`을 선택한다.
- wellness/state가 좋지 않으면 `SUGGEST_SMALL_ACTION`을 secondary action에 포함한다.
- Decision Agent가 follow-up 질문과 small action을 선택한다.
- avoid topic directive가 있으면 response constraints에 포함한다.

### 13.5 Follow-up Question Generator

- `"요즘 잠을 못 자고 불안해요"` 입력 후 질문은 "잠드는 데 오래 걸리는 편인가요, 아니면 자다가 자주 깨는 편인가요?" 계열이어야 한다.
- 질문은 한 턴에 최대 1개.
- 위기 상황에서는 follow-up question을 생성하지 않는다.
- 기존 next_follow_up이 있으면 새 질문보다 우선한다.

### 13.6 Small Action Plan

- 각 S1 intent에 대해 작은 행동 하나를 생성한다.
- action text는 raw user input을 포함하지 않는다.
- 생성된 action은 `status="suggested"`로 저장된다.
- 다음 턴에서 Proactive Recall이 이전 small action을 가져온다.

### 13.7 Proactive Recall

- 이전 small action을 가져온다.
- 반복 고민은 evidence_count 또는 최근 summary 기반으로 계산한다.
- 사용자 선호 응답 방식과 피하고 싶은 주제를 분리해 가져온다.
- raw conversation text는 recall result에 포함하지 않는다.

### 13.8 Session Dream Summary

- 5턴 또는 세션 종료 시 structured summary를 만든다.
- `main_issue`, `emotional_trend`, `risk_stage_start/end`, `last_small_action`, `next_follow_up`, `important_user_directives`를 포함한다.
- Session Dream Summary가 raw conversation을 저장하지 않는다.
- raw field key validator가 `raw_text`, `user_input`, `assistant_response`, `conversation`, `content` 저장을 막는다.

### 13.9 Main Pipeline / Demo Regression

- 기존 `process_message()` return keys는 유지한다.
- 기존 `counseling_hint`, `empathy_style_hint`, `wellness_hint`는 유지한다.
- `pipeline_details["agents"]`가 추가되어도 기존 테스트가 깨지지 않는다.
- Gradio Pipeline View에 raw dataset text나 raw memory transcript가 표시되지 않는다.

## 14. Presentation Appeal Points

교수님에게 설명할 핵심은 다음이다.

1. 입력을 바로 답변하지 않고 여러 Agent가 역할을 나눠 처리한다.
   - Safety Agent, Emotion Agent, Intent Agent, Dataset Strategy Agent, Memory Agent, Emotional State Agent, Decision Agent, Response Agent가 순서대로 동작한다.

2. Decision Agent가 이번 턴의 행동을 선택한다.
   - 단순히 감정/위험도 판정 후 답변하는 챗봇이 아니라, Intent, Emotion, Risk, Memory, Dataset Hint, Emotional State를 통합해 `ASK_FOLLOW_UP`, `SUGGEST_SMALL_ACTION`, `ESCALATE_SAFETY` 같은 행동을 고른다.

3. 사용자의 상태 벡터가 대화마다 업데이트된다.
   - `mood`, `anxiety`, `stress`, `sleep`, `energy`, `safety`, `rapport`가 매 턴 조금씩 변하고, 다음 응답의 톤과 선택에 영향을 준다.

4. Proactive Recall과 Session Dream Summary가 대화를 이어지게 만든다.
   - 지난번 제안한 작은 행동, 반복 고민, 최근 감정 흐름, 선호 응답 방식, 다음 follow-up을 기억한다.
   - raw 대화문을 저장하지 않고 구조화 요약만 저장해 privacy-preserving memory를 유지한다.

5. Safety Agent는 모든 Agent보다 우선한다.
   - 자해/자살/타해/즉시 위험 신호가 있으면 Decision Agent가 어떤 다른 행동도 선택하지 못하고 `ESCALATE_SAFETY`로 고정된다.

6. 기존 프로젝트의 장점을 유지한다.
   - Cloud for analysis, Local for generation 구조와 Two-Prompt System을 유지한다.
   - `LLM_TYPE=MOCK/CLOUD/LOCAL` 전환과 Gradio demo도 유지한다.
   - processed dataset hint만 사용하고 raw dataset text를 prompt/memory에 넣지 않는다.

요약 문장:

> 이 시스템은 사용자의 입력을 곧바로 답변하지 않고, Safety, Emotion, Intent, Dataset, Memory, State Agent가 각각 제한된 역할로 분석한 뒤 Decision Agent가 이번 턴의 행동을 선택하고 Response Agent가 실행한다. 그래서 단순 상담 챗봇이 아니라 상태, 기억, 위험도, 의도에 따라 다음 행동을 결정하는 Agent 구조다.
