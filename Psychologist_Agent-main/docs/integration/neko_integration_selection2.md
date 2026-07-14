# Memory 계층화 1단계 설계안

이 문서는 `Psychologist_Agent-main`의 기존 구조를 유지하면서, `N.E.K.O-main`의 memory 계층화 아이디어를 참고해 최소 기능으로 recent memory, fact memory, user directive, emotional state 데이터 구조를 설계한 것이다.

1단계에서는 실제 파이프라인 전체를 구현하지 않고, 기존 코드와 충돌하지 않는 데이터 구조와 확장 경계만 정의한다.

## 설계 원칙

- 기존 `src/memory/store.py`의 `MemoryStore`를 대체하지 않고 확장한다.
- 원문 대화는 저장하지 않는다.
- 저장 대상은 PII 마스킹 후 생성한 요약, 라벨, score, timestamp 같은 구조화 데이터로 제한한다.
- `SafetyGateway`와 `RiskChecker`/Risk Audit은 항상 memory보다 먼저 실행된다.
- memory는 응답 생성 보조 context일 뿐이며, crisis 판단이나 위기 응답을 덮어쓰지 않는다.
- 위기 상황에서는 memory 개인화를 최소화하거나 사용하지 않는다.

## 최소 메모리 계층

| 계층 | 목적 | 저장 데이터 | 원문 저장 여부 |
|---|---|---|---|
| Recent memory | 최근 상담 흐름 유지 | turn summary, key topics, emotional themes, risk stage | 저장 안 함 |
| Fact memory | 반복되는 사용자 맥락 저장 | stressor, preference, support style, boundary, concern | 저장 안 함 |
| User directive | 사용자가 명시한 경계/요청 저장 | avoid topic, preferred response style, do-not-mention term, TTL | 저장 안 함 |
| Emotional state | 최근 정서 흐름 추적 | emotion label, intensity, confidence, source, timestamp | 저장 안 함 |

## 파일 계획

| 파일 | 작업 | 역할 | 충돌 방지 방식 |
|---|---:|---|---|
| `src/memory/models.py` | 새로 생성 | `RecentMemoryEntry`, `FactMemoryEntry`, `UserDirective`, `EmotionalStateEntry`, `MemoryContext` dataclass 정의 | 기존 `store.py` 로직과 분리 |
| `src/memory/extractors.py` | 새로 생성 | 마스킹된 텍스트에서 topic, fact 후보, directive 후보, emotion label을 규칙 기반으로 추출 | LLM 호출 없이 최소 기능 |
| `src/memory/store.py` | 나중에 수정 | 기존 `MemoryStore`에 새 구조 저장/조회 메서드 추가 | 기존 `add`, `get_history`, `get_cloud_context`, `get_local_context` 유지 |
| `src/prompt/generator.py` | 나중에 수정 | `MemoryContext`를 선택적으로 받아 prompt section으로 삽입 | optional parameter로 추가해 기존 호출 유지 |
| `src/main.py` | 나중에 수정 | safety/risk 이후 memory context를 prompt 생성 전에 조회 | Safety Gateway와 Risk Audit 앞에는 배치하지 않음 |
| `tests/test_memory_layers.py` | 새로 생성 | 원문 미저장, directive TTL, emotion 누적, crisis 우선 정책 테스트 | 기존 테스트와 독립 |

## 데이터 모델 초안

### RecentMemoryEntry

```python
@dataclass
class RecentMemoryEntry:
    session_id: str
    summary: str
    key_topics: list[str]
    emotional_themes: list[str]
    risk_stage: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

역할:

- 최근 상담 턴을 요약 형태로 보관한다.
- 사용자 원문이나 assistant 원문을 저장하지 않는다.
- 기존 `MemoryStore._build_turn_summary()`와 자연스럽게 연결할 수 있다.

### FactMemoryEntry

```python
@dataclass
class FactMemoryEntry:
    fact_id: str
    session_id: str
    category: str
    label: str
    normalized_value: str
    confidence: float
    evidence_count: int
    first_seen_at: str
    last_seen_at: str
```

`category` 후보:

- `stressor`
- `preference`
- `support_style`
- `concern`
- `boundary`

역할:

- 반복되는 상담 맥락을 구조화해서 저장한다.
- 예: "수면 문제", "직장 스트레스", "조언보다 경청 선호"
- 진단명이나 치료 판단을 저장하지 않는다.

### UserDirective

```python
@dataclass
class UserDirective:
    directive_id: str
    session_id: str
    kind: str
    term: str
    active: bool
    hit_count: int
    created_at: str
    expires_at: str | None
```

`kind` 후보:

- `avoid_topic`
- `prefer_style`
- `do_not_mention`

역할:

- 사용자가 명시한 경계와 선호를 저장한다.
- 예: "가족 얘기는 하기 싫어", "해결책보다 들어줘"
- TTL을 둘 수 있게 해서 오래된 directive가 무기한 남지 않도록 한다.

### EmotionalStateEntry

```python
@dataclass
class EmotionalStateEntry:
    session_id: str
    label: str
    intensity: float
    confidence: float
    source: str
    risk_stage: str
    created_at: str
```

`label` 후보:

- `anxiety`
- `sadness`
- `loneliness`
- `anger`
- `overwhelm`
- `numbness`
- `hope`
- `crisis_signal`

`source` 후보:

- `message`
- `wellness_checkin`
- `risk_audit`

역할:

- 최근 정서 흐름을 추적한다.
- risk stage와 별도로 관리한다.
- 위기 신호는 항상 Safety Gateway와 Risk Audit 판단이 우선한다.

### MemoryContext

```python
@dataclass
class MemoryContext:
    recent_summaries: list[RecentMemoryEntry]
    facts: list[FactMemoryEntry]
    directives: list[UserDirective]
    emotional_trend: list[EmotionalStateEntry]
```

역할:

- prompt 생성기에 넘길 memory context를 하나로 묶는다.
- `src/prompt/generator.py`에는 optional parameter로 전달한다.
- 기존 prompt 생성 호출을 깨지 않도록 기본값은 `None`으로 둔다.

## 기존 파이프라인과의 관계

현재 `src/main.py`의 핵심 흐름은 유지한다.

```text
User Input
→ Safety Gateway
→ Risk Audit
→ PII Redaction
→ RAG
→ Cloud Analysis
→ Local Generation
→ Memory Update
```

나중에 memory context를 연결할 경우 목표 흐름은 다음과 같다.

```text
User Input
→ Safety Gateway
→ Risk Audit
→ PII Redaction
→ RAG
→ Memory Context 조회
→ Cloud Analysis
→ Local Generation
→ Masked/Structured Memory Update
```

위기 상황에서는 다음 원칙을 따른다.

```text
User Input
→ Safety Gateway detects crisis
→ Crisis Response
→ 최소 recent summary만 저장 가능
→ fact/directive/emotion 개인화 prompt에는 사용하지 않음
```

## Safety 우선 정책

memory 계층은 다음 판단을 수행하지 않는다.

- 자살/자해/폭력 위험 최종 판단
- crisis escalation 여부 결정
- 응급 연락처 안내 여부 결정
- 의료 진단 또는 치료 판단

위 판단은 기존 컴포넌트가 담당한다.

- `src/safety/gateway.py`
- `src/audit/risk_checker.py`
- `src/audit/crisis_handler.py`

memory는 이 결과를 metadata로 저장하거나 prompt 보조 context로 제공할 수는 있지만, 안전 판단을 변경하면 안 된다.

## 1단계 실제 구현 범위 제안

| 범위 | 포함 여부 |
|---|---:|
| dataclass 모델 추가 | 포함 |
| 규칙 기반 extractor 추가 | 포함 |
| 기존 `MemoryStore` 연결 | 최소 포함 또는 보류 |
| prompt 삽입 | 보류 |
| main pipeline 연결 | 보류 |
| persistent 저장 변경 | 보류 |
| LLM 기반 fact/reflection 추출 | 제외 |
| vector recall/BM25 | 제외 |
| 별도 memory server | 제외 |

1단계 실제 구현은 `src/memory/models.py`와 `src/memory/extractors.py`를 추가하고, 기존 파일 수정은 최소화하거나 다음 단계로 미루는 구성이 가장 안전하다.

## 다음 단계 후보

1. `src/memory/models.py` 생성
2. `src/memory/extractors.py` 생성
3. `tests/test_memory_layers.py` 생성
4. 이후 단계에서 `MemoryStore`에 저장/조회 메서드 추가
5. 그 다음 단계에서 `PromptGenerator`와 `src/main.py`에 optional 연결
