# MemoryStore 연결 2단계 설계안

이 문서는 1단계에서 추가한 memory models와 rule-based extractors를 `MemoryStore`에 연결하기 위한 설계안이다. 아직 실제 코드는 수정하지 않는다.

목표는 기존 `MemoryStore`의 동작을 깨지 않고, recent memory, fact memory, user directive, emotional state를 부가 레이어로 붙이는 것이다.

## 설계 원칙

- 기존 `src/memory/store.py`의 `add`, `get_history`, `get_cloud_context`, `get_local_context` 동작은 유지한다.
- 기존 `_sessions`, `_summaries`, `_profiles` 구조는 그대로 둔다.
- 새 memory 계층은 별도 내부 필드에 저장한다.
- `src/main.py`와 `src/prompt/generator.py`에는 아직 연결하지 않는다.
- `MemoryStore`는 safety 판단을 수행하지 않는다.
- `SafetyGateway`와 Risk Audit 결과는 항상 memory보다 우선한다.
- memory는 호출자가 넘긴 `risk_stage`, `risk_level` 같은 metadata를 저장하거나 context로 제공할 뿐, 위기 판단을 변경하지 않는다.
- 저장 대상은 원문이 아니라 PII 마스킹 후 생성한 요약/라벨/구조화 데이터다.

## 추가 내부 필드

`MemoryStore.__init__()`에 다음 필드를 추가하는 방식이 적합하다.

| 필드 | 타입 | 역할 |
|---|---|---|
| `_recent_entries` | `Dict[str, List[RecentMemoryEntry]]` | 세션별 최근 요약 메모리 |
| `_fact_entries` | `Dict[str, List[FactMemoryEntry]]` | 세션별 구조화 fact 후보 |
| `_directives` | `Dict[str, List[UserDirective]]` | 세션별 사용자 경계/선호 |
| `_emotional_states` | `Dict[str, List[EmotionalStateEntry]]` | 세션별 감정 상태 흐름 |

이 필드들은 기존 `_sessions`, `_summaries`, `_metadata`, `_profiles`와 독립적으로 둔다.

## 추가 메서드 제안

| 메서드 | 역할 | 기존 동작 영향 |
|---|---|---|
| `add_structured_memory(session_id, masked_text, risk_stage="관심", source="message")` | 1단계 extractor를 호출해 recent/fact/directive/emotion 레이어를 갱신 | 기존 `add()`와 분리 가능 |
| `get_memory_context(session_id, max_recent=5, max_facts=8, max_directives=5, max_emotions=5)` | `MemoryContext` 객체를 구성해 반환 | 기존 context API와 별도 |
| `get_recent_memory(session_id, limit=5)` | 최근 구조화 요약만 반환 | 영향 없음 |
| `get_fact_memory(session_id, limit=8, categories=None)` | fact 후보 반환, category 필터 지원 | 영향 없음 |
| `get_user_directives(session_id, active_only=True, limit=5)` | 활성 directive 반환 | 영향 없음 |
| `get_emotional_trend(session_id, limit=5)` | 최근 감정 상태 반환 | 영향 없음 |
| `clear_structured_memory(session_id)` | 새 메모리 레이어만 삭제 | `clear_session()`에서 함께 호출 가능 |
| `_merge_fact_entries(session_id, new_facts)` | 같은 `category + normalized_value` fact 중복 병합 | 내부 helper |
| `_merge_directives(session_id, new_directives)` | 같은 `kind + term` directive hit_count 갱신 | 내부 helper |
| `_trim_structured_memory(session_id)` | 각 레이어 최대 길이 유지 | 내부 helper |

## `add()`와의 관계

기존 `add()` 시그니처는 바꾸지 않는다.

현재 형태:

```python
async def add(
    self,
    session_id: str,
    user_input: str,
    response: str,
    user_metadata: Optional[Dict[str, Any]] = None,
    response_metadata: Optional[Dict[str, Any]] = None
) -> None:
    ...
```

2단계에서 가장 보수적인 방식은 `add()` 내부에 새 로직을 연결하지 않고, `add_structured_memory()`만 추가하는 것이다. 그러면 `src/main.py`를 수정하기 전까지 기존 동작 영향이 없다.

나중에 opt-in 연결을 할 경우에는 `add()` 마지막에 다음 흐름을 추가할 수 있다.

```python
masked_text = sanitized_user_input
risk_stage = (user_metadata or {}).get("risk_stage", "관심")
await self.add_structured_memory(
    session_id=session_id,
    masked_text=masked_text,
    risk_stage=risk_stage,
    source="message",
)
```

권장 순서:

1. 2단계: `MemoryStore`에 새 필드와 메서드만 추가
2. 3단계: `add()` 끝에서 구조화 메모리 업데이트를 opt-in으로 연결
3. 4단계: `PromptGenerator`에 `MemoryContext` optional 삽입
4. 5단계: `src/main.py`에서 safety/risk 이후 prompt 생성 전에 memory context 조회

## 메서드 상세

### add_structured_memory

```python
async def add_structured_memory(
    self,
    session_id: str,
    masked_text: str,
    risk_stage: str = "관심",
    source: str = "message",
) -> None:
    ...
```

역할:

- `build_recent_memory_entry()` 호출
- `extract_fact_candidates()` 호출
- `extract_user_directives()` 호출
- `extract_emotional_states()` 호출
- 반환된 dataclass들을 내부 레이어에 저장
- 입력은 반드시 원문이 아니라 마스킹된 텍스트여야 함

주의:

- 이 메서드는 crisis 판단을 하지 않는다.
- `risk_stage`는 외부 safety/risk 컴포넌트가 판단한 값을 metadata처럼 사용한다.

### get_memory_context

```python
async def get_memory_context(
    self,
    session_id: str,
    max_recent: int = 5,
    max_facts: int = 8,
    max_directives: int = 5,
    max_emotions: int = 5,
) -> MemoryContext:
    ...
```

역할:

- prompt generator가 나중에 사용할 수 있는 단일 객체를 반환한다.
- 기존 `get_cloud_context()`와 `get_local_context()`와 분리한다.
- context 크기를 호출 시점에 제한한다.

### get_recent_memory

```python
async def get_recent_memory(
    self,
    session_id: str,
    limit: int = 5,
) -> List[RecentMemoryEntry]:
    ...
```

역할:

- 세션별 recent structured memory를 최신순 또는 삽입순 tail로 반환한다.
- 기존 `get_history()`를 대체하지 않는다.

### get_fact_memory

```python
async def get_fact_memory(
    self,
    session_id: str,
    limit: int = 8,
    categories: Optional[List[str]] = None,
) -> List[FactMemoryEntry]:
    ...
```

역할:

- 세션별 fact 후보를 반환한다.
- 필요 시 `category` 필터를 지원한다.
- 반복 fact는 `_merge_fact_entries()`에서 병합한다.

### get_user_directives

```python
async def get_user_directives(
    self,
    session_id: str,
    active_only: bool = True,
    limit: int = 5,
) -> List[UserDirective]:
    ...
```

역할:

- 사용자 경계/선호 directive를 반환한다.
- 기본값은 활성 directive만 반환한다.

### get_emotional_trend

```python
async def get_emotional_trend(
    self,
    session_id: str,
    limit: int = 5,
) -> List[EmotionalStateEntry]:
    ...
```

역할:

- 최근 감정 상태 흐름을 반환한다.
- risk 판단에는 사용하지 않고 prompt context나 분석 보조로만 사용한다.

### clear_structured_memory

```python
async def clear_structured_memory(
    self,
    session_id: str,
) -> None:
    ...
```

역할:

- 새 memory 레이어만 삭제한다.
- `clear_session()`이 호출될 때 함께 호출하면 세션 삭제 의미와 일치한다.

## 중복 병합 정책

| 레이어 | 병합 기준 | 병합 방식 |
|---|---|---|
| Recent memory | 병합 없음 | append 후 최근 N개 유지 |
| Fact memory | `(category, normalized_value)` | `evidence_count += 1`, `last_seen_at` 갱신, `confidence`는 max |
| Directive | `(kind, term)` | `hit_count += 1`, 기존 항목 유지 |
| Emotional state | 병합 없음 | append 후 최근 N개 유지 |

`UserDirective`에는 현재 `last_seen_at` 필드가 없다. 2단계에서는 모델을 바꾸지 않고 `hit_count`만 갱신하는 방식이 안전하다. 필요하면 후속 단계에서 모델 확장을 검토한다.

## trim 정책

초기값은 다음 정도로 충분하다.

| 레이어 | 최대 보관 수 |
|---|---:|
| Recent memory | 20 |
| Fact memory | 50 |
| User directive | 20 |
| Emotional state | 30 |

이 값은 `MemoryConfig`에 바로 넣기보다, 2단계에서는 `MemoryStore` 내부 상수로 시작하는 편이 단순하다. 나중에 운영 필요가 생기면 `MemoryConfig`로 승격한다.

## clear 동작

기존 `clear_session(session_id)`는 새 레이어도 함께 삭제하는 것이 자연스럽다.

```python
await self.clear_structured_memory(session_id)
```

이 변경은 기존 API 의미와 충돌하지 않는다. 세션 삭제는 해당 세션의 모든 memory 삭제를 뜻하기 때문이다.

## persistence 정책

2단계에서는 구조화 memory persistence를 연결하지 않는다.

이유:

- 기존 `_persist_session()` 포맷을 건드리면 호환성 리스크가 생긴다.
- 구조화 memory는 아직 prompt와 main pipeline에 연결되지 않았다.
- 원문 미저장 정책 검증 후 별도 schema로 저장하는 편이 안전하다.

따라서 2단계는 in-memory only가 적합하다.

## 최소 구현 대상

| 파일 | 변경 여부 | 내용 |
|---|---:|---|
| `src/memory/store.py` | 수정 예정 | 새 내부 필드와 구조화 memory 메서드 추가 |
| `src/memory/models.py` | 수정 없음 | 1단계 모델 그대로 사용 |
| `src/memory/extractors.py` | 수정 없음 | 1단계 extractor 그대로 사용 |
| `src/main.py` | 수정 없음 | 아직 연결하지 않음 |
| `src/prompt/generator.py` | 수정 없음 | 아직 연결하지 않음 |
| `tests/test_memory_store_layers.py` | 새로 생성 가능 | 기존 API 보존과 새 메서드 동작 테스트 |

## 테스트 설계

| 테스트 | 확인 내용 |
|---|---|
| `test_existing_history_api_unchanged` | `add()`, `get_history()`, `get_cloud_context()`, `get_local_context()` 기존 반환 유지 |
| `test_add_structured_memory_uses_masked_text_only` | raw input 없이 masked summary/labels만 저장 |
| `test_get_memory_context_returns_layers` | `MemoryContext`에 recent/fact/directive/emotion 포함 |
| `test_fact_memory_merges_duplicates` | 같은 fact 반복 시 `evidence_count` 증가 |
| `test_directive_memory_merges_duplicates` | 같은 directive 반복 시 `hit_count` 증가 |
| `test_clear_session_clears_structured_memory` | 세션 삭제 시 새 레이어도 삭제 |

## 결론

2단계는 `MemoryStore`에 구조화 memory 레이어를 병렬로 추가하는 단계다. 기존 history/profile API는 그대로 유지하고, 새 메서드들은 별도 opt-in API로 제공한다.

핵심은 다음 세 가지다.

- 기존 API를 깨지 않는다.
- 원문 저장 없이 마스킹된 텍스트 기반 구조화 데이터만 저장한다.
- Safety Gateway와 Risk Audit은 memory보다 항상 우선한다.
