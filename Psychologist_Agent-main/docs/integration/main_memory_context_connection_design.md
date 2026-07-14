# Main Pipeline MemoryContext Connection Design

## Goal

`src/main.py`에서 `MemoryStore.get_memory_context(session_id)`를 호출해 `PromptGenerator.gen_cloud_prompt()`와 `PromptGenerator.gen_local_prompt()`에 전달한다.

단, crisis 상황에서는 memory 기반 개인화를 최소화하고 즉시 안전 확보와 전문 지원 연결을 우선한다.

## Proposed Pipeline Position

권장 위치는 SafetyGateway와 초기 Risk Audit 이후, cloud/local prompt 생성 전이다.

```text
1. SafetyGateway check
2. SafetyGateway crisis면 즉시 return
3. 초기 Risk Audit
4. 초기 Risk Audit crisis면 즉시 return
5. Counseling / empathy / wellness hints
6. PII redaction
7. RAG retrieval
8. Conversation history / user profile 조회
9. MemoryContext 조회
10. Cloud prompt 생성
11. Cloud analysis
12. Cloud 이후 Risk Audit
13. Cloud 이후 Risk Audit crisis면 즉시 return
14. Local prompt 생성
15. Response generation
16. Memory update
```

이 위치가 적절한 이유:

- SafetyGateway에서 crisis가 확정되면 LLM prompt 생성 자체를 건너뛰므로 memory context가 필요 없다.
- 초기 Risk Audit에서 crisis가 확정되는 경우에도 personalization보다 즉시 안전 응답이 우선이다.
- cloud/local prompt가 생성되기 직전에 memory context를 조회하면 최신 세션 memory를 반영할 수 있다.

## Normal Flow Integration

정상 non-crisis 경로에서만 memory context를 조회한다.

```python
try:
    memory_context = await self.memory_store.get_memory_context(session_id)
except Exception as e:
    logger.warning(f"Memory context unavailable: {e}")
    memory_context = None
```

Cloud prompt 생성에 전달한다.

```python
cloud_prompt = self.prompt_generator.gen_cloud_prompt(
    sanitized_input=sanitized_input,
    rag_context=rag_context,
    history=cloud_history,
    user_profile=user_profile.to_json() if user_profile else None,
    memory_context=memory_context,
)
```

Local prompt 생성에도 같은 context를 전달한다.

```python
local_prompt = self.prompt_generator.gen_local_prompt(
    user_input=user_input,
    cloud_analysis=cloud_analysis.to_dict(),
    rag_context=rag_context,
    history=local_history,
    therapeutic_guidance=wellness_recommendation.support_hint if wellness_recommendation else "",
    additional_context={
        "wellness_support_hint": wellness_recommendation.support_hint if wellness_recommendation else "",
        "wellness_risk_stage": wellness_recommendation.risk_stage if wellness_recommendation else "",
    } if wellness_recommendation else None,
    memory_context=memory_context,
)
```

## Crisis Personalization Minimization

Crisis 경로에서는 memory context를 조회하지 않거나, 조회했더라도 prompt에 넣지 않는다.

권장안은 조회하지 않는 것이다.

```text
SafetyGateway crisis
-> memory_context 조회 없음
-> safety_result.response 반환

초기 Risk Audit crisis
-> memory_context 조회 없음
-> crisis_handler response 반환

Cloud 이후 Risk Audit crisis
-> crisis response에는 memory_context 사용하지 않음
-> crisis_handler response 반환
```

Cloud 이후 Risk Audit의 경우 정상 흐름에서 이미 `memory_context`를 조회했을 수 있다. 그래도 crisis response 생성에는 사용하지 않는다.

현재 구조상 crisis 응답은 `crisis_handler.get_response()`를 통해 반환되므로, `PromptGenerator.gen_crisis_prompt()`나 local generation에 memory context를 전달하지 않으면 된다.

## Failure Handling

Memory context는 보조 feature이므로 조회 실패가 전체 상담 흐름을 중단하면 안 된다.

권장 fallback:

```python
try:
    memory_context = await self.memory_store.get_memory_context(session_id)
except Exception as e:
    logger.warning(f"Memory context unavailable: {e}")
    memory_context = None
```

이렇게 하면 memory layer 장애가 있어도 cloud/local prompt는 기존 방식으로 생성된다.

## Pipeline Details

디버깅 목적으로 `pipeline_details`에 raw memory 내용을 넣지 않고 count만 기록한다.

```python
if memory_context:
    result["pipeline_details"]["memory_context"] = {
        "recent_summaries": len(memory_context.recent_summaries),
        "facts": len(memory_context.facts),
        "directives": len([d for d in memory_context.directives if d.active]),
        "emotional_trend": len(memory_context.emotional_trend),
    }
else:
    result["pipeline_details"]["memory_context"] = {
        "available": False,
    }
```

주의:

- raw summary text를 `pipeline_details`에 넣지 않는다.
- fact value나 directive term을 그대로 넣지 않는다.
- emotional label도 필요하면 count만 남기고 raw prompt section은 기록하지 않는다.

## Test Plan

4단계 구현 시 권장 테스트:

- 정상 non-crisis 흐름에서 `get_memory_context(session_id)`가 호출된다.
- cloud prompt 생성에 `memory_context`가 전달된다.
- local prompt 생성에 `memory_context`가 전달된다.
- SafetyGateway crisis 경로에서는 `get_memory_context()`가 호출되지 않는다.
- 초기 Risk Audit crisis 경로에서도 `get_memory_context()`가 호출되지 않는다.
- memory 조회 실패 시 `memory_context=None`으로 fallback되고 응답 흐름은 계속된다.

## Expected Outcome

이 설계는 `PromptGenerator`의 optional `memory_context` 지원을 실제 pipeline에 연결한다.

정상 상담 흐름에서는 structured memory가 cloud/local prompt에 반영되고, crisis 흐름에서는 memory 개인화를 줄여 안전 응답과 전문 지원 연결을 우선한다.
