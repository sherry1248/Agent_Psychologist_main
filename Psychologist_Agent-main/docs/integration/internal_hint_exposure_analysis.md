# Internal Hint Exposure Analysis

## Goal

사용자에게는 자연스러운 상담 응답만 보이고, 내부 힌트/가이드/참고문은 최종 응답에 직접 노출되지 않게 한다.

현재 MOCK 모드 실행에서 다음 문구가 최종 응답 또는 데모 화면에 노출되고 있다.

- `상담 참고`
- `공감 참고`
- `웰니스 참고`
- `심리상담 데이터 기반 힌트`

## Findings

### 1. `src/main.py` MOCK 응답 생성 로직

직접 원인은 `src/main.py`의 `_compose_mock_response()`다.

현재 구조:

```python
def _compose_mock_response(
    self,
    counseling_hint: str,
    empathy_style_hint: str,
    wellness_hint: str,
) -> str:
    segments = ["지금의 상태를 함께 살펴보고 있어요."]
    if counseling_hint:
        segments.append(f"상담 참고: {counseling_hint}")
    if empathy_style_hint:
        segments.append(f"공감 참고: {empathy_style_hint}")
    if wellness_hint:
        segments.append(f"웰니스 참고: {wellness_hint}")
    return "\n\n".join(segments)
```

`mock_mode=True`일 때 `process_message()`는 cloud/local prompt 흐름으로 가지 않고 이 함수로 응답을 만든 뒤 바로 return한다.

```python
if self.mock_mode:
    response_text = self._compose_mock_response(
        counseling_recommendation.intervention_hint,
        empathy_recommendation.empathy_style_hint,
        wellness_recommendation.support_hint if wellness_recommendation else "",
    )
    result["response"] = self._add_safety_notice(response_text)
    ...
    return result
```

따라서 `상담 참고`, `공감 참고`, `웰니스 참고` 라벨이 최종 `result["response"]`에 그대로 들어간다.

### 2. `src/main.py` wellness hint 병합 로직

추가 노출 가능성은 `_merge_wellness_hint()`에도 있다.

```python
def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
    if not support_hint or support_hint in response_text:
        return response_text
    if not response_text:
        return support_hint
    return f"{response_text}\n\n웰니스 참고: {support_hint}"
```

현재 호출부는 mock mode 조건 안에 있다.

```python
if self.mock_mode and wellness_recommendation and wellness_recommendation.support_hint:
    response_text = self._merge_wellness_hint(response_text, wellness_recommendation.support_hint)
```

하지만 함수 자체는 최종 응답에 `웰니스 참고:` 라벨을 붙이는 구조이므로, 이후 호출 조건이 바뀌면 같은 문제가 재발할 수 있다.

### 3. `demo/app.py` 화면 구성

데모 화면은 agent 응답과 별개로 내부 힌트 카드를 직접 렌더링한다.

```python
def build_general_markdown(summary: Dict[str, Any], response_text: str) -> str:
    risk_stage = summary.get("risk_stage", "관심")
    return "\n\n".join(
        [
            wrap_card("상담 응답 카드", f"- 위험 단계: {escape_text(risk_stage)}"),
            wrap_card("AI 상담 응답", safe_body_text(response_text)),
            wrap_card("심리상담 데이터 기반 힌트", safe_body_text(summary.get("counseling_hint") or "없음")),
            wrap_card("공감형 대화 기반 힌트", safe_body_text(summary.get("empathy_style_hint") or "없음")),
            wrap_card("웰니스 기반 힌트", safe_body_text(summary.get("wellness_hint") or "없음")),
        ]
    )
```

즉, `result["response"]`에서 내부 힌트를 제거해도 데모 UI에는 다음 카드가 계속 보일 수 있다.

- `심리상담 데이터 기반 힌트`
- `공감형 대화 기반 힌트`
- `웰니스 기반 힌트`

`build_summary()`도 내부 힌트를 summary dict에 담고 있다.

```python
return {
    ...
    "counseling_hint": result.get("counseling_hint", ""),
    "empathy_style_hint": result.get("empathy_style_hint", ""),
    "wellness_hint": result.get("wellness_hint", "") or wellness.get("support_hint", ""),
    ...
}
```

현재 JSON output은 `visible=False`라 일반 사용자 화면에는 직접 보이지 않을 수 있지만, `build_general_markdown()`이 이 summary를 사용해 힌트 카드를 표시한다.

### 4. `src/prompt/generator.py`

`src/prompt/generator.py`는 직접 원인이 아니다.

Prompt generator는 cloud/local model에게 guidance와 context를 전달하는 역할이고, 최종 응답 문자열에 `상담 참고`, `공감 참고`, `웰니스 참고` 같은 라벨을 직접 붙이지 않는다.

다만 local prompt의 다음 항목은 모델 내부 지시로 사용된다.

- `therapeutic_guidance`
- `cloud_analysis`
- `rag_context`
- `memory_context`

실제 모델이 이 지시를 따라 “분석 요약” 같은 표현을 사용자에게 말할 가능성은 있지만, 현재 관찰된 정확한 문구는 `src/main.py`와 `demo/app.py`의 하드코딩된 문자열에서 나온다.

### 5. RAG / wellness hint 처리 흐름

RAG 자체가 최종 응답에 직접 붙는 구조는 아니다.

문제는 wellness hint가 다음 두 방식으로 사용자 화면에 노출되는 점이다.

- MOCK 응답: `_compose_mock_response()`가 `웰니스 참고: ...`로 직접 삽입
- Demo UI: `build_general_markdown()`이 `웰니스 기반 힌트` 카드로 직접 표시

## Files To Modify

최소 수정 대상:

1. `src/main.py`
2. `demo/app.py`
3. 관련 테스트

수정 가능성이 높은 테스트:

- `tests/test_multi_dataset_agent.py`
- `tests/test_wellness_recommender.py`

현재 일부 테스트가 내부 힌트 노출을 기대하고 있다.

```text
tests/test_multi_dataset_agent.py
- "상담 참고" in result["response"]

tests/test_wellness_recommender.py
- "웰니스 참고:" in result["response"]
```

## Minimal Fix Proposal

### `src/main.py`

`_compose_mock_response()`에서 내부 라벨을 제거한다.

안전한 최소안:

```python
def _compose_mock_response(
    self,
    counseling_hint: str,
    empathy_style_hint: str,
    wellness_hint: str,
) -> str:
    return (
        "지금 많이 부담스러웠을 수 있어요. "
        "우선 지금 느끼는 감정을 그대로 인정하고, "
        "오늘 할 수 있는 가장 작은 한 가지부터 정해보면 좋겠습니다."
    )
```

조금 더 힌트를 자연스럽게 반영하는 안:

```python
def _compose_mock_response(
    self,
    counseling_hint: str,
    empathy_style_hint: str,
    wellness_hint: str,
) -> str:
    segments = [
        "지금의 상태를 함께 살펴보면, 우선 지금 느끼는 부담을 인정하는 것부터 시작해도 괜찮아요."
    ]
    if counseling_hint:
        segments.append(counseling_hint)
    if empathy_style_hint:
        segments.append("지금처럼 느끼는 건 충분히 이해할 만한 반응일 수 있어요.")
    if wellness_hint:
        segments.append(wellness_hint)
    return " ".join(segments)
```

권장안은 첫 번째다. 내부 hint 문장을 그대로 사용자 응답에 넣지 않아 정보 누출 가능성이 가장 낮다.

`_merge_wellness_hint()`도 `웰니스 참고:` 라벨을 제거하거나, 최종 응답에 support hint를 직접 붙이지 않도록 바꾼다.

예:

```python
def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
    return response_text
```

또는 자연 문장으로만 병합한다.

```python
def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
    if not support_hint or support_hint in response_text:
        return response_text
    if not response_text:
        return support_hint
    return f"{response_text}\n\n{support_hint}"
```

### `demo/app.py`

`build_general_markdown()`에서 내부 힌트 카드 3개를 제거한다.

```python
def build_general_markdown(summary: Dict[str, Any], response_text: str) -> str:
    risk_stage = summary.get("risk_stage", "관심")
    return "\n\n".join(
        [
            wrap_card("상담 응답 카드", f"- 위험 단계: {escape_text(risk_stage)}"),
            wrap_card("AI 상담 응답", safe_body_text(response_text)),
        ]
    )
```

내부 hint 값은 필요하면 hidden JSON 또는 `pipeline_details`에만 유지한다.

## Test Update Proposal

기존 테스트는 내부 힌트가 응답에 노출되는 것을 기대하므로 수정해야 한다.

### `tests/test_multi_dataset_agent.py`

기존:

```python
assert "상담 참고" in result["response"]
```

변경:

```python
assert "상담 참고" not in result["response"]
assert "공감 참고" not in result["response"]
assert result["counseling_hint"]
assert result["empathy_style_hint"]
```

즉, 내부 힌트는 result metadata에는 남아도 최종 응답에는 노출되지 않아야 한다.

### `tests/test_wellness_recommender.py`

기존:

```python
assert "웰니스 참고:" in result["response"]
```

변경:

```python
assert "웰니스 참고:" not in result["response"]
assert result["wellness_hint"]
```

## Expected Outcome

수정 후 기대 동작:

- 사용자 최종 응답에는 자연스러운 상담 문장만 보인다.
- `상담 참고`, `공감 참고`, `웰니스 참고` 라벨은 `result["response"]`에 들어가지 않는다.
- 데모 화면에는 내부 힌트 카드가 표시되지 않는다.
- 내부 hint 값은 필요 시 metadata/debug 용도로만 유지된다.
