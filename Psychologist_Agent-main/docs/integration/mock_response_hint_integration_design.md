# Mock Response Hint Integration Design

## Goal

현재 MOCK 모드 응답에서 내부 힌트가 사용자에게 그대로 노출된다.

노출되는 예:

- `상담 참고`
- `공감 참고`
- `웰니스 참고`
- `심리상담 데이터 기반 힌트`

목표는 내부 힌트를 단순히 제거하는 것이 아니라, 다음 세 힌트를 자연스러운 MOCK 상담 응답 생성에 반영하는 것이다.

- `counseling_hint`
- `empathy_style_hint`
- `wellness_hint`

최종 사용자 응답은 다음 흐름을 갖는다.

```text
감정 확인 -> 공감 -> 작은 실행 단계 제안 -> 안전 안내
```

내부 hint metadata는 `result`에 유지해도 되지만, 최종 사용자 응답과 기본 demo 화면에는 직접 노출하지 않는다.

## Current Problem

### `src/main.py`

`_compose_mock_response()`가 내부 hint를 라벨과 함께 최종 응답에 직접 붙인다.

```python
segments = ["지금의 상태를 함께 살펴보고 있어요."]
if counseling_hint:
    segments.append(f"상담 참고: {counseling_hint}")
if empathy_style_hint:
    segments.append(f"공감 참고: {empathy_style_hint}")
if wellness_hint:
    segments.append(f"웰니스 참고: {wellness_hint}")
return "\n\n".join(segments)
```

`_merge_wellness_hint()`도 내부 라벨을 붙이는 구조다.

```python
return f"{response_text}\n\n웰니스 참고: {support_hint}"
```

### `demo/app.py`

`build_general_markdown()`이 내부 hint 카드를 기본 사용자 화면에 직접 보여준다.

```python
wrap_card("심리상담 데이터 기반 힌트", ...)
wrap_card("공감형 대화 기반 힌트", ...)
wrap_card("웰니스 기반 힌트", ...)
```

## Minimal Design

수정 대상:

- `src/main.py`
- `demo/app.py`
- 관련 테스트

관련 테스트 예상:

- `tests/test_multi_dataset_agent.py`
- `tests/test_wellness_recommender.py`

## `src/main.py` Design

### `_compose_mock_response()`

내부 hint 라벨을 제거하고, hint를 응답 구성 재료로만 사용한다.

권장 구조:

```python
def _compose_mock_response(
    self,
    counseling_hint: str,
    empathy_style_hint: str,
    wellness_hint: str,
) -> str:
    parts = [
        "지금 느끼는 부담이 꽤 컸을 것 같아요.",
        "이런 상태에서는 마음이 복잡해지고, 무엇부터 해야 할지 막막하게 느껴질 수 있습니다.",
    ]

    action_candidates = [
        hint.strip()
        for hint in (counseling_hint, wellness_hint)
        if hint and hint.strip()
    ]

    if action_candidates:
        parts.append(action_candidates[0])
    else:
        parts.append("지금 당장 해결하려 하기보다, 오늘 할 수 있는 가장 작은 한 가지를 정해보세요.")

    parts.append(
        "만약 스스로를 해치고 싶은 생각이 들거나 지금 안전하지 않다고 느껴진다면, "
        "혼자 버티지 말고 109, 119, 112 또는 가까운 응급실/지역 정신건강복지센터에 바로 연결하세요."
    )

    return "\n\n".join(parts)
```

응답 역할:

- 첫 문장: 감정 확인
- 두 번째 문장: 공감
- 세 번째 문장: `counseling_hint` 또는 `wellness_hint` 기반 작은 실행 단계
- 마지막 문장: 안전 안내

`empathy_style_hint`는 그대로 노출하지 않고 공감 톤 선택에만 사용한다.

예:

```python
if empathy_style_hint:
    parts[1] = "지금의 반응은 이상하거나 약한 것이 아니라, 많이 버텨온 마음이 보내는 신호일 수 있어요."
```

### `_merge_wellness_hint()`

MOCK 응답에서 이미 `wellness_hint`가 `_compose_mock_response()`에 반영되므로 중복 삽입을 피한다.

권장안:

```python
def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
    return response_text
```

대안:

```python
def _merge_wellness_hint(self, response_text: str, support_hint: str) -> str:
    if not support_hint or support_hint in response_text:
        return response_text
    if not response_text:
        return support_hint
    return f"{response_text}\n\n{support_hint}"
```

권장안은 첫 번째다. 이유는 `_compose_mock_response()`가 이미 wellness hint를 입력으로 받기 때문이다.

## `demo/app.py` Design

### `build_general_markdown()`

기본 사용자 화면에서 내부 hint 카드를 제거한다.

현재 구조:

```python
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

제안 구조:

```python
return "\n\n".join(
    [
        wrap_card("상담 응답 카드", f"- 위험 단계: {escape_text(risk_stage)}"),
        wrap_card("AI 상담 응답", safe_body_text(response_text)),
    ]
)
```

`build_summary()`는 그대로 둘 수 있다.

이유:

- 내부 hint metadata는 hidden JSON/debug 용도로 유지 가능하다.
- 기본 사용자 화면에서는 자연스러운 상담 응답만 보여야 한다.

## Test Update Design

기존 테스트가 내부 hint 라벨 노출을 기대한다면 수정한다.

예상 변경:

```python
assert "상담 참고" not in result["response"]
assert "공감 참고" not in result["response"]
assert "웰니스 참고" not in result["response"]
```

metadata는 유지되어야 한다.

```python
assert result["counseling_hint"]
assert result["empathy_style_hint"]
assert result["wellness_hint"]
```

MOCK 응답 구조 검증:

```python
response = result["response"]
assert "109" in response or "119" in response or "112" in response
assert len(response) > 0
```

## Expected Outcome

수정 후 기대 동작:

- 최종 MOCK 응답에는 내부 라벨이 노출되지 않는다.
- `counseling_hint`, `empathy_style_hint`, `wellness_hint`는 응답 생성에 자연스럽게 반영된다.
- 응답은 `감정 확인 -> 공감 -> 작은 실행 단계 제안 -> 안전 안내` 흐름을 갖는다.
- demo 기본 화면에는 내부 hint 카드가 표시되지 않는다.
- 내부 hint metadata는 `result`나 hidden summary에는 유지할 수 있다.
