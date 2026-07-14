# Dataset Usage Policy

이 프로젝트는 데이터셋을 역할별로 분리해서 사용합니다.

- counseling dataset: 상담 사례와 개입 힌트 참고용입니다.
- empathy dialogue dataset: 공감 말투, 감정 라벨, 공감 라벨 참고용입니다.
- wellness dataset: 체크인 점수 기반 support_hint 추천용입니다.
- child-adolescent counseling dataset: 위기 단계 매핑과 Safety Gateway 보조 규칙 참고용입니다.

운영 원칙은 다음과 같습니다.

- 사용자 대화 원문은 데이터셋 저장이나 모델 재학습에 직접 사용하지 않습니다.
- 위험 판단은 항상 Safety Gateway가 최우선입니다.
- 위험 단계에서는 어떤 retriever/recommender보다 crisis_response가 먼저 반환되어야 합니다.
- MOCK API는 유지하며, 웹 데모에서 먼저 동작 확인할 수 있어야 합니다.

샘플 데이터는 `data/raw/`의 `*.jsonl` 파일을 우선 사용하고, 실제 데이터가 준비되면 `data/processed/`로 확장할 수 있습니다.
