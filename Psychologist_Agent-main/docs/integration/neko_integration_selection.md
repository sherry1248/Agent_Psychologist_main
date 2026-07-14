# N.E.K.O 기능 통합 선별안

이 문서는 `N.E.K.O-main`의 README와 프로젝트 구조를 기준으로, `Psychologist_Agent-main`에 통합할 만한 기능만 선별한 것이다. 목적은 심리상담 AI 에이전트에 기억 시스템, 감정 상태 추적, 프롬프트 구조를 추가하는 것이다.

다음 기능은 명시적으로 제외한다.

- Live2D, VRM, MMD, Avatar 감정 매핑
- Steam, Workshop, UGC 관련 구조
- Docker 배포 구조
- frontend, plugin-manager, react-neko-chat
- plugin SDK, plugin market, plugin server
- Computer Use, CUA, browser automation, OpenClaw, OpenFang
- 외부 활동 감지, 화면 이해, 소셜/음악/게임 기반 proactive 대화 시작
- 별도 memory server 구조 전체
- 캐릭터 roleplay/persona 강화 기능

## 통합 추천 기능

### 1. 다층 메모리 개념

N.E.K.O의 recent / fact / reflection / persona 계층은 상담 에이전트에 맞게 축소 적용할 가치가 있다.

`Psychologist_Agent-main`에는 이미 `src/memory/store.py`에 요약 메모리와 `UserProfile`이 있으므로, 전체 메모리 서버를 이식하지 말고 기존 `MemoryStore`를 확장하는 방식이 적합하다.

추천 구조:

- Recent memory: 최근 상담 턴 요약 유지
- Fact memory: 사용자가 반복 언급한 생활 맥락, 스트레스 요인, 선호 지원 방식 저장
- Reflection memory: 여러 fact를 묶어 반복 패턴으로 추론하되 확정 표현은 피함
- Profile memory: 진단명이 아니라 상담 선호, 조심해야 할 주제, 반복 정서 패턴 중심으로 저장

참고 대상:

- `N.E.K.O-main/memory/recent.py`
- `N.E.K.O-main/memory/facts.py`
- `N.E.K.O-main/memory/reflection.py`
- `N.E.K.O-main/memory/persona.py`
- `N.E.K.O-main/docs/architecture/memory-system.md`

### 2. 사용자 지시와 금지 주제 기억

`memory/user_directives.py`는 상담 도메인에 특히 유용하다. 사용자가 명시한 경계와 선호를 다음 세션에서도 존중할 수 있다.

예시:

- "그 얘기 다시 꺼내지 마"
- "가족 얘기는 하기 싫어"
- "조언보다 들어주기만 해줘"

통합 방향:

- 장기 프로필과 별도 저장
- TTL이 있는 directive memory로 관리
- 원문 전체가 아니라 `kind`, `term`, `created_at`, `expire_at`, `hit_count` 같은 구조화 데이터만 저장
- 프롬프트에는 활성 directive만 짧게 삽입

참고 대상:

- `N.E.K.O-main/memory/user_directives.py`
- `N.E.K.O-main/config/prompts/prompts_memory.py`

### 3. 감정 상태 추적

N.E.K.O의 감정 분석은 Avatar 표정용이지만, 핵심 아이디어는 상담 에이전트에 맞게 바꿔 쓸 수 있다.

N.E.K.O의 `happy`, `sad`, `angry`, `surprised`, `neutral` 분류를 그대로 쓰기보다 심리상담 목적의 상태 라벨로 재설계하는 것이 낫다.

추천 감정 상태:

- anxiety
- sadness
- loneliness
- anger
- overwhelm
- numbness
- hope
- crisis_signal

통합 방향:

- `risk_stage`와 별도의 `EmotionalState`로 관리
- wellness checkin 값과 사용자 메시지 기반 감정 추정을 결합
- 최근 정서 흐름을 cloud/local prompt에 짧게 삽입
- 위기 신호는 항상 기존 Safety Gateway와 Risk Audit이 우선

참고 대상:

- `N.E.K.O-main/config/prompts/prompts_emotion.py`
- `N.E.K.O-main/main_logic/activity/llm_enrichment.py`

### 4. 프롬프트 구조화

N.E.K.O의 프롬프트 파일 분리 방식은 참고할 만하다. 다만 `Psychologist_Agent-main`에는 이미 `prompts/prompt_templates.yaml`, `src/prompt/templates.py`, `src/prompt/generator.py`가 있으므로 YAML 중심 구조를 유지하는 것이 적합하다.

추가 추천 섹션:

- `[Safety Boundary]`
- `[Recent Summary Memory]`
- `[Long-term Support Profile]`
- `[Emotional State Trend]`
- `[User Directives / Avoid Topics]`
- `[RAG Clinical Knowledge]`
- `[Cloud Analysis]`
- `[Response Rules]`

통합 방향:

- `PromptGenerator`에서 memory context를 별도 dict로 받아 렌더링
- cloud prompt에는 장기 프로필, 최근 요약, 감정 흐름을 포함
- local prompt에는 짧은 최근 맥락, directive, 응답 톤 힌트를 포함
- 위기 응답 prompt에는 memory를 최소화하고 안전 안내를 우선

참고 대상:

- `N.E.K.O-main/config/prompts/prompts_memory.py`
- `N.E.K.O-main/config/prompts/prompts_emotion.py`
- `N.E.K.O-main/docs/design/llm-prompt-budget.md`

### 5. 프롬프트 예산과 압축 정책

상담 에이전트는 안전 문구, RAG, 메모리, 대화 히스토리가 함께 들어가므로 입력 컨텍스트가 쉽게 커질 수 있다. N.E.K.O의 prompt budget 설계는 축소 적용할 가치가 있다.

추천 정책:

- 최근 대화: local 3턴, cloud 10턴 유지
- 장기 메모리: 관련도 높은 top-k 요약만 삽입
- 감정 추적: 최근 5개 상태만 삽입
- user directives: 활성 항목만 삽입
- RAG: 기존 `max_rag_context_length` 유지 또는 명시적 토큰 cap 추가
- 단일 사용자 입력이 길 경우 앞/뒤를 보존하는 truncation 고려

참고 대상:

- `N.E.K.O-main/docs/design/llm-prompt-budget.md`

## 통합하지 말아야 할 기능

다음 항목은 심리상담 AI 에이전트의 목적과 맞지 않거나, 요청 범위에서 제외된 기능이다.

- Live2D / VRM / MMD / Avatar 관련 파일과 감정-애니메이션 매핑
- Steam, Workshop, UGC 업로드/공유 기능
- Docker 관련 배포 파일
- 모든 frontend 앱과 정적 UI 자산
- plugin SDK, plugin marketplace, plugin server
- Computer Use, CUA, browser automation, task executor
- OpenClaw, OpenFang, browser_use_adapter
- proactive companion의 화면/앱/OS 활동 감지
- 소셜 미디어, 음악, 게임, 밈 기반 대화 시작 기능
- N.E.K.O의 별도 `memory_server.py` 프로세스 구조

## 권장 구현 순서

1. `src/memory/store.py` 확장
   - `EmotionalState`, `UserDirective`, `MemoryLayer` 데이터 구조 추가
   - 원문 저장 없이 PII 마스킹 후 구조화 요약만 저장

2. `src/prompt/generator.py` 확장
   - 감정 상태, 장기 메모리, directive 섹션을 prompt context로 삽입
   - cloud/local prompt별로 포함 범위를 다르게 제한

3. `prompts/prompt_templates.yaml` 수정
   - 상담용 구조화 섹션 추가
   - Safety Boundary와 User Directives를 명시

4. `src/main.py` 연결
   - Safety Gateway 이후, cloud/local prompt 생성 전에 memory context 구성
   - 위기 상황에서는 memory 기반 개인화보다 안전 응답 우선

5. 테스트 추가
   - PII 원문 미저장
   - 위기 상황에서 memory보다 safety 우선
   - 금지 주제 directive 반영
   - 감정 상태 누적과 prompt 삽입
   - 장기 메모리 top-k 제한

## N.E.K.O 관련 파일별 참고 구조

아래 표는 `N.E.K.O-main/memory`, `N.E.K.O-main/config/prompts`, `N.E.K.O-main/main_logic`의 파일을 기준으로 정리한 것이다. `가져오기` 판단은 코드 전체 이식이 아니라 `Psychologist_Agent-main`에 참고하거나 축소 구현할 가치가 있는지를 뜻한다.

### memory

| 파일 | 역할 | 가져오기 | 적용 방향 |
|---|---|---:|---|
| `memory/__init__.py` | memory 하위 모듈의 공통 규칙과 LLM 호출 정책 설명 | 부분 | 모델 tier, 하드코딩 금지, LLM 설정 분리 원칙만 참고 |
| `memory/recent.py` | 최근 대화 저장, 요약, 압축, review fingerprint 관리 | 예 | 기존 `MemoryStore`의 최근 요약/압축 로직 강화 |
| `memory/facts.py` | 대화에서 atomic fact 추출, 중복 제거, 중요도 관리 | 예 | 상담 맥락 fact 추출: 스트레스 요인, 선호 지원, 반복 이슈 |
| `memory/reflection.py` | 여러 fact를 상위 insight/reflection으로 합성 | 예 | 단정적 진단이 아닌 “반복 경향” 요약으로 축소 적용 |
| `memory/persona.py` | 장기 persona/profile 정보 관리, mention suppress, contradiction 처리 | 부분 | 캐릭터 persona가 아니라 사용자 지원 프로필로 재설계 |
| `memory/user_directives.py` | 사용자가 명시한 금지 주제/요청을 TTL로 저장 | 예 | 상담 경계, 피하고 싶은 주제, 응답 선호 저장 |
| `memory/anti_repeat.py` | 최근 AI 출력 기반 반복 주제 감지, BM25 힌트 생성 | 부분 | 상담 응답의 반복 조언 방지용 soft hint로만 참고 |
| `memory/hybrid_recall.py` | BM25 + embedding + RRF 기반 hybrid memory recall | 부분 | 장기 메모리 top-k 검색 설계 참고, 초기 구현은 단순 검색 권장 |
| `memory/recall.py` | vector coarse rank + LLM rerank 기반 memory recall | 부분 | 고도화 단계에서만 참고. 초기에는 과함 |
| `memory/embeddings.py` | 로컬 embedding service와 vector 계산 | 부분 | 기존 RAG/embedding 구조와 충돌 여부 확인 후 참고 |
| `memory/embeddings_fallback.py` | embedding 비활성/실패 시 no-op fallback | 부분 | 장애 시 메모리 검색을 degrade하는 패턴 참고 |
| `memory/embedding_worker.py` | 백그라운드 embedding warmup/backfill | 아니오 | 현재 규모에는 과함. 별도 worker 도입 불필요 |
| `memory/timeindex.py` | 시간 인덱스 기반 원문/압축 메모리 저장 | 부분 | “최근/과거/특정 시점” 조회가 필요해질 때 참고 |
| `memory/temporal.py` | fact/reflection의 시간 범위, 과거화, 시간 감쇠 처리 | 예 | 감정 상태와 스트레스 요인의 최신성 판단에 유용 |
| `memory/evidence.py` | evidence score, 강화/감쇠 점수 계산 | 부분 | memory confidence/reinforcement 개념만 차용 |
| `memory/evidence_handlers.py` | evidence 이벤트 처리와 상태 갱신 | 부분 | 명시적 확인/부정 feedback 처리 구조 참고 |
| `memory/evidence_analytics.py` | evidence 상태 분석/통계 | 아니오 | 상담 MVP에는 불필요 |
| `memory/fact_dedup.py` | fact 중복/유사 항목 병합 | 예 | 장기 메모리 오염 방지를 위해 축소 구현 권장 |
| `memory/refine.py` | persona/reflection cluster refine, LLM 결의 | 아니오 | 복잡도 높음. 나중에 운영 데이터가 쌓인 뒤 검토 |
| `memory/settings.py` | legacy settings 파일 접근 | 아니오 | N.E.K.O 캐릭터 설정용이라 상담 프로젝트와 맞지 않음 |
| `memory/stop_names.py` | 이름/별칭 제거 후 keyword/BM25 처리 | 부분 | 개인정보/사용자명 제거 후 키워드 추출하는 방식 참고 |
| `memory/event_log.py` | memory 이벤트 로그 저장 | 부분 | audit와 분리된 memory event log가 필요할 때 참고 |
| `memory/outbox.py` | 백그라운드 memory task 영속 큐 | 아니오 | 현재 단일 프로세스 상담 파이프라인에는 과함 |
| `memory/cursors.py` | background 처리 위치 cursor 관리 | 아니오 | outbox/worker를 도입하지 않으면 불필요 |
| `memory/archive_shards.py` | 오래된 memory archive shard 관리 | 아니오 | 장기 운영/대용량 이전에는 불필요 |
| `memory/store/.gitkeep` | 빈 저장소 디렉터리 유지 | 아니오 | 구조 참고 대상 아님 |

### config/prompts

| 파일 | 역할 | 가져오기 | 적용 방향 |
|---|---|---:|---|
| `config/prompts/__init__.py` | prompt 패키지 초기화 | 아니오 | 참고 필요 없음 |
| `config/prompts/prompts_memory.py` | 요약, fact 추출, reflection, persona, review 관련 prompt | 예 | 상담용 memory extraction/summarization prompt로 재작성 |
| `config/prompts/prompts_emotion.py` | 감정 분석 prompt와 다국어 emotion keyword | 예 | 상담용 감정 라벨 체계로 바꿔 사용 |
| `config/prompts/prompts_directives.py` | 사용자 금지/요청 directive 추출 prompt | 예 | “피하고 싶은 주제”, “응답 방식 선호” 추출에 직접 참고 |
| `config/prompts/prompts_response.py` | 응답 후처리/형식 관련 prompt | 부분 | 상담 응답의 톤, 안전 공지, 짧은 답변 규칙에 참고 |
| `config/prompts/prompts_sys.py` | 시스템 prompt 공통 조각, 다국어 locale helper, agent capability 문구 | 부분 | `_loc` 같은 locale fallback 패턴만 참고 |
| `config/prompts/prompts_activity.py` | 사용자 활동 상태와 proactive prompt용 문구 | 부분 | 활동 감지는 제외. 감정/상태 context section 구조만 참고 |
| `config/prompts/prompts_proactive.py` | proactive 대화 후보/생성 prompt | 아니오 | 상담 에이전트 목적과 다름 |
| `config/prompts/prompts_agent.py` | agent/tool 실행 관련 prompt | 아니오 | Computer Use/도구 실행 제외 |
| `config/prompts/prompts_avatar_interaction.py` | Avatar 상호작용 prompt | 아니오 | Live2D/Avatar 제외 |
| `config/prompts/prompts_chara.py` | 캐릭터 성격/역할 prompt | 아니오 | 상담 에이전트에는 캐릭터 roleplay 강화 불필요 |
| `config/prompts/prompts_voice.py` | 음성/TTS 관련 prompt | 아니오 | 현재 목적과 무관 |
| `config/prompts/prompts_card_assist.py` | 카드/보조 UI 관련 prompt | 아니오 | 프론트엔드/UI 제외 |
| `config/prompts/prompts_game.py` | 게임 관련 prompt | 아니오 | 게임 기능 제외 |
| `config/prompts/prompts_game_route.py` | 게임 라우팅 prompt | 아니오 | 게임 기능 제외 |
| `config/prompts/prompts_galgame.py` | galgame/스토리 prompt | 아니오 | 상담 목적과 무관 |

### main_logic

| 파일 | 역할 | 가져오기 | 적용 방향 |
|---|---|---:|---|
| `main_logic/core.py` | N.E.K.O 전체 대화 오케스트레이션, TTS, 화면/프론트 연동 포함 | 부분 | 전체 이식 금지. pipeline orchestration 순서만 참고 |
| `main_logic/omni_offline_client.py` | 일반 chat completion 기반 대화 클라이언트 | 부분 | provider 호출 추상화, 반복 응답 방지 일부만 참고 |
| `main_logic/omni_realtime_client.py` | realtime 음성/멀티모달 클라이언트 | 아니오 | 음성/프론트/실시간 Avatar 흐름 제외 |
| `main_logic/session_state.py` | 사용자 입력과 proactive 응답의 turn owner 상태기계 | 부분 | 상담 세션의 turn/race 관리가 필요할 때 참고 |
| `main_logic/lifecycle_bus.py` | 세션 내부 lifecycle event pub/sub | 부분 | 나중에 memory/risk/audit 이벤트 분리 시 참고 |
| `main_logic/cross_server.py` | main server에서 memory/monitor/commenter 서버로 메시지 전달 | 아니오 | 별도 서버 구조 제외 |
| `main_logic/proactive_delivery.py` | proactive 메시지 전달과 선점 처리 | 아니오 | proactive companion 제외 |
| `main_logic/tool_calling.py` | provider-agnostic tool call schema/registry | 아니오 | 도구 실행 제외 |
| `main_logic/agent_bridge.py` | agent server로 analyze/plan 이벤트 발행 | 아니오 | Computer Use/agent server 제외 |
| `main_logic/agent_event_bus.py` | main server와 agent server 간 ZMQ 이벤트 버스 | 아니오 | 별도 agent server 제외 |
| `main_logic/tts_client.py` | TTS 합성 및 음성 출력 | 아니오 | TTS 목적 아님 |
| `main_logic/mirror_meta.py` | 외부 컨트롤러가 만든 메시지를 chat history에 반영하는 schema | 아니오 | 게임/플러그인/외부 컨트롤러 흐름 제외 |
| `main_logic/__init__.py` | 패키지 초기화 | 아니오 | 참고 필요 없음 |

### main_logic/activity

| 파일 | 역할 | 가져오기 | 적용 방향 |
|---|---|---:|---|
| `main_logic/activity/llm_enrichment.py` | 최근 대화 기반 활동/열린 주제 LLM 보강 | 부분 | OS 활동 추정은 제외. 최근 대화에서 “미해결 감정 주제” 추출 방식만 참고 |
| `main_logic/activity/snapshot.py` | 활동 상태 snapshot 타입과 prompt 렌더링 | 부분 | 감정 상태 snapshot 타입 설계에 참고 |
| `main_logic/activity/state_machine.py` | OS/대화/음성 신호 기반 활동 상태 규칙 엔진 | 아니오 | 화면/OS 활동 추정 제외 |
| `main_logic/activity/system_signals.py` | Windows idle, CPU, foreground window 수집 | 아니오 | Computer/OS 감지 제외 |
| `main_logic/activity/tracker.py` | 사용자 활동 tracker와 snapshot 생성 | 아니오 | proactive 활동 추적 제외 |
| `main_logic/activity/__init__.py` | activity 패키지 초기화 | 아니오 | 참고 필요 없음 |

## Psychologist_Agent-main에 맞춘 구조 제안

N.E.K.O의 구조를 그대로 복사하지 말고, 현재 프로젝트의 경계에 맞춰 다음처럼 흡수하는 것이 적합하다.

| 목표 | Psychologist 쪽 위치 | 참고할 N.E.K.O 구조 |
|---|---|---|
| 최근 대화 요약/압축 | `src/memory/store.py` | `memory/recent.py`, `prompts_memory.py` |
| 장기 fact 저장 | `src/memory/store.py` 또는 `src/memory/facts.py` 신설 | `memory/facts.py`, `fact_dedup.py` |
| 반복 정서 패턴 | `src/memory/store.py` 또는 `src/memory/reflection.py` 신설 | `memory/reflection.py`, `temporal.py` |
| 상담 지원 프로필 | 기존 `UserProfile` 확장 | `memory/persona.py` 일부 |
| 사용자 금지 주제/선호 | `src/memory/directives.py` 신설 | `memory/user_directives.py`, `prompts_directives.py` |
| 감정 상태 추적 | `src/memory/emotion_state.py` 신설 또는 `MemoryStore` 내부 | `prompts_emotion.py`, `activity/snapshot.py` |
| prompt context 조립 | `src/prompt/generator.py` | `prompts_memory.py`, `prompts_sys.py` |
| prompt 예산 관리 | `src/prompt/generator.py`, `PromptConfig` | `docs/design/llm-prompt-budget.md` |
| 세션 이벤트 분리 | 필요 시 `src/session/manager.py` 확장 | `main_logic/lifecycle_bus.py`, `session_state.py` |

## 결론

N.E.K.O에서 가져올 것은 동반자 플랫폼 기능 전체가 아니라, 다음 네 가지 설계 요소다.

- 메모리 계층화
- 명시적 사용자 지시 기억
- 감정 상태 요약과 추적
- 프롬프트 예산 및 섹션화

이 네 가지는 `Psychologist_Agent-main`의 기존 안전 중심 파이프라인과 잘 맞는다. 반면 Avatar, Steam, Docker, frontend, plugin, Computer Use 계열은 상담 에이전트의 핵심 목적과 맞지 않으므로 통합 대상에서 제외한다.
