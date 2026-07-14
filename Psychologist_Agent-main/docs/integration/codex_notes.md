# Psychologist Agent 실행 정리

파일은 수정하지 않고 프로젝트 구조를 읽은 기준으로 정리한 내용입니다.

## 프로젝트 구조

- `src/main.py`: 상담 Agent 전체 파이프라인 진입점
- `demo/app.py`: Gradio 데모 UI
- `src/api/routes.py`: Android/외부 클라이언트용 FastAPI, `/api/v1/chat`
- `src/inference/server.py`: 로컬 GGUF 모델 추론 서버, `/generate`, `/chat`
- `android/`: Kotlin/Compose Android 앱
- `requirements.txt`: Python 의존성

## 리눅스 실행 순서

```bash
cd /home/juhyeon/AI_projects_backup/Agent_Psychologist_main/Psychologist_Agent-main

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

MOCK 모드로 먼저 실행하는 것이 가장 안전합니다. `.env`를 자동으로 읽는 코드는 대부분에 없어서, 셸 환경변수로 지정하는 편이 확실합니다.

```bash
export LLM_TYPE=MOCK
```

## CLI 예제 실행

```bash
python -m src.main
```

README의 `python src/main.py`보다 `python -m src.main`을 권장합니다. 코드가 `from src...` 절대 임포트를 쓰기 때문입니다.

## Gradio 데모 실행

```bash
python -m demo.app
```

브라우저에서 보통 다음 주소로 접속합니다.

```text
http://localhost:7860
```

## Android/외부 클라이언트용 FastAPI 실행

Android 앱이나 외부 클라이언트를 붙일 FastAPI 서버는 `src.api.routes:create_app` 쪽입니다.

```bash
LLM_TYPE=MOCK uvicorn "src.api.routes:create_app" --factory --host 0.0.0.0 --port 8080
```

헬스 체크:

```bash
curl http://localhost:8080/api/v1/health
```

채팅 API 테스트:

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"오늘 너무 지쳤어요.","wellness_checkin":{"mood_score":4,"anxiety_score":6,"sleep_quality":3}}'
```

주의: `src/inference/server.py`는 앱 API 서버가 아니라 로컬 모델 추론 서버입니다. 이 서버는 `/generate`, `/chat` 형태의 저수준 추론 API를 제공합니다.

## 테스트 실행

```bash
LLM_TYPE=MOCK pytest -q
```

## Android 실행

```bash
cd android
./gradlew assembleDebug
```

Android Studio에서는 `android/` 폴더를 열면 됩니다. 에뮬레이터에서 Python API 서버에 붙을 때 기본 주소는 `http://10.0.2.2:8080/`입니다.

현재 `android/app/src/main/java/com/psychologist/agent/AppConfig.kt`에서 `USE_MOCK_API`가 `true`라 실제 백엔드 연결 전에는 앱 내부 Mock API로 동작합니다.

## 주의점

- `models/` 폴더와 GGUF 모델 파일은 현재 저장소에 없습니다.
- `LLM_TYPE=LOCAL`로 실제 로컬 모델을 쓰려면 `LOCAL_MODEL_PATH` 또는 기본 경로 `models/psychologist-8b-q4_k_m.gguf`에 모델이 필요합니다.
- `requirements.txt`는 `torch`, `vllm`, `llama-cpp-python`, `easyocr`, `sentence-transformers` 등 무거운 패키지를 포함합니다.
- 처음 설치와 첫 실행은 시간이 걸릴 수 있습니다.
