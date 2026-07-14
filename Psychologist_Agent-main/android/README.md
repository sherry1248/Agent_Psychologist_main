# Android 앱 구조

이 폴더는 Python 기반 심리 상담 AI Agent와 연결되는 Android 클라이언트의 기본 골격입니다.

## 설계 원칙

- Kotlin + Jetpack Compose 기준으로 작성합니다.
- MVVM 구조를 사용합니다.
- Python 백엔드와는 REST API로 연결합니다.
- 실제 서버가 없을 때는 Mock API로도 동작하도록 설계합니다.
- 위기 대응은 자동 전화나 자동 신고가 아니라, 사용자가 직접 누를 때만 연결되도록 합니다.

## 현재 포함된 핵심 구성

- 상담 채팅 화면
- 오늘의 감정 체크 화면
- 위기 도움 화면
- 긴급 연락처 등록 화면
- 개인정보 보호 설정 화면
- API Client / Repository / ViewModel 분리 구조
- 민감정보 마스킹 유틸리티

## 백엔드 연결 방식

기본 API 계약은 Python 쪽의 `POST /api/v1/chat` 응답을 기준으로 맞췄습니다.

- `response`: 상담 응답 텍스트
- `session_id`: 대화 세션 식별자
- `risk_level`: `low`, `moderate`, `high`, `critical`
- `requires_crisis_response`: 위기 안내가 필요한지 여부

Android 앱은 이 값을 읽어 일반 채팅 응답 또는 위기 안내 카드로 전환합니다.

## 실행 전 준비

1. Android Studio에서 `android/` 폴더를 열어 프로젝트로 인식시킵니다.
2. `app/build.gradle.kts`의 `baseUrl`을 로컬 Python 서버 주소로 맞춥니다.
3. 에뮬레이터에서 실행할 때는 보통 `http://10.0.2.2:8080/`을 사용합니다.

## 폴더 역할

- `data/model`: 앱에서 쓰는 데이터 구조
- `data/network`: Python API 호출 또는 Mock API
- `data/repository`: 데이터 접근과 저장 정책
- `data/security`: 민감정보 마스킹
- `ui/viewmodel`: 화면 상태와 이벤트 처리
- `ui/screens`: 실제 화면 컴포저블
- `ui/navigation`: 화면 이동 구조
- `ui/theme`: 색상과 테마
