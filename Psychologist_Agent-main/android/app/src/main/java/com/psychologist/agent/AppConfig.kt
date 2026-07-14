package com.psychologist.agent

/**
 * 앱 전체에서 공유하는 단순 설정값입니다.
 * Mock API 기본 동작을 명시해 두고, 실제 서버 연결 시 false로 바꾸면 됩니다.
 */
object AppConfig {
    const val USE_MOCK_API: Boolean = true
    const val DEFAULT_BASE_URL: String = "http://10.0.2.2:8080/"
}
