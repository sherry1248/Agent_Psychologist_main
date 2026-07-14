package com.psychologist.agent.data.security

/**
 * 서버로 보내기 전에 민감정보를 최대한 가리는 간단한 마스킹 유틸리티입니다.
 * 완벽한 보안 도구는 아니므로, 실제 운영 전에는 더 정교한 규칙을 추가해야 합니다.
 */
object SensitiveDataMasker {
    private val emailPattern = Regex("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
    private val phonePattern = Regex("(01[016789])[-. ]?\\d{3,4}[-. ]?\\d{4}")
    private val residentPattern = Regex("\\b\\d{6}[- ]?\\d{7}\\b")
    private val longNumberPattern = Regex("\\b\\d{4,}\\b")

    fun mask(text: String): String {
        return text
            .replace(emailPattern, "[이메일]")
            .replace(phonePattern, "[전화번호]")
            .replace(residentPattern, "[주민번호]")
            .replace(longNumberPattern, "[숫자]")
    }
}
