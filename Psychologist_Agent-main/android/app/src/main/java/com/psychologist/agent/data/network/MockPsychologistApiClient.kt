package com.psychologist.agent.data.network

/**
 * 실제 Python 서버가 없어도 화면과 흐름을 확인할 수 있도록 해주는 Mock API입니다.
 * 위험 표현이 들어오면 위험 단계와 위기 안내 플래그를 함께 돌려줍니다.
 */
class MockPsychologistApiClient : PsychologistApiClient {
    override suspend fun sendChat(request: ChatApiRequest): ChatApiResponse {
        val lowerMessage = request.message.lowercase()
        val isDanger = lowerMessage.contains("죽") || lowerMessage.contains("자해") || lowerMessage.contains("끝내") || lowerMessage.contains("위험")
        val isCaution = lowerMessage.contains("불안") || lowerMessage.contains("우울") || lowerMessage.contains("외롭") || lowerMessage.contains("잠")

        return if (isDanger) {
            ChatApiResponse(
                response = "지금은 혼자 버티기보다 즉시 사람의 도움을 받는 것이 중요합니다. 아래 위기 도움 화면을 눌러 109, 119, 112 또는 가까운 사람에게 바로 연결하세요.",
                sessionId = request.sessionId ?: "mock-session-danger",
                riskLevel = "high",
                requiresCrisisResponse = true,
            )
        } else if (isCaution) {
            ChatApiResponse(
                response = "말해 주셔서 고맙습니다. 지금의 감정을 천천히 정리해 보면서, 오늘의 감정 체크도 함께 해 보면 좋겠습니다.",
                sessionId = request.sessionId ?: "mock-session-caution",
                riskLevel = "moderate",
                requiresCrisisResponse = false,
            )
        } else {
            ChatApiResponse(
                response = "지금 느끼는 상황을 조금 더 자세히 말씀해 주시면, 함께 정리해 보겠습니다.",
                sessionId = request.sessionId ?: "mock-session-attention",
                riskLevel = "low",
                requiresCrisisResponse = false,
            )
        }
    }

    override suspend fun getHealth(): HealthApiResponse {
        return HealthApiResponse(status = "healthy", version = "mock", mode = "MOCK")
    }
}
