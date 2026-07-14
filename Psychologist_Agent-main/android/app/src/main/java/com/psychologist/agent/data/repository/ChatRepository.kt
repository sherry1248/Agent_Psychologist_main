package com.psychologist.agent.data.repository

import com.psychologist.agent.data.model.ChatMessage
import com.psychologist.agent.data.model.ChatResult
import com.psychologist.agent.data.model.MessageRole
import com.psychologist.agent.data.model.RiskStage
import com.psychologist.agent.data.network.ChatApiRequest
import com.psychologist.agent.data.network.PsychologistApiClient
import com.psychologist.agent.data.security.SensitiveDataMasker

/**
 * 채팅 화면이 사용할 상담 데이터 접근 계층입니다.
 * 서버 호출 전 민감정보를 마스킹하고, 저장 금지 모드에서는 기록을 남기지 않습니다.
 */
class ChatRepository(
    private val apiClient: PsychologistApiClient,
    private val privacyRepository: PrivacyRepository,
) {
    private val conversation = mutableListOf<ChatMessage>()

    fun currentConversation(): List<ChatMessage> = conversation.toList()

    suspend fun sendUserMessage(message: String, sessionId: String?): ChatResult {
        val privacySettings = privacyRepository.settings.value
        val outgoingMessage = if (privacySettings.autoMaskSensitiveInfo) {
            SensitiveDataMasker.mask(message)
        } else {
            message
        }

        if (privacySettings.saveHistory) {
            conversation += ChatMessage(role = MessageRole.USER, content = outgoingMessage)
        }

        val response = apiClient.sendChat(
            ChatApiRequest(
                message = outgoingMessage,
                sessionId = sessionId,
            )
        )

        if (privacySettings.saveHistory) {
            conversation += ChatMessage(
                role = MessageRole.ASSISTANT,
                content = response.response,
                isCrisisMessage = response.requiresCrisisResponse,
            )
        }

        return ChatResult(
            responseText = response.response,
            sessionId = response.sessionId,
            riskStage = RiskStage.fromApiValue(response.riskLevel),
            requiresCrisisResponse = response.requiresCrisisResponse,
        )
    }

    fun clearConversation() {
        conversation.clear()
    }
}
