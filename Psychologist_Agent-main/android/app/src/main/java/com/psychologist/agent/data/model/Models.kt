package com.psychologist.agent.data.model

import java.util.UUID

/**
 * 앱 전반에서 공유하는 핵심 데이터 모델입니다.
 * 화면 상태와 서버 응답을 변환하는 기준이 됩니다.
 */

enum class RiskStage(val label: String) {
    ATTENTION("관심"),
    CAUTION("주의"),
    DANGER("위험");

    companion object {
        fun fromApiValue(value: String?): RiskStage {
            val normalized = value.orEmpty().trim().lowercase()
            return when {
                normalized.contains("critical") || normalized.contains("high") -> DANGER
                normalized.contains("moderate") || normalized.contains("medium") -> CAUTION
                else -> ATTENTION
            }
        }
    }
}

enum class MessageRole {
    USER,
    ASSISTANT,
    SYSTEM,
}

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String,
    val isCrisisMessage: Boolean = false,
)

data class ChatResult(
    val responseText: String,
    val sessionId: String,
    val riskStage: RiskStage,
    val requiresCrisisResponse: Boolean,
)

data class EmotionCheckEntry(
    val mood: Int,
    val anxiety: Int,
    val loneliness: Int,
    val sleepHours: Int,
    val eatingStatus: String,
    val consentToTrack: Boolean,
    val recordedAtMillis: Long = System.currentTimeMillis(),
)

data class EmergencyContact(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val relation: String,
    val phoneNumber: String,
)

data class PrivacySettings(
    val saveHistory: Boolean = false,
    val lockEnabled: Boolean = false,
    val pinCode: String = "",
    val allowRiskNotifications: Boolean = false,
    val autoMaskSensitiveInfo: Boolean = true,
)

data class CrisisAction(
    val title: String,
    val description: String,
    val phoneNumber: String? = null,
)
