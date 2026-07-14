package com.psychologist.agent.data.network

import com.google.gson.annotations.SerializedName

/**
 * Python 백엔드와 주고받는 REST API DTO입니다.
 * 필드 이름은 현재 FastAPI 응답 형식과 맞췄습니다.
 */

data class ChatApiRequest(
    @SerializedName("message") val message: String,
    @SerializedName("session_id") val sessionId: String? = null,
)

data class ChatApiResponse(
    @SerializedName("response") val response: String,
    @SerializedName("session_id") val sessionId: String,
    @SerializedName("risk_level") val riskLevel: String,
    @SerializedName("requires_crisis_response") val requiresCrisisResponse: Boolean,
)

data class HealthApiResponse(
    @SerializedName("status") val status: String,
    @SerializedName("version") val version: String? = null,
    @SerializedName("mode") val mode: String? = null,
)
