package com.psychologist.agent.data.network

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

/**
 * API Client는 네트워크 호출의 진입점입니다.
 * 실제 앱에서는 Retrofit 구현체를, 오프라인에서는 Mock 구현체를 사용합니다.
 */

interface PsychologistApiClient {
    suspend fun sendChat(request: ChatApiRequest): ChatApiResponse

    suspend fun getHealth(): HealthApiResponse
}

interface PsychologistApiService {
    @POST("api/v1/chat")
    suspend fun sendChat(@Body request: ChatApiRequest): ChatApiResponse

    @GET("api/v1/health")
    suspend fun getHealth(): HealthApiResponse
}
