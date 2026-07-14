package com.psychologist.agent.data.network

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

/**
 * 실제 Python 서버와 통신하는 Retrofit 기반 구현체입니다.
 * 기본 뼈대만 두고, 모킹이 어려운 구간은 Mock API로 먼저 개발할 수 있게 설계합니다.
 */
class RetrofitPsychologistApiClient(
    baseUrl: String,
) : PsychologistApiClient {
    private val service: PsychologistApiService

    init {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }

        val client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        service = retrofit.create(PsychologistApiService::class.java)
    }

    override suspend fun sendChat(request: ChatApiRequest): ChatApiResponse {
        return service.sendChat(request)
    }

    override suspend fun getHealth(): HealthApiResponse {
        return service.getHealth()
    }
}
