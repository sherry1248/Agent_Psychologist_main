package com.psychologist.agent

import android.content.Context
import com.psychologist.agent.data.network.MockPsychologistApiClient
import com.psychologist.agent.data.network.PsychologistApiClient
import com.psychologist.agent.data.network.RetrofitPsychologistApiClient
import com.psychologist.agent.data.repository.ChatRepository
import com.psychologist.agent.data.repository.EmergencyContactRepository
import com.psychologist.agent.data.repository.EmotionCheckRepository
import com.psychologist.agent.data.repository.PrivacyRepository

/**
 * 앱에서 사용할 Repository와 API Client를 한곳에 모아 둔 간단한 조립 지점입니다.
 * 실제 배포 전에는 Mock API를 끄고 Retrofit API Client를 쓰면 됩니다.
 */
class AppContainer private constructor(
    val apiClient: PsychologistApiClient,
    val privacyRepository: PrivacyRepository,
    val chatRepository: ChatRepository,
    val emotionCheckRepository: EmotionCheckRepository,
    val emergencyContactRepository: EmergencyContactRepository,
) {
    companion object {
        fun create(context: Context): AppContainer {
            val privacyRepository = PrivacyRepository()
            val apiClient = if (AppConfig.USE_MOCK_API) {
                MockPsychologistApiClient()
            } else {
                RetrofitPsychologistApiClient(AppConfig.DEFAULT_BASE_URL)
            }

            val chatRepository = ChatRepository(apiClient, privacyRepository)

            return AppContainer(
                apiClient = apiClient,
                privacyRepository = privacyRepository,
                chatRepository = chatRepository,
                emotionCheckRepository = EmotionCheckRepository(privacyRepository),
                emergencyContactRepository = EmergencyContactRepository(),
            )
        }
    }
}
