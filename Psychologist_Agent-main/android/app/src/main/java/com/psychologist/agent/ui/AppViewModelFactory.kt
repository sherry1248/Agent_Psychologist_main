package com.psychologist.agent.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.psychologist.agent.AppContainer
import com.psychologist.agent.ui.viewmodels.ChatViewModel
import com.psychologist.agent.ui.viewmodels.CrisisViewModel
import com.psychologist.agent.ui.viewmodels.EmergencyContactsViewModel
import com.psychologist.agent.ui.viewmodels.EmotionCheckViewModel
import com.psychologist.agent.ui.viewmodels.PrivacyViewModel

/**
 * Hilt 없이 ViewModel을 생성하기 위한 단순 팩토리입니다.
 */
class AppViewModelFactory(
    private val container: AppContainer,
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(ChatViewModel::class.java) -> ChatViewModel(container.chatRepository) as T
            modelClass.isAssignableFrom(EmotionCheckViewModel::class.java) -> EmotionCheckViewModel(container.emotionCheckRepository) as T
            modelClass.isAssignableFrom(CrisisViewModel::class.java) -> CrisisViewModel() as T
            modelClass.isAssignableFrom(EmergencyContactsViewModel::class.java) -> EmergencyContactsViewModel(container.emergencyContactRepository) as T
            modelClass.isAssignableFrom(PrivacyViewModel::class.java) -> PrivacyViewModel(container.privacyRepository, container.chatRepository) as T
            else -> throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
        }
    }
}
