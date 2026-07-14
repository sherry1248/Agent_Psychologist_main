package com.psychologist.agent.ui.viewmodels

import androidx.lifecycle.ViewModel
import com.psychologist.agent.data.model.PrivacySettings
import com.psychologist.agent.data.repository.ChatRepository
import com.psychologist.agent.data.repository.PrivacyRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * 개인정보 보호 설정과 저장 삭제 동작을 관리합니다.
 */
data class PrivacyUiState(
    val saveHistory: Boolean = false,
    val lockEnabled: Boolean = false,
    val pinCode: String = "",
    val allowRiskNotifications: Boolean = false,
    val autoMaskSensitiveInfo: Boolean = true,
    val deleteMessage: String? = null,
)

class PrivacyViewModel(
    private val privacyRepository: PrivacyRepository,
    private val chatRepository: ChatRepository,
) : ViewModel() {
    private val _uiState = MutableStateFlow(PrivacyUiState())
    val uiState: StateFlow<PrivacyUiState> = _uiState

    init {
        load()
    }

    fun load() {
        val settings = privacyRepository.settings.value
        _uiState.value = PrivacyUiState(
            saveHistory = settings.saveHistory,
            lockEnabled = settings.lockEnabled,
            pinCode = settings.pinCode,
            allowRiskNotifications = settings.allowRiskNotifications,
            autoMaskSensitiveInfo = settings.autoMaskSensitiveInfo,
        )
    }

    fun onSaveHistoryChange(value: Boolean) {
        privacyRepository.updateSettings { it.copy(saveHistory = value) }
        _uiState.update { it.copy(saveHistory = value) }
    }

    fun onLockEnabledChange(value: Boolean) {
        privacyRepository.updateSettings { it.copy(lockEnabled = value) }
        _uiState.update { it.copy(lockEnabled = value) }
    }

    fun onPinCodeChange(value: String) {
        privacyRepository.updateSettings { it.copy(pinCode = value) }
        _uiState.update { it.copy(pinCode = value) }
    }

    fun onAllowRiskNotificationsChange(value: Boolean) {
        privacyRepository.updateSettings { it.copy(allowRiskNotifications = value) }
        _uiState.update { it.copy(allowRiskNotifications = value) }
    }

    fun onAutoMaskSensitiveInfoChange(value: Boolean) {
        privacyRepository.updateSettings { it.copy(autoMaskSensitiveInfo = value) }
        _uiState.update { it.copy(autoMaskSensitiveInfo = value) }
    }

    fun deleteLocalData() {
        privacyRepository.clearAllData()
        chatRepository.clearConversation()
        _uiState.update {
            it.copy(
                saveHistory = false,
                lockEnabled = false,
                pinCode = "",
                allowRiskNotifications = false,
                autoMaskSensitiveInfo = true,
                deleteMessage = "로컬 대화 기록과 설정을 삭제했습니다."
            )
        }
    }
}
