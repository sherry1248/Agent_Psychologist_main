package com.psychologist.agent.ui.viewmodels

import androidx.lifecycle.ViewModel
import com.psychologist.agent.data.model.EmotionCheckEntry
import com.psychologist.agent.data.repository.EmotionCheckRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * 오늘의 감정 체크 화면 상태를 관리합니다.
 */
data class EmotionCheckUiState(
    val mood: Int = 5,
    val anxiety: Int = 5,
    val loneliness: Int = 5,
    val sleepHours: Int = 7,
    val eatingStatus: String = "보통",
    val consentToTrack: Boolean = false,
    val lastSavedMessage: String? = null,
)

class EmotionCheckViewModel(
    private val repository: EmotionCheckRepository,
) : ViewModel() {
    private val _uiState = MutableStateFlow(EmotionCheckUiState())
    val uiState: StateFlow<EmotionCheckUiState> = _uiState

    fun onMoodChange(value: Int) {
        _uiState.update { it.copy(mood = value) }
    }

    fun onAnxietyChange(value: Int) {
        _uiState.update { it.copy(anxiety = value) }
    }

    fun onLonelinessChange(value: Int) {
        _uiState.update { it.copy(loneliness = value) }
    }

    fun onSleepHoursChange(value: Int) {
        _uiState.update { it.copy(sleepHours = value) }
    }

    fun onEatingStatusChange(value: String) {
        _uiState.update { it.copy(eatingStatus = value) }
    }

    fun onConsentChange(value: Boolean) {
        _uiState.update { it.copy(consentToTrack = value) }
    }

    fun save() {
        val state = _uiState.value
        repository.save(
            EmotionCheckEntry(
                mood = state.mood,
                anxiety = state.anxiety,
                loneliness = state.loneliness,
                sleepHours = state.sleepHours,
                eatingStatus = state.eatingStatus,
                consentToTrack = state.consentToTrack,
            )
        )

        _uiState.update {
            it.copy(
                lastSavedMessage = if (state.consentToTrack) {
                    "기록을 저장했습니다. 이 값은 의학적 진단이 아니라 자기 점검용 지표입니다."
                } else {
                    "동의가 없어 장기 추적은 저장하지 않았습니다."
                }
            )
        }
    }
}
