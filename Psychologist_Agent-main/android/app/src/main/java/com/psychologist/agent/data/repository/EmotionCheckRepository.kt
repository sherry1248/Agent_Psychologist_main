package com.psychologist.agent.data.repository

import com.psychologist.agent.data.model.EmotionCheckEntry
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * 오늘의 감정 체크 기록을 저장합니다.
 * 사용자가 동의한 경우에만 장기 추적을 남기도록 설계합니다.
 */
class EmotionCheckRepository(
    private val privacyRepository: PrivacyRepository,
) {
    private val _history = MutableStateFlow<List<EmotionCheckEntry>>(emptyList())
    val history: StateFlow<List<EmotionCheckEntry>> = _history

    fun save(entry: EmotionCheckEntry) {
        if (!entry.consentToTrack) return
        if (!privacyRepository.settings.value.saveHistory) return
        _history.value = _history.value + entry
    }
}
