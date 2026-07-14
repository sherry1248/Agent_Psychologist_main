package com.psychologist.agent.data.repository

import com.psychologist.agent.data.model.PrivacySettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * 개인정보 보호 설정을 관리합니다.
 * 저장소는 우선 메모리 기반으로 두고, 이후 DataStore로 교체하기 쉽게 만들었습니다.
 */
class PrivacyRepository {
    private val _settings = MutableStateFlow(PrivacySettings())
    val settings: StateFlow<PrivacySettings> = _settings

    fun updateSettings(transform: (PrivacySettings) -> PrivacySettings) {
        _settings.value = transform(_settings.value)
    }

    fun clearAllData() {
        _settings.value = PrivacySettings()
    }
}
