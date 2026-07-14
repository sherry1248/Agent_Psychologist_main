package com.psychologist.agent.ui.viewmodels

import androidx.lifecycle.ViewModel
import com.psychologist.agent.data.model.EmergencyContact
import com.psychologist.agent.data.repository.EmergencyContactRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

/**
 * 긴급 연락처 등록 화면의 입력값과 목록을 관리합니다.
 */
data class EmergencyContactsUiState(
    val name: String = "",
    val relation: String = "",
    val phoneNumber: String = "",
    val contacts: List<EmergencyContact> = emptyList(),
)

class EmergencyContactsViewModel(
    private val repository: EmergencyContactRepository,
) : ViewModel() {
    private val _uiState = MutableStateFlow(EmergencyContactsUiState(contacts = repository.contacts.value))
    val uiState: StateFlow<EmergencyContactsUiState> = _uiState

    fun onNameChange(value: String) {
        _uiState.update { it.copy(name = value) }
    }

    fun onRelationChange(value: String) {
        _uiState.update { it.copy(relation = value) }
    }

    fun onPhoneNumberChange(value: String) {
        _uiState.update { it.copy(phoneNumber = value) }
    }

    fun addContact() {
        val state = _uiState.value
        if (state.name.isBlank() || state.phoneNumber.isBlank()) return

        repository.addContact(
            EmergencyContact(
                name = state.name,
                relation = state.relation,
                phoneNumber = state.phoneNumber,
            )
        )

        _uiState.update {
            it.copy(
                name = "",
                relation = "",
                phoneNumber = "",
                contacts = repository.contacts.value,
            )
        }
    }

    fun deleteContact(id: String) {
        repository.removeContact(id)
        _uiState.update { it.copy(contacts = repository.contacts.value) }
    }
}
