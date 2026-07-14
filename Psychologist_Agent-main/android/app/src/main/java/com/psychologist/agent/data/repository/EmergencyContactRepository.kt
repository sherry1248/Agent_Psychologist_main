package com.psychologist.agent.data.repository

import com.psychologist.agent.data.model.EmergencyContact
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * 사용자가 직접 등록한 긴급 연락처를 관리합니다.
 * 자동 연락이 아니라, 위기 시 연락을 권유하는 용도로 사용합니다.
 */
class EmergencyContactRepository {
    private val _contacts = MutableStateFlow<List<EmergencyContact>>(emptyList())
    val contacts: StateFlow<List<EmergencyContact>> = _contacts

    fun addContact(contact: EmergencyContact) {
        _contacts.value = _contacts.value + contact
    }

    fun removeContact(contactId: String) {
        _contacts.value = _contacts.value.filterNot { it.id == contactId }
    }

    fun clear() {
        _contacts.value = emptyList()
    }
}
