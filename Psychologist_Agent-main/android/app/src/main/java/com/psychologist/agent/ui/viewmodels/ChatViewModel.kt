package com.psychologist.agent.ui.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.psychologist.agent.data.model.ChatMessage
import com.psychologist.agent.data.model.ChatResult
import com.psychologist.agent.data.model.MessageRole
import com.psychologist.agent.data.model.RiskStage
import com.psychologist.agent.data.repository.ChatRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * 상담 화면 상태와 전송 이벤트를 관리합니다.
 */
data class ChatUiState(
    val inputText: String = "",
    val sessionId: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val isSending: Boolean = false,
    val crisisCardText: String? = null,
    val riskStage: RiskStage = RiskStage.ATTENTION,
)

class ChatViewModel(
    private val chatRepository: ChatRepository,
) : ViewModel() {
    private val _uiState = MutableStateFlow(ChatUiState(messages = chatRepository.currentConversation()))
    val uiState: StateFlow<ChatUiState> = _uiState

    fun onInputChange(text: String) {
        _uiState.update { it.copy(inputText = text) }
    }

    fun sendMessage() {
        val message = _uiState.value.inputText.trim()
        if (message.isEmpty()) return

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isSending = true,
                    messages = it.messages + ChatMessage(role = MessageRole.USER, content = message),
                    inputText = "",
                    crisisCardText = null,
                )
            }

            val result = chatRepository.sendUserMessage(
                message = message,
                sessionId = _uiState.value.sessionId,
            )

            _uiState.update { current ->
                val newMessages = if (result.requiresCrisisResponse) {
                    current.messages
                } else {
                    current.messages + ChatMessage(
                        role = MessageRole.ASSISTANT,
                        content = result.responseText,
                    )
                }

                current.copy(
                    isSending = false,
                    sessionId = result.sessionId,
                    messages = newMessages,
                    riskStage = result.riskStage,
                    crisisCardText = if (result.requiresCrisisResponse) result.responseText else null,
                )
            }
        }
    }

    fun clearConversation() {
        chatRepository.clearConversation()
        _uiState.value = ChatUiState()
    }
}
