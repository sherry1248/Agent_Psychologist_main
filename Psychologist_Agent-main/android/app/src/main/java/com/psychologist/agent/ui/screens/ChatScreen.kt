package com.psychologist.agent.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.psychologist.agent.data.model.MessageRole
import com.psychologist.agent.data.model.RiskStage
import com.psychologist.agent.ui.viewmodels.ChatViewModel

/**
 * 상담 채팅 화면입니다.
 * 사용자는 고민을 입력하고, 위험 단계가 높으면 위기 안내 카드가 먼저 노출됩니다.
 */
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("상담 채팅", style = MaterialTheme.typography.headlineSmall)
        Text("이 앱은 전문 진단 도구가 아니라 정서 지원용 대화 도우미입니다.")

        state.crisisCardText?.let { crisisText ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("위기 안내", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(text = crisisText)
                }
            }
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.55f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(state.messages) { message ->
                val title = when (message.role) {
                    MessageRole.USER -> "나"
                    MessageRole.ASSISTANT -> "AI"
                    MessageRole.SYSTEM -> "시스템"
                }

                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(title, style = MaterialTheme.typography.labelLarge)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(message.content)
                    }
                }
            }
        }

        if (state.riskStage == RiskStage.DANGER) {
            Text("현재 위험 단계가 높습니다. 위기 도움 화면을 바로 확인해 주세요.", color = MaterialTheme.colorScheme.error)
        }

        if (state.isSending) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                CircularProgressIndicator(modifier = Modifier.height(20.dp))
                Text("응답을 준비하고 있습니다...")
            }
        }

        OutlinedTextField(
            value = state.inputText,
            onValueChange = viewModel::onInputChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("고민을 입력하세요") },
        )

        Button(
            onClick = viewModel::sendMessage,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("보내기")
        }
    }
}
