package com.psychologist.agent.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.psychologist.agent.ui.viewmodels.PrivacyViewModel

/**
 * 개인정보 보호 기능을 모아둔 화면입니다.
 * 대화 기록 저장 안 함, 앱 잠금, 수동 삭제, 민감정보 마스킹을 제어합니다.
 */
@Composable
fun PrivacySettingsScreen(viewModel: PrivacyViewModel) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("개인정보 보호")

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                RowItem(title = "대화 기록 저장 안 함", checked = state.saveHistory, onCheckedChange = viewModel::onSaveHistoryChange)
                RowItem(title = "앱 잠금 사용", checked = state.lockEnabled, onCheckedChange = viewModel::onLockEnabledChange)
                RowItem(title = "위험 알림 동의", checked = state.allowRiskNotifications, onCheckedChange = viewModel::onAllowRiskNotificationsChange)
                RowItem(title = "민감정보 자동 마스킹", checked = state.autoMaskSensitiveInfo, onCheckedChange = viewModel::onAutoMaskSensitiveInfoChange)

                OutlinedTextField(
                    value = state.pinCode,
                    onValueChange = viewModel::onPinCodeChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("PIN 코드") },
                )
            }
        }

        Button(onClick = viewModel::deleteLocalData, modifier = Modifier.fillMaxWidth()) {
            Text("기록 삭제")
        }

        state.deleteMessage?.let { Text(it) }
    }
}

@Composable
private fun RowItem(title: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    androidx.compose.foundation.layout.Row(modifier = Modifier.fillMaxWidth()) {
        Text(title, modifier = Modifier.fillMaxWidth(0.85f))
        Checkbox(checked = checked, onCheckedChange = onCheckedChange)
    }
}
