package com.psychologist.agent.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.psychologist.agent.ui.viewmodels.EmotionCheckViewModel

/**
 * 오늘의 감정 체크 화면입니다.
 * 의학적 진단이 아니라, 사용자가 자기 상태를 간단히 점검하는 용도입니다.
 */
@Composable
fun EmotionCheckScreen(viewModel: EmotionCheckViewModel) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("오늘의 감정 체크", style = MaterialTheme.typography.headlineSmall)
        Text("이 결과는 의학적 진단이 아니라 자기 점검용 지표입니다.")

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("기분: ${state.mood}")
                Slider(value = state.mood.toFloat(), onValueChange = { viewModel.onMoodChange(it.toInt()) }, valueRange = 0f..10f)

                Text("불안: ${state.anxiety}")
                Slider(value = state.anxiety.toFloat(), onValueChange = { viewModel.onAnxietyChange(it.toInt()) }, valueRange = 0f..10f)

                Text("외로움: ${state.loneliness}")
                Slider(value = state.loneliness.toFloat(), onValueChange = { viewModel.onLonelinessChange(it.toInt()) }, valueRange = 0f..10f)

                Text("수면 시간: ${state.sleepHours}시간")
                Slider(value = state.sleepHours.toFloat(), onValueChange = { viewModel.onSleepHoursChange(it.toInt()) }, valueRange = 0f..12f)

                OutlinedTextField(
                    value = state.eatingStatus,
                    onValueChange = viewModel::onEatingStatusChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("식사 상태") },
                )

                Column {
                    Text("장기 감정 추적에 활용할 수 있도록 동의할까요?")
                    Text("동의한 경우에만 저장되며, 원하지 않으면 기록하지 않습니다.")
                    Checkbox(checked = state.consentToTrack, onCheckedChange = viewModel::onConsentChange)
                }
            }
        }

        Button(onClick = viewModel::save, modifier = Modifier.fillMaxWidth()) {
            Text("저장")
        }

        state.lastSavedMessage?.let { message ->
            Text(message)
        }
    }
}
