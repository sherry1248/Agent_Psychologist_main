package com.psychologist.agent.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.psychologist.agent.ui.viewmodels.EmergencyContactsViewModel

/**
 * 사용자가 가족, 친구, 상담센터 연락처를 직접 등록하는 화면입니다.
 */
@Composable
fun EmergencyContactsScreen(viewModel: EmergencyContactsViewModel) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("긴급 연락처 등록")

        OutlinedTextField(
            value = state.name,
            onValueChange = viewModel::onNameChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("이름") },
        )

        OutlinedTextField(
            value = state.relation,
            onValueChange = viewModel::onRelationChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("관계") },
        )

        OutlinedTextField(
            value = state.phoneNumber,
            onValueChange = viewModel::onPhoneNumberChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("전화번호") },
        )

        Button(onClick = viewModel::addContact, modifier = Modifier.fillMaxWidth()) {
            Text("연락처 추가")
        }

        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(state.contacts) { contact ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.fillMaxWidth(0.75f)) {
                            Text(contact.name)
                            Text(contact.relation)
                            Text(contact.phoneNumber)
                        }
                        Button(onClick = { viewModel.deleteContact(contact.id) }) {
                            Text("삭제")
                        }
                    }
                }
            }
        }
    }
}
