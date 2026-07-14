package com.psychologist.agent.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.psychologist.agent.ui.viewmodels.CrisisViewModel

/**
 * 위기 도움 화면입니다.
 * 앱이 자동으로 전화하거나 신고하지 않고, 버튼을 눌렀을 때만 전화 앱이 열립니다.
 */
@Composable
fun CrisisHelpScreen(viewModel: CrisisViewModel) {
    val context = LocalContext.current
    val actions = viewModel.crisisActions

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("위기 도움", style = MaterialTheme.typography.headlineSmall)
        Text("이 화면의 버튼은 사용자가 직접 누를 때만 연결됩니다.")

        actions.forEach { action ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(action.title, style = MaterialTheme.typography.titleMedium)
                    Text(action.description)

                    if (action.phoneNumber != null) {
                        Button(onClick = {
                            val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:${action.phoneNumber}"))
                            context.startActivity(intent)
                        }) {
                            Text("전화 걸기")
                        }
                    } else {
                        Text("긴급 연락처 화면에서 직접 등록한 사람에게 연락할 수 있습니다.")
                    }
                }
            }
        }
    }
}
