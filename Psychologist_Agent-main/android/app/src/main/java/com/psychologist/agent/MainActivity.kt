package com.psychologist.agent

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.psychologist.agent.ui.PsychologistAgentApp
import com.psychologist.agent.ui.theme.PsychologistAgentTheme

/**
 * 앱의 시작 화면입니다.
 * 여기서는 Compose UI를 띄우고, 이후 화면들은 Navigation으로 이동합니다.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PsychologistAgentTheme {
                PsychologistAgentApp()
            }
        }
    }
}
