package com.psychologist.agent.ui

import androidx.compose.runtime.Composable
import com.psychologist.agent.ui.navigation.AppNavGraph

/**
 * 앱의 메인 진입점입니다.
 * 실제 화면 이동은 ui/navigation/AppNavGraph.kt에서 관리합니다.
 */
@Composable
fun PsychologistAgentApp() {
    AppNavGraph()
}
