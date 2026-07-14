package com.psychologist.agent.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Color(0xFF0F766E),
    onPrimary = Color.White,
    secondary = Color(0xFF2F5D62),
    onSecondary = Color.White,
    tertiary = Color(0xFFB45309),
    background = Color(0xFFF7F7F2),
    surface = Color(0xFFFFFFFF),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF5EEAD4),
    onPrimary = Color(0xFF062925),
    secondary = Color(0xFF9BD3D0),
    onSecondary = Color(0xFF0D1B1E),
    tertiary = Color(0xFFFDBA74),
)

/**
 * 차분한 톤의 기본 테마입니다.
 * 상담 앱 특성상 자극적인 색보다 안정감을 주는 색 조합을 사용합니다.
 */
@Composable
fun PsychologistAgentTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        content = content,
    )
}
