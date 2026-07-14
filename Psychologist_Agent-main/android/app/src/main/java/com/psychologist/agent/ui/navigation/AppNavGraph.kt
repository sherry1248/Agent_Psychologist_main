package com.psychologist.agent.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.psychologist.agent.PsychologistAgentApplication
import com.psychologist.agent.ui.screens.ChatScreen
import com.psychologist.agent.ui.screens.CrisisHelpScreen
import com.psychologist.agent.ui.screens.EmergencyContactsScreen
import com.psychologist.agent.ui.screens.EmotionCheckScreen
import com.psychologist.agent.ui.screens.PrivacySettingsScreen
import com.psychologist.agent.ui.viewmodel.AppViewModelFactory
import com.psychologist.agent.ui.viewmodels.ChatViewModel
import com.psychologist.agent.ui.viewmodels.CrisisViewModel
import com.psychologist.agent.ui.viewmodels.EmergencyContactsViewModel
import com.psychologist.agent.ui.viewmodels.EmotionCheckViewModel
import com.psychologist.agent.ui.viewmodels.PrivacyViewModel

private data class BottomNavItem(val route: String, val label: String)

private val bottomItems = listOf(
    BottomNavItem("chat", "상담"),
    BottomNavItem("checkin", "감정 체크"),
    BottomNavItem("crisis", "위기 도움"),
    BottomNavItem("contacts", "연락처"),
    BottomNavItem("privacy", "보호 설정"),
)

/**
 * 앱의 전체 화면 이동을 담당하는 네비게이션 그래프입니다.
 */
@Composable
fun AppNavGraph() {
    val context = LocalContext.current
    val application = context.applicationContext as PsychologistAgentApplication
    val container = remember { application.container }
    val factory = remember { AppViewModelFactory(container) }
    val navController = rememberNavController()

    Scaffold(
        bottomBar = { BottomBar(navController) }
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = "chat",
            modifier = Modifier.padding(padding)
        ) {
            composable("chat") {
                val viewModel: ChatViewModel = viewModel(factory = factory)
                ChatScreen(viewModel = viewModel)
            }
            composable("checkin") {
                val viewModel: EmotionCheckViewModel = viewModel(factory = factory)
                EmotionCheckScreen(viewModel = viewModel)
            }
            composable("crisis") {
                val viewModel: CrisisViewModel = viewModel(factory = factory)
                CrisisHelpScreen(viewModel = viewModel)
            }
            composable("contacts") {
                val viewModel: EmergencyContactsViewModel = viewModel(factory = factory)
                EmergencyContactsScreen(viewModel = viewModel)
            }
            composable("privacy") {
                val viewModel: PrivacyViewModel = viewModel(factory = factory)
                PrivacySettingsScreen(viewModel = viewModel)
            }
        }
    }
}

@Composable
private fun BottomBar(navController: NavHostController) {
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    NavigationBar {
        bottomItems.forEach { item ->
            NavigationBarItem(
                selected = currentRoute == item.route,
                onClick = {
                    navController.navigate(item.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                label = { Text(item.label) },
                icon = {}
            )
        }
    }
}