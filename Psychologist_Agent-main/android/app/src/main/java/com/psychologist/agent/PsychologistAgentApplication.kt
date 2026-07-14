package com.psychologist.agent

import android.app.Application

/**
 * 앱 전역에서 필요한 의존성을 준비하는 Application 클래스입니다.
 * 초보자용 골격이므로 Hilt 대신 수동 주입 구조를 사용합니다.
 */
class PsychologistAgentApplication : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer.create(this)
    }
}
