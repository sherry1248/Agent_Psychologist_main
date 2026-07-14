package com.psychologist.agent.ui.viewmodels

import androidx.lifecycle.ViewModel
import com.psychologist.agent.data.model.CrisisAction

/**
 * 위기 도움 화면에 보여 줄 안내 카드 목록을 제공합니다.
 */
class CrisisViewModel : ViewModel() {
    val crisisActions: List<CrisisAction> = listOf(
        CrisisAction(
            title = "109 자살예방상담전화",
            description = "자살 위기나 극심한 고통이 있을 때 상담 연결을 시도합니다.",
            phoneNumber = "109",
        ),
        CrisisAction(
            title = "119 긴급 구조",
            description = "생명 위험이나 즉각적인 응급상황일 때 사용합니다.",
            phoneNumber = "119",
        ),
        CrisisAction(
            title = "112 경찰 신고",
            description = "안전이 위협받는 상황에서 직접 누르면 전화 앱이 열립니다.",
            phoneNumber = "112",
        ),
        CrisisAction(
            title = "가까운 사람에게 연락하기",
            description = "가족, 친구, 상담센터에 직접 연락해 도움을 요청합니다.",
        ),
        CrisisAction(
            title = "긴급 연락처 보기",
            description = "사용자가 등록한 연락처를 확인하고 직접 연결할 수 있습니다.",
        ),
    )
}
