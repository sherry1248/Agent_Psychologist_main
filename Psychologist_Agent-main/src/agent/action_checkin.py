"""Deterministic status detection for the previously suggested small action."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict


FAILURE_MARKERS = (
    "못 했어", "못했어", "안 했어", "안했어", "못하겠어",
    "귀찮았어", "시간이 없었어", "까먹었어", "하기 어려웠어",
)
PARTIAL_MARKERS = (
    "아직 다 못", "하다 말", "하긴 했는데",
)
COMPLETION_MARKERS = (
    "다 했어", "해냈어", "했어", "해봤어", "정리했어", "적어봤어",
    "끝냈어", "완료", "만들었어", "써봤어", "기록했어",
)
NUMBER_PATTERN = re.compile(r"(\d+)\s*(?:개|가지|번|회|줄|명|분)")
PARTIAL_ACTION_PATTERN = re.compile(
    r"(?:조금|일부(?:만)?|정도|반만).{0,8}"
    r"(?:했어|해봤어|정리했어|적어봤어|만들었어|써봤어|기록했어)"
)
EXPLICIT_COUNTED_COMPLETION_PATTERN = re.compile(
    r"\S+\s+\d+\s*개\s+(?:정리했어|끝냈어|완료했어)"
)


@dataclass(frozen=True)
class ActionCheckinResult:
    status: str = "none"
    detected: bool = False
    reason: str = "no_action_checkin_marker"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "detected": self.detected,
            "reason": self.reason,
        }


def _first_number(text: str) -> int | None:
    match = NUMBER_PATTERN.search(text)
    return int(match.group(1)) if match else None


def classify_action_checkin(
    user_input: str,
    has_previous_action: bool,
    previous_action_text: str = "",
) -> ActionCheckinResult:
    """Classify progress markers only when a previous structured action exists."""
    if not has_previous_action:
        return ActionCheckinResult(reason="no_previous_small_action")

    compact = " ".join((user_input or "").split())
    if any(marker in compact for marker in FAILURE_MARKERS):
        return ActionCheckinResult("not_completed", True, "failure_marker")
    if any(marker in compact for marker in PARTIAL_MARKERS) or PARTIAL_ACTION_PATTERN.search(compact):
        return ActionCheckinResult("partial", True, "partial_progress_marker")

    completed_count = _first_number(compact)
    target_count = _first_number(previous_action_text)
    if completed_count is not None and target_count is not None:
        status = "completed" if completed_count >= target_count else "partial"
        return ActionCheckinResult(status, True, "numeric_target_comparison")
    if completed_count is not None:
        if EXPLICIT_COUNTED_COMPLETION_PATTERN.search(compact):
            return ActionCheckinResult("completed", True, "explicit_counted_completion")
        return ActionCheckinResult("partial", True, "numeric_progress_without_target")
    if any(marker in compact for marker in COMPLETION_MARKERS):
        return ActionCheckinResult("completed", True, "completion_marker")
    return ActionCheckinResult()
