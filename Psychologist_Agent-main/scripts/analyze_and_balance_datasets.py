#!/usr/bin/env python3
"""Analyze processed counseling datasets and build safe balanced hint records.

The script reads existing JSON/JSONL files from data/processed by default,
does not modify raw or processed data, and writes derived evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_processing_report.md"
DEFAULT_MAX_SAMPLES_PER_LABEL = 120
DEFAULT_PROFILE_CONFIGS = {
    "small": {
        "max_samples_per_label": 250,
        "target_min_records": 3000,
        "target_max_records": 3500,
    },
    "balanced": {
        "max_samples_per_label": 400,
        "target_min_records": 5000,
        "target_max_records": 6000,
    },
    "broad": {
        "max_samples_per_label": 500,
        "target_min_records": 7000,
        "target_max_records": 8500,
    },
}
DEFAULT_SCENARIO_MINIMUM = 120
DEFAULT_LIVE_PROFILE = "broad"
PROFILE_OUTPUT_NAMES = {
    "small": "balanced_safe_hints_small.jsonl",
    "balanced": "balanced_safe_hints_balanced.jsonl",
    "broad": "balanced_safe_hints_broad.jsonl",
}
DEFAULT_LIVE_OUTPUT_NAME = "balanced_safe_hints.jsonl"
SAFE_SUMMARY_MAX_CHARS = 140

TEXT_FIELDS = (
    "user_input",
    "question",
    "prompt",
    "input",
    "message",
    "counselor_response",
    "answer",
    "response",
    "support_hint",
    "intervention_hint",
    "empathy_style_hint",
)
LABEL_FIELDS = (
    "category",
    "topic",
    "label",
    "label_or_topic",
    "emotion_label",
    "empathy_label",
    "intent",
    "risk_stage",
)
SOURCE_FIELDS = ("source", "source_dataset", "dataset")
SAFETY_TERMS = (
    "자살",
    "죽고",
    "죽어",
    "자해",
    "위험",
    "crisis",
    "suicide",
    "self-harm",
)
COVERAGE_CHECKS = {
    "sleep": ("sleep", "수면", "잠", "불면"),
    "anxiety": ("anxiety", "불안"),
    "stress": ("stress", "스트레스"),
    "professor/teacher criticism": (
        "professor",
        "teacher",
        "교수",
        "선생",
        "비판",
        "비난",
        "혼났",
    ),
    "exam/assignment pressure": ("exam", "assignment", "시험", "과제"),
    "sadness/crying": ("sadness", "crying", "cry", "슬픔", "우울", "눈물", "울"),
    "anger/frustration": ("anger", "frustration", "분노", "짜증", "화"),
    "self-blame": ("self-blame", "self blame", "자책", "내 탓"),
    "relationship hurt": ("relationship", "친구", "연인", "관계", "상처"),
    "crisis/safety": ("crisis", "suicide", "self-harm", "자살", "자해", "위험"),
}
SCENARIO_BUCKETS = {
    "academic_pressure": ("공부", "시험", "과제", "성적", "기말", "중간", "마감"),
    "criticism_scolding": ("교수", "선생님", "상사", "혼났", "지적", "꾸중", "욕먹"),
    "sleep_problem": ("잠", "수면", "불면", "자주 깨", "숙면"),
    "anxiety_stress": ("불안", "스트레스", "압박", "긴장", "부담"),
    "anger_frustration": ("짜증", "화나", "억울", "분노", "답답"),
    "sadness_crying": ("울고 싶", "눈물", "속상", "슬픔"),
    "self_blame": ("한심", "내 탓", "못난", "자책", "죄책감", "자존감"),
    "relationship_hurt": ("친구", "관계", "서운", "무시", "연락", "외로움"),
    "low_energy": ("무기력", "의욕", "피로", "지침", "활력"),
    "recovery_improvement": ("괜찮아졌", "나아졌", "좋아졌", "회복"),
    "crisis_safety": ("죽고 싶", "자해", "극단", "살기 싫"),
}
SCENARIO_MINIMUMS = {
    scenario: DEFAULT_SCENARIO_MINIMUM
    for scenario in SCENARIO_BUCKETS
}
SCENARIO_MINIMUMS["crisis_safety"] = 30
SOURCE_MAX_SHARE = 0.55


def discover_dataset_files(processed_dir: Path) -> List[Path]:
    if not processed_dir.exists():
        return []
    return sorted(
        path
        for path in processed_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    yield {"_invalid_record": True}
                    continue
                if isinstance(value, dict):
                    yield value
                else:
                    yield {"_invalid_record": True}
        return

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        yield {"_invalid_record": True}
        return

    if isinstance(value, list):
        for item in value:
            yield item if isinstance(item, dict) else {"_invalid_record": True}
    elif isinstance(value, dict):
        records = value.get("records") or value.get("data") or value.get("items")
        if isinstance(records, list):
            for item in records:
                yield item if isinstance(item, dict) else {"_invalid_record": True}
        else:
            yield value
    else:
        yield {"_invalid_record": True}


def clean_text(value: Any) -> str:
    text = " ".join(str(value or "").replace("\ufeff", " ").split()).strip()
    text = re.sub(r"\[[A-Z_]+_REDACTED\]", "", text)
    return " ".join(text.split()).strip()


def record_text(record: Dict[str, Any]) -> str:
    parts = [clean_text(record.get(field)) for field in TEXT_FIELDS]
    return " ".join(part for part in parts if part)


def normalize_text_for_duplicate(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[^\w가-힣]", "", lowered)
    return lowered


def source_name(path: Path, record: Dict[str, Any]) -> str:
    for field in SOURCE_FIELDS:
        value = clean_text(record.get(field))
        if value:
            return value
    return path.stem


def matched_scenarios_from_text(text: str) -> List[str]:
    lowered = clean_text(text).lower()
    return [
        scenario
        for scenario, keywords in SCENARIO_BUCKETS.items()
        if any(keyword.lower() in lowered for keyword in keywords)
    ]


def label_or_topic(record: Dict[str, Any]) -> str:
    source = clean_text(record.get("source") or record.get("source_dataset") or record.get("dataset")).lower()
    emotion = clean_text(record.get("emotion_label"))
    empathy = clean_text(record.get("empathy_label"))
    topic = clean_text(record.get("topic"))
    category = clean_text(record.get("category"))

    if source == "empathy" and emotion and empathy:
        return f"{emotion}/{empathy}"
    if topic:
        return topic
    if category:
        scenarios = matched_scenarios_from_text(record_text(record))
        if scenarios:
            return f"{category}/{scenarios[0]}"
        return category

    for field in LABEL_FIELDS:
        value = clean_text(record.get(field))
        if value:
            return value
    return "UNLABELED"


def is_valid_record(record: Dict[str, Any]) -> bool:
    if record.get("_invalid_record"):
        return False
    return bool(record_text(record)) and label_or_topic(record) != "UNLABELED"


def truncate_summary(text: str, max_chars: int = SAFE_SUMMARY_MAX_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    boundary = max(text.rfind(".", 0, max_chars), text.rfind("?", 0, max_chars), text.rfind("!", 0, max_chars))
    if boundary >= max_chars // 2:
        return text[: boundary + 1].strip()
    return text[:max_chars].rstrip() + "..."


def infer_intent_hint(record: Dict[str, Any], label: str) -> str:
    topic = label.lower()
    if any(token in topic for token in ("불안", "anxiety", "stress", "스트레스")):
        return "불안이나 스트레스 완화를 원하는 정서 지원 요청으로 해석한다."
    if any(token in topic for token in ("depression", "우울", "슬픔", "무기력")):
        return "저하된 기분을 표현하며 공감과 안정화가 필요한 요청으로 해석한다."
    if any(token in topic for token in ("sleep", "수면", "잠")):
        return "수면 문제와 생활 리듬 부담을 다루는 지원 요청으로 해석한다."
    if clean_text(record.get("empathy_label")):
        return "감정 반영과 공감 표현을 우선하는 대화로 해석한다."
    return "상담 맥락을 확인하고 감정, 원인, 다음 행동을 단계적으로 탐색한다."


def infer_emotion_hint(record: Dict[str, Any], label: str) -> str:
    emotion = clean_text(record.get("emotion_label"))
    if emotion:
        return f"표면 감정은 {emotion}이며, 먼저 판단 없이 반영한다."
    topic = label.lower()
    if any(token in topic for token in ("불안", "anxiety")):
        return "불안과 긴장을 먼저 인정하고 안정감을 높이는 표현을 사용한다."
    if any(token in topic for token in ("depression", "우울", "슬픔")):
        return "우울감이나 슬픔을 축소하지 않고 지지적으로 반영한다."
    if any(token in topic for token in ("분노", "anger")):
        return "분노를 안전하게 표현하도록 돕고 즉각적인 판단을 피한다."
    return "명시 감정이 불분명하면 현재 느낌을 확인하는 질문으로 시작한다."


def infer_cause_hint(record: Dict[str, Any], label: str) -> str:
    text = record_text(record)
    topic = label.lower()
    if any(token in text for token in ("잠", "수면", "피곤", "불면")) or "sleep" in topic:
        return "수면, 피로, 생활 리듬을 원인 후보로 점검한다."
    if any(token in text for token in ("회사", "일", "공부", "시험", "과제")):
        return "일, 학업, 성취 압박을 원인 후보로 점검한다."
    if any(token in text for token in ("친구", "가족", "엄마", "아빠", "연인", "관계")):
        return "대인관계나 가족 맥락을 원인 후보로 점검한다."
    return "원인을 단정하지 않고 최근 변화, 반복 패턴, 부담 요인을 확인한다."


def infer_action_hint(record: Dict[str, Any], label: str) -> str:
    for field in ("intervention_hint", "support_hint", "empathy_style_hint"):
        value = truncate_summary(record.get(field), 90)
        if value:
            return value
    if "주의" in label or "위험" in label:
        return "일반 조언보다 안전 확인과 도움 연결을 우선한다."
    return "감정 확인 뒤 오늘 가능한 작은 행동 하나만 제안한다."


def infer_safety_hint(record: Dict[str, Any], label: str) -> str:
    text = f"{record_text(record)} {label}".lower()
    if any(term in text for term in SAFETY_TERMS) or "위험" in label:
        return "위험 신호 가능성이 있으므로 Safety/Risk Agent가 일반 상담보다 먼저 평가한다."
    if "주의" in label:
        return "주의 단계로 간주하고 악화 신호를 확인하며 전문 도움 연결을 열어둔다."
    return "명시적 위기 신호는 낮지만 안전 질문은 필요 시 즉시 우선한다."


def infer_short_summary(record: Dict[str, Any], label: str) -> str:
    source = source_name(Path("unknown_dataset"), record) or "unknown_dataset"
    if clean_text(record.get("risk_stage")):
        return f"{source} 데이터의 {label} 패턴을 안전 우선 상담 힌트로 요약한 기록이다."
    if clean_text(record.get("emotion_label")):
        return f"{source} 데이터의 {label} 감정 표현을 공감 반응 힌트로 요약한 기록이다."
    return f"{source} 데이터의 {label} 상담 패턴을 원인 탐색과 작은 행동 힌트로 요약한 기록이다."


def safe_hint_record(path: Path, record: Dict[str, Any]) -> Dict[str, str]:
    label = label_or_topic(record)
    return {
        "source_dataset": source_name(path, record),
        "label_or_topic": label,
        "intent_hint": infer_intent_hint(record, label),
        "emotion_hint": infer_emotion_hint(record, label),
        "cause_hint": infer_cause_hint(record, label),
        "action_hint": infer_action_hint(record, label),
        "safety_hint": infer_safety_hint(record, label),
        "short_summary": infer_short_summary(record, label),
    }


def item_source(item: Dict[str, Any]) -> str:
    return source_name(item["_path"], item["_record"])


def interleave_by_source(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped_by_source[item_source(item)].append(item)

    ordered: List[Dict[str, Any]] = []
    sources = sorted(grouped_by_source)
    index = 0
    while True:
        progressed = False
        for source in sources:
            source_items = grouped_by_source[source]
            if index < len(source_items):
                ordered.append(source_items[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return ordered


def item_text_for_scenarios(item: Dict[str, Any]) -> str:
    record = item["_record"]
    parts = [
        record_text(record),
        item.get("_label", ""),
        source_name(item["_path"], record),
    ]
    return " ".join(clean_text(part) for part in parts if part)


def scenario_matches(item: Dict[str, Any]) -> List[str]:
    return matched_scenarios_from_text(item_text_for_scenarios(item))


def analyze_records(files: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "files": [str(path) for path in files],
        "total_records": 0,
        "usable_records": 0,
        "empty_invalid_records": 0,
        "duplicate_like_records": 0,
        "before_distribution": Counter(),
        "source_distribution": Counter(),
    }
    seen_normalized = set()
    usable: List[Dict[str, Any]] = []

    for path in files:
        for record in iter_json_records(path):
            stats["total_records"] += 1
            if not is_valid_record(record):
                stats["empty_invalid_records"] += 1
                continue

            text = record_text(record)
            normalized = normalize_text_for_duplicate(text)
            if not normalized or normalized in seen_normalized:
                stats["duplicate_like_records"] += 1
                continue
            seen_normalized.add(normalized)

            label = label_or_topic(record)
            stats["usable_records"] += 1
            stats["before_distribution"][label] += 1
            stats["source_distribution"][source_name(path, record)] += 1
            usable.append({"_path": path, "_record": record, "_label": label})

    return usable, stats


def balance_records(
    usable: Sequence[Dict[str, Any]],
    max_samples_per_label: int,
    target_min_records: int,
    target_max_records: int,
) -> Tuple[List[Dict[str, str]], Counter, List[str], Dict[str, Dict[str, Any]], Counter]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in usable:
        grouped[item["_label"]].append(item)

    selected_items: List[Dict[str, Any]] = []
    selected_keys = set()
    after_distribution: Counter = Counter()
    source_distribution: Counter = Counter()
    kept_low_frequency_labels: List[str] = []

    def item_key(item: Dict[str, Any]) -> Tuple[str, str]:
        record = item["_record"]
        return (
            str(item["_path"]),
            normalize_text_for_duplicate(record_text(record)),
        )

    def label_limit(label: str) -> int:
        return min(len(grouped.get(label, [])), max_samples_per_label)

    def source_share_allows(item: Dict[str, Any]) -> bool:
        if not selected_items:
            return True
        source = item_source(item)
        projected_total = len(selected_items) + 1
        return source_distribution[source] / projected_total <= SOURCE_MAX_SHARE

    def add_item(
        item: Dict[str, Any],
        *,
        enforce_label_limit: bool = False,
        enforce_source_share: bool = False,
    ) -> bool:
        if len(selected_items) >= target_max_records:
            return False
        if enforce_label_limit and after_distribution[item["_label"]] >= label_limit(item["_label"]):
            return False
        if enforce_source_share and not source_share_allows(item):
            return False
        key = item_key(item)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected_items.append(item)
        after_distribution[item["_label"]] += 1
        source_distribution[item_source(item)] += 1
        return True

    scenario_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in usable:
        for scenario in scenario_matches(item):
            scenario_index[scenario].append(item)

    for scenario, minimum in SCENARIO_MINIMUMS.items():
        current_count = sum(
            1 for item in selected_items if scenario in scenario_matches(item)
        )
        if current_count >= minimum:
            continue
        scenario_items = interleave_by_source(scenario_index.get(scenario, []))
        for enforce_source_share in (True, False):
            for item in scenario_items:
                if add_item(
                    item,
                    enforce_label_limit=True,
                    enforce_source_share=enforce_source_share,
                ):
                    current_count += 1
                    if current_count >= minimum:
                        break
            if current_count >= minimum:
                break

    for label in sorted(grouped):
        items = grouped[label]
        if len(items) <= max_samples_per_label:
            kept_low_frequency_labels.append(label)
            for item in items:
                add_item(item)

    positions: Dict[str, int] = defaultdict(int)
    ordered_labels = sorted(grouped)
    while len(selected_items) < target_min_records:
        progressed = False
        ordered_labels = sorted(
            ordered_labels,
            key=lambda label: (after_distribution[label], label),
        )
        for label in ordered_labels:
            if len(selected_items) >= target_min_records:
                break
            if after_distribution[label] >= label_limit(label):
                continue
            items = grouped[label]
            while positions[label] < len(items):
                item = items[positions[label]]
                positions[label] += 1
                if add_item(
                    item,
                    enforce_label_limit=True,
                    enforce_source_share=True,
                ):
                    progressed = True
                    break
        if not progressed:
            break

    after_distribution = Counter(item["_label"] for item in selected_items)
    source_distribution = Counter(item_source(item) for item in selected_items)
    scenario_coverage = compute_scenario_coverage(selected_items, scenario_index)
    balanced = [
        safe_hint_record(item["_path"], item["_record"])
        for item in selected_items
    ]

    return balanced, after_distribution, kept_low_frequency_labels, scenario_coverage, source_distribution


def compute_scenario_coverage(
    selected_items: Sequence[Dict[str, Any]],
    scenario_index: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    for scenario, minimum in SCENARIO_MINIMUMS.items():
        matched_items = [
            item for item in selected_items if scenario in scenario_matches(item)
        ]
        examples = []
        for item in matched_items[:3]:
            summary = truncate_summary(record_text(item["_record"]), 90)
            if summary:
                examples.append(summary)
        available = len(scenario_index.get(scenario, []))
        required = min(minimum, available)
        coverage[scenario] = {
            "count": len(matched_items),
            "available": available,
            "minimum": minimum,
            "required": required,
            "passed": len(matched_items) >= required,
            "examples": examples,
        }
    return coverage


def write_jsonl(path: Path, records: Iterable[Dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def check_coverage(records: Sequence[Dict[str, str]]) -> Dict[str, bool]:
    combined_text = "\n".join(" ".join(record.values()).lower() for record in records)
    return {
        name: any(keyword.lower() in combined_text for keyword in keywords)
        for name, keywords in COVERAGE_CHECKS.items()
    }


def markdown_distribution(counter: Counter, limit: Optional[int] = None) -> str:
    if not counter:
        return "- none"
    rows = []
    for label, count in counter.most_common(limit):
        rows.append(f"- {label}: {count}")
    return "\n".join(rows)


def markdown_coverage(results: Dict[str, bool]) -> str:
    if not results:
        return "- none"
    return "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in results.items())


def markdown_scenario_coverage(results: Dict[str, Dict[str, Any]]) -> str:
    if not results:
        return "- none"
    rows = []
    for scenario in SCENARIO_BUCKETS:
        result = results.get(scenario, {})
        status = "PASS" if result.get("passed") else "FAIL"
        examples = result.get("examples") or []
        rows.append(
            f"- {scenario}: {result.get('count', 0)} selected / "
            f"{result.get('available', 0)} available, minimum {result.get('minimum', 0)} "
            f"(required {result.get('required', 0)}) - {status}"
        )
        for example in examples:
            rows.append(f"  - example: {example}")
    return "\n".join(rows)


def markdown_profile_comparison(profile_results: Dict[str, Dict[str, Any]]) -> str:
    if not profile_results:
        return "- none"

    sections: List[str] = []
    for profile, result in profile_results.items():
        after_distribution: Counter = result.get("after_distribution", Counter())
        source_distribution: Counter = result.get("source_distribution", Counter())
        scenario_coverage = result.get("scenario_coverage", {})
        max_label_count = max(after_distribution.values()) if after_distribution else 0
        target_min = result.get("target_min_records", 0)
        target_max = result.get("target_max_records", 0)
        sections.extend(
            [
                f"### {profile}",
                f"- Total records: {result['count']}",
                f"- Label cap: {result['max_samples_per_label']}",
                f"- Target range: {target_min}-{target_max}",
                f"- Max label count: {max_label_count}",
                "- Source distribution:",
                markdown_distribution(source_distribution),
                "- Top 30 label_or_topic distribution:",
                markdown_distribution(after_distribution, limit=30),
                "- Scenario bucket coverage:",
                markdown_scenario_coverage(scenario_coverage),
                "",
            ]
        )
    return "\n".join(sections).strip()


def build_report(
    stats: Dict[str, Any],
    after_distribution: Counter,
    kept_low_frequency_labels: Sequence[str],
    output_path: Path,
    max_samples_per_label: int,
    profile_results: Dict[str, Dict[str, Any]],
    live_profile: str,
    coverage_results: Dict[str, bool],
    scenario_coverage: Dict[str, Dict[str, Any]],
    source_distribution: Counter,
) -> str:
    before_distribution: Counter = stats["before_distribution"]
    removed_total = stats["empty_invalid_records"] + stats["duplicate_like_records"]
    top_frequent = before_distribution.most_common(12)
    low_frequency = [label for label, count in before_distribution.items() if count <= max_samples_per_label]

    return "\n".join(
        [
            "# Dataset Processing Report",
            "",
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Input Files",
            markdown_distribution(Counter({path: 1 for path in stats["files"]})),
            "",
            "## Record Quality Summary",
            f"- Total records read: {stats['total_records']}",
            f"- Usable records after validation and deduplication: {stats['usable_records']}",
            f"- Empty or invalid records removed: {stats['empty_invalid_records']}",
            f"- Duplicate-like records removed: {stats['duplicate_like_records']}",
            f"- Total removed records: {removed_total}",
            "",
            "## Before Label Distribution",
            markdown_distribution(before_distribution),
            "",
            "## After Label Distribution",
            markdown_distribution(after_distribution),
            "",
            "## Top Frequent Labels Before Balancing",
            markdown_distribution(Counter(dict(top_frequent))),
            "",
            "## Kept Low-Frequency Labels",
            markdown_distribution(Counter({label: before_distribution[label] for label in sorted(low_frequency)})),
            "",
            "## Balanced Output",
            f"- Live profile: {live_profile}",
            f"- Live max samples per label: {max_samples_per_label}",
            f"- Balanced safe hints path: {output_path}",
            f"- Live output records: {sum(after_distribution.values())}",
            "",
            "## Source Dataset Distribution After Balancing",
            markdown_distribution(source_distribution),
            "",
            "## Profile Outputs",
            *[
                f"- {profile}: {result['count']} records, cap {result['max_samples_per_label']}, path {result['path']}"
                for profile, result in profile_results.items()
            ],
            "",
            "The previous 2416-record live safe hint file was fast, but it may be too narrow for common student counseling situations. The broad profile is selected as the default live profile because it keeps better coverage while remaining much smaller than the original processed datasets.",
            "",
            "Live chat should cache the broad safe hint dataset from `data/derived/balanced_safe_hints.jsonl` instead of loading or scanning the full `data/processed` corpus on each turn.",
            "",
            "## Profile Comparison",
            markdown_profile_comparison(profile_results),
            "",
            "## Coverage Check",
            markdown_coverage(coverage_results),
            "",
            "## Scenario Bucket Coverage",
            markdown_scenario_coverage(scenario_coverage),
            "",
            "## Agent Pipeline Use",
            "- `intent_hint` supports intent detection without exposing raw counseling turns.",
            "- `emotion_hint` supports emotional state interpretation and empathetic wording.",
            "- `cause_hint` supports cause candidate selection before follow-up generation.",
            "- `action_hint` supports small, low-burden action recommendations.",
            "- `safety_hint` keeps risk triage explicit so Safety/Risk checks run before normal counseling flow.",
            "",
            "The balanced file is a derived evidence artifact. It is intentionally safe and compact: it keeps only dataset origin, label/topic, summarized hints, and a short summary rather than long raw counseling text.",
            "",
            "## Low-Frequency Preservation Rationale",
            "Labels at or below the cap are kept intact because rare topics are often clinically and product-wise important. Downsampling only overrepresented labels reduces response-policy bias while retaining coverage of less common concerns.",
            "",
        ]
    )


def run_analysis(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    derived_dir: Path = DEFAULT_DERIVED_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    max_samples_per_label: Optional[int] = None,
    output_name: str = DEFAULT_LIVE_OUTPUT_NAME,
    profile_sample_caps: Optional[Dict[str, int]] = None,
    live_profile: str = DEFAULT_LIVE_PROFILE,
) -> Dict[str, Any]:
    files = discover_dataset_files(processed_dir)
    usable, stats = analyze_records(files)

    profile_configs = {
        profile: dict(config)
        for profile, config in DEFAULT_PROFILE_CONFIGS.items()
    }
    if profile_sample_caps is not None:
        profile_configs = {
            profile: {
                "max_samples_per_label": cap,
                "target_min_records": 0,
                "target_max_records": max(1, stats["usable_records"]),
            }
            for profile, cap in profile_sample_caps.items()
        }
    if max_samples_per_label is not None:
        profile_configs[live_profile]["max_samples_per_label"] = max_samples_per_label
    if live_profile not in profile_configs:
        raise ValueError(f"live_profile must be one of {sorted(profile_configs)}")

    profile_results: Dict[str, Dict[str, Any]] = {}
    for profile, config in profile_configs.items():
        sample_cap = int(config["max_samples_per_label"])
        target_min_records = int(config["target_min_records"])
        target_max_records = int(config["target_max_records"])
        (
            balanced,
            after_distribution,
            kept_low_frequency_labels,
            scenario_coverage,
            source_distribution,
        ) = balance_records(
            usable,
            max_samples_per_label=sample_cap,
            target_min_records=target_min_records,
            target_max_records=target_max_records,
        )
        profile_output_name = PROFILE_OUTPUT_NAMES.get(profile, f"balanced_safe_hints_{profile}.jsonl")
        profile_output_path = derived_dir / profile_output_name
        profile_count = write_jsonl(profile_output_path, balanced)
        profile_results[profile] = {
            "path": profile_output_path,
            "count": profile_count,
            "max_samples_per_label": sample_cap,
            "target_min_records": target_min_records,
            "target_max_records": target_max_records,
            "records": balanced,
            "after_distribution": after_distribution,
            "kept_low_frequency_labels": kept_low_frequency_labels,
            "scenario_coverage": scenario_coverage,
            "source_distribution": source_distribution,
        }

    live_result = profile_results[live_profile]
    output_path = derived_dir / output_name
    output_count = write_jsonl(output_path, live_result["records"])
    coverage_results = check_coverage(live_result["records"])
    report = build_report(
        stats,
        live_result["after_distribution"],
        live_result["kept_low_frequency_labels"],
        output_path,
        live_result["max_samples_per_label"],
        profile_results,
        live_profile,
        coverage_results,
        live_result["scenario_coverage"],
        live_result["source_distribution"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return {
        "input_files": files,
        "output_path": output_path,
        "report_path": report_path,
        "output_count": output_count,
        "profile_counts": {profile: result["count"] for profile, result in profile_results.items()},
        "profile_paths": {profile: result["path"] for profile, result in profile_results.items()},
        "profile_results": profile_results,
        "coverage_results": coverage_results,
        "coverage_passed": all(coverage_results.values()),
        "scenario_coverage": live_result["scenario_coverage"],
        "source_distribution": live_result["source_distribution"],
        "stats": stats,
        "after_distribution": live_result["after_distribution"],
        "kept_low_frequency_labels": live_result["kept_low_frequency_labels"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-samples-per-label", type=int, default=None)
    parser.add_argument("--output-name", default=DEFAULT_LIVE_OUTPUT_NAME)
    parser.add_argument("--live-profile", choices=tuple(DEFAULT_PROFILE_CONFIGS), default=DEFAULT_LIVE_PROFILE)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_analysis(
        processed_dir=args.processed_dir,
        derived_dir=args.derived_dir,
        report_path=args.report_path,
        max_samples_per_label=args.max_samples_per_label,
        output_name=args.output_name,
        live_profile=args.live_profile,
    )
    print(f"Input files: {len(result['input_files'])}")
    for profile, count in result["profile_counts"].items():
        print(f"{profile.title()} safe hints: {result['profile_paths'][profile]} ({count} records)")
    print(f"Balanced safe hints: {result['output_path']}")
    print(f"Dataset processing report: {result['report_path']}")
    print(f"Output records: {result['output_count']}")
    print(f"Coverage check passed: {result['coverage_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
