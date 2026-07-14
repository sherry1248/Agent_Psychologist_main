#!/usr/bin/env python3
"""Compare safe-hint dataset profiles without loading application components."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
PROFILE_NAMES = ("small", "balanced", "broad", "recommended")
PROFILE_PATHS = {
    name: DERIVED_DIR / f"balanced_safe_hints_{name}.jsonl"
    for name in PROFILE_NAMES
}
LABEL_FIELDS = ("label_or_topic", "label", "intent", "category", "topic")
SCENARIOS: Dict[str, Tuple[str, ...]] = {
    "academic": ("시험", "공부", "암기", "과제", "학업"),
    "sleep": ("잠", "수면", "숙면", "자다가", "깨"),
    "criticism": ("지적", "혼났", "교수님", "상사", "평가"),
    "self_blame": ("한심", "자책", "내가 문제", "못난", "자괴감"),
    "sadness": ("눈물", "슬프", "아무것도 하기 싫", "우울"),
    "anger": ("짜증", "화나", "답답", "억울"),
    "recovery": ("괜찮아졌", "나아졌", "조금 괜찮", "회복"),
    "crisis": ("죽고 싶", "자살", "사라지고 싶", "끝내고 싶"),
}


@dataclass(frozen=True)
class ProfileStats:
    name: str
    path: Path
    line_count: int
    invalid_lines: int
    labels: Counter[str]
    coverage_counts: Dict[str, int]
    matched_keywords: Dict[str, Tuple[str, ...]]

    @property
    def keyword_coverage_score(self) -> int:
        return sum(len(values) for values in self.matched_keywords.values())


def _record_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for value in record.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(item for item in value if isinstance(item, str))
    return " ".join(parts).lower()


def _record_label(record: Dict[str, Any]) -> str:
    for field in LABEL_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def evaluate_profile(name: str, path: Path) -> ProfileStats:
    labels: Counter[str] = Counter()
    coverage_counts = {scenario: 0 for scenario in SCENARIOS}
    matched_keywords = {scenario: set() for scenario in SCENARIOS}
    line_count = 0
    invalid_lines = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(record, dict):
                invalid_lines += 1
                continue

            label = _record_label(record)
            if label:
                labels[label] += 1
            text = _record_text(record)
            for scenario, keywords in SCENARIOS.items():
                record_matched = False
                for keyword in keywords:
                    if keyword.lower() in text:
                        matched_keywords[scenario].add(keyword)
                        record_matched = True
                if record_matched:
                    coverage_counts[scenario] += 1

    return ProfileStats(
        name=name,
        path=path,
        line_count=line_count,
        invalid_lines=invalid_lines,
        labels=labels,
        coverage_counts=coverage_counts,
        matched_keywords={
            scenario: tuple(keyword for keyword in SCENARIOS[scenario] if keyword in values)
            for scenario, values in matched_keywords.items()
        },
    )


def _print_profile(stats: ProfileStats) -> None:
    print(f"\n[{stats.name}] {stats.path.name}")
    print(f"lines: {stats.line_count:,} (invalid: {stats.invalid_lines:,})")
    if stats.labels:
        print(f"distinct labels: {len(stats.labels):,}")
        print("top labels:")
        for label, count in stats.labels.most_common(10):
            print(f"  {label}: {count:,}")
    else:
        print("labels: no label/intent/category field found")
    print("scenario keyword coverage:")
    for scenario in SCENARIOS:
        keywords = stats.matched_keywords[scenario]
        coverage = f"{len(keywords)}/{len(SCENARIOS[scenario])} keywords"
        matched = ", ".join(keywords) if keywords else "none"
        print(f"  {scenario}: {coverage}; record hits={stats.coverage_counts[scenario]:,}; matched={matched}")


def _print_coverage_table(results: Iterable[ProfileStats]) -> None:
    results = list(results)
    columns = list(SCENARIOS)
    widths = {column: max(len(column), 8) for column in columns}
    profile_width = max(9, *(len(result.name) for result in results))
    header = f"{'profile':<{profile_width}} " + " ".join(
        f"{column:>{widths[column]}}" for column in columns
    )
    print("\nCoverage table (record hit counts)")
    print(header)
    print("-" * len(header))
    for result in results:
        values = " ".join(
            f"{result.coverage_counts[column]:>{widths[column]},}" for column in columns
        )
        print(f"{result.name:<{profile_width}} {values}")


def recommend_profile(results: List[ProfileStats]) -> Tuple[ProfileStats, List[str]]:
    viable = [result for result in results if result.coverage_counts["crisis"] > 0]
    warnings: List[str] = []
    if not viable:
        viable = list(results)
        warnings.append("WARNING: every profile is missing crisis keyword coverage.")

    by_name = {result.name: result for result in viable}
    strongest_score = max(result.keyword_coverage_score for result in viable)
    strongest = [result for result in viable if result.keyword_coverage_score == strongest_score]

    recommended = by_name.get("recommended")
    broad = by_name.get("broad")
    balanced = by_name.get("balanced")
    if recommended and recommended.keyword_coverage_score == strongest_score:
        choice = recommended
    elif broad and broad.keyword_coverage_score == strongest_score:
        broad_too_large = bool(balanced and broad.line_count > balanced.line_count * 1.5)
        clearly_better = bool(
            not balanced
            or broad.keyword_coverage_score > balanced.keyword_coverage_score
            or sum(broad.coverage_counts.values()) > sum(balanced.coverage_counts.values()) * 1.1
        )
        choice = balanced if broad_too_large or not clearly_better else broad
    elif balanced:
        choice = balanced
    else:
        choice = min(strongest, key=lambda result: (result.line_count, result.name))

    crisis_hits = choice.coverage_counts["crisis"]
    if crisis_hits == 0:
        warnings.append(f"WARNING: {choice.name} has no crisis coverage.")
    elif (
        len(choice.matched_keywords["crisis"]) < 2
        or crisis_hits < max(5, choice.line_count // 1000)
    ):
        warnings.append(
            f"WARNING: {choice.name} crisis coverage is low "
            f"({len(choice.matched_keywords['crisis'])}/{len(SCENARIOS['crisis'])} keywords, "
            f"{crisis_hits} record hits)."
        )
    return choice, warnings


def main() -> int:
    found = [(name, path) for name, path in PROFILE_PATHS.items() if path.exists()]
    missing = [path.name for path in PROFILE_PATHS.values() if not path.exists()]
    if not found:
        print(f"No profile files found in {DERIVED_DIR}")
        return 1

    print("Found profiles: " + ", ".join(path.name for _, path in found))
    if missing:
        print("Missing profiles: " + ", ".join(missing))

    results = [evaluate_profile(name, path) for name, path in found]
    for result in results:
        _print_profile(result)
    _print_coverage_table(results)

    choice, warnings = recommend_profile(results)
    print(f"\nRecommended profile: {choice.name} ({choice.path.name})")
    print(
        f"Reason: keyword coverage score={choice.keyword_coverage_score}, "
        f"crisis record hits={choice.coverage_counts['crisis']:,}, lines={choice.line_count:,}."
    )
    for warning in warnings:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
