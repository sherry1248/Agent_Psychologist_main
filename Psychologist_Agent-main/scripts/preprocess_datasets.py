#!/usr/bin/env python3
"""Preprocess raw counseling, empathy, and wellness datasets into JSONL files.

The script keeps raw data untouched and writes loader-compatible files under
data/processed by default.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "processed"

ALLOWED_EMOTIONS = {"기쁨", "당황", "분노", "불안", "상처", "슬픔"}
ALLOWED_EMPATHY_LABELS = {"동조", "조언", "위로", "격려"}
EMPATHY_PRIORITY = ("위로", "격려", "조언", "동조")

NUMERIC_FIELDS = (
    "mood_score",
    "anxiety_score",
    "loneliness_score",
    "sleep_quality",
    "meal_status",
    "energy_score",
    "stress_score",
)

XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass
class PreprocessStats:
    counseling: int = 0
    empathy: int = 0
    wellness: int = 0


class TextCleaner:
    """Small regex-only cleaner to avoid adding preprocessing dependencies."""

    _patterns = (
        (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL_REDACTED]"),
        (re.compile(r"https?://\S+|www\.\S+"), "[URL_REDACTED]"),
        (re.compile(r"\b\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4}\b"), "[PHONE_REDACTED]"),
        (re.compile(r"\b\d{6}[- ]?[1-4]\d{6}\b"), "[ID_REDACTED]"),
        (re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b"), "[DATE_REDACTED]"),
        (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), "[TIME_REDACTED]"),
        (re.compile(r"@[^\s,.;:!?(){}\[\]<>]+"), "[MENTION_REDACTED]"),
    )

    def __init__(self, max_user_chars: int = 240, max_response_chars: int = 360):
        self.max_user_chars = max_user_chars
        self.max_response_chars = max_response_chars

    def clean_user(self, value: Any) -> str:
        return self.clean(value, self.max_user_chars)

    def clean_response(self, value: Any) -> str:
        return self.clean(value, self.max_response_chars)

    def clean(self, value: Any, max_chars: int) -> str:
        if value is None:
            return ""
        text = " ".join(str(value).replace("\ufeff", " ").split()).strip()
        if not text:
            return ""
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return self._truncate(text, max_chars)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        boundary = max(text.rfind(".", 0, max_chars), text.rfind("?", 0, max_chars), text.rfind("!", 0, max_chars))
        if boundary >= max_chars // 2:
            return text[: boundary + 1].strip()
        return text[:max_chars].rstrip()


def load_json_bytes(payload: bytes) -> Dict[str, Any]:
    return json.loads(payload.decode("utf-8-sig"), parse_constant=lambda _value: None)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_zip_json_members(zip_path: Path) -> Iterator[Dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".json"):
                continue
            try:
                payload = load_json_bytes(archive.read(member))
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload


def intervention_hint(paragraph: Dict[str, Any]) -> str:
    label_hints = [
        ("sympathy_support", "감정을 먼저 인정하고 지지하세요."),
        ("clarification_reflection", "내담자의 표현을 반영하고 핵심 감정을 명료화하세요."),
        ("cognitive_restructuring", "자동사고를 점검하고 대안적 해석을 함께 탐색하세요."),
        ("information_provision", "필요한 정보를 짧고 부담 없이 제공하세요."),
        ("goal_setting", "작고 구체적인 다음 행동을 정하세요."),
        ("behavioral_intervention", "실행 가능한 행동 실험이나 루틴을 제안하세요."),
        ("task_assignment", "부담이 낮은 과제를 하나만 제안하세요."),
        ("training_of_coping_skills", "호흡, 기록, 거리두기 같은 대처 기술을 안내하세요."),
        ("emotional_regulation_education_training", "감정 조절 방법을 차분하게 안내하세요."),
        ("acceptance_change", "변화 가능성을 서두르지 않고 수용적으로 다루세요."),
    ]
    selected = [hint for key, hint in label_hints if _positive(paragraph.get(key))]
    if selected:
        return " ".join(selected[:2])
    return "감정을 먼저 확인하고, 작은 실행 단계를 하나만 제안하세요."


def _positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip() not in {"", "0", "N", "n", "False", "false"}
    return False


def normalize_category(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return "GENERAL"
    return text.upper()


def generate_counseling_records(data_dir: Path, cleaner: TextCleaner) -> Iterator[Dict[str, Any]]:
    index = 1
    root = data_dir / "counseling"
    for zip_path in sorted(root.rglob("*.zip")):
        if "라벨링데이터" not in str(zip_path):
            continue
        for payload in read_zip_json_members(zip_path):
            category = normalize_category(payload.get("class"))
            paragraphs = payload.get("paragraph") or []
            if not isinstance(paragraphs, list):
                continue
            for current, following in zip(paragraphs, paragraphs[1:]):
                if not isinstance(current, dict) or not isinstance(following, dict):
                    continue
                if current.get("paragraph_speaker") != "내담자":
                    continue
                if following.get("paragraph_speaker") != "상담사":
                    continue
                user_input = cleaner.clean_user(current.get("paragraph_text"))
                counselor_response = cleaner.clean_response(following.get("paragraph_text"))
                if not user_input or not counselor_response:
                    continue
                yield {
                    "id": f"counseling_{index:06d}",
                    "user_input": user_input,
                    "counselor_response": counselor_response,
                    "category": category,
                    "intervention_hint": intervention_hint(following),
                    "source": "counseling",
                    "privacy_flags": ["redacted", "truncated", "no_raw_id", "no_filename", "no_timestamp"],
                }
                index += 1


def primary_empathy_label(value: Any, fallback: Any = None) -> str:
    labels: List[str] = []
    for candidate in (value, fallback):
        if isinstance(candidate, list):
            labels.extend(str(item).strip() for item in candidate)
        elif isinstance(candidate, str):
            labels.append(candidate.strip())
    labels = [label for label in labels if label in ALLOWED_EMPATHY_LABELS]
    for preferred in EMPATHY_PRIORITY:
        if preferred in labels:
            return preferred
    return labels[0] if labels else "위로"


def empathy_style_hint(emotion: str, empathy_label: str) -> str:
    emotion_text = {
        "기쁨": "기쁨을 함께 반영하고 자연스러운 흐름을 이어가세요.",
        "당황": "당황스러움을 인정하고 천천히 정리하도록 도우세요.",
        "분노": "분노를 판단하지 말고 안전한 표현을 돕는 방향으로 반응하세요.",
        "불안": "불안을 가볍게 여기지 않고 차분하게 인정하세요.",
        "상처": "상처받은 마음을 먼저 받아들이고 지지하세요.",
        "슬픔": "슬픔을 충분히 담아내고 조용히 지지하세요.",
    }.get(emotion, "감정을 먼저 확인하고 편안하게 반응하세요.")
    label_text = {
        "동조": "상대의 감정에 맞장구치며 함께하세요.",
        "조언": "감정 확인 뒤 작은 실천을 제안하세요.",
        "위로": "지금의 고됨을 인정하며 위로하세요.",
        "격려": "조금씩 해낼 수 있다는 점을 격려하세요.",
    }.get(empathy_label, "상담 맥락에 맞는 공감 표현을 사용하세요.")
    return f"{emotion_text} {label_text}"


def generate_empathy_records(data_dir: Path, cleaner: TextCleaner) -> Iterator[Dict[str, Any]]:
    index = 1
    root = data_dir / "empathy"
    for zip_path in sorted(root.rglob("*.zip")):
        if "라벨링데이터" not in str(zip_path):
            continue
        for payload in read_zip_json_members(zip_path):
            info = payload.get("info") or {}
            if not isinstance(info, dict):
                info = {}
            emotion = str(info.get("speaker_emotion") or "").strip()
            if emotion not in ALLOWED_EMOTIONS:
                continue
            utterances = payload.get("utterances") or []
            if not isinstance(utterances, list):
                continue
            for current, following in zip(utterances, utterances[1:]):
                if not isinstance(current, dict) or not isinstance(following, dict):
                    continue
                if current.get("role") != "speaker" or following.get("role") != "listener":
                    continue
                user_input = cleaner.clean_user(current.get("text"))
                if not user_input:
                    continue
                empathy_label = primary_empathy_label(following.get("listener_empathy"), info.get("listener_behavior"))
                yield {
                    "id": f"empathy_{index:06d}",
                    "user_input": user_input,
                    "emotion_label": emotion,
                    "empathy_label": empathy_label,
                    "empathy_style_hint": empathy_style_hint(emotion, empathy_label),
                    "source": "empathy",
                    "privacy_flags": ["redacted", "truncated", "no_raw_id", "no_filename", "no_timestamp", "no_voter_id"],
                }
                index += 1


def xlsx_col_number(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter.upper()) - 64
    return number


def read_xlsx_rows(path: Path) -> Iterator[tuple[str, List[str]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheets = []
        for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relmap[rel_id]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            sheets.append((sheet.attrib["name"], target))

        for sheet_name, target in sheets:
            worksheet = ET.fromstring(archive.read(target))
            for row in worksheet.findall("a:sheetData/a:row", XLSX_NS):
                values: List[str] = []
                for cell in row.findall("a:c", XLSX_NS):
                    idx = xlsx_col_number(cell.attrib.get("r", "A")) - 1
                    while len(values) < idx:
                        values.append("")
                    values.append(read_xlsx_cell(cell, shared_strings))
                yield sheet_name, values


def read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", XLSX_NS):
        parts = [node.text or "" for node in item.findall(".//a:t", XLSX_NS)]
        strings.append("".join(parts))
    return strings


def read_xlsx_cell(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None or value_node.text is None:
        inline_text = cell.find(".//a:t", XLSX_NS)
        return (inline_text.text or "").strip() if inline_text is not None else ""
    value = value_node.text
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(value)].strip()
        except (IndexError, ValueError):
            return ""
    return str(value).strip()


def rows_as_dicts(path: Path) -> Iterator[Dict[str, str]]:
    current_context: Dict[str, str] = {}
    headers_by_sheet: Dict[str, List[str]] = {}
    for sheet_name, row in read_xlsx_rows(path):
        if not any(row):
            continue
        if sheet_name not in headers_by_sheet:
            headers_by_sheet[sheet_name] = [cell.strip() for cell in row]
            current_context = {}
            continue
        headers = headers_by_sheet[sheet_name]
        record = {
            headers[idx]: row[idx].strip()
            for idx in range(min(len(headers), len(row)))
            if headers[idx]
        }
        for key in ("핵심증상", "intent", "keyword(임상키워드)", "구분"):
            if record.get(key):
                current_context[key] = record[key]
            elif key in current_context:
                record[key] = current_context[key]
        yield record


def infer_risk_stage(topic: str, question: str = "") -> str:
    text = f"{topic} {question}".lower()
    danger_markers = ("죽", "자살", "자해", "위험", "해치", "사라지고", "suicide", "self-harm")
    warning_markers = ("우울", "불안", "불면", "분노", "외로", "스트레스", "공황", "중독", "슬픔")
    if any(marker in text for marker in danger_markers):
        return "위험"
    if any(marker in text for marker in warning_markers):
        return "주의"
    return "관심"


def derive_scores(topic: str, risk_stage: str) -> Dict[str, int]:
    if risk_stage == "위험":
        scores = {
            "mood_score": 2,
            "anxiety_score": 8,
            "loneliness_score": 7,
            "sleep_quality": 2,
            "meal_status": 3,
            "energy_score": 2,
            "stress_score": 8,
        }
    elif risk_stage == "주의":
        scores = {
            "mood_score": 4,
            "anxiety_score": 6,
            "loneliness_score": 5,
            "sleep_quality": 4,
            "meal_status": 5,
            "energy_score": 4,
            "stress_score": 6,
        }
    else:
        scores = {
            "mood_score": 7,
            "anxiety_score": 3,
            "loneliness_score": 3,
            "sleep_quality": 7,
            "meal_status": 7,
            "energy_score": 6,
            "stress_score": 3,
        }

    lowered = topic.lower()
    if any(marker in lowered for marker in ("수면", "불면", "sleep")):
        scores.update({"sleep_quality": 3, "stress_score": max(scores["stress_score"], 5)})
    if any(marker in lowered for marker in ("불안", "걱정", "anxiety")):
        scores.update({"anxiety_score": 8, "stress_score": max(scores["stress_score"], 6)})
    if any(marker in lowered for marker in ("외로", "loneliness")):
        scores.update({"loneliness_score": 8})
    if any(marker in lowered for marker in ("우울", "슬픔", "mood")):
        scores.update({"mood_score": 3, "energy_score": 3})
    if any(marker in lowered for marker in ("스트레스", "업무", "직장", "stress", "work")):
        scores.update({"stress_score": 8})
    return scores


def first_sentence(text: str, max_chars: int = 160) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return "지금의 상태를 차분히 확인하고 오늘 할 수 있는 작은 행동 하나를 제안하세요."
    for sep in (".", "?", "!", "。"):
        pos = text.find(sep)
        if 0 <= pos < max_chars:
            return text[: pos + 1].strip()
    return text[:max_chars].strip()


def wellness_record(
    index: int,
    question: str,
    answer: str,
    topic: str,
    cleaner: TextCleaner,
) -> Optional[Dict[str, Any]]:
    cleaned_question = cleaner.clean_user(question)
    cleaned_answer = cleaner.clean_response(answer)
    cleaned_topic = cleaner.clean(topic, 80) or "general"
    if not cleaned_question or not cleaned_answer:
        return None
    risk_stage = infer_risk_stage(cleaned_topic, cleaned_question)
    scores = derive_scores(cleaned_topic, risk_stage)
    record: Dict[str, Any] = {
        "id": f"wellness_{index:06d}",
        "question": cleaned_question,
        "answer": cleaned_answer,
        "topic": cleaned_topic,
        "risk_stage": risk_stage,
        "support_hint": first_sentence(cleaned_answer),
    }
    record.update(scores)
    record["source"] = "wellness"
    record["privacy_flags"] = ["redacted", "truncated", "no_raw_id", "no_filename", "no_timestamp"]
    return record


def generate_wellness_records(data_dir: Path, cleaner: TextCleaner) -> Iterator[Dict[str, Any]]:
    index = 1
    for xlsx_path in sorted((data_dir / "wellness").glob("*.xlsx")):
        for row in rows_as_dicts(xlsx_path):
            emitted: List[tuple[str, str, str]] = []
            if "유저" in row or "챗봇" in row:
                emitted.append((row.get("유저", ""), row.get("챗봇", ""), row.get("구분", "general")))
            else:
                topic = row.get("핵심증상") or row.get("intent") or row.get("keyword(임상키워드)") or "general"
                base_question = row.get("utterance(2차)") or row.get("utterance") or ""
                base_answer = row.get("response(공감)") or ""
                emitted.append((base_question, base_answer, topic))
                emitted.append((row.get("utterance(긍정)", ""), row.get("긍정에 대한 챗봇 답변", ""), topic))
                emitted.append((row.get("utterance(부정)", ""), row.get("부정에 대한 챗봇 답변", ""), topic))

            for question, answer, topic in emitted:
                record = wellness_record(index, question, answer, topic, cleaner)
                if record is None:
                    continue
                yield record
                index += 1


def preprocess(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> PreprocessStats:
    cleaner = TextCleaner()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = PreprocessStats()
    stats.counseling = write_jsonl(
        output_dir / "counseling_processed.jsonl",
        generate_counseling_records(data_dir, cleaner),
    )
    stats.empathy = write_jsonl(
        output_dir / "empathy_processed.jsonl",
        generate_empathy_records(data_dir, cleaner),
    )
    stats.wellness = write_jsonl(
        output_dir / "wellness_processed.jsonl",
        generate_wellness_records(data_dir, cleaner),
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for processed JSONL files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = preprocess(args.data_dir, args.output_dir)
    print(f"counseling records: {stats.counseling}")
    print(f"empathy records: {stats.empathy}")
    print(f"wellness records: {stats.wellness}")
    print(f"output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
