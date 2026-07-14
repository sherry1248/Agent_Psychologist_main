from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.preprocess_datasets import preprocess
from src.counseling.dataset_loader import CounselingDatasetLoader
from src.empathy.dataset_loader import EmpathyDatasetLoader
from src.wellness.dataset_loader import WellnessDatasetLoader


def _write_zip_json(path: Path, member_name: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, json.dumps(payload, ensure_ascii=False))


def _write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    shared_strings: list[str] = []
    shared_index: dict[str, int] = {}

    def shared(value: str) -> int:
        if value not in shared_index:
            shared_index[value] = len(shared_strings)
            shared_strings.append(value)
        return shared_index[value]

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            if not value:
                continue
            col = chr(ord("A") + col_idx - 1)
            cells.append(f'<c r="{col}{row_idx}" t="s"><v>{shared(value)}</v></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
        + "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
        + "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_xml)


def test_preprocess_datasets_outputs_loader_compatible_jsonl(tmp_path: Path):
    data_dir = tmp_path / "data"
    output_dir = data_dir / "processed"

    _write_zip_json(
        data_dir
        / "counseling"
        / "3.개방데이터"
        / "1.데이터"
        / "Training"
        / "02.라벨링데이터"
        / "TL_sample.zip",
        "label_sample.json",
        {
            "id": "raw-counseling-id",
            "filename": "raw-file-name",
            "class": "DEPRESSION",
            "paragraph": [
                {
                    "paragraph_speaker": "내담자",
                    "paragraph_text": "제 이메일은 user@example.com 이고 요즘 너무 우울해요.",
                },
                {
                    "paragraph_speaker": "상담사",
                    "paragraph_text": "우울한 마음을 먼저 확인해볼게요.",
                    "sympathy_support": 1,
                },
            ],
        },
    )

    _write_zip_json(
        data_dir
        / "empathy"
        / "01-1.정식개방데이터"
        / "Training"
        / "02.라벨링데이터"
        / "TL_sample.zip",
        "empathy_sample.json",
        {
            "info": {
                "id": "raw-empathy-id",
                "speaker_emotion": "불안",
                "listener_behavior": ["동조"],
                "votes": [{"voter_id": "voter-1"}],
            },
            "utterances": [
                {"role": "speaker", "text": "내 전화번호는 010-1234-5678 이고 너무 불안해."},
                {"role": "listener", "text": "많이 불안했겠다.", "listener_empathy": ["위로"]},
            ],
        },
    )

    _write_minimal_xlsx(
        data_dir / "wellness" / "wellness.xlsx",
        [
            ["구분", "유저", "챗봇"],
            ["정신증상/불안", "요즘 걱정이 너무 많아요.", "걱정이 많아 힘드셨겠어요."],
        ],
    )

    stats = preprocess(data_dir=data_dir, output_dir=output_dir)

    assert stats.counseling == 1
    assert stats.empathy == 1
    assert stats.wellness == 1

    counseling_text = (output_dir / "counseling_processed.jsonl").read_text(encoding="utf-8")
    empathy_text = (output_dir / "empathy_processed.jsonl").read_text(encoding="utf-8")
    wellness_text = (output_dir / "wellness_processed.jsonl").read_text(encoding="utf-8")

    assert "raw-counseling-id" not in counseling_text
    assert "raw-file-name" not in counseling_text
    assert "user@example.com" not in counseling_text
    assert "raw-empathy-id" not in empathy_text
    assert "voter-1" not in empathy_text
    assert "010-1234-5678" not in empathy_text

    assert CounselingDatasetLoader(output_dir / "counseling_processed.jsonl").load_records()
    assert EmpathyDatasetLoader(output_dir / "empathy_processed.jsonl").load_records()
    assert WellnessDatasetLoader(output_dir / "wellness_processed.jsonl").load_records()

    wellness_record = json.loads(wellness_text.strip())
    assert wellness_record["risk_stage"] == "주의"
    assert all(field in wellness_record for field in ("mood_score", "anxiety_score", "stress_score"))
