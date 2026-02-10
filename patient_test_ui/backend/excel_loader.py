"""Utility to parse Excel files into visit-based conversation data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import zipfile
import xml.etree.ElementTree as ET


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _load_shared_strings(z: zipfile.ZipFile) -> Dict[int, str]:
    shared_strings: Dict[int, str] = {}
    try:
        content = z.read("xl/sharedStrings.xml")
    except KeyError:
        return shared_strings

    root = ET.fromstring(content)
    for idx, si in enumerate(root.findall("main:si", NS)):
        text = "".join(
            node.text or ""
            for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )
        shared_strings[idx] = text
    return shared_strings


def _parse_cell_text(cell: ET.Element, shared_strings: Dict[int, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        is_node = cell.find("main:is", NS)
        if is_node is None:
            return ""
        return "".join(
            node.text or ""
            for node in is_node.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        )

    value_node = cell.find("main:v", NS)
    if value_node is None or value_node.text is None:
        return ""

    if cell_type == "s":
        return shared_strings.get(int(value_node.text), "")

    return value_node.text


def load_visit_rows_from_excel(file_path: str | Path) -> List[Dict[str, Any]]:
    """Parse the provided Excel file into a list of row dictionaries."""

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with zipfile.ZipFile(file_path) as z:
        shared_strings = _load_shared_strings(z)
        try:
            sheet_xml = z.read("xl/worksheets/sheet1.xml")
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("Excel 工作表缺少 sheet1") from exc

    sheet = ET.fromstring(sheet_xml)
    sheet_data = sheet.find("main:sheetData", NS)
    if sheet_data is None:
        raise ValueError("Excel 中没有任何数据")

    rows = sheet_data.findall("main:row", NS)
    if not rows:
        raise ValueError("Excel 数据为空")

    header_cells = rows[0].findall("main:c", NS)
    headers: List[str] = []
    for cell in header_cells:
        headers.append(_parse_cell_text(cell, shared_strings).strip())

    # Remove empty trailing headers
    while headers and headers[-1] == "":
        headers.pop()

    records: List[Dict[str, Any]] = []
    for row in rows[1:]:
        values: List[str] = []
        for cell in row.findall("main:c", NS):
            values.append(_parse_cell_text(cell, shared_strings))
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        record = {
            headers[idx]: values[idx].strip() if idx < len(values) else ""
            for idx in range(len(headers))
            if headers[idx]
        }
        # Skip completely empty rows
        if any(value for value in record.values()):
            records.append(record)

    return records

