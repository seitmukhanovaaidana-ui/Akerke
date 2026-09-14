"""
Чтение "сырых" лабораторных отчётов в формате Word (.docx) - как они
приходят из лаборатории (см. ПРИЛОЖЕНИЕ 3 "результаты электрических
свойств породы с измерением капиллярного давления").

Каждый образец в таком отчёте оформлен отдельной таблицей вида
"Наименование | Значение | Водонасыщенность (Sw) | Капиллярное давление | ...":
первые строки таблицы (Месторождение, Скважина, № образца, Глубина,
Пористость, Проницаемость, ...) идут в столбце "Наименование"/"Значение",
а параллельно в столбцах Sw/Pc для этих же строк лежат сами точки
капилляриметрии.

В отличие от Excel-версии, в исходном Word-отчёте нет столбца "Горизонт" -
он проставлялся отдельно, уже после переноса данных в Excel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HEADER_MARKER = "Наименование"
LABEL_TARGETS = {
    "№ образца": "sample",
    "Глубина": "depth",
    "Пористость": "porosity_pct",
    "Проницаемость": "perm_mD",
    "Скважина": "well",
}


def _to_float(text: str) -> float | None:
    text = text.strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_lab_data_from_docx(path: str | Path) -> pd.DataFrame:
    """Извлекает точки измерений (Sw, Pc, пористость, проницаемость, ...) из .docx-отчёта."""
    import docx  # локальный импорт: нужен только для .docx

    doc = docx.Document(str(path))

    rows: list[dict] = []
    for table in doc.tables:
        if not table.rows or table.rows[0].cells[0].text.strip() != HEADER_MARKER:
            continue

        header_cells = [c.text.strip() for c in table.rows[0].cells]
        col_sw = col_pc = None
        for idx, text in enumerate(header_cells):
            if "одонасыщ" in text:
                col_sw = idx
            if "апил" in text:  # ловит и "Капиллярное", и опечатку "Капилярное"
                col_pc = idx
        if col_sw is None or col_pc is None:
            continue

        meta: dict[str, str] = {}
        data_pts: list[tuple[float, float]] = []
        for row in table.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            label = cells[0]
            for key, name in LABEL_TARGETS.items():
                if key in label:
                    meta[name] = cells[1]

            sw = _to_float(cells[col_sw]) if col_sw < len(cells) else None
            pc = _to_float(cells[col_pc]) if col_pc < len(cells) else None
            if sw is not None and pc is not None:
                data_pts.append((sw, pc))

        for sw, pc in data_pts:
            rows.append(
                {
                    "well": meta.get("well"),
                    "sample": meta.get("sample"),
                    "horizon": None,
                    "depth": _to_float(meta["depth"]) if "depth" in meta else None,
                    "porosity_pct": _to_float(meta["porosity_pct"]) if "porosity_pct" in meta else None,
                    "perm_mD": _to_float(meta["perm_mD"]) if "perm_mD" in meta else None,
                    "Sw": sw,
                    "Pc_lab_MPa": pc,
                }
            )

    return pd.DataFrame(rows)
