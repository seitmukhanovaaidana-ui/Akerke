"""
Чтение "сырых" отчётов лаборатории по ОФП (относительным фазовым
проницаемостям) в формате Word (.docx) - как они присылаются лабораторией
(см., например, "ПРИЛОЖЕНИЕ 4. Результаты определения относительных фазовых
проницаемостей").

Каждый образец в таком отчёте оформлен парой таблиц:

1. "Наименование / Значение" - метаданные образца: скважина, № модели,
   пористость, проницаемость, остаточная водонасыщенность (Swi),
   остаточная нефтенасыщенность (Sow/Sor).
2. "Qнефти / Qводы / Sw / krw / krow" - сами точки кривой ОФП.

Одна метаданные-таблица иногда предшествует НЕСКОЛЬКИМ таблицам с точками
(два образца измерены в одном керне) - в этом случае каждой такой таблице
присваивается свой model_id с буквенным суффиксом (300-4, 300-4b, ...).

Swmax и krwmax берутся как последняя точка кривой (Sw при krow=0), а
krow_swc - как krow в первой точке (Sw=Swi) - так же, как эти величины
получены в исходном Excel-файле (лист "Параметризация_Кори").
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _to_float(text: str) -> float | None:
    text = text.strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_meta_table(table) -> dict:
    meta: dict = {}
    paren_well: str | None = None
    field_well: str | None = None
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        if len(cells) < 2:
            continue
        label, value = cells[0], cells[1]
        label_l = label.lower()

        if "модел" in label_l and "образц" in label_l:
            # обычно "Модель №1"; изредка "модель 3 (302)" - число в скобках это скважина
            m = re.search(r"\d+", value)
            if m:
                meta["model_num"] = m.group()
            m_paren = re.search(r"\((\d+)\)", value)
            if m_paren:
                paren_well = m_paren.group(1)
        elif "скв" in label_l:
            m = re.search(r"\d+", value)
            if m:
                field_well = m.group()
        elif "месторожден" in label_l and re.fullmatch(r"\d+", value.strip()):
            # опечатка в отчёте: номер скважины иногда попадает в поле "Месторождение"
            field_well = field_well or value.strip()
        elif "пористост" in label_l:
            meta["porosity_pct"] = _to_float(value)
        elif "проницаем" in label_l and ("газу" in label_l or "пластовой" in label_l):
            meta["perm_mD"] = _to_float(value)
        elif "остаточная водонасыщ" in label_l or "(swi)" in label_l:
            meta["swir"] = _to_float(value)
        elif "остаточная нефтенасыщ" in label_l or "(sow)" in label_l:
            meta["sor"] = _to_float(value)

    well = field_well or paren_well
    if well is not None:
        meta["well"] = well
    return meta


def _find_curve_columns(header: list[str]) -> tuple[int, int, int] | None:
    idx_sw = idx_krw = idx_krow = None
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if hl == "sw":
            idx_sw = i
        elif hl == "krw":
            idx_krw = i
        elif hl == "krow":
            idx_krow = i
    if idx_sw is None or idx_krw is None or idx_krow is None:
        return None
    return idx_sw, idx_krw, idx_krow


def _process_data_table(table, meta: dict, model_counts: dict, rows: list[dict]) -> None:
    header = [c.text.strip() for c in table.rows[0].cells]
    cols = _find_curve_columns(header)
    if cols is None:
        return
    idx_sw, idx_krw, idx_krow = cols

    pts: list[tuple[float, float, float]] = []
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        if max(idx_sw, idx_krw, idx_krow) >= len(cells):
            continue
        sw = _to_float(cells[idx_sw])
        krw = _to_float(cells[idx_krw])
        krow = _to_float(cells[idx_krow])
        if sw is not None and krw is not None and krow is not None:
            pts.append((sw, krw, krow))

    if len(pts) < 2:
        return

    well = meta.get("well", "")
    model_num = meta.get("model_num", "?")
    key = (well, model_num)
    count = model_counts.get(key, 0)
    model_counts[key] = count + 1
    suffix = "" if count == 0 else chr(ord("a") + count)
    model_id = f"{well}-{model_num}{suffix}"

    swmax = pts[-1][0]
    krwmax = pts[-1][1]
    krow_swc = pts[0][2]

    for sw, krw, krow in pts:
        rows.append(
            {
                "model": model_id,
                "well": well,
                "Sw": sw,
                "krw": krw,
                "krow": krow,
                "Swir": meta.get("swir"),
                "Sor": meta.get("sor"),
                "Swmax": swmax,
                "krwmax": krwmax,
                "krow_swc": krow_swc,
                "porosity_pct": meta.get("porosity_pct"),
                "perm_mD": meta.get("perm_mD"),
            }
        )


def load_ofp_data_from_docx(path: str | Path) -> pd.DataFrame:
    """Извлекает точки кривых ОФП (Sw, krw, krow + Swir/Sor/Swmax/krwmax/krow_swc) из .docx-отчёта."""
    import docx  # локальный импорт: нужен только для .docx

    doc = docx.Document(str(path))

    rows: list[dict] = []
    model_counts: dict = {}
    pending_meta: dict | None = None

    for table in doc.tables:
        if not table.rows:
            continue
        header = [c.text.strip() for c in table.rows[0].cells]

        if len(header) == 2 and header[0].strip() == "Наименование":
            pending_meta = _parse_meta_table(table)
            continue

        if pending_meta is not None and _find_curve_columns(header) is not None:
            _process_data_table(table, pending_meta, model_counts, rows)

    return pd.DataFrame(rows)
