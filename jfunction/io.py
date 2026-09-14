"""Загрузка лабораторных данных из CSV или Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["Sw", "Pc_lab_MPa", "porosity_pct", "perm_mD"]
RECOMMENDED_COLUMNS = ["well", "sample", "horizon"]


def load_lab_data(path: str | Path) -> pd.DataFrame:
    """
    Читает "длинную" таблицу лабораторных данных: одна строка = одна точка
    измерения (один Sw / Pc) для одного образца.

    Обязательные столбцы:
        Sw            - водонасыщенность, д.ед.
        Pc_lab_MPa    - капиллярное давление в лаборатории, МПа
        porosity_pct  - пористость образца, %
        perm_mD       - проницаемость образца, мД

    Желательные столбцы (для группировки/автоопределения Swir):
        well          - номер/название скважины
        sample        - номер образца
        horizon       - пласт/горизонт (используется для расчёта a,b по группам)

    Необязательный столбец:
        Swir          - остаточная водонасыщенность образца, если её
                        не нужно определять автоматически.

    Поддерживаются файлы .csv, .xlsx/.xls (в двух видах - см. ниже), а
    также "сырые" лабораторные отчёты в формате Word (.docx) - см.
    jfunction.docx_io.

    Для .xlsx/.xls автоматически распознаётся один из двух форматов:
      - "длинная" таблица с готовыми столбцами (Sw, Pc_lab_MPa, ...);
      - "сырой" перенос лабораторного отчёта - по одному блоку на образец
        ("Наименование"/"Значение" + Sw/Pc, как в Word-отчёте лаборатории),
        см. jfunction.raw_excel_io.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            from .raw_excel_io import load_lab_data_from_raw_excel

            df = load_lab_data_from_raw_excel(path)
    elif suffix == ".docx":
        from .docx_io import load_lab_data_from_docx

        df = load_lab_data_from_docx(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле {path} не хватает обязательных столбцов: {missing}.\n"
            f"Обязательные столбцы: {REQUIRED_COLUMNS}\n"
            f"Желательные столбцы: {RECOMMENDED_COLUMNS}"
        )

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sample" in df.columns:
        for col in ("well", "sample"):
            if col in df.columns:
                df[col] = df[col].ffill()

    return df
