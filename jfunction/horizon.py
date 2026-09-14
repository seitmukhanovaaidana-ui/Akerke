"""
Разметка образцов по горизонтам через таблицу глубин пласта.

Полезно, когда в исходных лабораторных данных (например, при загрузке
"сырого" Word-отчёта) нет столбца horizon, но есть глубина образца, а
границы горизонтов (кровля/подошва) по каждой скважине известны отдельно.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["well", "depth_from", "depth_to", "horizon"]


def load_horizon_map(path: str | Path) -> pd.DataFrame:
    """
    Читает таблицу границ горизонтов по скважинам.

    Обязательные столбцы:
        well        - номер/название скважины
        depth_from  - глубина кровли пласта, м
        depth_to    - глубина подошвы пласта, м
        horizon     - название горизонта (Мел, Юра и т.п.)

    Поддерживаются файлы .csv и .xlsx/.xls.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла разметки горизонтов: {path.suffix}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле разметки горизонтов не хватает столбцов: {missing}. "
            f"Нужны столбцы: {REQUIRED_COLUMNS}"
        )
    return df


def assign_horizon_by_depth(df: pd.DataFrame, horizon_map: pd.DataFrame) -> pd.DataFrame:
    """
    Проставляет/переопределяет столбец horizon по глубине образца (depth)
    и границам пласта для соответствующей скважины (well).

    Образцы, чья глубина не попала ни в один интервал заданной скважины,
    остаются без горизонта (None).
    """
    if "depth" not in df.columns or "well" not in df.columns:
        raise ValueError("Для разметки по глубине нужны столбцы 'well' и 'depth' в исходных данных.")

    out = df.copy()
    horizons = []
    for _, row in out.iterrows():
        well, depth = row["well"], row["depth"]
        match = horizon_map[
            (horizon_map["well"].astype(str) == str(well))
            & (horizon_map["depth_from"] <= depth)
            & (depth < horizon_map["depth_to"])
        ]
        horizons.append(match["horizon"].iloc[0] if not match.empty else None)

    out["horizon"] = horizons
    return out
