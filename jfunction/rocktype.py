"""Простое разбиение образцов на типы породы по проницаемости (rock typing)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_by_permeability(perm_mD: pd.Series, breakpoints: list[float]) -> pd.Series:
    """
    Делит значения проницаемости на интервалы по заданным границам (мД).

    Например, breakpoints=[1, 10, 100] даёт классы:
    "< 1", "1-10", "10-100", ">= 100".
    """
    edges = [-np.inf, *sorted(breakpoints), np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo == -np.inf:
            labels.append(f"< {hi:g}")
        elif hi == np.inf:
            labels.append(f">= {lo:g}")
        else:
            labels.append(f"{lo:g}-{hi:g}")
    return pd.cut(perm_mD, bins=edges, labels=labels, right=False)
