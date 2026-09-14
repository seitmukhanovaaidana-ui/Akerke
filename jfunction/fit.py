"""
Подбор коэффициентов a, b экспоненциальной модели J(Sw) = a * exp(b * SWn)
методом наименьших квадратов (линеаризация: ln(J) = ln(a) + b * SWn).

Это тот же метод МНК, что был реализован вручную в Excel
(столбцы AV1:AY11 листа "ZH2026(аналог Грана)").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitResult:
    a: float
    b: float
    r2: float
    n: int

    def predict(self, swn):
        """J(SWn) по подобранной модели."""
        return self.a * np.exp(self.b * np.asarray(swn, dtype=float))

    def __str__(self) -> str:
        return f"J(SWn) = {self.a:.4f} * exp({self.b:.4f} * SWn)  (n={self.n}, R²={self.r2:.4f})"


def fit_exponential(swn, j) -> FitResult:
    """
    МНК-регрессия y = ln(J) от x = SWn.

    Точки, где J <= 0 (ln(J) не определён) или SWn/J = NaN, отбрасываются -
    так же, как в исходных Excel-формулах (IF(J>0, LN(J), "")).
    """
    x = np.asarray(swn, dtype=float)
    j = np.asarray(j, dtype=float)

    mask = np.isfinite(x) & np.isfinite(j) & (j > 0)
    x = x[mask]
    y = np.log(j[mask])
    n = x.size

    if n < 2:
        raise ValueError("Недостаточно точек с J > 0 для построения регрессии (нужно минимум 2).")

    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()

    b = (n * sxy - sx * sy) / (n * sxx - sx**2)
    ln_a = (sy - b * sx) / n
    a = np.exp(ln_a)

    y_pred = ln_a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(a=float(a), b=float(b), r2=float(r2), n=n)


def fit_by_group(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """
    Считает a, b, R², n для всей выборки ("Все образцы") и,
    если указан group_col (например 'horizon'), дополнительно для каждой группы.

    Возвращает DataFrame со столбцами: group, n, a, b, r2.
    """
    rows = []

    overall = fit_exponential(df["SWn"], df["J"])
    rows.append({"group": "Все образцы", "n": overall.n, "a": overall.a,
                 "b": overall.b, "r2": overall.r2})

    if group_col is not None and group_col in df.columns:
        for name, sub in df.groupby(group_col, dropna=True):
            try:
                res = fit_exponential(sub["SWn"], sub["J"])
            except ValueError:
                continue
            rows.append({"group": str(name), "n": res.n, "a": res.a,
                         "b": res.b, "r2": res.r2})

    return pd.DataFrame(rows)
