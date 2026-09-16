"""
Кубы концевых точек: корреляция Swir/Sor/krwmax (концевые точки Кори) с
пористостью и проницаемостью по каждому горизонту отдельно, и применение
этой корреляции к произвольному массиву значений Кп/k (то есть - к кубу
пористости/проницаемости из геологической модели).

Источник данных - сводная таблица по образцам вида листа "ОФП"
("Таблица 2.4.2 - Относительная проницаемость в системе вода-нефть"):
одна строка на образец/модель, со столбцами скважина, модель, горизонт,
пористость, проницаемость, Swir, Sor, ОФП по воде при Sor (krwmax).

Методика повторяет то, что было сделано вручную в исходном Excel (12
диаграмм рассеяния с линиями тренда на листе "ОФП"): для каждой пары
(горизонт, концевая точка) перебираются 4 типа зависимости от Кп и от k
(линейная, логарифмическая, степенная, экспоненциальная), выбирается та,
что даёт максимальный R².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SUMMARY_COLUMNS = {
    "well": 2,
    "model": 3,
    "depth": 4,
    "horizon": 6,
    "porosity_pct": 7,
    "perm_mD": 8,
    "Swir": 12,
    "Sor": 13,
    "krwmax": 14,
    "krow_swc": 15,
}

ENDPOINTS = ("Swir", "Sor", "krwmax")
X_VARS = ("porosity_pct", "perm_mD")


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_endpoint_summary_xlsx(path: str, sheet_name: str = "ОФП") -> pd.DataFrame:
    """
    Читает сводную таблицу по образцам (лист "ОФП" - Таблица 2.4.2):
    well, model, horizon, porosity_pct, perm_mD, Swir, Sor, krwmax, krow_swc.
    """
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"На листе не найдено «{sheet_name}». Доступные листы: {wb.sheetnames}")
    ws = wb[sheet_name]

    rows: list[dict] = []
    for r in range(5, ws.max_row + 1):
        well = ws.cell(row=r, column=SUMMARY_COLUMNS["well"]).value
        horizon = ws.cell(row=r, column=SUMMARY_COLUMNS["horizon"]).value
        if well is None or horizon is None:
            continue
        row = {key: ws.cell(row=r, column=col).value for key, col in SUMMARY_COLUMNS.items()}
        for key in ("depth", "porosity_pct", "perm_mD", "Swir", "Sor", "krwmax", "krow_swc"):
            row[key] = _to_float(row[key])
        row["well"] = str(well).strip()
        row["horizon"] = str(horizon).strip()
        rows.append(row)

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Correlation:
    horizon: str
    endpoint: str
    x_var: str
    form: str  # "linear" | "log" | "power" | "exp"
    a: float
    b: float
    r2: float
    n: int
    x_min: float
    x_max: float

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if self.form == "linear":
            return self.a + self.b * x
        if self.form == "log":
            return self.a + self.b * np.log(x)
        if self.form == "power":
            return self.a * np.power(x, self.b)
        if self.form == "exp":
            return self.a * np.exp(self.b * x)
        raise ValueError(f"Неизвестная форма зависимости: {self.form}")


def _r2(y, pred) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _fit_form(form: str, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Возвращает (a, b, r2) для заданной формы или None, если форма неприменима к данным."""
    try:
        if form == "linear":
            b, a = np.polyfit(x, y, 1)
            pred = a + b * x
        elif form == "log":
            if np.any(x <= 0):
                return None
            b, a = np.polyfit(np.log(x), y, 1)
            pred = a + b * np.log(x)
        elif form == "power":
            if np.any(x <= 0) or np.any(y <= 0):
                return None
            b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
            a = np.exp(ln_a)
            pred = a * np.power(x, b)
        elif form == "exp":
            if np.any(y <= 0):
                return None
            b, ln_a = np.polyfit(x, np.log(y), 1)
            a = np.exp(ln_a)
            pred = a * np.exp(b * x)
        else:
            raise ValueError(form)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(a), float(b), _r2(y, pred)


def fit_best_correlation(x, y, horizon: str, endpoint: str, x_var: str) -> Correlation | None:
    """Перебирает linear/log/power/exp, возвращает лучшую по R² (или None, если данных мало)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None

    best = None
    for form in ("linear", "log", "power", "exp"):
        fit = _fit_form(form, x, y)
        if fit is None:
            continue
        a, b, r2 = fit
        if best is None or (r2 == r2 and r2 > best[1]):  # r2==r2 отсекает NaN
            best = (form, r2, a, b)

    if best is None:
        return None
    form, r2, a, b = best
    return Correlation(
        horizon=horizon, endpoint=endpoint, x_var=x_var, form=form,
        a=a, b=b, r2=r2, n=len(x), x_min=float(x.min()), x_max=float(x.max()),
    )


def fit_endpoint_cubes(df: pd.DataFrame, min_samples: int = 4) -> list[Correlation]:
    """
    Для каждого горизонта и каждой концевой точки (Swir, Sor, krwmax)
    подбирает лучшую корреляцию отдельно от пористости и от проницаемости
    (горизонты с числом образцов меньше min_samples пропускаются).
    """
    results: list[Correlation] = []
    for horizon, sub in df.groupby("horizon"):
        if len(sub) < min_samples:
            continue
        for endpoint in ENDPOINTS:
            for x_var in X_VARS:
                sub2 = sub.dropna(subset=[x_var, endpoint])
                if len(sub2) < min_samples:
                    continue
                corr = fit_best_correlation(sub2[x_var], sub2[endpoint], horizon, endpoint, x_var)
                if corr is not None:
                    results.append(corr)
    return results


def apply_correlation(corr: Correlation, x_values) -> np.ndarray:
    """Применяет корреляцию к массиву значений Кп/k - то есть строит "куб" концевой точки."""
    return corr.predict(x_values)
