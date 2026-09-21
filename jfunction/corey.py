"""
Расчёт степеней Кори (nw, now) по данным ОФП (Sw, krw, krow).

Методика повторяет лист "Параметризация_Кори" исходного Excel-файла:

    Sw*  = (Sw - Swir) / (Swmax - Swir)
    Krw* = krw  / krwmax
    Kro* = krow / krow_swc

    ln(Krw*)     = nw  * ln(Sw*)        (Krw* = Sw*^nw)
    ln(1-Sw*)... = now * ln(1 - Sw*)    (Kro* = (1-Sw*)^now)

nw и now находятся методом наименьших квадратов БЕЗ свободного члена
(прямая через ноль) - точно так же, как в исходном Excel (формулы вида
SUMPRODUCT(ln(Sw*)*ln(Krw*)) / SUMPRODUCT(ln(Sw*)^2)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Sw", "krw", "krow", "Swir", "Sor", "Swmax", "krwmax", "krow_swc")


@dataclass(frozen=True)
class CoreyFitResult:
    nw: float
    now: float
    r2_w: float
    r2_o: float
    n: int

    def predict_krw_star(self, sw_star):
        sw_star = np.clip(np.asarray(sw_star, dtype=float), 0, None)
        return np.power(sw_star, self.nw)

    def predict_kro_star(self, sw_star):
        one_minus = np.clip(1 - np.asarray(sw_star, dtype=float), 0, None)
        return np.power(one_minus, self.now)


def add_corey_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Sw*, Krw*, Kro* и их логарифмы (столбцы L-R в исходном Excel)."""
    df = df.copy()
    df["Sw_star"] = (df["Sw"] - df["Swir"]) / (df["Swmax"] - df["Swir"])
    df["Krw_star"] = df["krw"] / df["krwmax"]
    df["Kro_star"] = df["krow"] / df["krow_swc"]

    sw_star = df["Sw_star"].to_numpy()
    krw_star = df["Krw_star"].to_numpy()
    kro_star = df["Kro_star"].to_numpy()

    valid_w = np.isfinite(sw_star) & np.isfinite(krw_star) & (sw_star > 0) & (krw_star > 0)
    valid_o = np.isfinite(sw_star) & np.isfinite(kro_star) & ((1 - sw_star) > 0) & (kro_star > 0)

    df["ln_Sw_star"] = np.where(valid_w, np.log(np.where(valid_w, sw_star, 1.0)), 0.0)
    df["ln_Krw_star"] = np.where(valid_w, np.log(np.where(valid_w, krw_star, 1.0)), 0.0)
    df["ln_1m_Sw_star"] = np.where(valid_o, np.log(np.where(valid_o, 1 - sw_star, 1.0)), 0.0)
    df["ln_Kro_star"] = np.where(valid_o, np.log(np.where(valid_o, kro_star, 1.0)), 0.0)
    return df


def _r2_through_origin(y: np.ndarray, y_pred: np.ndarray, x: np.ndarray) -> float:
    """R² для точек, где x != 0 (т.е. точка реально участвовала в регрессии)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    y = y[mask]
    y_pred = y_pred[mask]
    if y.size < 2:
        return float("nan")
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_corey_model(df: pd.DataFrame) -> CoreyFitResult:
    """МНК-регрессия nw, now через ноль по столбцам ln_Sw_star/ln_Krw_star/ln_1m_Sw_star/ln_Kro_star."""
    ln_sw = df["ln_Sw_star"].to_numpy()
    ln_krw = df["ln_Krw_star"].to_numpy()
    ln_1msw = df["ln_1m_Sw_star"].to_numpy()
    ln_kro = df["ln_Kro_star"].to_numpy()

    denom_w = np.sum(ln_sw * ln_sw)
    denom_o = np.sum(ln_1msw * ln_1msw)
    if denom_w == 0 or denom_o == 0:
        raise ValueError(
            "Недостаточно точек с положительными Sw*/Krw* (или (1-Sw*)/Kro*) для расчёта степеней Кори."
        )

    nw = float(np.sum(ln_sw * ln_krw) / denom_w)
    now = float(np.sum(ln_1msw * ln_kro) / denom_o)

    r2_w = _r2_through_origin(ln_krw, nw * ln_sw, ln_sw)
    r2_o = _r2_through_origin(ln_kro, now * ln_1msw, ln_1msw)

    return CoreyFitResult(nw=nw, now=now, r2_w=r2_w, r2_o=r2_o, n=len(df))


def evaluate_fixed_corey(nw: float, now: float, df: pd.DataFrame) -> CoreyFitResult:
    """
    Не подбирает nw, now, а считает R²w/R²o по ВСЕМ точкам выборки для
    введённых пользователем nw, now (чтобы можно было покрутить степени
    Кори вручную и сразу увидеть, насколько хуже/лучше они ложатся на
    фактические данные, чем автоматически подобранная медиана).
    """
    df = add_corey_derived_columns(df)
    ln_sw = df["ln_Sw_star"].to_numpy()
    ln_krw = df["ln_Krw_star"].to_numpy()
    ln_1msw = df["ln_1m_Sw_star"].to_numpy()
    ln_kro = df["ln_Kro_star"].to_numpy()

    r2_w = _r2_through_origin(ln_krw, nw * ln_sw, ln_sw)
    r2_o = _r2_through_origin(ln_kro, now * ln_1msw, ln_1msw)
    return CoreyFitResult(nw=nw, now=now, r2_w=r2_w, r2_o=r2_o, n=len(df))


def fit_corey_by_model(df: pd.DataFrame, group_col: str = "model") -> pd.DataFrame:
    """
    Считает nw, now, R²w, R²o, n для каждой модели/образца (group_col) отдельно,
    а также сводные "единые" параметры Кори по всей выборке: nw/now - медиана
    по образцам, Swir/Sor/krwmax - среднее по образцам (как в блоке "Единые
    параметры Кори (по керну)" исходного Excel).

    Возвращает DataFrame со столбцами:
        model, well, n, Swir, Sor, Swmax, krwmax, nw, r2_w, now, r2_o
    """
    df = add_corey_derived_columns(df)
    rows = []
    for name, sub in df.groupby(group_col, dropna=True, sort=False):
        try:
            fit = fit_corey_model(sub)
        except ValueError:
            continue
        rows.append(
            {
                "model": str(name),
                "well": sub["well"].iloc[0] if "well" in sub.columns else "",
                "horizon": sub["horizon"].iloc[0] if "horizon" in sub.columns else "",
                "n": fit.n,
                "Swir": sub["Swir"].iloc[0],
                "Sor": sub["Sor"].iloc[0],
                "Swmax": sub["Swmax"].iloc[0],
                "krwmax": sub["krwmax"].iloc[0],
                "nw": fit.nw,
                "r2_w": fit.r2_w,
                "now": fit.now,
                "r2_o": fit.r2_o,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class UnifiedCoreyParams:
    nw: float
    now: float
    swir: float
    sor: float
    krwmax: float
    krow_swc: float = 1.0
    n_models: int = 0


def unified_corey_params(per_model: pd.DataFrame) -> UnifiedCoreyParams | None:
    """Единые параметры Кори по всей выборке образцов: nw/now - медиана, Swir/Sor/krwmax - среднее."""
    if per_model.empty:
        return None
    return UnifiedCoreyParams(
        nw=float(per_model["nw"].median()),
        now=float(per_model["now"].median()),
        swir=float(per_model["Swir"].mean()),
        sor=float(per_model["Sor"].mean()),
        krwmax=float(per_model["krwmax"].mean()),
        n_models=len(per_model),
    )


def summarize_corey_by_horizon(per_model: pd.DataFrame) -> pd.DataFrame:
    """
    Единые параметры Кори (nw/now - медиана, Swir/Sor/krwmax - среднее),
    посчитанные ОТДЕЛЬНО для каждого горизонта - то же самое, что
    unified_corey_params(), но по группам "horizon", а не по всей выборке
    сразу. Требует столбец "horizon" в per_model (см. fit_corey_by_model);
    если горизонт не был извлечён из отчёта, возвращает пустой DataFrame.
    """
    columns = ["horizon", "n_models", "nw", "now", "Swir", "Sor", "krwmax"]
    if per_model.empty or "horizon" not in per_model.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for horizon, sub in per_model.groupby("horizon", sort=False):
        if not horizon:
            continue
        rows.append(
            {
                "horizon": horizon,
                "n_models": len(sub),
                "nw": float(sub["nw"].median()),
                "now": float(sub["now"].median()),
                "Swir": float(sub["Swir"].mean()),
                "Sor": float(sub["Sor"].mean()),
                "krwmax": float(sub["krwmax"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)
