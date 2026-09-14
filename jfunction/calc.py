"""
Формулы расчёта Pc(рез), J-функции и SWn.

Единицы измерения (как в исходном Excel):
    Sw          - д.ед. (0..1)
    Pc_lab_MPa  - капиллярное давление в лаборатории, МПа
    porosity_pct- пористость, %
    perm_mD     - проницаемость, мД
    theta_*_deg - угол смачивания, градусы
    gamma_*     - поверхностное натяжение, дин/см

Формулы (см. лист "ZH2026(аналог Грана)", столбцы G, H, I):
    Pc(рез)[атм] = Pc_lab[МПа] * 9.8692327 * (gamma_res*cos(theta_res))
                                            / (gamma_lab*cos(theta_lab))

    J(Sw) = coeff * Pc(рез) * perm_mD^perm_power / (porosity_pct/100)^poro_power
                   / (gamma_res * cos(theta_res))

    SWn = (Sw - Swir) / (1 - Swir)

где Swir - остаточная (необразованная) водонасыщенность образца,
    coeff = 3.183 (переводной коэффициент, как в исходном файле),
    perm_power, poro_power - степени при проницаемости и пористости
    (по умолчанию 0.5/0.5 - классическая формула Леверетта, sqrt(k/phi);
    в Petrel это поля "Power for permeability term" / "Power for porosity
    term" на вкладке J-function parameters, и они могут отличаться от 0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MPA_TO_ATM = 9.8692327


@dataclass(frozen=True)
class JFunctionConstants:
    """Константы J-функции (одинаковые для всех образцов одной скважины/файла)."""

    theta_lab_deg: float = 30.0
    gamma_lab: float = 48.0
    theta_res_deg: float = 30.0
    gamma_res: float = 30.0
    coeff: float = 3.183
    perm_power: float = 0.5
    poro_power: float = 0.5

    @property
    def cos_theta_lab(self) -> float:
        return math.cos(math.radians(self.theta_lab_deg))

    @property
    def cos_theta_res(self) -> float:
        return math.cos(math.radians(self.theta_res_deg))


def compute_pc_res(pc_lab_mpa, const: JFunctionConstants):
    """Пересчёт капиллярного давления из лабораторных условий в резервуарные (атм)."""
    ratio = (const.gamma_res * const.cos_theta_res) / (const.gamma_lab * const.cos_theta_lab)
    return pc_lab_mpa * MPA_TO_ATM * ratio


def compute_j(pc_res_atm, porosity_pct, perm_mD, const: JFunctionConstants):
    """
    Значение J-функции. При perm_power = poro_power = 0.5 (по умолчанию)
    это классическая формула Леверетта J = Pc/(σcosθ) * sqrt(k/φ).
    """
    return (
        const.coeff
        * pc_res_atm
        * np.power(perm_mD, const.perm_power)
        * np.power(porosity_pct / 100.0, -const.poro_power)
        / (const.gamma_res * const.cos_theta_res)
    )


def compute_swn(sw, swir):
    """Нормализованная водонасыщенность SWn = (Sw - Swir) / (1 - Swir)."""
    return (sw - swir) / (1 - swir)


def _derive_swir(df: pd.DataFrame, sample_cols: list[str]) -> pd.Series:
    """
    Swir по умолчанию = Sw при максимальном Pc в пределах одного образца
    (то есть последняя точка капилляриметрии - как в исходном Excel,
    где Swir берётся из последней строки блока образца).
    """
    idx_max_pc = df.groupby(sample_cols)["Pc_lab_MPa"].transform("idxmax")
    return df.loc[idx_max_pc, "Sw"].to_numpy()


def add_derived_columns(
    df: pd.DataFrame,
    const: JFunctionConstants,
    sample_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Добавляет к таблице лабораторных данных столбцы Pc_res_atm, Swir, SWn, J.

    Параметры
    ---------
    df : DataFrame с обязательными столбцами
        Sw, Pc_lab_MPa, porosity_pct, perm_mD
        и, если Swir не задан явно, столбцами для группировки по образцу
        (по умолчанию ['well', 'sample']).
    const : JFunctionConstants
    sample_cols : список столбцов, идентифицирующих один образец
        (нужен, чтобы правильно найти Swir для каждого образца).

    Возвращает копию df с добавленными столбцами.
    """
    required = {"Sw", "Pc_lab_MPa", "porosity_pct", "perm_mD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В исходных данных не хватает столбцов: {sorted(missing)}")

    out = df.copy()

    if "Swir" not in out.columns:
        sample_cols = sample_cols or ["well", "sample"]
        missing_sample_cols = [c for c in sample_cols if c not in out.columns]
        if missing_sample_cols:
            raise ValueError(
                "Нет столбца Swir, а для его автоматического определения "
                f"не хватает столбцов группировки по образцу: {missing_sample_cols}. "
                "Либо добавьте столбец Swir в исходные данные, либо укажите "
                "правильные sample_cols."
            )
        out["Swir"] = _derive_swir(out, sample_cols)

    out["Pc_res_atm"] = compute_pc_res(out["Pc_lab_MPa"], const)
    out["SWn"] = compute_swn(out["Sw"], out["Swir"])
    out["J"] = compute_j(out["Pc_res_atm"], out["porosity_pct"], out["perm_mD"], const)
    return out
