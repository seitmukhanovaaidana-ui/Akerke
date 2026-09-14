"""
Расчёт J-функции Леверетта, нормализованной водонасыщенности (SWn)
и коэффициентов a, b экспоненциального тренда J(Sw) = a * exp(b * SWn).

Логика полностью повторяет методику, которая использовалась вручную
в Excel-файле J-Function_Gran.xlsx (лист "ZH2026(аналог Грана)").
"""

from .calc import add_derived_columns, compute_pc_res, compute_j, compute_swn
from .fit import fit_exponential, fit_by_group

__all__ = [
    "add_derived_columns",
    "compute_pc_res",
    "compute_j",
    "compute_swn",
    "fit_exponential",
    "fit_by_group",
]
