"""
Экспорт единых параметров Кори (nw, now, Swir, Sor, krwmax, krow_swc) в
формат ключевых слов симулятора (Eclipse/тНавигатор) - COREYWO (корреляция)
или SWOF (таблица), с обозначениями по ГОСТ-подобной SCAL-номенклатуре
(SWL, SWCR, SWU, SOWCR, KRW, KRWR, KRO, KRORW, PCOW, NW, NOW):

    SWL, SWCR   - наименьшая / критическая водонасыщенность (Krw=0) - Swir
    SWU         - максимальная водонасыщенность (по умолчанию 1)
    SOWCR       - остаточная нефтенасыщенность в системе вода-нефть - Sor
    KRW, KRWR   - макс. ОФП по воде / ОФП по воде при Sowcr - krwmax
    KRO, KRORW  - макс. ОФП по нефти / ОФП по нефти при Swcr - krow_swc
    PCOW        - капиллярное давление при Swl (здесь не считается - 0)
    NW, NOW     - показатели степени Кори (наши nw, now)

Модель Кори здесь двухточечная (без отдельной "критической" точки внутри
диапазона): SWCR совпадает с SWL, поэтому KRORW = KRO.
"""

from __future__ import annotations

import numpy as np


def format_coreywo(
    nw: float,
    now: float,
    swir: float,
    sor: float,
    krwmax: float,
    krow_swc: float = 1.0,
    swu: float = 1.0,
    pcow: float = 0.0,
) -> str:
    """Строка ключевого слова COREYWO (корреляция Кори для системы вода-нефть)."""
    swl = swcr = swir
    sowcr = sor
    krolw = krorw = krow_swc
    krwr = krwu = krwmax

    header = "-- SWL    SWU    SWCR   SOWCR  KROLW  KRORW  KRWR   KRWU   PCOW   NOW    NW     NP  SPC0"
    values = (
        f"   {swl:<6.4f} {swu:<6.4f} {swcr:<6.4f} {sowcr:<6.4f} "
        f"{krolw:<6.4f} {krorw:<6.4f} {krwr:<6.4f} {krwu:<6.4f} "
        f"{pcow:<6.4f} {now:<6.4f} {nw:<6.4f} 0  /"
    )
    return "COREYWO\n" + header + "\n" + values + "\n"


def format_swof(
    nw: float,
    now: float,
    swir: float,
    sor: float,
    krwmax: float,
    krow_swc: float = 1.0,
    n_points: int = 20,
) -> str:
    """Строка ключевого слова SWOF (таблица Sw/Krw/Krow/Pcow), рассчитанная по формуле Кори."""
    swmax_eff = 1.0 - sor
    sw_grid = np.linspace(swir, swmax_eff, n_points)
    sw_star = np.clip((sw_grid - swir) / (swmax_eff - swir), 0, 1)
    krw = krwmax * np.power(sw_star, nw)
    krow = krow_swc * np.power(1 - sw_star, now)

    lines = ["SWOF", "-- Sw       Krw        Krow       Pcow"]
    for sw, kw, ko in zip(sw_grid, krw, krow):
        lines.append(f"   {sw:<10.4f} {kw:<10.4f} {ko:<10.4f} 0")
    lines.append("/")
    return "\n".join(lines) + "\n"
