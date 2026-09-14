"""
Проверка расчёта J, SWn, Pc(рез) на эталонных значениях,
взятых напрямую из ячеек исходного файла J-Function_Gran.xlsx
(лист "ZH2026(аналог Грана)", образец №011103002K02H, строки 5-12).
"""

import numpy as np
import pandas as pd

from jfunction.calc import JFunctionConstants, add_derived_columns

CONST = JFunctionConstants(theta_lab_deg=30, gamma_lab=48, theta_res_deg=30, gamma_res=30, coeff=3.183)

# (Sw, Pc_lab_MPa, ожидаемый Pc_res_atm, ожидаемый SWn, ожидаемый J) - из ячеек H,I,G листа Excel
REFERENCE_ROWS = [
    (1.00, 0.000, 0.0, 1.0, 0.0),
    (0.95, 0.006, 0.037009622625, 0.9166666666666666, 0.019650464921429495),
    (0.88, 0.026, 0.160375031375, 0.8, 0.08515201465952782),
    (0.82, 0.103, 0.6353318550625, 0.7, 0.33733298115120636),
    (0.68, 0.233, 1.4372070119375, 0.4666666666666667, 0.7630930544488453),
    (0.56, 0.414, 2.5536639611249994, 0.2666666666666667, 1.3558820795786348),
    (0.45, 0.647, 3.9908709730625, 0.08333333333333331, 2.1189751340274805),
    (0.40, 1.268, 7.8213669147500005, 0.0, 4.152798253395433),
]


def test_matches_excel_reference_sample():
    df = pd.DataFrame(
        {
            "well": ["300"] * 8,
            "sample": ["011103002K02H"] * 8,
            "Sw": [r[0] for r in REFERENCE_ROWS],
            "Pc_lab_MPa": [r[1] for r in REFERENCE_ROWS],
            "porosity_pct": 31.04,
            "perm_mD": 5.83,
        }
    )

    result = add_derived_columns(df, CONST)

    expected_pc_res = [r[2] for r in REFERENCE_ROWS]
    expected_swn = [r[3] for r in REFERENCE_ROWS]
    expected_j = [r[4] for r in REFERENCE_ROWS]

    assert result["Swir"].iloc[0] == 0.40
    np.testing.assert_allclose(result["Pc_res_atm"], expected_pc_res, rtol=1e-9)
    np.testing.assert_allclose(result["SWn"], expected_swn, rtol=1e-9)
    np.testing.assert_allclose(result["J"], expected_j, rtol=1e-9)


def test_custom_perm_and_poro_powers():
    """Petrel's "Power for permeability/porosity term" can differ from 0.5."""
    const = JFunctionConstants(
        theta_lab_deg=30, gamma_lab=48, theta_res_deg=30, gamma_res=30, coeff=3.183,
        perm_power=1.0, poro_power=0.0,
    )
    df = pd.DataFrame(
        {
            "well": ["300"],
            "sample": ["s1"],
            "Sw": [0.95],
            "Pc_lab_MPa": [0.006],
            "porosity_pct": [31.04],
            "perm_mD": [5.83],
            "Swir": [0.4],
        }
    )
    result = add_derived_columns(df, const)

    pc_res = 0.037009622625  # то же Pc(рез), от constants не зависит perm/poro power
    expected_j = const.coeff * pc_res * (5.83**1.0) / ((31.04 / 100.0) ** 0.0) / (const.gamma_res * const.cos_theta_res)
    np.testing.assert_allclose(result["J"].iloc[0], expected_j, rtol=1e-9)


def test_swn_zero_at_swir_and_one_at_sw_equal_one():
    df = pd.DataFrame(
        {
            "well": ["A"] * 3,
            "sample": ["s1"] * 3,
            "Sw": [1.0, 0.7, 0.4],
            "Pc_lab_MPa": [0.0, 0.2, 1.0],
            "porosity_pct": 25.0,
            "perm_mD": 10.0,
        }
    )
    result = add_derived_columns(df, CONST)
    assert result["SWn"].iloc[0] == 1.0  # Sw=1 -> SWn=1
    assert result["SWn"].iloc[-1] == 0.0  # Sw=Swir -> SWn=0
