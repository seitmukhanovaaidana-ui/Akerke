"""Проверка сопоставления Swir между данными J-функции и данными ОФП по номеру керна."""

import pandas as pd

from jfunction.swir_crosscheck import crosscheck_swir


def test_matches_by_sample_within_well():
    jf_df = pd.DataFrame(
        {
            "well": ["300", "300", "301"],
            "sample": ["011103058J02H", "999999999X01H", "011101014J02H"],
            "Swir": [0.30, 0.50, 0.26],
            "horizon": ["Мел", "Мел", "Юра"],
            "porosity_pct": [34.8, 20.0, 28.7],
            "perm_mD": [112.5, 5.0, 164.2],
        }
    )
    ofp_df = pd.DataFrame(
        {
            "model": ["300-1", "300-1", "301-1", "301-1"],
            "well": ["300", "300", "301", "301"],
            "samples": ["011103058J02H, 011103058J03H"] * 2 + ["011101014J02H"] * 2,
            "Swir": [0.262, 0.262, 0.260, 0.260],
            "Sor": [0.224, 0.224, 0.360, 0.360],
        }
    )

    result = crosscheck_swir(jf_df, ofp_df)

    assert len(result) == 2  # третий образец (999999999X01H) не найден в ОФП
    assert set(result["sample"]) == {"011103058J02H", "011101014J02H"}

    row_300 = result[result["sample"] == "011103058J02H"].iloc[0]
    assert row_300["model_OFP"] == "300-1"
    assert row_300["Swir_Pc"] == 0.30
    assert row_300["Swir_OFP"] == 0.262
    assert abs(row_300["diff"] - 0.038) < 1e-9

    row_301 = result[result["sample"] == "011101014J02H"].iloc[0]
    assert abs(row_301["diff"]) < 1e-9  # точное совпадение


def test_missing_columns_raise():
    import pytest

    jf_df = pd.DataFrame({"well": ["300"], "sample": ["A"]})
    ofp_df = pd.DataFrame({"well": ["300"], "samples": ["A"], "Swir": [0.3]})
    with pytest.raises(ValueError):
        crosscheck_swir(jf_df, ofp_df)
