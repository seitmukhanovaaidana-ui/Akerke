"""Проверка подбора корреляций концевых точек (Swir/Sor/krwmax) от Кп/k по горизонтам."""

import numpy as np
import pandas as pd
import pytest

from jfunction.endpoint_cubes import (
    apply_correlation,
    classify_formation,
    fit_best_correlation,
    fit_endpoint_cubes,
    group_by_formation,
)


def test_fit_best_correlation_recovers_exact_exponential():
    x = np.linspace(10, 30, 8)
    y = 0.5 * np.exp(-0.02 * x)

    corr = fit_best_correlation(x, y, horizon="H1", endpoint="Swir", x_var="porosity_pct")

    assert corr is not None
    assert corr.form == "exp"
    assert corr.a == pytest.approx(0.5, rel=1e-4)
    assert corr.b == pytest.approx(-0.02, rel=1e-3)
    assert corr.r2 == pytest.approx(1.0, abs=1e-6)


def test_fit_best_correlation_none_with_too_few_points():
    corr = fit_best_correlation([1, 2], [0.1, 0.2], "H1", "Sor", "perm_mD")
    assert corr is None


def test_fit_endpoint_cubes_skips_small_horizons():
    df = pd.DataFrame(
        {
            "horizon": ["A"] * 6 + ["B"] * 2,
            "porosity_pct": [10, 15, 20, 25, 30, 35, 10, 20],
            "perm_mD": [1, 5, 20, 50, 150, 400, 2, 30],
            "Swir": [0.4, 0.35, 0.3, 0.27, 0.24, 0.2, 0.4, 0.3],
            "Sor": [0.3, 0.28, 0.26, 0.25, 0.23, 0.2, 0.3, 0.25],
            "krwmax": [0.2, 0.22, 0.25, 0.28, 0.3, 0.33, 0.2, 0.25],
        }
    )

    fits = fit_endpoint_cubes(df, min_samples=4)

    horizons = {c.horizon for c in fits}
    assert horizons == {"A"}
    assert len(fits) == 6  # 3 endpoints x 2 x_vars for horizon A


def test_apply_correlation_matches_predict():
    x = np.linspace(10, 30, 8)
    y = 0.5 * np.exp(-0.02 * x)
    corr = fit_best_correlation(x, y, "H1", "Swir", "porosity_pct")

    result = apply_correlation(corr, [15, 20, 25])
    expected = corr.predict([15, 20, 25])
    assert np.allclose(result, expected)


def test_classify_formation():
    assert classify_formation("апт") == "мел"
    assert classify_formation("II-alb (альбский)") == "мел"
    assert classify_formation("мел (не уточнён)") == "мел"
    assert classify_formation("III-nc (неокомский)") == "мел"
    assert classify_formation("Ю-VI") == "юра"
    assert classify_formation("Ю-III") == "юра"
    assert classify_formation("юра") == "юра"
    assert classify_formation("I1-J2 (юрский)") == "юра"
    assert classify_formation("что-то непонятное") is None


def test_group_by_formation_drops_unclassified_and_merges_groups():
    df = pd.DataFrame(
        {
            "horizon": ["апт", "Ю-III", "Ю-V", "неизвестно"],
            "porosity_pct": [30, 25, 28, 20],
        }
    )
    grouped = group_by_formation(df)

    assert len(grouped) == 3
    assert set(grouped["horizon"]) == {"мел", "юра"}
    assert list(grouped.loc[grouped["horizon"] == "юра", "porosity_pct"]) == [25, 28]
