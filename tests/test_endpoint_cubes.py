"""Проверка подбора корреляций концевых точек (Swir/Sor/krwmax) от Кп/k по горизонтам."""

import numpy as np
import pandas as pd
import pytest

from jfunction.endpoint_cubes import apply_correlation, fit_best_correlation, fit_endpoint_cubes


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
