import numpy as np

from jfunction.fit import fit_exponential


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    a_true, b_true = 12.5, -4.1

    swn = np.linspace(0, 1, 50)
    j = a_true * np.exp(b_true * swn)

    result = fit_exponential(swn, j)

    assert np.isclose(result.a, a_true, rtol=1e-6)
    assert np.isclose(result.b, b_true, rtol=1e-6)
    assert result.r2 > 0.999
    assert result.n == 50


def test_fit_ignores_non_positive_j():
    swn = np.array([0.0, 0.2, 0.4, 0.6])
    j = np.array([0.0, 1.0, 2.0, 4.0])  # первая точка J=0 должна быть отброшена

    result = fit_exponential(swn, j)
    assert result.n == 3
