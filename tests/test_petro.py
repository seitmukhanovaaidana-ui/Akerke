"""Проверка чтения петрофизики керна и подбора k=a*exp(b*Кп) по горизонтам."""

import numpy as np
import pandas as pd
import pytest

from jfunction.petro import fit_poro_perm_by_horizon


def _synthetic_horizon(horizon, n, a, b, seed):
    rng = np.random.default_rng(seed)
    poro = rng.uniform(15, 35, n)
    perm = a * np.exp(b * poro)
    return pd.DataFrame({"horizon": horizon, "poro_open": poro, "perm_gas": perm})


def test_fit_recovers_exact_relationship_per_horizon():
    df = pd.concat(
        [
            _synthetic_horizon("A", 6, 0.5, 0.15, seed=1),
            _synthetic_horizon("B", 6, 2.0, 0.10, seed=2),
        ],
        ignore_index=True,
    )

    fits = fit_poro_perm_by_horizon(df, min_samples=5)

    assert set(fits["horizon"]) == {"A", "B"}
    row_a = fits[fits["horizon"] == "A"].iloc[0]
    assert row_a["a"] == pytest.approx(0.5, abs=1e-6)
    assert row_a["b"] == pytest.approx(0.15, abs=1e-6)
    assert row_a["r2"] == pytest.approx(1.0, abs=1e-6)


def test_horizon_below_min_samples_excluded():
    df = pd.concat(
        [
            _synthetic_horizon("A", 6, 0.5, 0.15, seed=1),
            _synthetic_horizon("Rare", 3, 1.0, 0.1, seed=3),
        ],
        ignore_index=True,
    )

    fits = fit_poro_perm_by_horizon(df, min_samples=5)

    assert "Rare" not in set(fits["horizon"])
