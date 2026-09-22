"""Проверка чтения петрофизики керна и подбора k=a*exp(b*Кп) по горизонтам."""

import numpy as np
import pandas as pd
import pytest

from jfunction.petro import (
    combine_small_formations,
    fit_poro_perm_by_horizon,
    fit_poro_perm_single,
    group_by_strat,
)


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


def test_fit_poro_perm_single_combines_arbitrary_subset():
    """Произвольная комбинация нескольких горизонтов - как один пользовательский набор."""
    df = pd.concat(
        [
            _synthetic_horizon("Ю-II", 6, 0.5, 0.15, seed=1),
            _synthetic_horizon("Ю-VI", 6, 0.5, 0.15, seed=2),
            _synthetic_horizon("Ю-III", 6, 5.0, 0.05, seed=3),
        ],
        ignore_index=True,
    )
    combined = df[df["horizon"].isin(["Ю-II", "Ю-VI"])]

    fit = fit_poro_perm_single(combined, label="Ю-II + Ю-VI")

    assert fit is not None
    assert fit["horizon"] == "Ю-II + Ю-VI"
    assert fit["n"] == 12
    assert fit["a"] == pytest.approx(0.5, abs=1e-2)
    assert fit["b"] == pytest.approx(0.15, abs=1e-2)


def test_fit_poro_perm_single_none_below_min_samples():
    df = _synthetic_horizon("Ю-II", 2, 0.5, 0.15, seed=1)
    assert fit_poro_perm_single(df, label="Ю-II", min_samples=3) is None


def test_group_by_strat_uses_authoritative_column_not_horizon_name():
    """"K1al2-1" не распознаётся эвристикой по названию, но strat="мел" уже проставлен."""
    df = pd.DataFrame(
        {
            "horizon": ["I альбский", "K1al2-1", "Ю-II", "Ю-VI", "Ю-VI", "Q1"],
            "strat": ["мел", "мел", "юра", "юра", "юра", "четверт"],
            "poro_open": [20, 21, 22, 23, 24, 25],
            "perm_gas": [1, 2, 3, 4, 5, 6],
        }
    )

    grouped = group_by_strat(df)

    assert set(grouped["horizon"]) == {"мел", "юра"}
    assert (grouped["horizon"] == "мел").sum() == 2
    assert (grouped["horizon"] == "юра").sum() == 3  # "четверт" отброшен


def test_group_by_strat_missing_column_raises():
    df = pd.DataFrame({"horizon": ["Ю-II"], "poro_open": [20], "perm_gas": [1]})
    with pytest.raises(ValueError):
        group_by_strat(df)


def test_group_by_strat_combines_enough_samples_for_fit():
    """Мел-горизонты по отдельности малочисленны, но вместе проходят порог min_samples."""
    df = pd.concat(
        [
            _synthetic_horizon("I альбский", 4, 0.3, 0.1, seed=10),
            _synthetic_horizon("K1al2-1", 4, 0.3, 0.1, seed=11),
        ],
        ignore_index=True,
    )
    df["strat"] = "мел"

    grouped = group_by_strat(df)
    fits = fit_poro_perm_by_horizon(grouped, min_samples=5)

    assert list(fits["horizon"]) == ["мел"]
    assert fits.iloc[0]["n"] == 8


def test_combine_small_formations_merges_only_requested_formation():
    df = pd.DataFrame(
        {
            "horizon": ["I альбский", "K1al2-1", "Ю-II", "Ю-II", "Ю-VI"],
            "strat": ["мел", "мел", "юра", "юра", "юра"],
            "poro_open": [20, 21, 22, 23, 24],
            "perm_gas": [1, 2, 3, 4, 5],
        }
    )

    combined = combine_small_formations(df, formations=("мел",))

    assert list(combined["horizon"]) == ["мел", "мел", "Ю-II", "Ю-II", "Ю-VI"]


def test_combine_small_formations_lets_small_mel_group_pass_threshold():
    df = pd.concat(
        [
            _synthetic_horizon("I альбский", 4, 0.3, 0.1, seed=10),
            _synthetic_horizon("K1al2-1", 4, 0.3, 0.1, seed=11),
            _synthetic_horizon("Ю-II", 14, 0.5, 0.15, seed=1),
        ],
        ignore_index=True,
    )
    df["strat"] = ["мел"] * 8 + ["юра"] * 14

    combined = combine_small_formations(df)
    fits = fit_poro_perm_by_horizon(combined, min_samples=5)

    assert set(fits["horizon"]) == {"мел", "Ю-II"}
    assert fits.loc[fits["horizon"] == "мел", "n"].iloc[0] == 8


def test_combine_small_formations_without_strat_column_is_noop():
    df = pd.DataFrame({"horizon": ["Ю-II"], "poro_open": [20], "perm_gas": [1]})
    combined = combine_small_formations(df)
    assert list(combined["horizon"]) == ["Ю-II"]
