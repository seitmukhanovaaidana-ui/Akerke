"""Проверка расчёта степеней Кори (nw, now) по данным ОФП."""

import numpy as np
import pandas as pd
import pytest

from jfunction.corey import add_corey_derived_columns, fit_corey_by_model, fit_corey_model, unified_corey_params


def _synthetic_model(model, well, swir, swmax, krwmax, nw, now, n=8):
    sw_star = np.linspace(0.001, 0.999, n)
    sw = swir + sw_star * (swmax - swir)
    krw = krwmax * sw_star**nw
    krow = 1.0 * (1 - sw_star) ** now
    return pd.DataFrame(
        {
            "model": model,
            "well": well,
            "Sw": sw,
            "krw": krw,
            "krow": krow,
            "Swir": swir,
            "Sor": 1 - swmax,
            "Swmax": swmax,
            "krwmax": krwmax,
            "krow_swc": 1.0,
        }
    )


def test_fit_corey_model_recovers_exact_exponents():
    df = _synthetic_model("A-1", "100", swir=0.2, swmax=0.8, krwmax=0.3, nw=2.0, now=1.5)
    df = add_corey_derived_columns(df)
    fit = fit_corey_model(df)

    assert fit.nw == pytest.approx(2.0)
    assert fit.now == pytest.approx(1.5)
    assert fit.r2_w == pytest.approx(1.0)
    assert fit.r2_o == pytest.approx(1.0)


def test_fit_corey_by_model_and_unified_params():
    df = pd.concat(
        [
            _synthetic_model("A-1", "100", swir=0.2, swmax=0.8, krwmax=0.3, nw=2.0, now=1.5),
            _synthetic_model("A-2", "100", swir=0.25, swmax=0.75, krwmax=0.28, nw=2.4, now=1.7),
        ],
        ignore_index=True,
    )

    per_model = fit_corey_by_model(df, group_col="model")
    assert set(per_model["model"]) == {"A-1", "A-2"}
    assert per_model.loc[per_model["model"] == "A-1", "nw"].iloc[0] == pytest.approx(2.0)
    assert per_model.loc[per_model["model"] == "A-2", "now"].iloc[0] == pytest.approx(1.7)

    unified = unified_corey_params(per_model)
    assert unified is not None
    assert unified.n_models == 2
    assert unified.nw == pytest.approx((2.0 + 2.4) / 2)
    assert unified.swir == pytest.approx((0.2 + 0.25) / 2)
