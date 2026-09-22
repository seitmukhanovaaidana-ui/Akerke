"""Проверка подбора корреляций концевых точек (Swir/Sor/krwmax) от Кп/k по горизонтам."""

import numpy as np
import pandas as pd
import pytest

from jfunction.endpoint_cubes import (
    Correlation,
    apply_correlation,
    check_apply_inputs,
    classify_formation,
    fit_all_forms,
    fit_best_correlation,
    fit_endpoint_cubes,
    group_by_formation,
    load_endpoint_summary_docx,
    quality_flags,
    quality_level,
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


def test_quality_flags_clean_fit_is_ok():
    x = np.linspace(10, 30, 10)
    y = 0.5 * np.exp(-0.02 * x)
    corr = fit_best_correlation(x, y, "H1", "Swir", "porosity_pct")

    assert quality_flags(corr) == []
    assert quality_level(corr) == "ok"


def test_quality_flags_small_sample_is_warning():
    corr = Correlation(
        horizon="H1", endpoint="Swir", x_var="porosity_pct", form="linear",
        a=0.4, b=-0.005, r2=0.95, n=4, x_min=10, x_max=30,
    )
    flags = quality_flags(corr)
    assert any("выборка" in f for f in flags)
    assert quality_level(corr) == "warning"


def test_quality_flags_low_r2_is_bad():
    corr = Correlation(
        horizon="H1", endpoint="Swir", x_var="porosity_pct", form="linear",
        a=0.4, b=-0.005, r2=0.2, n=8, x_min=10, x_max=30,
    )
    flags = quality_flags(corr)
    assert any("низкий R²" in f for f in flags)
    assert quality_level(corr) == "bad"


def test_quality_flags_out_of_physical_range_is_bad():
    # линейная зависимость, дающая отрицательные значения Sor в конце диапазона
    corr = Correlation(
        horizon="H1", endpoint="Sor", x_var="porosity_pct", form="linear",
        a=0.3, b=-0.02, r2=0.9, n=10, x_min=5, x_max=40,
    )
    flags = quality_flags(corr)
    assert any("физический диапазон" in f for f in flags)
    assert quality_level(corr) == "bad"


def test_quality_flags_unstable_extreme_coefficients_flagged():
    # аналог реального случая Ю-VIб: огромный a, отрицательный b -
    # в пределах диапазона данных прогноз в норме, но неустойчив за его границами
    corr = Correlation(
        horizon="Ю-VIб", endpoint="Swir", x_var="porosity_pct", form="exp",
        a=1.4e14, b=-3.5, n=5, r2=0.85, x_min=9.0, x_max=9.6,
    )
    flags = quality_flags(corr)
    assert any("выборка" in f for f in flags)
    assert any("экстраполя" in f for f in flags)
    assert quality_level(corr) in ("warning", "bad")


def test_check_apply_inputs_flags_extrapolation_and_physical_bounds():
    corr = Correlation(
        horizon="H1", endpoint="Swir", x_var="porosity_pct", form="linear",
        a=0.6, b=-0.02, r2=0.9, n=10, x_min=10, x_max=30,
    )
    warnings = check_apply_inputs(corr, [15, 50])
    assert any("вне диапазона" in w for w in warnings)

    result = apply_correlation(corr, [50])
    assert result[0] < 0
    assert any("физический диапазон" in w for w in warnings)


def test_check_apply_inputs_no_warnings_when_in_range_and_physical():
    corr = Correlation(
        horizon="H1", endpoint="Swir", x_var="porosity_pct", form="linear",
        a=0.6, b=-0.01, r2=0.9, n=10, x_min=10, x_max=30,
    )
    assert check_apply_inputs(corr, [15, 20, 25]) == []


def test_classify_formation_recognizes_valanginian_as_mel():
    assert classify_formation("Валанжин") == "мел"
    assert classify_formation("валанжинский ярус") == "мел"
    assert classify_formation("Триас") is None
    assert classify_formation("") is None


def test_load_endpoint_summary_docx_flat_format(tmp_path):
    """Формат 2 (Горизонт/Скважина/Идентификатор образца/Sw/krw/krow) - есть горизонт, нет Кп/k."""
    import docx

    doc = docx.Document()
    table = doc.add_table(rows=5, cols=6)
    header = table.rows[0].cells
    for i, name in enumerate(["Горизонт", "Скважина", "Идентификатор образца", "Sw", "krw", "krow"]):
        header[i].text = name
    rows = [
        ("Валанжин", "700", "Модель 1", "0.412", "0.000", "1.000"),
        ("Валанжин", "700", "Модель 1", "0.709", "0.155", "0.000"),
        ("Юра", "800", "Модель 2", "0.300", "0.000", "1.000"),
        ("Юра", "800", "Модель 2", "0.600", "0.200", "0.000"),
    ]
    for i, row in enumerate(rows, start=1):
        cells = table.rows[i].cells
        for j, value in enumerate(row):
            cells[j].text = value
    path = tmp_path / "ofp_flat.docx"
    doc.save(path)

    df = load_endpoint_summary_docx(path)

    assert len(df) == 2
    assert set(df["horizon"]) == {"Валанжин", "Юра"}
    assert df["porosity_pct"].isna().all()
    valanzhin = df[df["horizon"] == "Валанжин"].iloc[0]
    assert valanzhin["Swir"] == 0.412
    assert valanzhin["krwmax"] == 0.155


def test_load_endpoint_summary_docx_meta_format_has_no_horizon(tmp_path):
    """Формат 1 (метаданные + кривая) не содержит горизонт - столбец должен остаться пустым."""
    import docx

    doc = docx.Document()
    meta_rows = [
        ("Данные по керну (модели образцов)", "Данные по керну (модели образцов)"),
        ("№ модели образцов", "Модель №1"),
        ("Скважина", "300"),
        ("Пористость, %", "34,82"),
        ("Проницаемость по газу, мД", "112,50"),
        ("Остаточная водонасыщенность (Swi), доли ед.", "0,262"),
        ("Остаточная нефтенасыщенность (Sow), доли ед.", "0,224"),
    ]
    meta_table = doc.add_table(rows=1 + len(meta_rows), cols=2)
    meta_table.rows[0].cells[0].text = "Наименование"
    meta_table.rows[0].cells[1].text = "Значение"
    for i, (label, value) in enumerate(meta_rows, start=1):
        meta_table.rows[i].cells[0].text = label
        meta_table.rows[i].cells[1].text = value

    curve_table = doc.add_table(rows=3, cols=5)
    header = curve_table.rows[0].cells
    header[2].text = "Sw"
    header[3].text = "krw"
    header[4].text = "krow"
    for i, (sw, krw, krow) in enumerate([("0,26", "0,000", "1"), ("0,78", "0,308", "0,000")], start=1):
        cells = curve_table.rows[i].cells
        cells[2].text = sw
        cells[3].text = krw
        cells[4].text = krow

    path = tmp_path / "ofp_meta.docx"
    doc.save(path)

    df = load_endpoint_summary_docx(path)

    assert len(df) == 1
    assert df["horizon"].iloc[0] == ""
    assert df["porosity_pct"].iloc[0] == 34.82
    assert df["perm_mD"].iloc[0] == 112.50


def test_fit_all_forms_returns_every_applicable_form():
    x = np.linspace(10, 30, 8)
    y = 0.5 * np.exp(-0.02 * x)

    options = fit_all_forms(x, y, horizon="H1", endpoint="Swir", x_var="porosity_pct")

    assert set(options) == {"linear", "log", "power", "exp", "poly2"}
    assert options["exp"].a == pytest.approx(0.5, rel=1e-4)
    assert options["exp"].b == pytest.approx(-0.02, rel=1e-3)
    # каждая форма несёт свой собственный R², а не только у автоматически лучшей
    assert options["linear"].r2 != options["exp"].r2


def test_fit_all_forms_excludes_inapplicable_forms_for_nonpositive_x():
    x = np.array([-2, -1, 0, 1, 2, 3])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    options = fit_all_forms(x, y, "H1", "Swir", "porosity_pct")

    assert "log" not in options  # ln(x) не определён при x<=0
    assert "power" not in options  # x^b не определён при x<=0
    assert "linear" in options
    assert "poly2" in options


def test_fit_all_forms_poly2_requires_at_least_four_points():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])

    options = fit_all_forms(x, y, "H1", "Swir", "porosity_pct")
    assert "poly2" not in options


def test_poly2_predict_matches_quadratic_fit():
    x = np.linspace(1, 10, 6)
    y = 2 + 3 * x - 0.5 * x**2

    corr = fit_all_forms(x, y, "H1", "Swir", "porosity_pct")["poly2"]

    assert corr.a == pytest.approx(2, abs=1e-6)
    assert corr.b == pytest.approx(3, abs=1e-6)
    assert corr.c == pytest.approx(-0.5, abs=1e-6)
    assert corr.r2 == pytest.approx(1.0, abs=1e-6)
    assert corr.predict([4.0])[0] == pytest.approx(2 + 3 * 4 - 0.5 * 16, abs=1e-6)


def test_fit_best_correlation_never_picks_polynomial():
    """poly2 доступен только через fit_all_forms() для ручного выбора -
    автоматический подбор им не пользуется (лишний параметр завышает R² на
    маленькой выборке)."""
    x = np.linspace(10, 30, 5)
    y = 0.5 * np.exp(-0.02 * x)

    corr = fit_best_correlation(x, y, "H1", "Swir", "porosity_pct")
    assert corr.form != "poly2"
