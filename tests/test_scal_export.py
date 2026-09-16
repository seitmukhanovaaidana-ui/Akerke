"""Проверка форматирования ключевых слов COREYWO/SWOF для симулятора."""

from jfunction.scal_export import format_coreywo, format_swof


def test_format_coreywo_fields_and_order():
    text = format_coreywo(nw=2.1758, now=1.7347, swir=0.2912, sor=0.2511, krwmax=0.2535)

    assert text.startswith("COREYWO\n")
    lines = text.strip().splitlines()
    assert lines[1].startswith("-- SWL")
    values_line = lines[2]

    assert "0.2912" in values_line  # SWL = SWCR = Swir
    assert "0.2511" in values_line  # SOWCR = Sor
    assert "0.2535" in values_line  # KRWR = KRWU = krwmax
    assert "1.0000" in values_line  # KROLW = KRORW = krow_swc
    assert "2.1758" in values_line  # NW
    assert "1.7347" in values_line  # NOW
    assert values_line.strip().endswith("/")


def test_format_swof_table_matches_corey_formula():
    text = format_swof(nw=2.0, now=1.5, swir=0.2, sor=0.2, krwmax=0.3, n_points=5)

    assert text.startswith("SWOF\n")
    lines = [l for l in text.strip().splitlines() if l and not l.startswith("--") and l != "SWOF" and l != "/"]
    assert len(lines) == 5

    first = [float(x) for x in lines[0].split()]
    sw0, krw0, krow0 = first[0], first[1], first[2]
    assert sw0 == 0.2  # Sw = Swir -> Sw*=0
    assert krw0 == 0.0
    assert krow0 == 1.0  # (1-0)^1.5 = 1, krow_swc default 1.0

    last = [float(x) for x in lines[-1].split()]
    sw_last, krw_last, krow_last = last[0], last[1], last[2]
    assert abs(sw_last - 0.8) < 1e-6  # Swmax = 1 - Sor = 0.8
    assert abs(krw_last - 0.3) < 1e-6  # Sw*=1 -> krw=krwmax
    assert abs(krow_last - 0.0) < 1e-6
