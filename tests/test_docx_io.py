"""Проверка чтения "сырых" лабораторных отчётов в формате Word (.docx)."""

import docx

from jfunction.docx_io import load_lab_data_from_docx


def _make_sample_docx(path, rows):
    doc = docx.Document()
    table = doc.add_table(rows=1 + len(rows), cols=4)
    header = table.rows[0].cells
    header[0].text = "Наименование"
    header[1].text = "Значение"
    header[2].text = "Водонасыщенность (Sw), д.ед"
    header[3].text = "Капилярное давление, МПа"

    meta = {
        "Скважина": "300",
        "№ образца": "011103002K02H",
        "Пористость, %": "31,04",
        "Проницаемость по воде, мД": "5,83",
    }
    meta_rows = list(meta.items())

    for i, (sw, pc) in enumerate(rows, start=1):
        cells = table.rows[i].cells
        if i - 1 < len(meta_rows):
            label, value = meta_rows[i - 1]
            cells[0].text = label
            cells[1].text = value
        cells[2].text = sw
        cells[3].text = pc

    doc.save(path)


def test_reads_sample_table_with_comma_decimals(tmp_path):
    path = tmp_path / "report.docx"
    _make_sample_docx(
        path,
        rows=[("1,00", "0,000"), ("0,95", "0,006"), ("0,88", "0,026"), ("0,40", "1,268")],
    )

    df = load_lab_data_from_docx(path)

    assert len(df) == 4
    assert df["well"].iloc[0] == "300"
    assert df["sample"].iloc[0] == "011103002K02H"
    assert df["porosity_pct"].iloc[0] == 31.04
    assert df["perm_mD"].iloc[0] == 5.83
    assert list(df["Sw"]) == [1.00, 0.95, 0.88, 0.40]
    assert list(df["Pc_lab_MPa"]) == [0.000, 0.006, 0.026, 1.268]
    assert df["horizon"].isna().all()
