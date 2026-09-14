#!/usr/bin/env python3
"""
CLI для расчёта J-функции, SWn и коэффициентов a, b экспоненциального тренда
J(Sw) = a * exp(b * SWn) по лабораторным данным керна.

Пример запуска:
    python run.py --input examples/lab_data_example.csv --output-dir results

Формат входного файла (CSV или XLSX), одна строка = одна точка измерения:
    well, sample, horizon, Sw, Pc_lab_MPa, porosity_pct, perm_mD

Подробности столбцов и констант - см. README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jfunction.calc import JFunctionConstants, add_derived_columns
from jfunction.fit import FitResult, fit_by_group
from jfunction.horizon import assign_horizon_by_depth, load_horizon_map
from jfunction.io import load_lab_data
from jfunction.report import plot_j_function, save_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Путь к файлу с лабораторными данными (.csv или .xlsx)")
    p.add_argument("--output-dir", default="results", help="Папка для результатов (по умолчанию: results)")
    p.add_argument("--group-col", default="horizon",
                   help="Столбец для расчёта отдельных a,b по группам (например, по пласту). "
                        "По умолчанию 'horizon'. Укажите '' чтобы отключить.")

    p.add_argument("--theta-lab", type=float, default=30.0, help="Угол смачивания, лаб., град (по умолч. 30)")
    p.add_argument("--gamma-lab", type=float, default=48.0, help="Пов. натяжение, лаб., дин/см (по умолч. 48)")
    p.add_argument("--theta-res", type=float, default=30.0, help="Угол смачивания, резервуар, град (по умолч. 30)")
    p.add_argument("--gamma-res", type=float, default=30.0, help="Пов. натяжение, резервуар, дин/см (по умолч. 30)")
    p.add_argument("--coeff", type=float, default=3.183, help="Переводной коэффициент J-функции (по умолч. 3.183)")
    p.add_argument("--perm-power", type=float, default=0.5,
                   help="Степень при проницаемости (Power for permeability term в Petrel, по умолч. 0.5)")
    p.add_argument("--poro-power", type=float, default=0.5,
                   help="Степень при пористости (Power for porosity term в Petrel, по умолч. 0.5)")
    p.add_argument("--horizon-map", default=None,
                   help="Файл разметки горизонтов по глубине (столбцы well, depth_from, depth_to, "
                        "horizon) - проставляет/переопределяет horizon по глубине образца. "
                        "Полезно, когда исходные данные (например, .docx) не содержат горизонт.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    group_col = args.group_col or None

    const = JFunctionConstants(
        theta_lab_deg=args.theta_lab,
        gamma_lab=args.gamma_lab,
        theta_res_deg=args.theta_res,
        gamma_res=args.gamma_res,
        coeff=args.coeff,
        perm_power=args.perm_power,
        poro_power=args.poro_power,
    )

    print(f"Читаю лабораторные данные: {args.input}")
    df = load_lab_data(args.input)
    print(f"  строк: {len(df)}")

    if args.horizon_map:
        print(f"Проставляю горизонт по глубине из: {args.horizon_map}")
        df = assign_horizon_by_depth(df, load_horizon_map(args.horizon_map))

    print("Считаю Pc(рез), SWn, J(Sw) ...")
    df = add_derived_columns(df, const)

    print("Подбираю коэффициенты a, b экспоненциальной модели J(SWn) = a*exp(b*SWn) ...")
    coeffs = fit_by_group(df, group_col)
    print(coeffs.to_string(index=False))

    out_dir = Path(args.output_dir)
    points_path, coeffs_path = save_results(df, coeffs, out_dir)
    print(f"\nТаблица точек сохранена:      {points_path}")
    print(f"Таблица коэффициентов a,b:    {coeffs_path}")

    fits = {
        row["group"]: FitResult(a=row["a"], b=row["b"], r2=row["r2"], n=int(row["n"]))
        for _, row in coeffs.iterrows()
    }

    plot_path = plot_j_function(df, fits, out_dir / "j_function_plot.png", group_col)
    print(f"График сохранён:              {plot_path}")


if __name__ == "__main__":
    main()
