"""Сохранение результатов (таблицы + график) на диск."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fit import FitResult


def save_results(
    df: pd.DataFrame,
    coeffs: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Сохраняет таблицу точек (Sw, SWn, J, ...) и таблицу коэффициентов a,b в Excel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path = out_dir / "j_function_points.xlsx"
    coeffs_path = out_dir / "j_function_coefficients.xlsx"

    df.to_excel(points_path, index=False)
    coeffs.to_excel(coeffs_path, index=False)

    return points_path, coeffs_path


def plot_j_function(
    df: pd.DataFrame,
    fits: dict[str, FitResult],
    out_path: str | Path,
    group_col: str | None = None,
) -> Path:
    """
    Строит график J(SWn): точки по образцам (по группам, если задан group_col)
    и подобранные экспоненциальные кривые.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    if group_col is not None and group_col in df.columns:
        groups = list(df.groupby(group_col, dropna=True))
    else:
        groups = [("Все образцы", df)]

    colors = plt.cm.tab10.colors
    swn_grid = np.linspace(max(df["SWn"].min(), 0), df["SWn"].max(), 200)

    for i, (name, sub) in enumerate(groups):
        color = colors[i % len(colors)]
        ax.scatter(sub["SWn"], sub["J"], s=14, alpha=0.6, color=color, label=f"{name} (данные)")
        fit = fits.get(str(name))
        if fit is not None:
            ax.plot(
                swn_grid,
                fit.predict(swn_grid),
                color=color,
                linewidth=2,
                label=f"{name}: {fit}",
            )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("SWn (нормализованная водонасыщенность)")
    ax.set_ylabel("J(Sw)")
    ax.set_title("J-функция и экспоненциальный тренд J(SWn) = a·exp(b·SWn)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
