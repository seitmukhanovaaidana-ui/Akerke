"""
Настольное приложение (графический интерфейс) для расчёта J-функции.

Загружаете файл с лабораторными данными - программа сама считает
Pc(рез), SWn, J(Sw), подбирает коэффициенты a, b экспоненциального
тренда и строит график. Есть фильтры по скважине и горизонту.

Запуск: python run_gui.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .calc import JFunctionConstants, add_derived_columns
from .fit import fit_exponential
from .io import load_lab_data
from .report import save_results

ALL = "Все"
TABLE_COLUMNS = ("well", "sample", "horizon", "Sw", "Pc_lab_MPa", "SWn", "J")


class JFunctionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Расчёт J-функции, SWn и коэффициентов a, b")
        self.root.geometry("1050x720")

        self.df: pd.DataFrame | None = None
        self.const = JFunctionConstants()

        self._build_widgets()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить данные...", command=self.on_load).pack(side="left")
        self.file_label = ttk.Label(top, text="Файл не загружен")
        self.file_label.pack(side="left", padx=10)
        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export).pack(side="right")

        filt = ttk.LabelFrame(self.root, text="Фильтр", padding=8)
        filt.pack(fill="x", padx=8, pady=4)

        ttk.Label(filt, text="Скважина:").grid(row=0, column=0, sticky="w")
        self.well_var = tk.StringVar(value=ALL)
        self.well_combo = ttk.Combobox(filt, textvariable=self.well_var, state="readonly", values=[ALL], width=15)
        self.well_combo.grid(row=0, column=1, padx=6)
        self.well_combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())

        ttk.Label(filt, text="Горизонт:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.horizon_var = tk.StringVar(value=ALL)
        self.horizon_combo = ttk.Combobox(
            filt, textvariable=self.horizon_var, state="readonly", values=[ALL], width=15
        )
        self.horizon_combo.grid(row=0, column=3, padx=6)
        self.horizon_combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())

        self.result_label = ttk.Label(
            self.root, text="Загрузите файл с лабораторными данными.", font=("Segoe UI", 11, "bold")
        )
        self.result_label.pack(fill="x", padx=8, pady=6)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)

        chart_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        notebook.add(chart_tab, text="График")
        notebook.add(table_tab, text="Таблица точек")

        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.tree = ttk.Treeview(table_tab, columns=TABLE_COLUMNS, show="headings")
        for col in TABLE_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        vsb = ttk.Scrollbar(table_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    # ------------------------------------------------------------- actions

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл с лабораторными данными",
            filetypes=[("Таблицы", "*.csv *.xlsx *.xls"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            df = load_lab_data(path)
            df = add_derived_columns(df, self.const)
        except Exception as exc:  # noqa: BLE001 - показываем пользователю любую ошибку загрузки
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        self.df = df
        self.file_label.config(text=f"{Path(path).name}  ({len(df)} строк)")

        wells = [ALL] + sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else [ALL]
        horizons = (
            [ALL] + sorted(df["horizon"].dropna().astype(str).unique()) if "horizon" in df.columns else [ALL]
        )
        self.well_combo["values"] = wells
        self.horizon_combo["values"] = horizons
        self.well_var.set(ALL)
        self.horizon_var.set(ALL)

        self.recompute()

    def _filtered(self) -> pd.DataFrame | None:
        if self.df is None:
            return None
        df = self.df
        if self.well_var.get() != ALL and "well" in df.columns:
            df = df[df["well"].astype(str) == self.well_var.get()]
        if self.horizon_var.get() != ALL and "horizon" in df.columns:
            df = df[df["horizon"].astype(str) == self.horizon_var.get()]
        return df

    def recompute(self) -> None:
        df = self._filtered()
        if df is None or df.empty:
            self.result_label.config(text="Нет данных для отображения.")
            return

        try:
            fit = fit_exponential(df["SWn"], df["J"])
        except ValueError as exc:
            self.result_label.config(text=str(exc))
            fit = None

        if fit is not None:
            self.result_label.config(
                text=f"J(SWn) = {fit.a:.4f} · exp({fit.b:.4f} · SWn)   (n={fit.n}, R²={fit.r2:.4f})"
            )

        self._update_plot(df, fit)
        self._update_table(df)

    def _update_plot(self, df: pd.DataFrame, fit) -> None:
        self.ax.clear()
        self.ax.scatter(df["SWn"], df["J"], s=14, alpha=0.6, label="данные")
        if fit is not None:
            swn_grid = np.linspace(max(df["SWn"].min(), 0), df["SWn"].max(), 200)
            self.ax.plot(swn_grid, fit.predict(swn_grid), color="red", linewidth=2, label="тренд")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("SWn")
        self.ax.set_ylabel("J(Sw)")
        self.ax.set_title("J(SWn) = a·exp(b·SWn)")
        self.ax.legend()
        self.ax.grid(True, which="both", alpha=0.3)
        self.canvas.draw()

    def _update_table(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, row in df.head(500).iterrows():
            self.tree.insert("", "end", values=[row.get(col, "") for col in TABLE_COLUMNS])

    def on_export(self) -> None:
        df = self._filtered()
        if df is None or df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return

        out_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if not out_dir:
            return

        try:
            fit = fit_exponential(df["SWn"], df["J"])
            coeffs = pd.DataFrame(
                [{"group": "Текущий фильтр", "n": fit.n, "a": fit.a, "b": fit.b, "r2": fit.r2}]
            )
            points_path, coeffs_path = save_results(df, coeffs, out_dir)
            messagebox.showinfo("Готово", f"Сохранено:\n{points_path}\n{coeffs_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))


def main() -> None:
    root = tk.Tk()
    JFunctionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
