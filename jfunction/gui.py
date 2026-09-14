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

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.const = JFunctionConstants()

        self._build_widgets()
        self._update_cos_labels()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить данные...", command=self.on_load).pack(side="left")
        self.file_label = ttk.Label(top, text="Файл не загружен")
        self.file_label.pack(side="left", padx=10)
        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export).pack(side="right")

        middle = ttk.Frame(self.root)
        middle.pack(fill="x", padx=8, pady=4)

        filt = ttk.LabelFrame(middle, text="Фильтр", padding=8)
        filt.pack(side="left", fill="y")

        ttk.Label(filt, text="Скважина:").grid(row=0, column=0, sticky="w")
        self.well_var = tk.StringVar(value=ALL)
        self.well_combo = ttk.Combobox(filt, textvariable=self.well_var, state="readonly", values=[ALL], width=15)
        self.well_combo.grid(row=0, column=1, padx=6, pady=2)
        self.well_combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())

        ttk.Label(filt, text="Горизонт:").grid(row=1, column=0, sticky="w")
        self.horizon_var = tk.StringVar(value=ALL)
        self.horizon_combo = ttk.Combobox(
            filt, textvariable=self.horizon_var, state="readonly", values=[ALL], width=15
        )
        self.horizon_combo.grid(row=1, column=1, padx=6, pady=2)
        self.horizon_combo.bind("<<ComboboxSelected>>", lambda _e: self.recompute())

        self._build_constants_panel(middle)

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

    def _build_constants_panel(self, parent: ttk.Widget) -> None:
        """Панель "Константы J-функции" - таблица, как в исходном Excel, с полями для правки."""
        panel = ttk.LabelFrame(parent, text="Константы J-функции", padding=8)
        panel.pack(side="left", fill="y", padx=(10, 0))

        self.theta_lab_var = tk.StringVar(value=str(self.const.theta_lab_deg))
        self.gamma_lab_var = tk.StringVar(value=str(self.const.gamma_lab))
        self.theta_res_var = tk.StringVar(value=str(self.const.theta_res_deg))
        self.gamma_res_var = tk.StringVar(value=str(self.const.gamma_res))
        self.coeff_var = tk.StringVar(value=str(self.const.coeff))
        self.cos_lab_var = tk.StringVar()
        self.cos_res_var = tk.StringVar()

        rows = [
            ("Угол смач-ти лаб (θ_лаб), град", self.theta_lab_var, True),
            ("ПНС лаб (γ_лаб), дин/см", self.gamma_lab_var, True),
            ("Угол смач-ти рез (θ_рез), град", self.theta_res_var, True),
            ("ПНС рез (γ_рез), дин/см", self.gamma_res_var, True),
            ("cos θ_лаб", self.cos_lab_var, False),
            ("cos θ_рез", self.cos_res_var, False),
            ("Коэфф.", self.coeff_var, True),
        ]
        for r, (label, var, editable) in enumerate(rows):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            if editable:
                ttk.Entry(panel, textvariable=var, width=10, justify="right").grid(row=r, column=1, pady=1)
            else:
                ttk.Entry(panel, textvariable=var, width=10, justify="right", state="readonly").grid(
                    row=r, column=1, pady=1
                )

        ttk.Button(panel, text="Применить константы", command=self.on_apply_constants).grid(
            row=len(rows), column=0, columnspan=2, pady=(6, 0), sticky="we"
        )

    def _update_cos_labels(self) -> None:
        self.cos_lab_var.set(f"{self.const.cos_theta_lab:.6f}")
        self.cos_res_var.set(f"{self.const.cos_theta_res:.6f}")

    def on_apply_constants(self) -> None:
        try:
            new_const = JFunctionConstants(
                theta_lab_deg=float(self.theta_lab_var.get()),
                gamma_lab=float(self.gamma_lab_var.get()),
                theta_res_deg=float(self.theta_res_var.get()),
                gamma_res=float(self.gamma_res_var.get()),
                coeff=float(self.coeff_var.get()),
            )
        except ValueError:
            messagebox.showerror("Ошибка", "Все константы J-функции должны быть числами.")
            return

        self.const = new_const
        self._update_cos_labels()
        if self._recompute_from_raw():
            self.recompute()

    # ------------------------------------------------------------- actions

    def _recompute_from_raw(self) -> bool:
        """Пересчитывает Pc(рез), SWn, J из исходных данных с текущими константами."""
        if self.raw_df is None:
            return False
        try:
            self.df = add_derived_columns(self.raw_df, self.const)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка расчёта", str(exc))
            return False
        return True

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл с лабораторными данными",
            filetypes=[("Таблицы", "*.csv *.xlsx *.xls"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            raw_df = load_lab_data(path)
        except Exception as exc:  # noqa: BLE001 - показываем пользователю любую ошибку загрузки
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        self.raw_df = raw_df
        if not self._recompute_from_raw():
            return

        df = self.df
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
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlabel("SWn")
        self.ax.set_ylabel("J(Sw)")
        self.ax.set_title("J(SWn) = a·exp(b·SWn)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
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
