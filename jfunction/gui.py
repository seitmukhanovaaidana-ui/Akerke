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
from .corey import fit_corey_by_model, unified_corey_params
from .fit import evaluate_fixed_params, fit_by_group, fit_exponential
from .io import load_lab_data
from .ofp_docx_io import load_ofp_data_from_docx
from .report import save_results
from .rocktype import classify_by_permeability

ALL = "Все"
TABLE_COLUMNS = ("well", "sample", "horizon", "Sw", "Pc_lab_MPa", "SWn", "J")
PINNED_COLORS = ["green", "purple", "brown", "magenta", "gray", "olive", "cyan", "black"]
OFP_COLUMNS = ("model", "well", "n", "Swir", "Sor", "Swmax", "krwmax", "nw", "r2_w", "now", "r2_o")


def _fmt_num(value: float) -> str:
    """Целые числа показываем без ".0" (30, а не 30.0) - как в исходном Excel."""
    return str(int(value)) if float(value).is_integer() else str(value)


class JFunctionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Расчёт J-функции, SWn и коэффициентов a, b")
        self.root.geometry("1450x820")

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.const = JFunctionConstants()
        self.pinned_trends: list[dict] = []

        self.ofp_df: pd.DataFrame | None = None
        self.ofp_per_model: pd.DataFrame | None = None

        self._build_widgets()
        self._update_cos_labels()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        outer = ttk.Notebook(self.root)
        outer.pack(fill="both", expand=True)
        self.outer_notebook = outer

        jfunc_tab = ttk.Frame(outer)
        ofp_tab = ttk.Frame(outer)
        outer.add(jfunc_tab, text="J-функция")
        outer.add(ofp_tab, text="ОФП (Кори)")

        top = ttk.Frame(jfunc_tab, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить данные...", command=self.on_load).pack(side="left")
        self.file_label = ttk.Label(top, text="Файл не загружен")
        self.file_label.pack(side="left", padx=10)
        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export).pack(side="right")
        ttk.Button(top, text="Сохранить график...", command=self.on_save_chart).pack(side="right", padx=(0, 8))

        middle = ttk.Frame(jfunc_tab)
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
        self._build_result_panel(middle)

        self.result_label = tk.Label(
            jfunc_tab,
            text="Загрузите файл с лабораторными данными.",
            font=("Segoe UI", 13, "bold"),
            bg="#ED7D31",
            fg="white",
            anchor="w",
            padx=12,
            pady=8,
        )
        self.result_label.pack(fill="x", padx=8, pady=6)

        self._build_compare_panel(jfunc_tab)

        notebook = ttk.Notebook(jfunc_tab)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.notebook = notebook

        chart_tab = ttk.Frame(notebook)
        table_tab = ttk.Frame(notebook)
        rocktype_tab = ttk.Frame(notebook)
        notebook.add(chart_tab, text="График")
        notebook.add(table_tab, text="Таблица точек")
        notebook.add(rocktype_tab, text="Типы пород (k-φ)")
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.scatter = None
        self.hover_annotation = None
        self._plot_swn = None
        self._plot_j = None
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self.tree = ttk.Treeview(table_tab, columns=TABLE_COLUMNS, show="headings")
        for col in TABLE_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        vsb = ttk.Scrollbar(table_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._build_rocktype_tab(rocktype_tab)
        self._build_ofp_tab(ofp_tab)

    def _build_ofp_tab(self, parent: ttk.Widget) -> None:
        """Вкладка ОФП: загрузка Word-отчёта лаборатории и расчёт степеней Кори (nw, now)."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить ОФП-отчёт (.docx)...", command=self.on_load_ofp).pack(side="left")
        self.ofp_file_label = ttk.Label(top, text="Файл не загружен")
        self.ofp_file_label.pack(side="left", padx=10)
        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export_ofp).pack(side="right")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.ofp_figure = Figure(figsize=(6, 5), dpi=100)
        self.ofp_ax = self.ofp_figure.add_subplot(111)
        self.ofp_canvas = FigureCanvasTkAgg(self.ofp_figure, master=body)
        self.ofp_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        unified = ttk.LabelFrame(right, text="Единые параметры Кори (по керну)", padding=8)
        unified.pack(fill="x")

        self.ofp_nw_var = tk.StringVar(value="-")
        self.ofp_now_var = tk.StringVar(value="-")
        self.ofp_swir_var = tk.StringVar(value="-")
        self.ofp_sor_var = tk.StringVar(value="-")
        self.ofp_krwmax_var = tk.StringVar(value="-")
        self.ofp_nmodels_var = tk.StringVar(value="-")

        rows = [
            ("nw (медиана)", self.ofp_nw_var),
            ("now (медиана)", self.ofp_now_var),
            ("Swir обр. (среднее)", self.ofp_swir_var),
            ("Sor обр. (среднее)", self.ofp_sor_var),
            ("krwmax обр. (среднее)", self.ofp_krwmax_var),
            ("Число моделей/образцов", self.ofp_nmodels_var),
        ]
        for r, (label, var) in enumerate(rows):
            ttk.Label(unified, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            ttk.Entry(unified, textvariable=var, width=12, justify="right", state="readonly").grid(
                row=r, column=1, pady=1
            )

        ofp_columns = OFP_COLUMNS
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.ofp_tree = ttk.Treeview(tree_frame, columns=ofp_columns, show="headings", height=14)
        for col in ofp_columns:
            self.ofp_tree.heading(col, text=col)
            self.ofp_tree.column(col, width=60, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.ofp_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.ofp_tree.xview)
        self.ofp_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.ofp_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

    def _build_rocktype_tab(self, parent: ttk.Widget) -> None:
        """Кроссплот k-φ и разбиение образцов на типы породы по проницаемости."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Границы проницаемости, мД (через запятую):").pack(side="left")
        self.rocktype_breaks_var = tk.StringVar(value="1, 10, 100")
        ttk.Entry(top, textvariable=self.rocktype_breaks_var, width=20).pack(side="left", padx=6)
        ttk.Button(top, text="Разбить на типы породы", command=self.on_apply_rocktype).pack(side="left")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.rocktype_figure = Figure(figsize=(4.3, 3.6), dpi=100)
        self.rocktype_ax = self.rocktype_figure.add_subplot(111)
        self.rocktype_canvas = FigureCanvasTkAgg(self.rocktype_figure, master=body)
        self.rocktype_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        self.rocktype_jswn_figure = Figure(figsize=(4.3, 3.6), dpi=100)
        self.rocktype_jswn_ax = self.rocktype_jswn_figure.add_subplot(111)
        self.rocktype_jswn_canvas = FigureCanvasTkAgg(self.rocktype_jswn_figure, master=body)
        self.rocktype_jswn_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        rt_columns = ("group", "n", "a", "b", "r2")
        self.rocktype_tree = ttk.Treeview(body, columns=rt_columns, show="headings", height=15)
        for col in rt_columns:
            self.rocktype_tree.heading(col, text=col)
            self.rocktype_tree.column(col, width=70, anchor="center")
        self.rocktype_tree.pack(side="left", fill="y", padx=(8, 0))

    def on_apply_rocktype(self) -> None:
        df = self._filtered()
        if df is None or df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return

        try:
            breaks = [float(x.strip()) for x in self.rocktype_breaks_var.get().split(",") if x.strip()]
        except ValueError:
            messagebox.showerror("Ошибка", "Границы проницаемости должны быть числами через запятую.")
            return
        if not breaks:
            messagebox.showerror("Ошибка", "Укажите хотя бы одну границу проницаемости (мД).")
            return

        df = df.copy()
        df["rock_type"] = classify_by_permeability(df["perm_mD"], breaks)

        coeffs = fit_by_group(df, "rock_type")
        self._update_rocktype_table(coeffs)
        self._update_rocktype_plot(df, color_col="rock_type")
        self._update_rocktype_jswn_plot(df, color_col="rock_type")

    def _update_rocktype_table(self, coeffs: pd.DataFrame) -> None:
        self.rocktype_tree.delete(*self.rocktype_tree.get_children())
        for _, row in coeffs.iterrows():
            self.rocktype_tree.insert(
                "",
                "end",
                values=(row["group"], row["n"], f"{row['a']:.4f}", f"{row['b']:.4f}", f"{row['r2']:.4f}"),
            )

    def _update_rocktype_plot(self, df: pd.DataFrame, color_col: str | None = None) -> None:
        self.rocktype_ax.clear()
        samples = df.drop_duplicates(subset=["well", "sample"]) if "sample" in df.columns else df

        if color_col is not None and color_col in samples.columns:
            for i, (name, sub) in enumerate(samples.groupby(color_col, observed=True)):
                color = PINNED_COLORS[i % len(PINNED_COLORS)]
                self.rocktype_ax.scatter(
                    sub["porosity_pct"], sub["perm_mD"], s=20, alpha=0.7, color=color, label=str(name)
                )
            self.rocktype_ax.legend(fontsize=8, title="Тип породы", loc="best")
        else:
            self.rocktype_ax.scatter(samples["porosity_pct"], samples["perm_mD"], s=20, alpha=0.7)

        self.rocktype_ax.set_yscale("log")
        self.rocktype_ax.set_xlabel("Пористость, %")
        self.rocktype_ax.set_ylabel("Проницаемость, мД")
        self.rocktype_ax.set_title("Кроссплот k-φ")
        self.rocktype_ax.grid(True, which="both", alpha=0.3)
        self.rocktype_canvas.draw()

    def _update_rocktype_jswn_plot(self, df: pd.DataFrame, color_col: str | None = None) -> None:
        """График J(Sw) от SWn с отдельным трендом для каждой группы (типа породы/горизонта)."""
        self.rocktype_jswn_ax.clear()

        if color_col is not None and color_col in df.columns:
            groups = list(df.groupby(color_col, observed=True))
        else:
            groups = [("Все образцы", df)]

        swn_grid = np.linspace(max(df["SWn"].min(), 0), df["SWn"].max(), 200)
        for i, (name, sub) in enumerate(groups):
            color = PINNED_COLORS[i % len(PINNED_COLORS)]
            self.rocktype_jswn_ax.scatter(sub["SWn"], sub["J"], s=10, alpha=0.6, color=color, label=str(name))
            try:
                fit = fit_exponential(sub["SWn"], sub["J"])
            except ValueError:
                continue
            self.rocktype_jswn_ax.plot(swn_grid, fit.predict(swn_grid), color=color, linewidth=2)

        self.rocktype_jswn_ax.set_ylim(bottom=0)
        self.rocktype_jswn_ax.set_xlabel("SWn")
        self.rocktype_jswn_ax.set_ylabel("J(Sw)")
        self.rocktype_jswn_ax.set_title("J(SWn) по группам")
        self.rocktype_jswn_ax.legend(fontsize=7, loc="best")
        self.rocktype_jswn_ax.grid(True, alpha=0.3)
        self.rocktype_jswn_canvas.draw()

    def _build_constants_panel(self, parent: ttk.Widget) -> None:
        """Панель "Константы J-функции" - таблица, как в исходном Excel, с полями для правки."""
        panel = ttk.LabelFrame(parent, text="Константы J-функции", padding=8)
        panel.pack(side="left", fill="y", padx=(10, 0))

        self.theta_lab_var = tk.StringVar(value=_fmt_num(self.const.theta_lab_deg))
        self.gamma_lab_var = tk.StringVar(value=_fmt_num(self.const.gamma_lab))
        self.theta_res_var = tk.StringVar(value=_fmt_num(self.const.theta_res_deg))
        self.gamma_res_var = tk.StringVar(value=_fmt_num(self.const.gamma_res))
        self.coeff_var = tk.StringVar(value=_fmt_num(self.const.coeff))
        self.perm_power_var = tk.StringVar(value=_fmt_num(self.const.perm_power))
        self.poro_power_var = tk.StringVar(value=_fmt_num(self.const.poro_power))
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
            ("Power for permeability term", self.perm_power_var, True),
            ("Power for porosity term", self.poro_power_var, True),
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

    def _on_tab_changed(self, _event=None) -> None:
        """На вкладке "Типы пород" панель "Результат" (общий тренд) не нужна - там свои a,b по группам."""
        current = self.notebook.index(self.notebook.select())
        if current == 2:  # "Типы пород (k-φ)"
            self.result_panel.pack_forget()
        else:
            self.result_panel.pack(side="left", fill="y", padx=(10, 0))

    def _build_result_panel(self, parent: ttk.Widget) -> None:
        """Панель "Результат" - a, b, n, R² каждый в своём отдельном окошке."""
        panel = ttk.LabelFrame(parent, text="Результат: J(SWn) = a·exp(b·SWn)", padding=8)
        panel.pack(side="left", fill="y", padx=(10, 0))
        self.result_panel = panel

        self.a_var = tk.StringVar(value="-")
        self.b_var = tk.StringVar(value="-")
        self.n_var = tk.StringVar(value="-")
        self.r2_var = tk.StringVar(value="-")
        self.manual_ab_var = tk.BooleanVar(value=False)

        self.a_scale_var = tk.DoubleVar(value=0.01)
        self.b_scale_var = tk.DoubleVar(value=0.0)

        bold = ("Segoe UI", 11, "bold")
        ttk.Label(panel, text="a").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.a_entry = ttk.Entry(panel, textvariable=self.a_var, width=10, justify="right",
                                  state="readonly", font=bold)
        self.a_entry.grid(row=0, column=1, pady=3)
        self.a_entry.bind("<Return>", lambda _e: self._on_manual_entry())
        self.a_scale = tk.Scale(
            panel, from_=0.01, to=200, resolution=0.01, orient="horizontal", length=170,
            variable=self.a_scale_var, showvalue=False, state="disabled", command=self._on_a_scale,
        )
        self.a_scale.grid(row=0, column=2, padx=(6, 0))

        ttk.Label(panel, text="b").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.b_entry = ttk.Entry(panel, textvariable=self.b_var, width=10, justify="right",
                                  state="readonly", font=bold)
        self.b_entry.grid(row=1, column=1, pady=3)
        self.b_entry.bind("<Return>", lambda _e: self._on_manual_entry())
        self.b_scale = tk.Scale(
            panel, from_=-20, to=5, resolution=0.01, orient="horizontal", length=170,
            variable=self.b_scale_var, showvalue=False, state="disabled", command=self._on_b_scale,
        )
        self.b_scale.grid(row=1, column=2, padx=(6, 0))

        ttk.Label(panel, text="n (число точек)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(panel, textvariable=self.n_var, width=10, justify="right", state="readonly").grid(
            row=2, column=1, pady=3
        )

        ttk.Label(panel, text="R² (качество подгонки)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        self.r2_entry = tk.Entry(
            panel, textvariable=self.r2_var, width=10, justify="right", state="readonly",
            relief="sunken", readonlybackground="white",
        )
        self.r2_entry.grid(row=3, column=1, pady=3)
        self.r2_hint = ttk.Label(panel, text="", foreground="gray")
        self.r2_hint.grid(row=3, column=2, sticky="w", padx=(6, 0))

        ttk.Checkbutton(
            panel, text="Задать a, b вручную (ползунками или числом)",
            variable=self.manual_ab_var, command=self.on_toggle_manual_ab,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 2))

        ttk.Button(panel, text="Применить a, b", command=self.recompute).grid(
            row=5, column=0, columnspan=3, pady=(2, 0), sticky="we"
        )

    def on_toggle_manual_ab(self) -> None:
        manual = self.manual_ab_var.get()
        state = "normal" if manual else "readonly"
        scale_state = "normal" if manual else "disabled"

        if manual:
            # Ставим ползунки туда, где сейчас находится a,b (из последнего автоподбора
            # или предыдущей ручной правки), чтобы не начинать с края шкалы.
            try:
                self.a_scale_var.set(float(self.a_var.get()))
            except ValueError:
                pass
            try:
                self.b_scale_var.set(float(self.b_var.get()))
            except ValueError:
                pass

        self.a_entry.config(state=state)
        self.b_entry.config(state=state)
        self.a_scale.config(state=scale_state)
        self.b_scale.config(state=scale_state)
        self.recompute()

    def _on_a_scale(self, value: str) -> None:
        if not self.manual_ab_var.get():
            return
        self.a_var.set(f"{float(value):.4f}")
        self.recompute()

    def _on_b_scale(self, value: str) -> None:
        if not self.manual_ab_var.get():
            return
        self.b_var.set(f"{float(value):.4f}")
        self.recompute()

    def _on_manual_entry(self) -> None:
        """Пользователь ввёл a или b числом и нажал Enter - пересчитать и подвинуть ползунки."""
        self.recompute()
        try:
            self.a_scale_var.set(float(self.a_var.get()))
        except ValueError:
            pass
        try:
            self.b_scale_var.set(float(self.b_var.get()))
        except ValueError:
            pass

    def _update_r2_hint(self, r2: float | None) -> None:
        if r2 is None or r2 != r2:  # NaN
            self.r2_entry.config(fg="black")
            self.r2_hint.config(text="")
        elif 0 <= r2 <= 1:
            self.r2_entry.config(fg="#1a7f37")
            self.r2_hint.config(text="✓ в диапазоне 0-1", foreground="#1a7f37")
        else:
            self.r2_entry.config(fg="#c0392b")
            self.r2_hint.config(text="✗ хуже среднего (< 0)", foreground="#c0392b")

    def _build_compare_panel(self, parent: ttk.Widget) -> None:
        """Панель "Сравнение трендов" - закреплённые варианты a,b поверх графика."""
        panel = ttk.LabelFrame(parent, text="Сравнение трендов на графике", padding=8)
        panel.pack(fill="x", padx=8, pady=(0, 4))

        btns = ttk.Frame(panel)
        btns.pack(side="left", fill="y", padx=(0, 10))
        ttk.Button(btns, text="Закрепить текущий тренд", command=self.on_pin_trend).pack(fill="x", pady=1)
        ttk.Button(btns, text="Удалить выбранный", command=self.on_unpin_trend).pack(fill="x", pady=1)
        ttk.Button(btns, text="Очистить всё", command=self.on_clear_pinned).pack(fill="x", pady=1)

        self.pinned_listbox = tk.Listbox(panel, height=4)
        self.pinned_listbox.pack(side="left", fill="both", expand=True)

    def on_pin_trend(self) -> None:
        try:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
        except ValueError:
            messagebox.showwarning("Нет тренда", "Сначала загрузите данные, чтобы получить тренд.")
            return

        parts = [f"a={a:.4g}, b={b:.4g}"]
        if self.well_var.get() != ALL:
            parts.append(f"скв.{self.well_var.get()}")
        if self.horizon_var.get() != ALL:
            parts.append(self.horizon_var.get())
        label = " | ".join(parts)

        self.pinned_trends.append({"label": label, "a": a, "b": b})
        self.pinned_listbox.insert("end", label)
        self.recompute()

    def on_unpin_trend(self) -> None:
        sel = self.pinned_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.pinned_listbox.delete(idx)
        del self.pinned_trends[idx]
        self.recompute()

    def on_clear_pinned(self) -> None:
        self.pinned_trends.clear()
        self.pinned_listbox.delete(0, "end")
        self.recompute()

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
                perm_power=float(self.perm_power_var.get()),
                poro_power=float(self.poro_power_var.get()),
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
            filetypes=[("Таблицы и Word-отчёты", "*.csv *.xlsx *.xls *.docx"), ("Все файлы", "*.*")],
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

        self.manual_ab_var.set(False)
        self.a_entry.config(state="readonly")
        self.b_entry.config(state="readonly")
        self.a_scale.config(state="disabled")
        self.b_scale.config(state="disabled")

        self.pinned_trends.clear()
        self.pinned_listbox.delete(0, "end")

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
            for var in (self.a_var, self.b_var, self.n_var, self.r2_var):
                var.set("-")
            self._update_r2_hint(None)
            return

        if self.manual_ab_var.get():
            try:
                a = float(self.a_var.get())
                b = float(self.b_var.get())
                if a <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "a и b должны быть числами, a > 0.")
                fit = None
            else:
                fit = evaluate_fixed_params(a, b, df["SWn"], df["J"])
        else:
            try:
                fit = fit_exponential(df["SWn"], df["J"])
            except ValueError as exc:
                self.result_label.config(text=str(exc))
                fit = None

        if fit is not None:
            self.result_label.config(
                text=(
                    f"y = {fit.a:.4f}·e^{fit.b:.4f}x    "
                    f"(J(SWn) = a·exp(b·SWn), n={fit.n}, R²={fit.r2:.4f})"
                )
            )
            if not self.manual_ab_var.get():
                self.a_var.set(f"{fit.a:.4f}")
                self.b_var.set(f"{fit.b:.4f}")
            self.n_var.set(str(fit.n))
            self.r2_var.set(f"{fit.r2:.4f}" if fit.r2 == fit.r2 else "-")  # NaN check
            self._update_r2_hint(fit.r2)
        else:
            for var in (self.a_var, self.b_var, self.n_var, self.r2_var):
                var.set("-")
            self._update_r2_hint(None)

        self._update_plot(df, fit)
        self._update_table(df)
        color_col = "horizon" if "horizon" in df.columns else None
        self._update_rocktype_plot(df, color_col=color_col)
        self._update_rocktype_jswn_plot(df, color_col=color_col)
        self.rocktype_tree.delete(*self.rocktype_tree.get_children())

    def _update_plot(self, df: pd.DataFrame, fit) -> None:
        self.ax.clear()
        self._plot_swn = df["SWn"].to_numpy()
        self._plot_j = df["J"].to_numpy()
        self.scatter = self.ax.scatter(self._plot_swn, self._plot_j, s=14, alpha=0.6, label="данные")
        self.hover_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.hover_annotation.set_visible(False)
        swn_grid = np.linspace(max(df["SWn"].min(), 0), df["SWn"].max(), 200)
        if fit is not None:
            self.ax.plot(swn_grid, fit.predict(swn_grid), color="red", linewidth=2, label="тренд")
            self.ax.text(
                0.4,
                0.7,
                f"y = {fit.a:.4f}e^{fit.b:.4f}x",
                transform=self.ax.transAxes,
                fontsize=15,
                color="black",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ED7D31", edgecolor="none", alpha=0.95),
            )

        for i, pinned in enumerate(self.pinned_trends):
            color = PINNED_COLORS[i % len(PINNED_COLORS)]
            curve = pinned["a"] * np.exp(pinned["b"] * swn_grid)
            self.ax.plot(swn_grid, curve, color=color, linewidth=2, linestyle="--", label=pinned["label"])

        self.ax.set_ylim(bottom=0)
        self.ax.set_xlabel("SWn")
        self.ax.set_ylabel("J(Sw)")
        self.ax.set_title("J(SWn) = a·exp(b·SWn)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _on_hover(self, event) -> None:
        if self.scatter is None or self.hover_annotation is None or event.inaxes != self.ax:
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        contained, info = self.scatter.contains(event)
        if contained and len(info.get("ind", [])) > 0:
            idx = info["ind"][0]
            x, y = self._plot_swn[idx], self._plot_j[idx]
            self.hover_annotation.xy = (x, y)
            self.hover_annotation.set_text(f"SWn = {x:.4f}\nJ(Sw) = {y:.4f}")
            self.hover_annotation.set_visible(True)
            self.canvas.draw_idle()
        elif self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def _update_table(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, row in df.head(500).iterrows():
            self.tree.insert("", "end", values=[row.get(col, "") for col in TABLE_COLUMNS])

    def on_save_chart(self) -> None:
        if self.df is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить график как...",
            defaultextension=".png",
            filetypes=[("Изображение PNG", "*.png"), ("PDF", "*.pdf"), ("Все файлы", "*.*")],
            initialfile="j_function_plot.png",
        )
        if not path:
            return

        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения графика", str(exc))
            return
        messagebox.showinfo("Готово", f"График сохранён:\n{path}")

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
            plot_path = Path(out_dir) / "j_function_plot.png"
            self.figure.savefig(plot_path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Готово", f"Сохранено:\n{points_path}\n{coeffs_path}\n{plot_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    # -------------------------------------------------------- ОФП (Кори)

    def on_load_ofp(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите Word-отчёт лаборатории по ОФП",
            filetypes=[("Word-документ", "*.docx"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            df = load_ofp_data_from_docx(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        if df.empty:
            messagebox.showwarning("Нет данных", "Не удалось найти в файле данные ОФП (Sw/krw/krow).")
            return

        try:
            per_model = fit_corey_by_model(df, group_col="model")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка расчёта", str(exc))
            return

        self.ofp_df = df
        self.ofp_per_model = per_model
        unified = unified_corey_params(per_model)

        self.ofp_file_label.config(
            text=f"{Path(path).name}  ({per_model['model'].nunique()} моделей, {len(df)} точек)"
        )
        self._update_ofp_table(per_model)
        self._update_ofp_unified(unified)
        self._update_ofp_plot(df, unified)

    def _update_ofp_table(self, per_model: pd.DataFrame) -> None:
        self.ofp_tree.delete(*self.ofp_tree.get_children())
        for _, row in per_model.iterrows():
            values = []
            for col in OFP_COLUMNS:
                v = row.get(col, "")
                if isinstance(v, float):
                    values.append("-" if v != v else f"{v:.4f}")
                else:
                    values.append(v)
            self.ofp_tree.insert("", "end", values=values)

    def _update_ofp_unified(self, unified) -> None:
        if unified is None:
            for var in (
                self.ofp_nw_var, self.ofp_now_var, self.ofp_swir_var,
                self.ofp_sor_var, self.ofp_krwmax_var, self.ofp_nmodels_var,
            ):
                var.set("-")
            return
        self.ofp_nw_var.set(f"{unified.nw:.4f}")
        self.ofp_now_var.set(f"{unified.now:.4f}")
        self.ofp_swir_var.set(f"{unified.swir:.4f}")
        self.ofp_sor_var.set(f"{unified.sor:.4f}")
        self.ofp_krwmax_var.set(f"{unified.krwmax:.4f}")
        self.ofp_nmodels_var.set(str(unified.n_models))

    def _update_ofp_plot(self, df: pd.DataFrame, unified) -> None:
        self.ofp_ax.clear()
        if df is not None and not df.empty:
            wells = sorted(df["well"].dropna().astype(str).unique())
            well_colors = {w: PINNED_COLORS[i % len(PINNED_COLORS)] for i, w in enumerate(wells)}
            for well, sub in df.groupby("well"):
                color = well_colors.get(str(well), "gray")
                self.ofp_ax.scatter(sub["Sw"], sub["krw"], s=14, alpha=0.6, color=color, marker="o")
                self.ofp_ax.scatter(sub["Sw"], sub["krow"], s=14, alpha=0.6, color=color, marker="^")

            self.ofp_ax.scatter([], [], color="gray", marker="o", label="krw (точки)")
            self.ofp_ax.scatter([], [], color="gray", marker="^", label="krow (точки)")

        if unified is not None:
            sw_grid = np.linspace(unified.swir, 1.0, 100)
            sw_star = np.clip((sw_grid - unified.swir) / (1.0 - unified.swir), 0, 1)
            krw_curve = unified.krwmax * np.power(sw_star, unified.nw)
            kro_curve = unified.krow_swc * np.power(1 - sw_star, unified.now)
            self.ofp_ax.plot(sw_grid, krw_curve, color="blue", linewidth=2, linestyle="--", label="krw (единая)")
            self.ofp_ax.plot(sw_grid, kro_curve, color="black", linewidth=2, linestyle="--", label="kro (единая)")

        self.ofp_ax.set_xlabel("Sw")
        self.ofp_ax.set_ylabel("Относительная проницаемость")
        self.ofp_ax.set_title("ОФП: krw/krow(Sw) и единая кривая Кори")
        self.ofp_ax.set_ylim(bottom=0)
        self.ofp_ax.legend(fontsize=8)
        self.ofp_ax.grid(True, alpha=0.3)
        self.ofp_canvas.draw()

    def on_export_ofp(self) -> None:
        if self.ofp_per_model is None or self.ofp_per_model.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите ОФП-отчёт.")
            return

        out_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов ОФП")
        if not out_dir:
            return

        try:
            out_dir_path = Path(out_dir)
            points_path = out_dir_path / "ofp_points.xlsx"
            coeffs_path = out_dir_path / "ofp_corey_coefficients.xlsx"
            plot_path = out_dir_path / "ofp_corey_plot.png"

            self.ofp_df.to_excel(points_path, index=False)
            self.ofp_per_model.to_excel(coeffs_path, index=False)
            self.ofp_figure.savefig(plot_path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Готово", f"Сохранено:\n{points_path}\n{coeffs_path}\n{plot_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))


def main() -> None:
    root = tk.Tk()
    JFunctionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
