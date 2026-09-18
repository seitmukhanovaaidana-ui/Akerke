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
from .corey import evaluate_fixed_corey, fit_corey_by_model, unified_corey_params
from .endpoint_cubes import (
    apply_correlation,
    check_apply_inputs,
    fit_endpoint_cubes,
    group_by_formation,
    load_endpoint_summary_xlsx,
    quality_flags,
    quality_level,
)
from .fit import evaluate_fixed_params, fit_by_group, fit_exponential
from .io import load_lab_data
from .ofp_docx_io import load_ofp_data_from_docx
from .petro import fit_poro_perm_by_horizon, load_core_petro_xlsx
from .report import save_results
from .rocktype import classify_by_permeability
from .scal_export import format_coreywo, format_swof
from .swir_crosscheck import crosscheck_swir

ALL = "Все"
TABLE_COLUMNS = ("well", "sample", "horizon", "Sw", "Pc_lab_MPa", "SWn", "J")
PINNED_COLORS = ["green", "purple", "brown", "magenta", "gray", "olive", "cyan", "black"]
OFP_COLUMNS = ("model", "well", "n", "Swir", "Sor", "Swmax", "krwmax", "nw", "r2_w", "now", "r2_o")
CROSSCHECK_COLUMNS = ("well", "sample", "model_OFP", "Swir_Pc", "Swir_OFP", "diff", "perm_mD", "porosity_pct")
PETRO_COLUMNS = ("horizon", "n", "a", "b", "r2", "poro_min", "poro_max", "perm_min", "perm_max")
CUBES_COLUMNS = ("horizon", "endpoint", "x_var", "form", "a", "b", "r2", "n")


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
        self.ofp_unified = None

        self.crosscheck_result: pd.DataFrame | None = None

        self.petro_df: pd.DataFrame | None = None
        self.petro_fits: pd.DataFrame | None = None

        self.cubes_df: pd.DataFrame | None = None
        self.cubes_active_df: pd.DataFrame | None = None
        self.cubes_fits: list = []
        self.cubes_selected: int | None = None
        self._cubes_last_input: list | None = None
        self._cubes_last_result = None

        self._build_widgets()
        self._update_cos_labels()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        outer = ttk.Notebook(self.root)
        outer.pack(fill="both", expand=True)
        self.outer_notebook = outer

        jfunc_tab = ttk.Frame(outer)
        ofp_tab = ttk.Frame(outer)
        crosscheck_tab = ttk.Frame(outer)
        petro_tab = ttk.Frame(outer)
        cubes_tab = ttk.Frame(outer)
        outer.add(jfunc_tab, text="J-функция")
        outer.add(ofp_tab, text="ОФП (Кори)")
        outer.add(crosscheck_tab, text="Сверка Swir")
        outer.add(petro_tab, text="Петрофизика по горизонтам")
        outer.add(cubes_tab, text="Кубы концевых точек")

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
        self._build_crosscheck_tab(crosscheck_tab)
        self._build_petro_tab(petro_tab)
        self._build_cubes_tab(cubes_tab)

    def _build_petro_tab(self, parent: ttk.Widget) -> None:
        """Петрофизика керна по горизонтам: k = a*exp(b*Кп) отдельно на каждый горизонт."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Button(
            top, text="Загрузить петрофизику керна (.xlsx)...", command=self.on_load_petro
        ).pack(side="left")
        self.petro_file_label = ttk.Label(top, text="Файл не загружен")
        self.petro_file_label.pack(side="left", padx=10)

        ttk.Label(top, text="Мин. образцов на горизонт:").pack(side="left", padx=(20, 4))
        self.petro_min_samples_var = tk.StringVar(value="5")
        ttk.Entry(top, textvariable=self.petro_min_samples_var, width=5).pack(side="left")
        ttk.Button(top, text="Построить", command=self.on_run_petro).pack(side="left", padx=(8, 0))

        ttk.Label(top, text="Горизонт (график):").pack(side="left", padx=(20, 4))
        self.petro_horizon_var = tk.StringVar(value="Все")
        self.petro_horizon_combo = ttk.Combobox(
            top, textvariable=self.petro_horizon_var, state="readonly", width=14, values=["Все"],
        )
        self.petro_horizon_combo.pack(side="left")
        self.petro_horizon_combo.bind("<<ComboboxSelected>>", self._on_petro_filter_change)

        ttk.Button(
            top, text="Экспортировать таблицу...", command=self.on_export_petro
        ).pack(side="right")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.petro_figure = Figure(figsize=(6, 5), dpi=100)
        self.petro_ax = self.petro_figure.add_subplot(111)
        self.petro_canvas = FigureCanvasTkAgg(self.petro_figure, master=body)
        self.petro_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(
            right,
            text="k = a·exp(b·Кп), МНК по ln(k) от Кп, отдельно на каждый горизонт\n"
                 "(горизонты с числом образцов меньше порога не показываются -\n"
                 "тренд на 1-3 точках не показателен).",
            justify="left", wraplength=280,
        ).pack(anchor="w", pady=(0, 6))

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True)
        self.petro_tree = ttk.Treeview(tree_frame, columns=PETRO_COLUMNS, show="headings", height=14)
        for col in PETRO_COLUMNS:
            self.petro_tree.heading(col, text=col)
            self.petro_tree.column(col, width=75, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.petro_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.petro_tree.xview)
        self.petro_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.petro_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.petro_tree.bind("<<TreeviewSelect>>", self._on_petro_tree_select)

        formula_frame = ttk.LabelFrame(right, text="Формула зависимости (для куба в Petrel)", padding=8)
        formula_frame.pack(fill="x", pady=(8, 0))

        self.petro_formula_var = tk.StringVar(value="Выберите горизонт в таблице или в фильтре выше.")
        ttk.Label(
            formula_frame, textvariable=self.petro_formula_var, wraplength=320, justify="left",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")

        self.petro_formula_expr_var = tk.StringVar(value="")
        ttk.Entry(formula_frame, textvariable=self.petro_formula_expr_var, state="readonly", width=40).pack(
            fill="x", pady=(6, 4)
        )
        ttk.Button(
            formula_frame, text="Скопировать формулу", command=self.on_copy_petro_formula
        ).pack(anchor="w")

    def _build_cubes_tab(self, parent: ttk.Widget) -> None:
        """Кубы концевых точек: корреляция Swir/Sor/krwmax от Кп/k по горизонтам + применение к массиву."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Button(
            top, text="Загрузить сводную таблицу ОФП (.xlsx)...", command=self.on_load_cubes
        ).pack(side="left")
        self.cubes_file_label = ttk.Label(top, text="Файл не загружен")
        self.cubes_file_label.pack(side="left", padx=10)

        ttk.Label(top, text="Группировка:").pack(side="left", padx=(20, 4))
        self.cubes_grouping_var = tk.StringVar(value="По горизонту")
        ttk.Combobox(
            top, textvariable=self.cubes_grouping_var, state="readonly", width=13,
            values=["По горизонту", "Мел/Юра"],
        ).pack(side="left")

        ttk.Label(top, text="Мин. образцов на группу:").pack(side="left", padx=(20, 4))
        self.cubes_min_samples_var = tk.StringVar(value="4")
        ttk.Entry(top, textvariable=self.cubes_min_samples_var, width=5).pack(side="left")
        ttk.Button(top, text="Построить", command=self.on_run_cubes).pack(side="left", padx=(8, 0))

        ttk.Button(top, text="Экспортировать таблицу...", command=self.on_export_cubes).pack(side="right")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.cubes_figure = Figure(figsize=(5.5, 5), dpi=100)
        self.cubes_ax = self.cubes_figure.add_subplot(111)
        self.cubes_canvas = FigureCanvasTkAgg(self.cubes_figure, master=body)
        self.cubes_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(
            right,
            text="Выберите строку в таблице - график слева покажет именно\n"
                 "эту зависимость. Формы (linear/log/power/exp) перебираются\n"
                 "автоматически, выбирается лучшая по R².",
            justify="left", wraplength=320,
        ).pack(anchor="w", pady=(0, 6))

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True)
        self.cubes_tree = ttk.Treeview(tree_frame, columns=CUBES_COLUMNS, show="headings", height=12)
        for col in CUBES_COLUMNS:
            self.cubes_tree.heading(col, text=col)
            self.cubes_tree.column(col, width=75, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.cubes_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.cubes_tree.xview)
        self.cubes_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.cubes_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.cubes_tree.bind("<<TreeviewSelect>>", self._on_cubes_select)
        self.cubes_tree.tag_configure("qc_ok", background="")
        self.cubes_tree.tag_configure("qc_warning", background="#fff3cd")
        self.cubes_tree.tag_configure("qc_bad", background="#f8d7da")

        self.cubes_qc_var = tk.StringVar(value="")
        self.cubes_qc_label = ttk.Label(right, textvariable=self.cubes_qc_var, wraplength=320, justify="left")
        self.cubes_qc_label.pack(anchor="w", pady=(6, 0))

        apply_frame = ttk.LabelFrame(right, text="Применить к массиву значений (построить «куб»)", padding=8)
        apply_frame.pack(fill="x", pady=(8, 0))

        ttk.Label(apply_frame, text="Значения Кп/k через запятую:").pack(anchor="w")
        self.cubes_input_var = tk.StringVar(value="")
        ttk.Entry(apply_frame, textvariable=self.cubes_input_var, width=40).pack(fill="x", pady=(2, 4))
        btn_row = ttk.Frame(apply_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Применить", command=self.on_apply_cube).pack(side="left")
        ttk.Button(
            btn_row, text="Сохранить результат в CSV...", command=self.on_save_cube_result
        ).pack(side="left", padx=(8, 0))
        self.cubes_output_var = tk.StringVar(value="")
        self.cubes_output_label = ttk.Label(
            apply_frame, textvariable=self.cubes_output_var, wraplength=320, foreground="#1a7f37"
        )
        self.cubes_output_label.pack(anchor="w", pady=(4, 0))

    def _build_crosscheck_tab(self, parent: ttk.Widget) -> None:
        """Сверка Swir: капилляриметрия (J-функция) vs ОФП по общим образцам керна."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Сравнить", command=self.on_run_crosscheck).pack(side="left")
        self.crosscheck_info_label = ttk.Label(
            top, text="Загрузите данные на вкладках «J-функция» и «ОФП (Кори)», затем нажмите «Сравнить»."
        )
        self.crosscheck_info_label.pack(side="left", padx=10)
        ttk.Button(
            top, text="Экспортировать таблицу...", command=self.on_export_crosscheck
        ).pack(side="right")

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.crosscheck_figure = Figure(figsize=(5.5, 5), dpi=100)
        self.crosscheck_ax = self.crosscheck_figure.add_subplot(111)
        self.crosscheck_canvas = FigureCanvasTkAgg(self.crosscheck_figure, master=body)
        self.crosscheck_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        stats = ttk.LabelFrame(right, text="Сводка", padding=8)
        stats.pack(fill="x")

        self.crosscheck_n_var = tk.StringVar(value="-")
        self.crosscheck_corr_var = tk.StringVar(value="-")
        self.crosscheck_mean_var = tk.StringVar(value="-")
        self.crosscheck_median_var = tk.StringVar(value="-")

        stats_rows = [
            ("Найдено общих образцов", self.crosscheck_n_var),
            ("Корреляция R", self.crosscheck_corr_var),
            ("Средняя |разница|", self.crosscheck_mean_var),
            ("Медиана |разница|", self.crosscheck_median_var),
        ]
        for r, (label, var) in enumerate(stats_rows):
            ttk.Label(stats, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            ttk.Entry(stats, textvariable=var, width=10, justify="right", state="readonly").grid(
                row=r, column=1, pady=1
            )

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.crosscheck_tree = ttk.Treeview(
            tree_frame, columns=CROSSCHECK_COLUMNS, show="headings", height=14
        )
        for col in CROSSCHECK_COLUMNS:
            self.crosscheck_tree.heading(col, text=col)
            self.crosscheck_tree.column(col, width=80, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.crosscheck_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.crosscheck_tree.xview)
        self.crosscheck_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.crosscheck_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

    def _build_ofp_tab(self, parent: ttk.Widget) -> None:
        """Вкладка ОФП: загрузка Word-отчёта лаборатории и расчёт степеней Кори (nw, now)."""
        top = ttk.Frame(parent, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="Загрузить ОФП-отчёт (.docx)...", command=self.on_load_ofp).pack(side="left")
        self.ofp_file_label = ttk.Label(top, text="Файл не загружен")
        self.ofp_file_label.pack(side="left", padx=10)
        ttk.Button(top, text="Экспортировать результаты...", command=self.on_export_ofp).pack(side="right")
        ttk.Button(top, text="Экспорт SWOF...", command=self.on_export_swof).pack(side="right", padx=(0, 8))
        ttk.Button(top, text="Экспорт COREYWO...", command=self.on_export_coreywo).pack(side="right", padx=(0, 8))

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
        self.ofp_r2w_var = tk.StringVar(value="-")
        self.ofp_r2o_var = tk.StringVar(value="-")
        self.ofp_manual_var = tk.BooleanVar(value=False)
        self.ofp_nw_scale_var = tk.DoubleVar(value=2.0)
        self.ofp_now_scale_var = tk.DoubleVar(value=1.5)

        bold = ("Segoe UI", 10, "bold")

        ttk.Label(unified, text="nw").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        self.ofp_nw_entry = ttk.Entry(
            unified, textvariable=self.ofp_nw_var, width=10, justify="right", state="readonly", font=bold
        )
        self.ofp_nw_entry.grid(row=0, column=1, pady=2)
        self.ofp_nw_entry.bind("<Return>", lambda _e: self._on_ofp_manual_entry())
        self.ofp_nw_scale = tk.Scale(
            unified, from_=0.1, to=6.0, resolution=0.01, orient="horizontal", length=150,
            variable=self.ofp_nw_scale_var, showvalue=False, state="disabled", command=self._on_ofp_nw_scale,
        )
        self.ofp_nw_scale.grid(row=0, column=2, padx=(6, 0))

        ttk.Label(unified, text="now").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        self.ofp_now_entry = ttk.Entry(
            unified, textvariable=self.ofp_now_var, width=10, justify="right", state="readonly", font=bold
        )
        self.ofp_now_entry.grid(row=1, column=1, pady=2)
        self.ofp_now_entry.bind("<Return>", lambda _e: self._on_ofp_manual_entry())
        self.ofp_now_scale = tk.Scale(
            unified, from_=0.1, to=6.0, resolution=0.01, orient="horizontal", length=150,
            variable=self.ofp_now_scale_var, showvalue=False, state="disabled", command=self._on_ofp_now_scale,
        )
        self.ofp_now_scale.grid(row=1, column=2, padx=(6, 0))

        readonly_rows = [
            ("Swir обр. (среднее)", self.ofp_swir_var),
            ("Sor обр. (среднее)", self.ofp_sor_var),
            ("krwmax обр. (среднее)", self.ofp_krwmax_var),
            ("R²_w (по всем точкам)", self.ofp_r2w_var),
            ("R²_o (по всем точкам)", self.ofp_r2o_var),
            ("Число моделей/образцов", self.ofp_nmodels_var),
        ]
        for i, (label, var) in enumerate(readonly_rows, start=2):
            ttk.Label(unified, text=label).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(unified, textvariable=var, width=10, justify="right", state="readonly").grid(
                row=i, column=1, pady=2
            )

        ttk.Checkbutton(
            unified, text="Задать nw, now вручную (ползунками или числом)",
            variable=self.ofp_manual_var, command=self.on_toggle_ofp_manual,
        ).grid(row=len(readonly_rows) + 2, column=0, columnspan=3, sticky="w", pady=(8, 0))

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
        self.ofp_unified = unified_corey_params(per_model)

        self.ofp_file_label.config(
            text=f"{Path(path).name}  ({per_model['model'].nunique()} моделей, {len(df)} точек)"
        )
        self._update_ofp_table(per_model)

        self.ofp_manual_var.set(False)
        self.ofp_nw_entry.config(state="readonly")
        self.ofp_now_entry.config(state="readonly")
        self.ofp_nw_scale.config(state="disabled")
        self.ofp_now_scale.config(state="disabled")

        self._update_ofp_unified(self.ofp_unified)
        self._recompute_ofp_fit()

    def on_toggle_ofp_manual(self) -> None:
        manual = self.ofp_manual_var.get()
        state = "normal" if manual else "readonly"
        scale_state = "normal" if manual else "disabled"

        if manual:
            try:
                self.ofp_nw_scale_var.set(float(self.ofp_nw_var.get()))
            except ValueError:
                pass
            try:
                self.ofp_now_scale_var.set(float(self.ofp_now_var.get()))
            except ValueError:
                pass

        self.ofp_nw_entry.config(state=state)
        self.ofp_now_entry.config(state=state)
        self.ofp_nw_scale.config(state=scale_state)
        self.ofp_now_scale.config(state=scale_state)

        if not manual and self.ofp_unified is not None:
            # вернуться к автоматической медиане при снятии галочки
            self.ofp_nw_var.set(f"{self.ofp_unified.nw:.4f}")
            self.ofp_now_var.set(f"{self.ofp_unified.now:.4f}")
            self.ofp_nw_scale_var.set(self.ofp_unified.nw)
            self.ofp_now_scale_var.set(self.ofp_unified.now)

        self._recompute_ofp_fit()

    def _on_ofp_nw_scale(self, value: str) -> None:
        if not self.ofp_manual_var.get():
            return
        self.ofp_nw_var.set(f"{float(value):.4f}")
        self._recompute_ofp_fit()

    def _on_ofp_now_scale(self, value: str) -> None:
        if not self.ofp_manual_var.get():
            return
        self.ofp_now_var.set(f"{float(value):.4f}")
        self._recompute_ofp_fit()

    def _on_ofp_manual_entry(self) -> None:
        """Пользователь ввёл nw или now числом и нажал Enter - пересчитать и подвинуть ползунки."""
        self._recompute_ofp_fit()
        try:
            self.ofp_nw_scale_var.set(float(self.ofp_nw_var.get()))
        except ValueError:
            pass
        try:
            self.ofp_now_scale_var.set(float(self.ofp_now_var.get()))
        except ValueError:
            pass

    def _recompute_ofp_fit(self) -> None:
        """Считает R²w/R²o по текущим nw, now (автоматическим или введённым вручную) и обновляет график."""
        if self.ofp_df is None or self.ofp_unified is None:
            return
        try:
            nw = float(self.ofp_nw_var.get())
            now = float(self.ofp_now_var.get())
            if nw <= 0 or now <= 0:
                raise ValueError
        except ValueError:
            self.ofp_r2w_var.set("-")
            self.ofp_r2o_var.set("-")
            return

        fit = evaluate_fixed_corey(nw, now, self.ofp_df)
        self.ofp_r2w_var.set(f"{fit.r2_w:.4f}" if fit.r2_w == fit.r2_w else "-")
        self.ofp_r2o_var.set(f"{fit.r2_o:.4f}" if fit.r2_o == fit.r2_o else "-")
        self._update_ofp_plot(self.ofp_df, self.ofp_unified, nw=nw, now=now)

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
                self.ofp_nw_var, self.ofp_now_var, self.ofp_swir_var, self.ofp_sor_var,
                self.ofp_krwmax_var, self.ofp_nmodels_var, self.ofp_r2w_var, self.ofp_r2o_var,
            ):
                var.set("-")
            return
        self.ofp_nw_var.set(f"{unified.nw:.4f}")
        self.ofp_now_var.set(f"{unified.now:.4f}")
        self.ofp_swir_var.set(f"{unified.swir:.4f}")
        self.ofp_sor_var.set(f"{unified.sor:.4f}")
        self.ofp_krwmax_var.set(f"{unified.krwmax:.4f}")
        self.ofp_nmodels_var.set(str(unified.n_models))
        self.ofp_nw_scale_var.set(unified.nw)
        self.ofp_now_scale_var.set(unified.now)

    def _update_ofp_plot(self, df: pd.DataFrame, unified, nw: float | None = None, now: float | None = None) -> None:
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
            nw_val = unified.nw if nw is None else nw
            now_val = unified.now if now is None else now
            # Swmax (верхняя граница Sw в опыте) = 1 - Sor - так же, как в исходном
            # Excel (формула Sw* = (Sw-Swir)/((1-Sor)-Swir)), а не Sw=1.
            swmax_eff = 1.0 - unified.sor
            sw_grid = np.linspace(unified.swir, swmax_eff, 100)
            sw_star = np.clip((sw_grid - unified.swir) / (swmax_eff - unified.swir), 0, 1)
            krw_curve = unified.krwmax * np.power(sw_star, nw_val)
            kro_curve = unified.krow_swc * np.power(1 - sw_star, now_val)
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

    def _current_ofp_endpoints(self) -> tuple[float, float, float, float, float] | None:
        """Текущие nw, now (авто-медиана или вручную) + Swir/Sor/krwmax (среднее по образцам)."""
        if self.ofp_unified is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите ОФП-отчёт.")
            return None
        try:
            nw = float(self.ofp_nw_var.get())
            now = float(self.ofp_now_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "nw и now должны быть числами.")
            return None
        return nw, now, self.ofp_unified.swir, self.ofp_unified.sor, self.ofp_unified.krwmax

    def on_export_coreywo(self) -> None:
        endpoints = self._current_ofp_endpoints()
        if endpoints is None:
            return
        nw, now, swir, sor, krwmax = endpoints
        text = format_coreywo(nw, now, swir, sor, krwmax)

        path = filedialog.asksaveasfilename(
            title="Сохранить COREYWO как...",
            defaultextension=".txt",
            filetypes=[("Текстовый файл", "*.txt"), ("Все файлы", "*.*")],
            initialfile="coreywo.txt",
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
            messagebox.showinfo("Готово", f"COREYWO сохранён:\n{path}\n\n{text}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    def on_export_swof(self) -> None:
        endpoints = self._current_ofp_endpoints()
        if endpoints is None:
            return
        nw, now, swir, sor, krwmax = endpoints
        text = format_swof(nw, now, swir, sor, krwmax)

        path = filedialog.asksaveasfilename(
            title="Сохранить SWOF как...",
            defaultextension=".txt",
            filetypes=[("Текстовый файл", "*.txt"), ("Все файлы", "*.*")],
            initialfile="swof.txt",
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
            messagebox.showinfo("Готово", f"SWOF сохранён:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    # -------------------------------------------------------- Сверка Swir

    def on_run_crosscheck(self) -> None:
        if self.df is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные на вкладке «J-функция».")
            return
        if self.ofp_df is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите ОФП-отчёт на вкладке «ОФП (Кори)».")
            return

        try:
            result = crosscheck_swir(self.df, self.ofp_df)
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self.crosscheck_result = result
        if result.empty:
            self.crosscheck_info_label.config(
                text="Общих образцов (керн, измеренный и в J-функции, и в ОФП) не найдено."
            )
        else:
            n_total = self.df["sample"].nunique() if "sample" in self.df.columns else "?"
            self.crosscheck_info_label.config(
                text=f"Найдено {len(result)} общих образцов из {n_total} в J-функции."
            )
        self._update_crosscheck_table(result)
        self._update_crosscheck_stats(result)
        self._update_crosscheck_plot(result)

    def _update_crosscheck_table(self, result: pd.DataFrame) -> None:
        self.crosscheck_tree.delete(*self.crosscheck_tree.get_children())
        for _, row in result.iterrows():
            values = []
            for col in CROSSCHECK_COLUMNS:
                v = row.get(col, "")
                if isinstance(v, float):
                    values.append("-" if v != v else f"{v:.4f}")
                else:
                    values.append(v)
            self.crosscheck_tree.insert("", "end", values=values)

    def _update_crosscheck_stats(self, result: pd.DataFrame) -> None:
        if result.empty:
            for var in (
                self.crosscheck_n_var, self.crosscheck_corr_var,
                self.crosscheck_mean_var, self.crosscheck_median_var,
            ):
                var.set("-")
            return
        self.crosscheck_n_var.set(str(len(result)))
        corr = result["Swir_Pc"].corr(result["Swir_OFP"])
        self.crosscheck_corr_var.set(f"{corr:.4f}" if corr == corr else "-")
        self.crosscheck_mean_var.set(f"{result['abs_diff'].mean():.4f}")
        self.crosscheck_median_var.set(f"{result['abs_diff'].median():.4f}")

    def _update_crosscheck_plot(self, result: pd.DataFrame) -> None:
        self.crosscheck_ax.clear()
        if not result.empty:
            wells = sorted(result["well"].dropna().astype(str).unique())
            well_colors = {w: PINNED_COLORS[i % len(PINNED_COLORS)] for i, w in enumerate(wells)}
            for well, sub in result.groupby("well"):
                color = well_colors.get(str(well), "gray")
                self.crosscheck_ax.scatter(
                    sub["Swir_Pc"], sub["Swir_OFP"], s=60, alpha=0.8, color=color,
                    edgecolor="black", label=f"скв. {well}",
                )
            lo = min(result["Swir_Pc"].min(), result["Swir_OFP"].min()) - 0.02
            hi = max(result["Swir_Pc"].max(), result["Swir_OFP"].max()) + 0.02
            self.crosscheck_ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Swir(Pc) = Swir(ОФП)")
            self.crosscheck_ax.set_xlim(lo, hi)
            self.crosscheck_ax.set_ylim(lo, hi)
            self.crosscheck_ax.legend(fontsize=8)

        self.crosscheck_ax.set_xlabel("Swir по капилляриметрии (J-функция)")
        self.crosscheck_ax.set_ylabel("Swir по ОФП")
        self.crosscheck_ax.set_title("Сверка Swir по общим образцам")
        self.crosscheck_ax.grid(True, alpha=0.3)
        self.crosscheck_canvas.draw()

    def on_export_crosscheck(self) -> None:
        if self.crosscheck_result is None or self.crosscheck_result.empty:
            messagebox.showwarning("Нет данных", "Сначала выполните сравнение (кнопка «Сравнить»).")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить таблицу сверки как...",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Все файлы", "*.*")],
            initialfile="swir_crosscheck.xlsx",
        )
        if not path:
            return
        try:
            self.crosscheck_result.to_excel(path, index=False)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    # ------------------------------------------------ Петрофизика по горизонтам

    def on_load_petro(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл с результатами петрофизического анализа керна",
            filetypes=[("Excel", "*.xlsx"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            df = load_core_petro_xlsx(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        if df.empty:
            messagebox.showwarning("Нет данных", "Не удалось найти данные в файле.")
            return

        self.petro_df = df
        self.petro_file_label.config(
            text=f"{Path(path).name}  ({len(df)} образцов, {df['horizon'].nunique()} горизонтов)"
        )
        self.on_run_petro()

    def on_run_petro(self) -> None:
        if self.petro_df is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите файл с петрофизикой керна.")
            return
        try:
            min_samples = int(self.petro_min_samples_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "«Мин. образцов на горизонт» должно быть целым числом.")
            return

        fits = fit_poro_perm_by_horizon(self.petro_df, min_samples=min_samples)
        self.petro_fits = fits
        self._update_petro_table(fits)

        horizons = sorted(fits["horizon"]) if not fits.empty else []
        self.petro_horizon_combo.config(values=["Все"] + horizons)
        self.petro_horizon_var.set("Все")
        self._set_petro_formula(None)
        self._update_petro_plot(self.petro_df, fits, horizon_filter="Все")

    def _update_petro_table(self, fits: pd.DataFrame) -> None:
        self.petro_tree.delete(*self.petro_tree.get_children())
        for _, row in fits.iterrows():
            values = []
            for col in PETRO_COLUMNS:
                v = row.get(col, "")
                if isinstance(v, float):
                    values.append("-" if v != v else f"{v:.4g}")
                else:
                    values.append(v)
            self.petro_tree.insert("", "end", values=values)

    def _on_petro_tree_select(self, _event=None) -> None:
        sel = self.petro_tree.selection()
        if not sel or self.petro_fits is None:
            return
        idx = self.petro_tree.index(sel[0])
        row = self.petro_fits.iloc[idx]
        self.petro_horizon_var.set(row["horizon"])
        self._update_petro_plot(self.petro_df, self.petro_fits, horizon_filter=row["horizon"])
        self._set_petro_formula(row)

    def _on_petro_filter_change(self, _event=None) -> None:
        if self.petro_df is None or self.petro_fits is None:
            return
        horizon = self.petro_horizon_var.get()
        self._update_petro_plot(self.petro_df, self.petro_fits, horizon_filter=horizon)
        if horizon == "Все":
            self._set_petro_formula(None)
        else:
            match = self.petro_fits[self.petro_fits["horizon"] == horizon]
            self._set_petro_formula(match.iloc[0] if not match.empty else None)

    def _set_petro_formula(self, row) -> None:
        if row is None:
            self.petro_formula_var.set("Выберите горизонт в таблице или в фильтре выше.")
            self.petro_formula_expr_var.set("")
            return
        a, b, r2, n = row["a"], row["b"], row["r2"], row["n"]
        self.petro_formula_var.set(
            f"{row['horizon']}: k = {a:.4g}·exp({b:.4g}·Кп)   (R²={r2:.3f}, n={n})"
        )
        self.petro_formula_expr_var.set(f"{a:.6g}*Exp({b:.6g}*$Poro)")

    def on_copy_petro_formula(self) -> None:
        expr = self.petro_formula_expr_var.get()
        if not expr:
            messagebox.showwarning("Нет формулы", "Сначала выберите горизонт в таблице или в фильтре.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(expr)

    def _update_petro_plot(self, df: pd.DataFrame, fits: pd.DataFrame, horizon_filter: str = "Все") -> None:
        self.petro_ax.clear()

        if horizon_filter != "Все":
            df = df[df["horizon"] == horizon_filter]
            fits = fits[fits["horizon"] == horizon_filter]

        horizons = list(fits["horizon"]) if not fits.empty else []
        if horizon_filter != "Все":
            color_map = {h: "steelblue" for h in horizons}
            line_color = "red"
        else:
            color_map = {h: PINNED_COLORS[i % len(PINNED_COLORS)] for i, h in enumerate(horizons)}
            line_color = None

        for horizon, sub in df.groupby("horizon"):
            sub = sub.dropna(subset=["poro_open", "perm_gas"])
            sub = sub[sub["perm_gas"] > 0]
            if sub.empty:
                continue
            color = color_map.get(horizon, "lightgray")
            label = f"{horizon} (n={len(sub)})" if horizon in color_map else None
            self.petro_ax.scatter(
                sub["poro_open"], sub["perm_gas"], s=20 if horizon_filter == "Все" else 40,
                alpha=0.7, color=color, edgecolor="black" if horizon_filter != "Все" else None, label=label,
            )

        for _, row in fits.iterrows():
            color = line_color or color_map.get(row["horizon"], "black")
            xx = np.linspace(row["poro_min"], row["poro_max"], 50)
            yy = row["a"] * np.exp(row["b"] * xx)
            self.petro_ax.plot(xx, yy, color=color, linewidth=2)

        self.petro_ax.set_yscale("log")
        self.petro_ax.set_xlabel("Пористость (открытая), %")
        self.petro_ax.set_ylabel("Проницаемость (газ), мД")
        title = "k = a·exp(b·Кп) по горизонтам" if horizon_filter == "Все" else f"{horizon_filter}: k = a·exp(b·Кп)"
        self.petro_ax.set_title(title)
        if horizon_filter == "Все":
            self.petro_ax.legend(fontsize=8)
        self.petro_ax.grid(True, which="both", alpha=0.3)
        self.petro_canvas.draw()

    def on_export_petro(self) -> None:
        if self.petro_fits is None or self.petro_fits.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные и нажмите «Построить».")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить таблицу зависимостей как...",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Все файлы", "*.*")],
            initialfile="poro_perm_by_horizon.xlsx",
        )
        if not path:
            return
        try:
            self.petro_fits.to_excel(path, index=False)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    # -------------------------------------------------------- Кубы концевых точек

    def on_load_cubes(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл со сводной таблицей ОФП (лист «ОФП», Таблица 2.4.2)",
            filetypes=[("Excel", "*.xlsx"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            df = load_endpoint_summary_xlsx(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка загрузки", str(exc))
            return

        if df.empty:
            messagebox.showwarning("Нет данных", "Не удалось найти данные на листе «ОФП».")
            return

        self.cubes_df = df
        self.cubes_file_label.config(
            text=f"{Path(path).name}  ({len(df)} образцов, {df['horizon'].nunique()} горизонтов)"
        )
        self.on_run_cubes()

    def on_run_cubes(self) -> None:
        if self.cubes_df is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите сводную таблицу ОФП.")
            return
        try:
            min_samples = int(self.cubes_min_samples_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "«Мин. образцов на группу» должно быть целым числом.")
            return

        if self.cubes_grouping_var.get() == "Мел/Юра":
            self.cubes_active_df = group_by_formation(self.cubes_df)
        else:
            self.cubes_active_df = self.cubes_df

        self.cubes_fits = fit_endpoint_cubes(self.cubes_active_df, min_samples=min_samples)
        self._update_cubes_table()
        self.cubes_selected = None
        self._cubes_last_input = None
        self._cubes_last_result = None
        self.cubes_output_var.set("")
        self.cubes_qc_var.set("")
        self.cubes_ax.clear()
        self.cubes_canvas.draw()

        if not self.cubes_fits:
            messagebox.showwarning(
                "Нет корреляций",
                "Ни для одного горизонта не набралось достаточно образцов "
                "(или не хватает данных Кп/k/концевых точек). Попробуйте "
                "уменьшить «Мин. образцов на горизонт».",
            )

    def _update_cubes_table(self) -> None:
        self.cubes_tree.delete(*self.cubes_tree.get_children())
        for corr in self.cubes_fits:
            self.cubes_tree.insert(
                "", "end",
                values=(
                    corr.horizon, corr.endpoint, corr.x_var, corr.form,
                    f"{corr.a:.4g}", f"{corr.b:.4g}", f"{corr.r2:.3f}", corr.n,
                ),
                tags=(f"qc_{quality_level(corr)}",),
            )

    def _on_cubes_select(self, _event=None) -> None:
        sel = self.cubes_tree.selection()
        if not sel:
            return
        idx = self.cubes_tree.index(sel[0])
        self.cubes_selected = idx
        corr = self.cubes_fits[idx]
        self._update_cubes_plot(corr)
        self._update_cubes_qc(corr)

    def _update_cubes_qc(self, corr) -> None:
        flags = quality_flags(corr)
        if not flags:
            self.cubes_qc_label.config(foreground="#1a7f37")
            self.cubes_qc_var.set("✓ контроль качества: замечаний нет")
            return
        level = quality_level(corr)
        color = "#c0392b" if level == "bad" else "#b7791f"
        mark = "✗" if level == "bad" else "⚠"
        self.cubes_qc_label.config(foreground=color)
        self.cubes_qc_var.set(f"{mark} " + "; ".join(flags))

    def _update_cubes_plot(self, corr) -> None:
        self.cubes_ax.clear()
        sub = self.cubes_active_df[self.cubes_active_df["horizon"] == corr.horizon].dropna(
            subset=[corr.x_var, corr.endpoint]
        )
        self.cubes_ax.scatter(sub[corr.x_var], sub[corr.endpoint], s=40, color="steelblue", edgecolor="black")

        xx = np.linspace(corr.x_min, corr.x_max, 100)
        yy = corr.predict(xx)
        self.cubes_ax.plot(xx, yy, color="red", linewidth=2)

        x_label = "Пористость, %" if corr.x_var == "porosity_pct" else "Проницаемость, мД"
        self.cubes_ax.set_xlabel(x_label)
        self.cubes_ax.set_ylabel(corr.endpoint)
        self.cubes_ax.set_title(f"{corr.horizon}: {corr.endpoint} = f(x), {corr.form}, R²={corr.r2:.3f}")
        if corr.x_var == "perm_mD":
            self.cubes_ax.set_xscale("log")
        self.cubes_ax.grid(True, alpha=0.3)
        self.cubes_canvas.draw()

    def on_apply_cube(self) -> None:
        if self.cubes_selected is None:
            messagebox.showwarning("Не выбрано", "Сначала выберите строку в таблице зависимостей.")
            return
        text = self.cubes_input_var.get().strip()
        if not text:
            messagebox.showwarning("Нет значений", "Введите значения Кп/k через запятую.")
            return
        try:
            values = [float(v.strip().replace(",", ".")) for v in text.split(",") if v.strip()]
        except ValueError:
            messagebox.showerror("Ошибка", "Все значения должны быть числами через запятую.")
            return

        corr = self.cubes_fits[self.cubes_selected]
        result = apply_correlation(corr, values)
        self._cubes_last_input = values
        self._cubes_last_result = result
        pairs = ", ".join(f"{x:g}→{y:.4f}" for x, y in zip(values, result))
        apply_warnings = check_apply_inputs(corr, values)
        if apply_warnings:
            msg = f"{corr.endpoint} по {corr.horizon}: {pairs}\n⚠ " + "; ".join(apply_warnings)
            self.cubes_output_label.config(foreground="#b7791f")
        else:
            msg = f"{corr.endpoint} по {corr.horizon}: {pairs}"
            self.cubes_output_label.config(foreground="#1a7f37")
        self.cubes_output_var.set(msg)

    def on_save_cube_result(self) -> None:
        if self._cubes_last_result is None or self.cubes_selected is None:
            messagebox.showwarning("Нет данных", "Сначала нажмите «Применить».")
            return
        corr = self.cubes_fits[self.cubes_selected]

        path = filedialog.asksaveasfilename(
            title="Сохранить результат как...",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")],
            initialfile=f"{corr.endpoint}_{corr.horizon}.csv",
        )
        if not path:
            return
        try:
            out = pd.DataFrame({corr.x_var: self._cubes_last_input, corr.endpoint: self._cubes_last_result})
            out.to_csv(path, index=False)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))

    def on_export_cubes(self) -> None:
        if not self.cubes_fits:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные и нажмите «Построить».")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить таблицу корреляций как...",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Все файлы", "*.*")],
            initialfile="endpoint_correlations.xlsx",
        )
        if not path:
            return
        try:
            rows = [
                {
                    "horizon": c.horizon, "endpoint": c.endpoint, "x_var": c.x_var, "form": c.form,
                    "a": c.a, "b": c.b, "r2": c.r2, "n": c.n, "x_min": c.x_min, "x_max": c.x_max,
                    "качество": quality_level(c), "замечания": "; ".join(quality_flags(c)),
                }
                for c in self.cubes_fits
            ]
            pd.DataFrame(rows).to_excel(path, index=False)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка экспорта", str(exc))


def main() -> None:
    root = tk.Tk()
    JFunctionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
