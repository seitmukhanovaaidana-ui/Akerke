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
from .fit import evaluate_fixed_params, fit_exponential
from .io import load_lab_data
from .report import save_results

ALL = "Все"
TABLE_COLUMNS = ("well", "sample", "horizon", "Sw", "Pc_lab_MPa", "SWn", "J")


def _fmt_num(value: float) -> str:
    """Целые числа показываем без ".0" (30, а не 30.0) - как в исходном Excel."""
    return str(int(value)) if float(value).is_integer() else str(value)


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
        ttk.Button(top, text="Сохранить график...", command=self.on_save_chart).pack(side="right", padx=(0, 8))

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
        self._build_result_panel(middle)

        self.result_label = tk.Label(
            self.root,
            text="Загрузите файл с лабораторными данными.",
            font=("Segoe UI", 13, "bold"),
            bg="#ED7D31",
            fg="white",
            anchor="w",
            padx=12,
            pady=8,
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

    def _build_result_panel(self, parent: ttk.Widget) -> None:
        """Панель "Результат" - a, b, n, R² каждый в своём отдельном окошке."""
        panel = ttk.LabelFrame(parent, text="Результат: J(SWn) = a·exp(b·SWn)", padding=8)
        panel.pack(side="left", fill="y", padx=(10, 0))

        self.a_var = tk.StringVar(value="-")
        self.b_var = tk.StringVar(value="-")
        self.n_var = tk.StringVar(value="-")
        self.r2_var = tk.StringVar(value="-")
        self.manual_ab_var = tk.BooleanVar(value=False)

        bold = ("Segoe UI", 11, "bold")
        ttk.Label(panel, text="a").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.a_entry = ttk.Entry(panel, textvariable=self.a_var, width=12, justify="right",
                                  state="readonly", font=bold)
        self.a_entry.grid(row=0, column=1, pady=3)
        self.a_entry.bind("<Return>", lambda _e: self.recompute())

        ttk.Label(panel, text="b").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.b_entry = ttk.Entry(panel, textvariable=self.b_var, width=12, justify="right",
                                  state="readonly", font=bold)
        self.b_entry.grid(row=1, column=1, pady=3)
        self.b_entry.bind("<Return>", lambda _e: self.recompute())

        ttk.Label(panel, text="n (число точек)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(panel, textvariable=self.n_var, width=12, justify="right", state="readonly").grid(
            row=2, column=1, pady=3
        )

        ttk.Label(panel, text="R² (качество подгонки)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(panel, textvariable=self.r2_var, width=12, justify="right", state="readonly").grid(
            row=3, column=1, pady=3
        )

        ttk.Checkbutton(
            panel, text="Задать a, b вручную", variable=self.manual_ab_var, command=self.on_toggle_manual_ab
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 2))

        ttk.Button(panel, text="Применить a, b", command=self.recompute).grid(
            row=5, column=0, columnspan=2, pady=(2, 0), sticky="we"
        )

    def on_toggle_manual_ab(self) -> None:
        state = "normal" if self.manual_ab_var.get() else "readonly"
        self.a_entry.config(state=state)
        self.b_entry.config(state=state)
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
        else:
            for var in (self.a_var, self.b_var, self.n_var, self.r2_var):
                var.set("-")

        self._update_plot(df, fit)
        self._update_table(df)

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
        if fit is not None:
            swn_grid = np.linspace(max(df["SWn"].min(), 0), df["SWn"].max(), 200)
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


def main() -> None:
    root = tk.Tk()
    JFunctionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
