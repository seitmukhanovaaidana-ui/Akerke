"""
Сверка остаточной водонасыщенности (Swir), полученной двумя независимыми
методами по одному и тому же керну:

    - Swir(Pc) - из капилляриметрии (J-функция): Sw в точке максимального Pc;
    - Swir(ОФП) - из эксперимента по определению относительных фазовых
      проницаемостей (Кори): насыщенность, при которой вода перестаёт
      двигаться.

Сопоставление образцов идёт по номеру керна (столбец "sample" в данных
J-функции и столбец "samples" - список номеров через запятую - в данных
ОФП), в пределах одной скважины.
"""

from __future__ import annotations

import pandas as pd


def crosscheck_swir(jf_df: pd.DataFrame, ofp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает таблицу образцов, для которых есть измерения и в J-функции,
    и в ОФП: well, sample, model_OFP, Swir_Pc, Swir_OFP, Sor_OFP, diff,
    abs_diff (+ horizon/porosity_pct/perm_mD из данных J-функции, если есть).
    """
    if "sample" not in jf_df.columns or "well" not in jf_df.columns or "Swir" not in jf_df.columns:
        raise ValueError("В данных J-функции нет столбцов well/sample/Swir.")
    if "samples" not in ofp_df.columns or "well" not in ofp_df.columns or "Swir" not in ofp_df.columns:
        raise ValueError("В данных ОФП нет столбцов well/samples/Swir (загрузите ОФП-отчёт заново).")

    agg = {"Swir_Pc": ("Swir", "first")}
    for col in ("horizon", "porosity_pct", "perm_mD"):
        if col in jf_df.columns:
            agg[col] = (col, "first")
    jf_samples = jf_df.dropna(subset=["sample"]).groupby(["well", "sample"], as_index=False).agg(**agg)

    ofp_models = ofp_df.groupby(["model", "well", "samples"], as_index=False).agg(
        Swir_OFP=("Swir", "first"), Sor_OFP=("Sor", "first")
    )

    def find_match(well, sample):
        well = str(well)
        sample = str(sample)
        candidates = ofp_models[ofp_models["well"].astype(str) == well]
        for _, row in candidates.iterrows():
            sample_list = [s.strip() for s in str(row["samples"]).split(",")]
            if sample in sample_list:
                return row["model"], row["Swir_OFP"], row["Sor_OFP"]
        return None, None, None

    matches = []
    for _, r in jf_samples.iterrows():
        model, swir_ofp, sor_ofp = find_match(r["well"], r["sample"])
        if model is None:
            continue
        row = dict(r)
        row["model_OFP"] = model
        row["Swir_OFP"] = swir_ofp
        row["Sor_OFP"] = sor_ofp
        matches.append(row)

    result = pd.DataFrame(matches)
    if not result.empty:
        result["diff"] = result["Swir_Pc"] - result["Swir_OFP"]
        result["abs_diff"] = result["diff"].abs()
        result = result.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return result
