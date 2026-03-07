from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from healthcare_ai.data import DatasetBundle, get_case_row


@dataclass
class CaseContext:
    case_row: pd.Series
    diagnoses: pd.DataFrame
    prescriptions: pd.DataFrame
    labs: pd.DataFrame
    abnormal_labs: pd.DataFrame


def _lab_reference_stats(labs: pd.DataFrame) -> pd.DataFrame:
    numeric_labs = labs.dropna(subset=["numeric_value", "lab_name"]).copy()
    if numeric_labs.empty:
        return pd.DataFrame(columns=["lab_name", "median", "q1", "q3"])
    stats = (
        numeric_labs.groupby("lab_name")["numeric_value"]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    stats["iqr"] = (stats["q3"] - stats["q1"]).replace(0, np.nan)
    return stats


def _flag_abnormal_labs(case_labs: pd.DataFrame, population_labs: pd.DataFrame) -> pd.DataFrame:
    if case_labs.empty:
        return case_labs.copy()
    stats = _lab_reference_stats(population_labs)
    merged = case_labs.merge(stats, on="lab_name", how="left")
    merged["iqr_distance"] = (
        (merged["numeric_value"] - merged["median"]).abs() / merged["iqr"]
    )
    abnormal = merged.loc[merged["iqr_distance"].fillna(0) >= 1.5].copy()
    abnormal["direction"] = np.where(
        abnormal["numeric_value"] >= abnormal["median"],
        "high",
        "low",
    )
    abnormal = abnormal.sort_values(["iqr_distance", "charttime"], ascending=[False, True])
    return abnormal


def build_case_context(bundle: DatasetBundle, hadm_id: int) -> CaseContext:
    case_row = get_case_row(bundle, hadm_id)
    diagnoses = (
        bundle.diagnoses.loc[bundle.diagnoses["hadm_id"] == hadm_id]
        .sort_values("seq_num")
        .copy()
    )
    prescriptions = (
        bundle.prescriptions.loc[bundle.prescriptions["hadm_id"] == hadm_id]
        .sort_values(["startdate", "drug"], na_position="last")
        .copy()
    )
    labs = (
        bundle.labs.loc[bundle.labs["hadm_id"] == hadm_id]
        .sort_values(["charttime", "lab_name"], na_position="last")
        .copy()
    )
    abnormal_labs = _flag_abnormal_labs(labs, bundle.labs)
    return CaseContext(
        case_row=case_row,
        diagnoses=diagnoses,
        prescriptions=prescriptions,
        labs=labs,
        abnormal_labs=abnormal_labs,
    )


def top_diagnosis_titles(diagnoses: pd.DataFrame, limit: int = 5) -> list[str]:
    titles = diagnoses["long_title"].fillna(diagnoses["short_title"]).dropna().astype(str)
    return titles.head(limit).tolist()


def top_medications(prescriptions: pd.DataFrame, limit: int = 8) -> list[str]:
    medications = prescriptions["drug"].dropna().astype(str).drop_duplicates()
    return medications.head(limit).tolist()


def lab_signal_strings(abnormal_labs: pd.DataFrame, limit: int = 6) -> list[str]:
    if abnormal_labs.empty:
        return []
    signals = []
    for _, row in abnormal_labs.head(limit).iterrows():
        value = row.get("numeric_value")
        unit = row.get("unit")
        if pd.isna(unit):
            unit = ""
        signal = (
            f"{row.get('lab_name', 'Unknown lab')}: {value:g} {unit}".strip()
            + f" ({row.get('direction', 'abnormal')})"
        )
        signals.append(signal)
    return signals
