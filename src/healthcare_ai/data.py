from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "bavehackathon/2026-healthcare-ai"
RAW_DATA_DIR = Path("data/raw")


@dataclass(frozen=True)
class DatasetBundle:
    clinical_cases: pd.DataFrame
    labs: pd.DataFrame
    prescriptions: pd.DataFrame
    diagnoses: pd.DataFrame
    lab_dictionary: pd.DataFrame
    diagnosis_dictionary: pd.DataFrame


def _download_csv(filename: str, cache_dir: Path | None = None) -> pd.DataFrame:
    destination = cache_dir or RAW_DATA_DIR
    destination.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=destination,
    )
    return pd.read_csv(local_path)


@lru_cache(maxsize=1)
def load_dataset() -> DatasetBundle:
    clinical_cases = _download_csv("clinical_cases.csv.gz")
    labs = _download_csv("labs_subset.csv.gz")
    prescriptions = _download_csv("prescriptions_subset.csv.gz")
    diagnoses = _download_csv("diagnoses_subset.csv.gz")
    lab_dictionary = _download_csv("lab_dictionary.csv.gz")
    diagnosis_dictionary = _download_csv("diagnosis_dictionary.csv.gz")

    labs = labs.merge(lab_dictionary, on="itemid", how="left")
    diagnoses = diagnoses.merge(diagnosis_dictionary, on="icd9_code", how="left")

    labs["numeric_value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    prescriptions["startdate"] = pd.to_datetime(prescriptions["startdate"], errors="coerce")
    prescriptions["enddate"] = pd.to_datetime(prescriptions["enddate"], errors="coerce")

    return DatasetBundle(
        clinical_cases=clinical_cases,
        labs=labs,
        prescriptions=prescriptions,
        diagnoses=diagnoses,
        lab_dictionary=lab_dictionary,
        diagnosis_dictionary=diagnosis_dictionary,
    )


def get_case_row(bundle: DatasetBundle, hadm_id: int) -> pd.Series:
    matches = bundle.clinical_cases.loc[bundle.clinical_cases["hadm_id"] == hadm_id]
    if matches.empty:
        raise KeyError(f"No case found for hadm_id={hadm_id}")
    return matches.iloc[0]


def available_hadm_ids(bundle: DatasetBundle) -> list[int]:
    return sorted(bundle.clinical_cases["hadm_id"].dropna().astype(int).tolist())
