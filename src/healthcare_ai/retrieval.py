from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from healthcare_ai.data import DatasetBundle


@dataclass(frozen=True)
class RetrievalIndex:
    case_lookup: pd.DataFrame
    matrix: object
    vectorizer: TfidfVectorizer


STOPWORDS = {
    "acute",
    "chronic",
    "disease",
    "failure",
    "history",
    "pain",
    "primary",
    "secondary",
    "status",
    "syndrome",
    "telemetry",
    "unspecified",
}


def _aggregate_text(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return " | ".join(cleaned.drop_duplicates().tolist())


def _split_aggregated(text: object) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def _shared_entities(source: object, candidate: object, limit: int = 3) -> list[str]:
    source_items = set(_split_aggregated(source))
    candidate_items = set(_split_aggregated(candidate))
    return sorted(source_items & candidate_items)[:limit]


def _shared_admission_terms(source: object, candidate: object, limit: int = 3) -> list[str]:
    if not isinstance(source, str) or not isinstance(candidate, str):
        return []
    source_terms = {
        term
        for term in re.findall(r"[a-z]{4,}", source.lower())
        if term not in STOPWORDS
    }
    candidate_terms = {
        term
        for term in re.findall(r"[a-z]{4,}", candidate.lower())
        if term not in STOPWORDS
    }
    return sorted(source_terms & candidate_terms)[:limit]


def _build_match_reasons(source_row: pd.Series, candidate_row: pd.Series) -> str:
    reasons: list[str] = []

    shared_diagnoses = _shared_entities(
        source_row.get("diagnosis_text", ""),
        candidate_row.get("diagnosis_text", ""),
    )
    if shared_diagnoses:
        reasons.append("Shared diagnoses: " + ", ".join(shared_diagnoses))

    shared_medications = _shared_entities(
        source_row.get("medication_text", ""),
        candidate_row.get("medication_text", ""),
    )
    if shared_medications:
        reasons.append("Shared medications: " + ", ".join(shared_medications))

    shared_labs = _shared_entities(
        source_row.get("lab_text", ""),
        candidate_row.get("lab_text", ""),
    )
    if shared_labs:
        reasons.append("Shared labs: " + ", ".join(shared_labs))

    shared_terms = _shared_admission_terms(
        source_row.get("admission_diagnosis", ""),
        candidate_row.get("admission_diagnosis", ""),
    )
    if shared_terms:
        reasons.append("Shared admission terms: " + ", ".join(shared_terms))

    if not reasons:
        reasons.append("Similarity driven by broader overlap across note, diagnosis, medication, and lab vocabulary")

    return " | ".join(reasons[:3])


def _build_case_lookup(bundle: DatasetBundle) -> pd.DataFrame:
    diagnoses = (
        bundle.diagnoses.groupby("hadm_id")
        .agg(
            diagnosis_text=("long_title", _aggregate_text),
            diagnosis_code_text=("icd9_code", _aggregate_text),
        )
        .reset_index()
    )
    prescriptions = (
        bundle.prescriptions.groupby("hadm_id")
        .agg(
            medication_text=("drug", _aggregate_text),
            route_text=("route", _aggregate_text),
        )
        .reset_index()
    )
    labs = (
        bundle.labs.groupby("hadm_id")
        .agg(
            lab_text=("lab_name", _aggregate_text),
            lab_category_text=("category", _aggregate_text),
        )
        .reset_index()
    )

    case_lookup = bundle.clinical_cases.merge(diagnoses, on="hadm_id", how="left")
    case_lookup = case_lookup.merge(prescriptions, on="hadm_id", how="left")
    case_lookup = case_lookup.merge(labs, on="hadm_id", how="left")

    for column in [
        "admission_diagnosis",
        "discharge_summary",
        "diagnosis_text",
        "diagnosis_code_text",
        "medication_text",
        "route_text",
        "lab_text",
        "lab_category_text",
    ]:
        case_lookup[column] = case_lookup[column].fillna("")

    case_lookup["retrieval_text"] = (
        "admission diagnosis "
        + case_lookup["admission_diagnosis"]
        + " diagnoses "
        + case_lookup["diagnosis_text"]
        + " diagnosis codes "
        + case_lookup["diagnosis_code_text"]
        + " medications "
        + case_lookup["medication_text"]
        + " medication routes "
        + case_lookup["route_text"]
        + " labs "
        + case_lookup["lab_text"]
        + " lab categories "
        + case_lookup["lab_category_text"]
        + " note "
        + case_lookup["discharge_summary"].str.slice(0, 1500)
    )
    return case_lookup


def build_retrieval_index(bundle: DatasetBundle) -> RetrievalIndex:
    case_lookup = _build_case_lookup(bundle)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(case_lookup["retrieval_text"])
    return RetrievalIndex(case_lookup=case_lookup, matrix=matrix, vectorizer=vectorizer)


def find_similar_cases(index: RetrievalIndex, hadm_id: int, top_n: int = 5) -> pd.DataFrame:
    matches = index.case_lookup.index[index.case_lookup["hadm_id"] == hadm_id].tolist()
    if not matches:
        raise KeyError(f"No case found for hadm_id={hadm_id}")

    row_idx = matches[0]
    source_row = index.case_lookup.iloc[row_idx]
    similarity_scores = linear_kernel(index.matrix[row_idx : row_idx + 1], index.matrix).ravel()
    ranked_indices = similarity_scores.argsort()[::-1]

    rows: list[dict[str, object]] = []
    for idx in ranked_indices:
        candidate = int(index.case_lookup.iloc[idx]["hadm_id"])
        if candidate == hadm_id:
            continue
        row = index.case_lookup.iloc[idx]
        rows.append(
            {
                "hadm_id": candidate,
                "case_id": row.get("case_id"),
                "age": row.get("age"),
                "gender": row.get("gender"),
                "admission_diagnosis": row.get("admission_diagnosis"),
                "similarity": round(float(similarity_scores[idx]), 3),
                "match_reasons": _build_match_reasons(source_row, row),
            }
        )
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows)