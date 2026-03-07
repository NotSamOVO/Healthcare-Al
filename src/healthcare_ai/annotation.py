from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from healthcare_ai.data import load_dataset
from healthcare_ai.features import build_case_context, lab_signal_strings, top_diagnosis_titles, top_medications
from healthcare_ai.summarizer import build_case_summary
from healthcare_ai.triage import score_case


def _safe_text(value: str, limit: int = 400) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def _build_annotation_row(hadm_id: int) -> dict[str, object]:
    bundle = load_dataset()
    context = build_case_context(bundle, hadm_id)
    triage = score_case(context)
    case = context.case_row
    summary = build_case_summary(context, triage)

    return {
        "hadm_id": int(case.get("hadm_id")),
        "case_id": case.get("case_id"),
        "subject_id": case.get("subject_id"),
        "age": case.get("age"),
        "gender": case.get("gender"),
        "admission_diagnosis": case.get("admission_diagnosis"),
        "predicted_urgency": triage.urgency,
        "triage_score": triage.score,
        "major_risks": " | ".join(triage.evidence[:5]),
        "top_diagnoses": " | ".join(top_diagnosis_titles(context.diagnoses, limit=6)),
        "key_medications": " | ".join(top_medications(context.prescriptions, limit=6)),
        "abnormal_labs_preview": " | ".join(lab_signal_strings(context.abnormal_labs, limit=6)),
        "handoff_summary": _safe_text(summary, limit=1000),
        "note_snippet": _safe_text(case.get("discharge_summary", ""), limit=600),
        "reviewer_urgency_label": "",
        "reviewer_summary_quality": "",
        "reviewer_retrieval_usefulness": "",
        "reviewer_major_issue": "",
        "reviewer_notes": "",
    }


def _build_sampling_frame(sample_size: int, seed: int) -> pd.DataFrame:
    bundle = load_dataset()
    hadm_series = bundle.clinical_cases["hadm_id"].dropna().astype(int)
    candidate_pool_size = min(len(hadm_series.index), max(sample_size * 8, 64))
    candidate_ids = hadm_series.sample(n=candidate_pool_size, random_state=seed).tolist()
    rows: list[dict[str, object]] = []
    for hadm_id in candidate_ids:
        context = build_case_context(bundle, hadm_id)
        triage = score_case(context)
        case = context.case_row
        rows.append(
            {
                "hadm_id": hadm_id,
                "predicted_urgency": triage.urgency,
                "triage_score": triage.score,
                "admission_diagnosis": case.get("admission_diagnosis"),
            }
        )
    return pd.DataFrame(rows)


def _stratified_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if frame.empty or sample_size >= len(frame.index):
        return frame

    groups = []
    urgency_order = ["Critical", "High", "Moderate", "Low"]
    per_group = max(sample_size // max(len(urgency_order), 1), 1)
    remaining = sample_size

    for urgency in urgency_order:
        group = frame.loc[frame["predicted_urgency"] == urgency]
        if group.empty:
            continue
        take = min(len(group.index), per_group, remaining)
        if take <= 0:
            continue
        groups.append(group.sample(n=take, random_state=seed))
        remaining -= take

    if remaining > 0:
        used_hadm_ids = set(pd.concat(groups)["hadm_id"].tolist()) if groups else set()
        leftovers = frame.loc[~frame["hadm_id"].isin(used_hadm_ids)]
        if not leftovers.empty:
            groups.append(leftovers.sample(n=min(remaining, len(leftovers.index)), random_state=seed))

    sampled = pd.concat(groups).drop_duplicates(subset=["hadm_id"]).sort_values(["predicted_urgency", "triage_score"], ascending=[True, False])
    return sampled.head(sample_size).reset_index(drop=True)


def export_annotation_set(sample_size: int = 24, seed: int = 42, output_path: str | None = None) -> Path:
    sampling_frame = _build_sampling_frame(sample_size=sample_size, seed=seed)
    sampled_ids = _stratified_sample(sampling_frame, sample_size=sample_size, seed=seed)["hadm_id"].tolist()
    sampled = pd.DataFrame([_build_annotation_row(int(hadm_id)) for hadm_id in sampled_ids])

    output = Path(output_path) if output_path else Path("data/annotations/triage_review_template.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a manual review template for triage validation.")
    parser.add_argument("--sample-size", type=int, default=24, help="Number of cases to include in the review set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/annotations/triage_review_template.csv",
        help="CSV path for the exported review template.",
    )
    args = parser.parse_args()

    output_path = export_annotation_set(sample_size=args.sample_size, seed=args.seed, output_path=args.output)
    print(f"Annotation template written to {output_path}")


if __name__ == "__main__":
    main()
