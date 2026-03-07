from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

VALID_URGENCY = {"Critical", "High", "Moderate", "Low"}
VALID_SUMMARY_QUALITY = {"Good", "Partial", "Poor"}
VALID_RETRIEVAL_USEFULNESS = {"Useful", "Mixed", "Not useful"}


def _clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _normalized_counts(series: pd.Series) -> dict[str, int]:
    cleaned = _clean_series(series)
    cleaned = cleaned[cleaned != ""]
    return dict(cleaned.value_counts().sort_index())


def _agreement_rate(frame: pd.DataFrame) -> float:
    reviewed = frame.loc[frame["reviewer_urgency_label"] != ""].copy()
    if reviewed.empty:
        return 0.0
    matches = reviewed["predicted_urgency"] == reviewed["reviewer_urgency_label"]
    return round(float(matches.mean()), 3)


def _support_rate(series: pd.Series, accepted: set[str]) -> float:
    cleaned = _clean_series(series)
    reviewed = cleaned[cleaned != ""]
    if reviewed.empty:
        return 0.0
    supported = reviewed.isin(accepted)
    return round(float(supported.mean()), 3)


def _confusion_rows(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    reviewed = frame.loc[
        (frame["reviewer_urgency_label"] != "")
        & (frame["predicted_urgency"] != frame["reviewer_urgency_label"])
    ].copy()
    if reviewed.empty:
        return []
    rows = []
    for _, row in reviewed.head(limit).iterrows():
        rows.append(
            {
                "hadm_id": int(row["hadm_id"]),
                "predicted_urgency": row["predicted_urgency"],
                "reviewer_urgency_label": row["reviewer_urgency_label"],
                "reviewer_major_issue": row.get("reviewer_major_issue", ""),
            }
        )
    return rows


def summarize_review_file(csv_path: str | Path) -> dict[str, object]:
    frame = pd.read_csv(csv_path)

    for column in [
        "predicted_urgency",
        "reviewer_urgency_label",
        "reviewer_summary_quality",
        "reviewer_retrieval_usefulness",
        "reviewer_major_issue",
        "reviewer_notes",
    ]:
        if column not in frame.columns:
            raise KeyError(f"Missing required review column: {column}")
        frame[column] = _clean_series(frame[column])

    reviewed_urgency = frame.loc[frame["reviewer_urgency_label"] != "", "reviewer_urgency_label"]
    invalid_urgency = sorted(set(reviewed_urgency) - VALID_URGENCY)
    reviewed_summary = frame.loc[frame["reviewer_summary_quality"] != "", "reviewer_summary_quality"]
    invalid_summary = sorted(set(reviewed_summary) - VALID_SUMMARY_QUALITY)
    reviewed_retrieval = frame.loc[
        frame["reviewer_retrieval_usefulness"] != "", "reviewer_retrieval_usefulness"
    ]
    invalid_retrieval = sorted(set(reviewed_retrieval) - VALID_RETRIEVAL_USEFULNESS)

    issues = _clean_series(frame["reviewer_major_issue"])
    issue_counter = Counter(issue for issue in issues if issue)

    return {
        "total_cases": int(len(frame.index)),
        "urgency_labels_completed": int((frame["reviewer_urgency_label"] != "").sum()),
        "summary_reviews_completed": int((frame["reviewer_summary_quality"] != "").sum()),
        "retrieval_reviews_completed": int((frame["reviewer_retrieval_usefulness"] != "").sum()),
        "urgency_agreement_rate": _agreement_rate(frame),
        "summary_support_rate": _support_rate(frame["reviewer_summary_quality"], {"Good", "Partial"}),
        "retrieval_support_rate": _support_rate(frame["reviewer_retrieval_usefulness"], {"Useful", "Mixed"}),
        "predicted_urgency_distribution": _normalized_counts(frame["predicted_urgency"]),
        "reviewer_urgency_distribution": _normalized_counts(frame["reviewer_urgency_label"]),
        "summary_quality_distribution": _normalized_counts(frame["reviewer_summary_quality"]),
        "retrieval_usefulness_distribution": _normalized_counts(frame["reviewer_retrieval_usefulness"]),
        "top_major_issues": issue_counter.most_common(10),
        "urgency_disagreements": _confusion_rows(frame),
        "invalid_values": {
            "reviewer_urgency_label": invalid_urgency,
            "reviewer_summary_quality": invalid_summary,
            "reviewer_retrieval_usefulness": invalid_retrieval,
        },
    }


def _print_distribution(title: str, values: dict[str, int]) -> None:
    print(title)
    if not values:
        print("  No completed labels")
        return
    for key, count in values.items():
        print(f"  {key}: {count}")


def print_review_report(report: dict[str, object]) -> None:
    print("Manual Review Report")
    print(f"Total cases in file: {report['total_cases']}")
    print(f"Urgency labels completed: {report['urgency_labels_completed']}")
    print(f"Summary reviews completed: {report['summary_reviews_completed']}")
    print(f"Retrieval reviews completed: {report['retrieval_reviews_completed']}")
    print(f"Urgency agreement rate: {report['urgency_agreement_rate']}")
    print(f"Summary support rate (Good or Partial): {report['summary_support_rate']}")
    print(f"Retrieval support rate (Useful or Mixed): {report['retrieval_support_rate']}")
    _print_distribution("Predicted urgency distribution:", report["predicted_urgency_distribution"])
    _print_distribution("Reviewer urgency distribution:", report["reviewer_urgency_distribution"])
    _print_distribution("Summary quality distribution:", report["summary_quality_distribution"])
    _print_distribution("Retrieval usefulness distribution:", report["retrieval_usefulness_distribution"])

    print("Top major issues:")
    if not report["top_major_issues"]:
        print("  No issues recorded")
    else:
        for issue, count in report["top_major_issues"]:
            print(f"  {count:>3}  {issue}")

    print("Urgency disagreements:")
    if not report["urgency_disagreements"]:
        print("  No urgency disagreements recorded")
    else:
        for row in report["urgency_disagreements"]:
            print(
                "  "
                f"hadm_id={row['hadm_id']} predicted={row['predicted_urgency']} "
                f"reviewer={row['reviewer_urgency_label']} issue={row['reviewer_major_issue']}"
            )

    invalid_values = report["invalid_values"]
    print("Invalid reviewer values:")
    for key, values in invalid_values.items():
        if values:
            print(f"  {key}: {', '.join(values)}")
        else:
            print(f"  {key}: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed manual review CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/annotations/triage_review_template.csv",
        help="Path to the completed annotation CSV.",
    )
    args = parser.parse_args()

    report = summarize_review_file(args.input)
    print_review_report(report)


if __name__ == "__main__":
    main()
