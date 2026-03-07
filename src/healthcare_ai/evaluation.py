from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from healthcare_ai.data import available_hadm_ids, load_dataset
from healthcare_ai.features import build_case_context, top_diagnosis_titles
from healthcare_ai.retrieval import build_retrieval_index, find_similar_cases
from healthcare_ai.triage import score_case


def _sample_hadm_ids(sample_size: int, seed: int) -> list[int]:
    bundle = load_dataset()
    hadm_ids = available_hadm_ids(bundle)
    if sample_size >= len(hadm_ids):
        return hadm_ids
    series = pd.Series(hadm_ids)
    sampled = series.sample(n=sample_size, random_state=seed)
    return sampled.sort_values().tolist()


def _diagnosis_set(context) -> set[str]:
    return {title.lower() for title in top_diagnosis_titles(context.diagnoses, limit=10)}


def evaluate(sample_size: int = 200, seed: int = 42, top_n: int = 3) -> dict[str, object]:
    bundle = load_dataset()
    retrieval_index = build_retrieval_index(bundle)
    selected_hadm_ids = _sample_hadm_ids(sample_size=sample_size, seed=seed)

    urgency_counter: Counter[str] = Counter()
    evidence_counter: Counter[str] = Counter()
    scores: list[int] = []
    abnormal_lab_counts: list[int] = []
    diagnosis_overlap_hits = 0
    urgency_match_hits = 0
    retrieved_pairs = 0

    for hadm_id in selected_hadm_ids:
        context = build_case_context(bundle, hadm_id)
        triage = score_case(context)
        urgency_counter[triage.urgency] += 1
        evidence_counter.update(triage.evidence)
        scores.append(triage.score)
        abnormal_lab_counts.append(len(context.abnormal_labs))

        source_diagnoses = _diagnosis_set(context)
        similar_cases = find_similar_cases(retrieval_index, hadm_id, top_n=top_n)
        for _, row in similar_cases.iterrows():
            retrieved_pairs += 1
            similar_context = build_case_context(bundle, int(row["hadm_id"]))
            similar_triage = score_case(similar_context)
            similar_diagnoses = _diagnosis_set(similar_context)
            if source_diagnoses & similar_diagnoses:
                diagnosis_overlap_hits += 1
            if triage.urgency == similar_triage.urgency:
                urgency_match_hits += 1

    evaluated_cases = len(selected_hadm_ids)
    return {
        "sample_size": evaluated_cases,
        "seed": seed,
        "retrieval_neighbors_per_case": top_n,
        "mean_triage_score": round(sum(scores) / evaluated_cases, 2) if evaluated_cases else 0.0,
        "median_triage_score": float(pd.Series(scores).median()) if scores else 0.0,
        "mean_abnormal_lab_count": round(sum(abnormal_lab_counts) / evaluated_cases, 2) if evaluated_cases else 0.0,
        "urgency_distribution": dict(sorted(urgency_counter.items())),
        "top_evidence_signals": evidence_counter.most_common(10),
        "retrieval_same_diagnosis_rate": round(diagnosis_overlap_hits / retrieved_pairs, 3) if retrieved_pairs else 0.0,
        "retrieval_same_urgency_rate": round(urgency_match_hits / retrieved_pairs, 3) if retrieved_pairs else 0.0,
        "retrieved_pairs_evaluated": retrieved_pairs,
    }


def _print_report(report: dict[str, object]) -> None:
    print("Baseline Evaluation")
    print(f"Sample size: {report['sample_size']}")
    print(f"Random seed: {report['seed']}")
    print(f"Neighbors per case: {report['retrieval_neighbors_per_case']}")
    print(f"Mean triage score: {report['mean_triage_score']}")
    print(f"Median triage score: {report['median_triage_score']}")
    print(f"Mean abnormal lab count: {report['mean_abnormal_lab_count']}")
    print("Urgency distribution:")
    for urgency, count in report["urgency_distribution"].items():
        print(f"  {urgency}: {count}")
    print("Top evidence signals:")
    for evidence, count in report["top_evidence_signals"]:
        print(f"  {count:>3}  {evidence}")
    print(
        "Retrieval agreement: "
        f"same diagnosis rate={report['retrieval_same_diagnosis_rate']}, "
        f"same urgency rate={report['retrieval_same_urgency_rate']}"
    )
    print(f"Retrieved pairs evaluated: {report['retrieved_pairs_evaluated']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline triage and retrieval pipeline.")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of admissions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case sampling.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of retrieved neighbors per case.")
    args = parser.parse_args()

    report = evaluate(sample_size=args.sample_size, seed=args.seed, top_n=args.top_n)
    _print_report(report)


if __name__ == "__main__":
    main()
