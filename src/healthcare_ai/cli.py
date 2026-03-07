from __future__ import annotations

import argparse

from healthcare_ai.data import available_hadm_ids, load_dataset
from healthcare_ai.features import build_case_context
from healthcare_ai.retrieval import build_retrieval_index, find_similar_cases
from healthcare_ai.summarizer import build_case_summary
from healthcare_ai.triage import score_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a hospital admission from the hackathon dataset.")
    parser.add_argument("--hadm-id", type=int, help="Hospital admission identifier to inspect.")
    parser.add_argument(
        "--show-first",
        action="store_true",
        help="Print the first available hadm_id if you do not already know one.",
    )
    parser.add_argument(
        "--show-similar",
        action="store_true",
        help="Also print the top similar admissions for the selected case.",
    )
    args = parser.parse_args()

    bundle = load_dataset()
    if args.show_first or args.hadm_id is None:
        hadm_id = available_hadm_ids(bundle)[0]
    else:
        hadm_id = args.hadm_id

    context = build_case_context(bundle, hadm_id)
    triage = score_case(context)
    print(build_case_summary(context, triage))

    if args.show_similar:
        index = build_retrieval_index(bundle)
        similar_cases = find_similar_cases(index, hadm_id)
        if similar_cases.empty:
            print("\nNo similar admissions found.")
        else:
            print("\nSimilar admissions:")
            print(similar_cases.to_string(index=False))


if __name__ == "__main__":
    main()
