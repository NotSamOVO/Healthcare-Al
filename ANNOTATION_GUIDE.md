# Annotation Guide

Use the generated CSV review set to create a lightweight hand-labeled validation artifact for the hackathon.

## How to generate the template

```powershell
python -m healthcare_ai.annotation --sample-size 24
```

The default output path is `data/annotations/triage_review_template.csv`.

## Recommended reviewers

- one teammate focuses on urgency labeling
- one teammate checks whether the handoff summary is clinically useful
- one teammate checks whether retrieval results are relevant and explainable

## Columns to complete

- `reviewer_urgency_label`: `Critical`, `High`, `Moderate`, or `Low`
- `reviewer_summary_quality`: `Good`, `Partial`, or `Poor`
- `reviewer_retrieval_usefulness`: `Useful`, `Mixed`, or `Not useful`
- `reviewer_major_issue`: short label such as `over-triage`, `missing risk`, `unclear summary`, `irrelevant retrieval`
- `reviewer_notes`: free-text notes for disagreements or error patterns

## Suggested review rubric

- Urgency: Does the model's urgency band match what a cautious clinical reviewer would prioritize?
- Summary quality: Does the handoff surface the right risks, findings, and next-step guidance?
- Retrieval usefulness: Do the similar cases look meaningfully comparable and are the match reasons plausible?

## What to report in the presentation

- number of reviewed cases
- agreement between predicted and reviewer urgency labels
- percentage of summaries marked `Good` or `Partial`
- percentage of retrieval results marked `Useful` or `Mixed`
- two or three representative failure modes and how you would fix them

## How to score the completed review file

```powershell
python -m healthcare_ai.review_report --input data/annotations/triage_review_template.csv
```

This prints:

- urgency agreement rate
- summary support rate (`Good` or `Partial`)
- retrieval support rate (`Useful` or `Mixed`)
- reviewer label distributions
- top recorded failure modes
- example urgency disagreements
