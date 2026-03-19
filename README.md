# Clinical Workflow Triage Assistant

This repository contains a hackathon-ready baseline for the 2026 Healthcare AI challenge. It focuses on Track 1: turning messy clinical documentation into structured signals that can fit a realistic clinician workflow.

The project uses the `bavehackathon/2026-healthcare-ai` dataset from Hugging Face and builds an interpretable admission review assistant that can:

- load and join the relational clinical dataset
- identify high-signal diagnoses, medications, and abnormal laboratory patterns
- assign an urgency band using transparent rules with case-level evidence
- generate a clinician handoff summary with risks, findings, and follow-up guidance
- retrieve similar historical admissions for case comparison
- explain retrieval matches using shared diagnoses, medications, labs, and admission terms
- answer case-specific questions with grounded local RAG over note, lab, medication, diagnosis, and similar-case evidence
- optionally upgrade grounded answers with an OpenAI-compatible LLM while preserving retrieved evidence display
- present results in a small Streamlit app for demos

## Project idea

Instead of relying on a black-box score, this baseline produces a structured triage brief with traceable evidence. That makes it more suitable for a hackathon focused on reliability and clinical workflow integration.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -e .
```

3. Launch the demo app:

```powershell
streamlit run src/healthcare_ai/app.py
```

4. Or inspect a case from the command line:

```powershell
python -m healthcare_ai.cli --hadm-id 123456
```

5. Run a baseline evaluation report:

```powershell
python -m healthcare_ai.evaluation --sample-size 200 --top-n 3
```

6. Export a manual review template for validation:

```powershell
python -m healthcare_ai.annotation --sample-size 24
```

7. Score a completed manual review file:

```powershell
python -m healthcare_ai.review_report --input data/annotations/triage_review_template.csv
```

## Optional LLM-backed RAG

The app now supports two grounded answer modes in the Grounded Q&A tab:

- `Local grounded`: deterministic answer synthesis from retrieved evidence only
- `LLM grounded`: OpenAI-compatible answer generation constrained to retrieved evidence

Both modes present a clinician-facing response with explicit evidence-title citations so the answer can be defended during demos.

For local development, you can store credentials in a repo-local `.env` file that is gitignored. Start by copying `.env.example` to `.env` and filling in your values.

To enable `LLM grounded`, set these environment variables before launching the app:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

Or use a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Optional for OpenAI-compatible providers:

```powershell
$env:OPENAI_BASE_URL="https://your-provider.example/v1"
```

If no API key is configured, the app falls back automatically to the local grounded mode.

## What the app shows

- case demographics and admission diagnosis
- urgency band with an interpretable score
- top evidence supporting the recommendation
- clinician handoff summary generated from the note and structured tables
- similar admissions based on diagnoses, labs, medications, and note text
- grounded Q&A over retrieved case evidence
- baseline metrics for urgency distribution, evidence triggers, and retrieval agreement
- manual review template for rapid hand-labeling before final submission
- diagnoses, prescriptions, and abnormal labs for the selected admission

## Demo flow

For presentations, walk through the app in this order:

1. Select an admission and show the urgency band.
2. Use the clinician handoff panel to explain why the case needs review.
3. Show the evidence trail and abnormal labs table as the reliability layer.
4. Open similar admissions to demonstrate case comparison and retrieval transparency.

## Reliability choices

- deterministic scoring instead of opaque predictions
- explicit evidence trail for every urgency recommendation
- conservative language: the tool prioritizes review support, not autonomous decisions
- structured joins across diagnoses, medications, labs, and notes
- retrieval is transparent and based on nearest-neighbor text similarity, not hidden embeddings

## Repository layout

```text
src/healthcare_ai/
  app.py
  cli.py
  data.py
  features.py
  summarizer.py
  triage.py
```


