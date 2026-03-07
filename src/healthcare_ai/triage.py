from __future__ import annotations

from dataclasses import dataclass

from healthcare_ai.features import CaseContext, lab_signal_strings, top_diagnosis_titles, top_medications

ACUTE_INSTABILITY_TERMS = {
    "sepsis": 3,
    "septic": 3,
    "shock": 3,
    "respiratory failure": 3,
    "pulmonary embol": 3,
    "intracranial hemorrhage": 3,
    "ventilator": 2,
    "intub": 2,
}

SERIOUS_DIAGNOSTIC_TERMS = {
    "stroke": 2,
    "infarction": 2,
    "gi bleed": 2,
    "heart failure": 2,
    "renal failure": 2,
    "kidney injury": 2,
    "arrhythmia": 2,
    "hypoxia": 2,
    "infection": 2,
    "pneumonia": 2,
}

MODERATE_CONCERN_TERMS = {
    "cellulitis": 1,
    "syncope": 1,
    "chest pain": 1,
    "copd": 1,
}

ESCALATION_THERAPIES = {
    "norepinephrine": 2,
    "epinephrine": 2,
}

MEDICATION_RISK_MEDS = {
    "vancomycin": 1,
    "heparin": 1,
    "warfarin": 1,
    "insulin": 1,
    "furosemide": 1,
}


@dataclass
class TriageResult:
    score: int
    urgency: str
    evidence: list[str]


def _score_terms(text: str, term_weights: dict[str, int], evidence_prefix: str) -> tuple[int, list[str], int]:
    lowered = text.lower()
    score = 0
    evidence: list[str] = []
    match_count = 0
    for term, weight in term_weights.items():
        if term in lowered:
            score += weight
            evidence.append(f"{evidence_prefix}: {term}")
            match_count += 1
    return score, evidence, match_count


def score_case(context: CaseContext) -> TriageResult:
    diagnosis_text = " | ".join(top_diagnosis_titles(context.diagnoses, limit=10))
    medication_text = " | ".join(top_medications(context.prescriptions, limit=12))
    note_text = str(context.case_row.get("discharge_summary", ""))
    admission_text = str(context.case_row.get("admission_diagnosis", ""))
    combined_text = " | ".join([diagnosis_text, medication_text, note_text, admission_text])

    score = 0
    evidence: list[str] = []
    acute_match_count = 0

    term_score, term_evidence, acute_match_count = _score_terms(
        combined_text,
        ACUTE_INSTABILITY_TERMS,
        "Acute instability",
    )
    score += term_score
    evidence.extend(term_evidence)

    term_score, term_evidence, _ = _score_terms(
        combined_text,
        SERIOUS_DIAGNOSTIC_TERMS,
        "Serious diagnostic signal",
    )
    score += term_score
    evidence.extend(term_evidence)

    term_score, term_evidence, _ = _score_terms(
        combined_text,
        MODERATE_CONCERN_TERMS,
        "Moderate concern",
    )
    score += term_score
    evidence.extend(term_evidence)

    abnormal_count = len(context.abnormal_labs)
    if abnormal_count >= 6:
        score += 2
        evidence.append(f"Abnormal lab burden: {abnormal_count} high-signal findings")
    elif abnormal_count >= 2:
        score += 1
        evidence.append(f"Abnormal lab burden: {abnormal_count} high-signal findings")

    treatment_intensity_score = 0
    for medication, weight in ESCALATION_THERAPIES.items():
        if medication in medication_text.lower():
            treatment_intensity_score += weight
            evidence.append(f"Treatment intensity: {medication}")
    score += min(treatment_intensity_score, 2)

    medication_risk_score = 0
    for medication, weight in MEDICATION_RISK_MEDS.items():
        if medication in medication_text.lower():
            medication_risk_score += weight
            evidence.append(f"Medication risk: {medication}")
    score += min(medication_risk_score, 1)

    diagnosis_count = len(context.diagnoses.index)
    if diagnosis_count >= 8:
        score += 1
        evidence.append(f"Case complexity: {diagnosis_count} diagnosis codes")
    elif diagnosis_count >= 4:
        score += 1
        evidence.append(f"Case complexity: {diagnosis_count} diagnosis codes")

    if (acute_match_count >= 2 and score >= 7) or (acute_match_count >= 1 and treatment_intensity_score >= 2 and score >= 7):
        urgency = "Critical"
    elif score >= 5:
        urgency = "High"
    elif score >= 3:
        urgency = "Moderate"
    else:
        urgency = "Low"

    if not evidence:
        evidence.append("No strong triage signals found; default to routine review")

    abnormal_lab_evidence = lab_signal_strings(context.abnormal_labs, limit=3)
    evidence.extend(f"Lab signal: {signal}" for signal in abnormal_lab_evidence)

    deduped_evidence = list(dict.fromkeys(evidence))
    return TriageResult(score=score, urgency=urgency, evidence=deduped_evidence)
