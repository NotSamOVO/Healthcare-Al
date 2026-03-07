from __future__ import annotations

import re

from healthcare_ai.features import CaseContext, lab_signal_strings, top_diagnosis_titles, top_medications
from healthcare_ai.triage import TriageResult

SECTION_PATTERN = re.compile(r"\n([A-Z][A-Z /]{3,40}):")


def _extract_note_snippet(note: str, max_chars: int = 450) -> str:
    if not note or note.strip() == "nan":
        return "No discharge summary available."
    normalized = re.sub(r"\s+", " ", note).strip()
    return normalized[:max_chars].rstrip() + ("..." if len(normalized) > max_chars else "")


def _find_sections(note: str) -> list[str]:
    if not note:
        return []
    sections = []
    for match in SECTION_PATTERN.finditer(note):
        heading = match.group(1).strip()
        if len(heading.split()) < 2:
            continue
        sections.append(heading.title())
        if len(sections) == 6:
            break
    return sections


def _format_list(items: list[str], fallback: str, limit: int = 5) -> str:
    if not items:
        return fallback
    return ", ".join(items[:limit])


def _recommended_follow_up(triage: TriageResult) -> str:
    if triage.urgency == "Critical":
        return "Immediate clinician review with attention to acute instability signals, treatment intensity, and abnormal lab burden."
    if triage.urgency == "High":
        return "Prioritized same-shift review to confirm major risks, reconcile medications, and verify pending data."
    if triage.urgency == "Moderate":
        return "Routine clinician review with focus on diagnostic clarification and follow-up of notable abnormalities."
    return "Standard review pathway; monitor for missing context or evolving risk signals."


def build_case_summary(context: CaseContext, triage: TriageResult) -> str:
    case = context.case_row
    diagnoses = top_diagnosis_titles(context.diagnoses)
    medications = top_medications(context.prescriptions)
    labs = lab_signal_strings(context.abnormal_labs)
    sections = _find_sections(str(case.get("discharge_summary", "")))
    note_snippet = _extract_note_snippet(str(case.get("discharge_summary", "")))
    patient_context = (
        f"{case.get('age', 'Unknown')} year old {case.get('gender', 'unknown gender')} "
        f"admitted for {case.get('admission_diagnosis', 'unknown reason')}."
    )

    summary_lines = [
        "CLINICIAN HANDOFF",
        f"Reason for review: {triage.urgency} priority case (triage score {triage.score}).",
        f"Patient context: {patient_context}",
        "Major risks: " + _format_list(triage.evidence[:5], "No major risk signals detected."),
        "Key diagnoses: " + _format_list(diagnoses, "No diagnosis codes linked."),
        "Medication context: " + _format_list(medications, "No medications linked."),
        "Abnormal labs to review: " + _format_list(labs, "No clear abnormal lab pattern detected.", limit=6),
        "Recommended follow-up: " + _recommended_follow_up(triage),
        "Detected note sections: " + _format_list(sections, "No obvious section headers found.", limit=6),
        "Source note snippet: " + note_snippet,
    ]
    return "\n".join(summary_lines)
