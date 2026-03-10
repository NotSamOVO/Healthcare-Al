"""Clinical note summarization and structured entity extraction."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Section-based extractive summarizer
# ---------------------------------------------------------------------------

# Common MIMIC-style section headers (case-insensitive matching)
_SECTION_RE = re.compile(
    r"\n\s*("
    r"CHIEF COMPLAINT|MAJOR SURGICAL OR INVASIVE PROCEDURE|"
    r"HISTORY OF PRESENT ILLNESS|PAST MEDICAL HISTORY|"
    r"SOCIAL HISTORY|FAMILY HISTORY|PHYSICAL EXAM|"
    r"PERTINENT RESULTS|BRIEF HOSPITAL COURSE|"
    r"HOSPITAL COURSE|MEDICATIONS ON ADMISSION|"
    r"DISCHARGE MEDICATIONS|DISCHARGE DIAGNOSIS|"
    r"DISCHARGE CONDITION|DISCHARGE INSTRUCTIONS|"
    r"FOLLOWUP INSTRUCTIONS|FOLLOW-UP|ALLERGIES|SERVICE"
    r")\s*:?\s*",
    re.IGNORECASE,
)

# Priority sections for the condensed summary (order matters)
_PRIORITY_SECTIONS = [
    "CHIEF COMPLAINT",
    "BRIEF HOSPITAL COURSE",
    "HOSPITAL COURSE",
    "DISCHARGE DIAGNOSIS",
    "DISCHARGE CONDITION",
    "MAJOR SURGICAL OR INVASIVE PROCEDURE",
    "HISTORY OF PRESENT ILLNESS",
    "DISCHARGE MEDICATIONS",
    "PERTINENT RESULTS",
    "ALLERGIES",
]


@dataclass
class NoteSummary:
    sections: dict[str, str]
    condensed: str
    section_count: int


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip de-id brackets."""
    text = re.sub(r"\[?\*\*[^*]*\*\*\]?", "___", text)  # de-id placeholders
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_sentences(text: str, max_sentences: int = 3, max_chars: int = 350) -> str:
    """Return the first few sentences up to a character limit."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = ""
    for sentence in sentences[:max_sentences]:
        proposal = f"{result} {sentence}".strip() if result else sentence
        if len(proposal) > max_chars:
            break
        result = proposal
    return result or text[:max_chars]


def parse_note_sections(note: str) -> dict[str, str]:
    """Split a discharge note into named sections."""
    if not note or note.strip() in ("", "nan"):
        return {}

    splits = _SECTION_RE.split(note)
    sections: dict[str, str] = {}

    # Content before first header → use as preamble
    if splits and splits[0].strip():
        sections["PREAMBLE"] = _clean_text(splits[0])

    # Pairs: header, content, header, content, ...
    idx = 1
    while idx < len(splits) - 1:
        header = splits[idx].strip().upper()
        body = _clean_text(splits[idx + 1])
        if body:
            sections[header] = body
        idx += 2

    return sections


def summarize_note(note: str) -> NoteSummary:
    """Produce an extractive condensed summary from a discharge note."""
    sections = parse_note_sections(note)
    if not sections:
        return NoteSummary(sections={}, condensed="No discharge summary available.", section_count=0)

    condensed_parts: list[str] = []
    for header in _PRIORITY_SECTIONS:
        body = sections.get(header)
        if not body:
            continue
        snippet = _first_sentences(body)
        condensed_parts.append(f"**{header.title()}:** {snippet}")
        if len(condensed_parts) >= 5:
            break

    if not condensed_parts:
        # Fall back to first available section
        for header, body in sections.items():
            condensed_parts.append(f"**{header.title()}:** {_first_sentences(body)}")
            if len(condensed_parts) >= 3:
                break

    return NoteSummary(
        sections=sections,
        condensed="\n\n".join(condensed_parts),
        section_count=len(sections),
    )


# ---------------------------------------------------------------------------
# Clinical entity extraction  (rule-based)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntities:
    conditions: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    vitals: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        for cat, items in [
            ("Condition", self.conditions),
            ("Procedure", self.procedures),
            ("Medication", self.medications),
            ("Vital / Measurement", self.vitals),
        ]:
            for item in items:
                rows.append({"category": cat, "entity": item})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["category", "entity"])


# ----- Condition patterns -----
_CONDITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"pneumonia|sepsis|septic shock|heart failure|"
        r"atrial fibrillation|hypertension|hypotension|"
        r"diabetes|renal failure|acute kidney injury|"
        r"respiratory failure|pulmonary embolism|"
        r"deep vein thrombosis|stroke|myocardial infarction|"
        r"congestive heart failure|chronic obstructive pulmonary disease|"
        r"COPD|coronary artery disease|CAD|anemia|"
        r"urinary tract infection|UTI|gastrointestinal bleed|"
        r"GI bleed|cirrhosis|hepatitis|pancreatitis|"
        r"altered mental status|encephalopathy|seizure|"
        r"fracture|cellulitis|bacteremia|endocarditis|"
        r"pleural effusion|pulmonary edema|aortic stenosis|"
        r"mitral regurgitation|cardiomyopathy|hypothyroidism|"
        r"hyperthyroidism|hyperkalemia|hyponatremia|"
        r"hyperglycemia|hypoglycemia|coagulopathy|"
        r"thrombocytopenia|leukocytosis|acidosis|alkalosis"
        r")\b",
        re.IGNORECASE,
    ),
]

# ----- Procedure patterns -----
_PROCEDURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"intubat(?:ed|ion)|extubat(?:ed|ion)|"
        r"tracheostomy|bronchoscopy|endoscopy|colonoscopy|"
        r"catheter(?:ization| placement)|angiogram|angioplasty|"
        r"bypass graft|CABG|stent(?:ing| placement)|"
        r"dialysis|hemodialysis|transfusion|"
        r"chest tube|thoracentesis|paracentesis|"
        r"lumbar puncture|central line|arterial line|"
        r"mechanical ventilation|ventilator|"
        r"CT scan|MRI|echocardiogram|EKG|ECG|"
        r"ultrasound|X-ray|biopsy|surgical repair|"
        r"amputation|debridement|lavage|drainage|"
        r"pacemaker|defibrillator|cardioversion"
        r")\b",
        re.IGNORECASE,
    ),
]

# ----- Medication patterns -----
_MEDICATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b("
        r"heparin|warfarin|coumadin|enoxaparin|lovenox|"
        r"aspirin|clopidogrel|plavix|metoprolol|atenolol|"
        r"lisinopril|enalapril|losartan|amlodipine|"
        r"furosemide|lasix|hydrochlorothiazide|spironolactone|"
        r"insulin|metformin|glipizide|"
        r"vancomycin|piperacillin|tazobactam|zosyn|"
        r"ceftriaxone|ciprofloxacin|levofloxacin|"
        r"meropenem|azithromycin|amoxicillin|"
        r"morphine|fentanyl|hydromorphone|dilaudid|"
        r"acetaminophen|tylenol|ibuprofen|"
        r"pantoprazole|omeprazole|famotidine|"
        r"prednisone|dexamethasone|methylprednisolone|"
        r"albuterol|ipratropium|levothyroxine|"
        r"amiodarone|digoxin|nitroglycerin|"
        r"potassium chloride|calcium gluconate|"
        r"docusate|senna|ondansetron|zofran|"
        r"lorazepam|ativan|midazolam|propofol|"
        r"norepinephrine|vasopressin|dopamine|phenylephrine"
        r")\b",
        re.IGNORECASE,
    ),
]

# ----- Vital / measurement patterns -----
_VITAL_RE = re.compile(
    r"\b("
    r"(?:blood pressure|BP|systolic|diastolic)\s*(?:of\s*)?[\d]+(?:/[\d]+)?\s*(?:mmHg)?|"
    r"(?:heart rate|HR|pulse)\s*(?:of\s*)?[\d]+\s*(?:bpm)?|"
    r"(?:temperature|temp|T)\s*(?:of\s*)?\d+\.?\d*\s*(?:[°]?[CF])?|"
    r"(?:respiratory rate|RR)\s*(?:of\s*)?[\d]+|"
    r"(?:O2 sat|SpO2|oxygen saturation|saturation|sat)\s*(?:of\s*)?[\d]+\s*%?|"
    r"(?:BMI)\s*(?:of\s*)?[\d]+\.?\d*|"
    r"(?:weight|wt)\s*(?:of\s*)?[\d]+\.?\d*\s*(?:kg|lbs?)?|"
    r"(?:INR)\s*(?:of\s*)?[\d]+\.?\d*|"
    r"(?:ejection fraction|EF)\s*(?:of\s*)?[\d]+\s*%?"
    r")\b",
    re.IGNORECASE,
)


def _unique_ordered(items: list[str]) -> list[str]:
    """De-dup while keeping first occurrence order, case-insensitive."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def extract_entities(note: str) -> ExtractedEntities:
    """Extract clinical entities from a free-text discharge note."""
    if not note or note.strip() in ("", "nan"):
        return ExtractedEntities()

    cleaned = _clean_text(note)

    conditions: list[str] = []
    for pattern in _CONDITION_PATTERNS:
        conditions.extend(m.group(0) for m in pattern.finditer(cleaned))

    procedures: list[str] = []
    for pattern in _PROCEDURE_PATTERNS:
        procedures.extend(m.group(0) for m in pattern.finditer(cleaned))

    medications: list[str] = []
    for pattern in _MEDICATION_PATTERNS:
        medications.extend(m.group(0) for m in pattern.finditer(cleaned))

    vitals = [m.group(0) for m in _VITAL_RE.finditer(cleaned)]

    return ExtractedEntities(
        conditions=_unique_ordered(conditions),
        procedures=_unique_ordered(procedures),
        medications=_unique_ordered(medications),
        vitals=_unique_ordered(vitals),
    )


# ---------------------------------------------------------------------------
# Optional LLM-powered summarization
# ---------------------------------------------------------------------------

def _llm_summarize(note: str) -> str | None:
    """Use OpenAI to produce an abstractive summary. Returns None on failure."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    prompt = (
        "You are a clinical documentation assistant. "
        "Summarize the following discharge note into a concise clinician-facing summary. "
        "Use this structure:\n"
        "• Chief complaint (1 line)\n"
        "• Hospital course (2-4 sentences)\n"
        "• Key procedures performed\n"
        "• Discharge diagnosis\n"
        "• Critical follow-up items\n\n"
        "Be factual. Do not invent information not present in the note.\n\n"
        f"Note:\n{note[:6000]}"
    )

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.15,
        )
        return response.output_text
    except Exception:
        return None


def summarize_note_llm(note: str) -> str | None:
    """Public wrapper for LLM summarization. Returns None when unavailable."""
    if not note or note.strip() in ("", "nan"):
        return None
    return _llm_summarize(note)
