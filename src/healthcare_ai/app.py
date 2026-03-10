from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from healthcare_ai.data import available_hadm_ids, load_dataset
from healthcare_ai.features import CaseContext, build_case_context
from healthcare_ai.rag import answer_grounded_question, build_case_rag_index, llm_rag_available
from healthcare_ai.retrieval import build_retrieval_index, find_similar_cases
from healthcare_ai.summarizer import build_case_summary
from healthcare_ai.triage import score_case

load_dotenv()

st.set_page_config(page_title="Clinical Workflow Triage Assistant", layout="wide")


def _check_data_completeness(context: CaseContext) -> list[dict[str, str]]:
    """Return a list of {'priority': ..., 'category': ..., 'detail': ...} dicts."""
    issues: list[dict[str, str]] = []

    # --- Missing data checks ---
    note = str(context.case_row.get("discharge_summary", "")).strip()
    if not note or note == "nan":
        issues.append({
            "priority": "High",
            "category": "Missing data",
            "detail": "No discharge summary — triage relies on structured data only.",
        })

    admission_dx = str(context.case_row.get("admission_diagnosis", "")).strip()
    if not admission_dx or admission_dx == "nan":
        issues.append({
            "priority": "High",
            "category": "Missing data",
            "detail": "Admission diagnosis field is empty.",
        })

    if context.diagnoses.empty:
        issues.append({
            "priority": "High",
            "category": "Missing data",
            "detail": "No diagnosis codes linked to this admission.",
        })

    if context.labs.empty:
        issues.append({
            "priority": "Moderate",
            "category": "Missing data",
            "detail": "No lab results linked to this admission.",
        })

    if context.prescriptions.empty:
        issues.append({
            "priority": "Low",
            "category": "Missing data",
            "detail": "No prescriptions linked to this admission.",
        })

    # --- Inconsistency checks ---
    # Prescription date order
    if not context.prescriptions.empty:
        rx = context.prescriptions.dropna(subset=["startdate", "enddate"])
        bad_rx = rx.loc[rx["enddate"] < rx["startdate"]]
        if not bad_rx.empty:
            drugs = ", ".join(bad_rx["drug"].head(3).tolist())
            issues.append({
                "priority": "High",
                "category": "Inconsistency",
                "detail": f"Prescription end date before start date for: {drugs}.",
            })

    # Labs with missing numeric values
    if not context.labs.empty:
        total_labs = len(context.labs)
        unparsed = int(context.labs["numeric_value"].isna().sum())
        if unparsed > 0:
            pct = round(100 * unparsed / total_labs)
            severity = "Moderate" if pct < 50 else "High"
            issues.append({
                "priority": severity,
                "category": "Inconsistency",
                "detail": f"{unparsed}/{total_labs} lab values ({pct}%) could not be parsed to numeric.",
            })

    # Labs with missing timestamps
    if not context.labs.empty:
        missing_time = int(context.labs["charttime"].isna().sum())
        if missing_time > 0:
            issues.append({
                "priority": "Moderate",
                "category": "Inconsistency",
                "detail": f"{missing_time} lab result(s) have no chart timestamp.",
            })

    # Diagnoses missing descriptions
    if not context.diagnoses.empty:
        missing_titles = int(
            context.diagnoses["long_title"].fillna(context.diagnoses["short_title"]).isna().sum()
        )
        if missing_titles > 0:
            issues.append({
                "priority": "Low",
                "category": "Inconsistency",
                "detail": f"{missing_titles} diagnosis code(s) have no description — unmapped ICD9.",
            })

    # Prescriptions missing dose info
    if not context.prescriptions.empty:
        missing_dose = int(context.prescriptions["dose_value"].isna().sum())
        if missing_dose > 0:
            issues.append({
                "priority": "Low",
                "category": "Inconsistency",
                "detail": f"{missing_dose} prescription(s) have no dose value recorded.",
            })

    # Age sanity check
    age = context.case_row.get("age")
    if pd.notna(age):
        age_val = int(age)
        if age_val < 0 or age_val > 120:
            issues.append({
                "priority": "High",
                "category": "Inconsistency",
                "detail": f"Patient age ({age_val}) is outside plausible range (0–120).",
            })

    # Sort by priority
    priority_order = {"High": 0, "Moderate": 1, "Low": 2}
    issues.sort(key=lambda x: priority_order.get(x["priority"], 9))
    return issues


def _render_data_quality(issues: list[dict[str, str]]) -> None:
    if not issues:
        return
    high = sum(1 for i in issues if i["priority"] == "High")
    moderate = sum(1 for i in issues if i["priority"] == "Moderate")
    low = sum(1 for i in issues if i["priority"] == "Low")
    label_parts = []
    if high:
        label_parts.append(f"{high} high")
    if moderate:
        label_parts.append(f"{moderate} moderate")
    if low:
        label_parts.append(f"{low} low")
    label = f"Data quality: {', '.join(label_parts)} issue(s) flagged"
    with st.expander(label, expanded=high > 0):
        df = pd.DataFrame(issues, columns=["priority", "category", "detail"])
        for _, row in df.iterrows():
            icon = {"High": "\u2757", "Moderate": "\u26a0\ufe0f", "Low": "\u2139\ufe0f"}.get(row["priority"], "")
            if row["priority"] == "High":
                st.error(f"{icon} **{row['priority']}** \u2014 {row['category']}: {row['detail']}")
            elif row["priority"] == "Moderate":
                st.warning(f"{icon} **{row['priority']}** \u2014 {row['category']}: {row['detail']}")
            else:
                st.info(f"{icon} **{row['priority']}** \u2014 {row['category']}: {row['detail']}")


def _parse_handoff_summary(summary: str) -> tuple[str, dict[str, str]]:
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    if not lines:
        return "", {}
    title = lines[0]
    sections: dict[str, str] = {}
    for line in lines[1:]:
        if ": " not in line:
            continue
        label, value = line.split(": ", 1)
        sections[label] = value
    return title, sections


def _render_priority_banner(urgency: str, recommendation: str) -> None:
    if urgency == "Critical":
        st.error(recommendation)
    elif urgency == "High":
        st.warning(recommendation)
    elif urgency == "Moderate":
        st.info(recommendation)
    else:
        st.success(recommendation)


def _render_handoff(summary: str) -> None:
    title, sections = _parse_handoff_summary(summary)
    if title:
        st.caption(title)

    _render_priority_banner(
        sections.get("Reason for review", "Low").split(" priority", 1)[0],
        sections.get("Recommended follow-up", "No follow-up recommendation available."),
    )

    st.markdown("#### Review Snapshot")
    snapshot = pd.DataFrame(
        [
            {"section": "Reason for review", "detail": sections.get("Reason for review", "")},
            {"section": "Patient context", "detail": sections.get("Patient context", "")},
            {"section": "Major risks", "detail": sections.get("Major risks", "")},
        ]
    )
    st.dataframe(snapshot, width="stretch", hide_index=True)

    st.markdown("#### Structured Handoff")
    left, right = st.columns(2)
    with left:
        st.markdown("**Clinical findings**")
        st.write(sections.get("Key diagnoses", ""))
        st.markdown("**Medication context**")
        st.write(sections.get("Medication context", ""))
        st.markdown("**Abnormal labs to review**")
        st.write(sections.get("Abnormal labs to review", ""))
    with right:
        st.markdown("**Recommended follow-up**")
        st.write(sections.get("Recommended follow-up", ""))
        st.markdown("**Detected note sections**")
        st.write(sections.get("Detected note sections", ""))
        st.markdown("**Source note snippet**")
        st.write(sections.get("Source note snippet", ""))


def _similar_case_cards(bundle, similar_cases: pd.DataFrame) -> None:
    if similar_cases.empty:
        st.info("No similar admissions found.")
        return

    for _, row in similar_cases.iterrows():
        similar_context = build_case_context(bundle, int(row["hadm_id"]))
        similar_triage = score_case(similar_context)
        with st.container(border=True):
            top_left, top_right = st.columns([1.3, 1])
            with top_left:
                st.markdown(
                    f"**Admission {int(row['hadm_id'])}**  \nDiagnosis: {row['admission_diagnosis']}"
                )
            with top_right:
                st.markdown(
                    f"Similarity: **{row['similarity']:.3f}**  \nUrgency: **{similar_triage.urgency}**"
                )
            st.caption(f"Age {row['age']} | Gender {row['gender']}")
            st.write(row["match_reasons"])


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_retrieval_index():
    return build_retrieval_index(get_bundle())


bundle = get_bundle()
retrieval_index = get_retrieval_index()
hadm_ids = available_hadm_ids(bundle)

st.title("Clinical Workflow Triage Assistant")
st.caption(
    "Interpretable admission triage over discharge summaries, diagnoses, prescriptions, and labs."
)

with st.sidebar:
    st.markdown("### RAG Mode")
    rag_mode = st.radio(
        "Grounded answer mode",
        options=["Local grounded", "LLM grounded"],
        index=0,
    )
    if rag_mode == "LLM grounded":
        if llm_rag_available():
            st.success("OPENAI_API_KEY detected. LLM grounded answers are enabled.")
        else:
            st.warning("OPENAI_API_KEY not set. The app will fall back to local grounded answers.")

selected_hadm_id = st.sidebar.selectbox("Admission", hadm_ids, index=0)
context = build_case_context(bundle, int(selected_hadm_id))
triage = score_case(context)
summary = build_case_summary(context, triage)
similar_cases = find_similar_cases(retrieval_index, int(selected_hadm_id), top_n=5)
rag_index = build_case_rag_index(context, similar_cases)
case = context.case_row

col1, col2, col3 = st.columns(3)
col1.metric("Urgency", triage.urgency)
col2.metric("Triage Score", triage.score)
col3.metric("Abnormal Lab Signals", len(context.abnormal_labs))

data_gaps = _check_data_completeness(context)
_render_data_quality(data_gaps)

overview_left, overview_right = st.columns([1.1, 1.2])

with overview_left:
    st.subheader("Case Overview")
    overview = pd.DataFrame(
        [
            {"field": "case_id", "value": str(case.get("case_id", ""))},
            {"field": "subject_id", "value": str(case.get("subject_id", ""))},
            {"field": "hadm_id", "value": str(case.get("hadm_id", ""))},
            {"field": "age", "value": str(case.get("age", ""))},
            {"field": "gender", "value": str(case.get("gender", ""))},
            {"field": "admission_diagnosis", "value": str(case.get("admission_diagnosis", ""))},
        ]
    )
    st.dataframe(overview, width="stretch", hide_index=True)

    st.subheader("Clinician Handoff")
    _render_handoff(summary)

with overview_right:
    st.subheader("Evidence Trail")
    st.caption("These are the interpretable signals that drove the current urgency band.")
    for item in triage.evidence:
        st.write(f"- {item}")

    st.subheader("Abnormal Labs")
    if context.abnormal_labs.empty:
        st.info("No strong abnormal lab deviations detected from dataset-level reference patterns.")
    else:
        lab_columns = [
            "charttime",
            "lab_name",
            "numeric_value",
            "unit",
            "direction",
            "iqr_distance",
            "category",
        ]
        st.dataframe(
            context.abnormal_labs[lab_columns].head(20),
            width="stretch",
            hide_index=True,
        )

st.subheader("Structured Data")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Diagnoses", "Medications", "All Labs", "Discharge Summary", "Similar Admissions", "Grounded Q&A"]
)

with tab1:
    st.dataframe(context.diagnoses, width="stretch", hide_index=True)

with tab2:
    st.dataframe(context.prescriptions, width="stretch", hide_index=True)

with tab3:
    st.dataframe(context.labs, width="stretch", hide_index=True)

with tab4:
    st.text(str(case.get("discharge_summary", "")))

with tab5:
    st.caption("Nearest-neighbor retrieval over diagnoses, medications, labs, admission diagnosis, and note text.")
    _similar_case_cards(bundle, similar_cases)

with tab6:
    st.caption("Ask a case-specific question. Answers are grounded only in retrieved note, lab, diagnosis, medication, and similar-case evidence.")
    default_question = "Why is this case urgent?"
    question = st.text_input("Question", value=default_question, key="rag_question")
    if question.strip():
        rag_result = answer_grounded_question(
            rag_index,
            question,
            use_llm=(rag_mode == "LLM grounded"),
        )
        st.markdown("#### Grounded Answer")
        st.text(rag_result["answer"])
        if rag_result.get("mode") == "llm-grounded":
            st.caption(f"Answer mode: LLM grounded via {rag_result.get('model', 'configured model')}")
        elif rag_result.get("mode") == "local-grounded":
            st.caption("Answer mode: local grounded retrieval summary")
        elif rag_result.get("mode") == "fallback-local":
            st.caption("Answer mode: local grounded fallback")
        st.markdown("#### Retrieved Evidence")
        if not rag_result["evidence"]:
            st.info("No evidence retrieved for this question.")
        else:
            for item in rag_result["evidence"]:
                with st.container(border=True):
                    st.markdown(
                        f"**{item['title']}**  \nSource: {item['source_type']} ({item['source_id']})  \nScore: {item['score']}"
                    )
                    st.write(item["content"])