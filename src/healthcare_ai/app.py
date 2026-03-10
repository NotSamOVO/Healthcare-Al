from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from healthcare_ai.data import available_hadm_ids, load_dataset
from healthcare_ai.features import CaseContext, build_case_context
from healthcare_ai.note_nlp import extract_entities, summarize_note, summarize_note_llm
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
            "detail": "No discharge summary found. The triage can only use diagnoses, labs, and prescriptions for this case.",
        })

    admission_dx = str(context.case_row.get("admission_diagnosis", "")).strip()
    if not admission_dx or admission_dx == "nan":
        issues.append({
            "priority": "High",
            "category": "Missing data",
            "detail": "No admission diagnosis recorded for this case.",
        })

    if context.diagnoses.empty:
        issues.append({
            "priority": "High",
            "category": "Missing data",
            "detail": "No diagnosis codes found. The system cannot assess clinical conditions for this case.",
        })

    if context.labs.empty:
        issues.append({
            "priority": "Moderate",
            "category": "Missing data",
            "detail": "No lab results found. Abnormal lab detection is unavailable for this case.",
        })

    if context.prescriptions.empty:
        issues.append({
            "priority": "Low",
            "category": "Missing data",
            "detail": "No prescriptions found. Medication risk scoring will be skipped for this case.",
        })

    # --- Inconsistency checks ---
    # Prescription date order
    if not context.prescriptions.empty:
        rx = context.prescriptions.dropna(subset=["startdate", "enddate"])
        bad_rx = rx.loc[rx["enddate"] < rx["startdate"]]
        if not bad_rx.empty:
            issues.append({
                "priority": "High",
                "category": "Inconsistency",
                "detail": f"{len(bad_rx)} medication(s) have end date before start date.",
                "items": bad_rx[["drug", "startdate", "enddate"]].reset_index(drop=True),
            })

    # Labs with missing numeric values
    if not context.labs.empty:
        total_labs = len(context.labs)
        text_only = context.labs[context.labs["numeric_value"].isna()]
        unparsed = len(text_only)
        if unparsed > 0:
            pct = round(100 * unparsed / total_labs)
            severity = "Moderate" if pct < 50 else "High"
            issues.append({
                "priority": severity,
                "category": "Inconsistency",
                "detail": f"{unparsed} of {total_labs} lab results ({pct}%) are text-only and cannot be checked for abnormality.",
                "items": text_only[["lab_name", "value", "unit", "charttime"]].reset_index(drop=True),
            })

    # Labs with missing timestamps
    if not context.labs.empty:
        no_time = context.labs[context.labs["charttime"].isna()]
        missing_time = len(no_time)
        if missing_time > 0:
            issues.append({
                "priority": "Moderate",
                "category": "Inconsistency",
                "detail": f"{missing_time} lab result(s) are missing a recorded date/time.",
                "items": no_time[["lab_name", "value", "unit"]].reset_index(drop=True),
            })

    # Diagnoses missing descriptions
    if not context.diagnoses.empty:
        no_title = context.diagnoses[
            context.diagnoses["long_title"].fillna(context.diagnoses["short_title"]).isna()
        ]
        missing_titles = len(no_title)
        if missing_titles > 0:
            issues.append({
                "priority": "Low",
                "category": "Inconsistency",
                "detail": f"{missing_titles} diagnosis code(s) have no readable name in the dictionary.",
                "items": no_title[["icd9_code", "seq_num"]].reset_index(drop=True),
            })

    # Prescriptions missing dose info
    if not context.prescriptions.empty:
        no_dose = context.prescriptions[context.prescriptions["dose_value"].isna()]
        missing_dose = len(no_dose)
        if missing_dose > 0:
            issues.append({
                "priority": "Low",
                "category": "Inconsistency",
                "detail": f"{missing_dose} prescription(s) are missing dosage information.",
                "items": no_dose[["drug", "route", "startdate", "enddate"]].reset_index(drop=True),
            })

    # Age sanity check
    age = context.case_row.get("age")
    if pd.notna(age):
        age_val = int(age)
        if age_val < 0 or age_val > 120:
            issues.append({
                "priority": "High",
                "category": "Inconsistency",
                "detail": f"Patient age is recorded as {age_val}, which is outside the expected range of 0–120.",
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
        summary_df = pd.DataFrame(
            [{"priority": i["priority"], "category": i["category"], "detail": i["detail"]} for i in issues]
        )
        summary_df.index = range(1, len(summary_df) + 1)
        summary_df.index.name = "#"
        st.dataframe(summary_df, width="stretch")
        st.divider()
        for idx, issue in enumerate(issues):
            icon = {"High": "\u2757", "Moderate": "\u26a0\ufe0f", "Low": "\u2139\ufe0f"}.get(issue["priority"], "")
            msg = f"{icon} **{issue['priority']}** — {issue['category']}: {issue['detail']}"
            if issue["priority"] == "High":
                st.error(msg)
            elif issue["priority"] == "Moderate":
                st.warning(msg)
            else:
                st.info(msg)
            if "items" in issue and isinstance(issue["items"], pd.DataFrame) and not issue["items"].empty:
                st.dataframe(issue["items"], width="stretch", hide_index=True)


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


@st.cache_data(show_spinner="Building case queue — triaging all admissions...")
def build_case_queue(_bundle) -> pd.DataFrame:
    all_ids = available_hadm_ids(_bundle)
    rows: list[dict[str, object]] = []
    for hadm_id in all_ids:
        ctx = build_case_context(_bundle, hadm_id)
        tri = score_case(ctx)
        case = ctx.case_row
        rows.append({
            "hadm_id": int(hadm_id),
            "case_id": case.get("case_id"),
            "age": case.get("age"),
            "gender": case.get("gender"),
            "admission_diagnosis": case.get("admission_diagnosis"),
            "urgency": tri.urgency,
            "triage_score": tri.score,
            "abnormal_labs": len(ctx.abnormal_labs),
            "diagnosis_count": len(ctx.diagnoses),
            "medication_count": len(ctx.prescriptions.drop_duplicates(subset=["drug"])) if not ctx.prescriptions.empty else 0,
            "top_evidence": tri.evidence[0] if tri.evidence else "",
        })
    df = pd.DataFrame(rows)
    urgency_rank = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
    df["_urgency_rank"] = df["urgency"].map(urgency_rank)
    df = df.sort_values(["_urgency_rank", "triage_score"], ascending=[True, False]).drop(columns=["_urgency_rank"])
    return df.reset_index(drop=True)


bundle = get_bundle()
retrieval_index = get_retrieval_index()
hadm_ids = available_hadm_ids(bundle)

st.title("Clinical Workflow Triage Assistant")
st.caption(
    "Interpretable admission triage over discharge summaries, diagnoses, prescriptions, and labs."
)

with st.sidebar:
    st.markdown("### View")
    app_mode = st.radio("Mode", options=["Case Review", "Case Queue"], index=0)
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

if app_mode == "Case Queue":
    st.subheader("Case Queue — Review by Priority")
    st.caption("All 2,000 admissions triaged and ranked. Filter by urgency, sort by score, then select a case to review.")

    queue_df = build_case_queue(bundle)

    filter_col, stats_col = st.columns([1, 1])
    with filter_col:
        urgency_filter = st.multiselect(
            "Filter by urgency",
            options=["Critical", "High", "Moderate", "Low"],
            default=["Critical", "High", "Moderate", "Low"],
        )
    with stats_col:
        counts = queue_df["urgency"].value_counts()
        st.markdown(
            f"**Critical:** {counts.get('Critical', 0)} · "
            f"**High:** {counts.get('High', 0)} · "
            f"**Moderate:** {counts.get('Moderate', 0)} · "
            f"**Low:** {counts.get('Low', 0)}"
        )

    filtered = queue_df[queue_df["urgency"].isin(urgency_filter)].copy()
    filtered.index = range(1, len(filtered) + 1)
    filtered.index.name = "#"

    st.dataframe(
        filtered,
        width="stretch",
        height=600,
        column_config={
            "hadm_id": st.column_config.NumberColumn("Admission ID", format="%d"),
            "triage_score": st.column_config.NumberColumn("Score"),
            "abnormal_labs": st.column_config.NumberColumn("Abnormal Labs"),
            "diagnosis_count": st.column_config.NumberColumn("Diagnoses"),
            "medication_count": st.column_config.NumberColumn("Medications"),
        },
    )

    st.info("To review a specific case, copy the Admission ID and select it from the dropdown in **Case Review** mode.")

else:
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
    tab1, tab2, tab3, tab4, tab4b, tab5, tab6 = st.tabs(
        ["Diagnoses", "Medications", "All Labs", "Note Summary", "Extracted Entities", "Similar Admissions", "Grounded Q&A"]
    )

    with tab1:
        st.dataframe(context.diagnoses, width="stretch", hide_index=True)

    with tab2:
        st.dataframe(context.prescriptions, width="stretch", hide_index=True)

    with tab3:
        st.dataframe(context.labs, width="stretch", hide_index=True)

    with tab4:
        raw_note = str(case.get("discharge_summary", ""))
        note_summary = summarize_note(raw_note)

        if note_summary.condensed and note_summary.condensed != "No discharge summary available.":
            st.markdown("#### Condensed Summary")
            st.caption(f"{note_summary.section_count} sections detected in the discharge note.")
            st.markdown(note_summary.condensed)

            if rag_mode == "LLM grounded":
                with st.expander("LLM-generated summary"):
                    llm_summary = summarize_note_llm(raw_note)
                    if llm_summary:
                        st.markdown(llm_summary)
                    else:
                        st.info("LLM summarization unavailable (no API key or connection error).")

            with st.expander("All detected sections"):
                for header, body in note_summary.sections.items():
                    st.markdown(f"**{header.title()}**")
                    st.write(body[:600] + ("..." if len(body) > 600 else ""))

            with st.expander("Full discharge note"):
                st.text(raw_note)
        else:
            st.info("No discharge summary available for this case.")

    with tab4b:
        entities = extract_entities(str(case.get("discharge_summary", "")))
        entity_df = entities.to_dataframe()
        if entity_df.empty:
            st.info("No clinical entities extracted from the discharge note.")
        else:
            cat_counts = entity_df["category"].value_counts()
            st.caption(
                f"Extracted {len(entity_df)} entities: "
                + ", ".join(f"{count} {cat.lower()}" for cat, count in cat_counts.items())
                + "."
            )
            for cat in ["Condition", "Procedure", "Medication", "Vital / Measurement"]:
                cat_df = entity_df[entity_df["category"] == cat]
                if cat_df.empty:
                    continue
                st.markdown(f"#### {cat}s ({len(cat_df)})")
                st.dataframe(cat_df[["entity"]].reset_index(drop=True), width="stretch", hide_index=True)

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