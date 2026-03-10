from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from healthcare_ai.data import available_hadm_ids, load_dataset
from healthcare_ai.features import build_case_context
from healthcare_ai.rag import answer_grounded_question, build_case_rag_index, llm_rag_available
from healthcare_ai.retrieval import build_retrieval_index, find_similar_cases
from healthcare_ai.summarizer import build_case_summary
from healthcare_ai.triage import score_case

load_dotenv()

st.set_page_config(page_title="Clinical Workflow Triage Assistant", layout="wide")


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