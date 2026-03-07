from __future__ import annotations

from dataclasses import dataclass
import os
import re

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from healthcare_ai.features import CaseContext, lab_signal_strings, top_diagnosis_titles, top_medications
from healthcare_ai.triage import score_case


@dataclass(frozen=True)
class RagChunk:
    source_type: str
    source_id: str
    title: str
    content: str


@dataclass(frozen=True)
class RagIndex:
    chunks: list[RagChunk]
    matrix: object
    vectorizer: TfidfVectorizer


QUERY_EXPANSIONS = {
    "urgent": "urgent priority risk review critical high unstable concern",
    "urgency": "urgent priority risk review critical high unstable concern",
    "why": "reason evidence because due to driven by",
    "medication": "medication drug therapy treatment",
    "labs": "lab laboratory abnormal result chemistry hematology",
    "diagnosis": "diagnosis problem condition disease",
}

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv()
    _ENV_LOADED = True


def _evidence_payload(evidence: list[tuple[RagChunk, float]]) -> list[dict[str, object]]:
    return [
        {
            "title": chunk.title,
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "score": round(score, 3),
            "content": chunk.content,
        }
        for chunk, score in evidence
    ]


def _local_grounded_answer(query: str, evidence: list[tuple[RagChunk, float]]) -> str:
    primary_titles = ", ".join(chunk.title for chunk, _ in evidence[:2])
    answer_lines = [
        "Clinical response",
        f"Question: {query}",
        "Answer: The strongest support comes from the current case triage rationale and case summary evidence.",
        f"Supporting evidence: {primary_titles}.",
    ]
    for chunk, _ in evidence[:2]:
        answer_lines.append(f"- {chunk.title}: {chunk.content}")
    answer_lines.append(
        "Citations: " + ", ".join(f"[{chunk.title}]" for chunk, _ in evidence[:3])
    )
    return "\n".join(answer_lines)


def _split_note_into_chunks(note: str, max_chars: int = 420) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(note or "")).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        proposal = f"{current} {sentence}".strip() if current else sentence
        if len(proposal) <= max_chars:
            current = proposal
            continue
        if current:
            chunks.append(current)
        current = sentence[:max_chars]
    if current:
        chunks.append(current)
    return chunks[:6]


def build_case_rag_index(context: CaseContext, similar_cases: pd.DataFrame) -> RagIndex:
    case = context.case_row
    triage = score_case(context)
    chunks: list[RagChunk] = [
        RagChunk(
            source_type="current_case",
            source_id=str(case.get("hadm_id")),
            title="Current case overview",
            content=(
                f"Admission diagnosis: {case.get('admission_diagnosis', '')}. "
                f"Top diagnoses: {', '.join(top_diagnosis_titles(context.diagnoses, limit=8))}. "
                f"Key medications: {', '.join(top_medications(context.prescriptions, limit=8))}. "
                f"Abnormal labs: {', '.join(lab_signal_strings(context.abnormal_labs, limit=8))}."
            ),
        ),
        RagChunk(
            source_type="triage_reasoning",
            source_id=str(case.get("hadm_id")),
            title="Current triage rationale",
            content=(
                f"Urgency is {triage.urgency}. "
                f"Triage score is {triage.score}. "
                f"Top evidence: {', '.join(triage.evidence[:8])}."
            ),
        )
    ]

    for idx, snippet in enumerate(_split_note_into_chunks(str(case.get("discharge_summary", ""))), start=1):
        chunks.append(
            RagChunk(
                source_type="current_note",
                source_id=f"{case.get('hadm_id')}-note-{idx}",
                title=f"Current note excerpt {idx}",
                content=snippet,
            )
        )

    for _, row in similar_cases.head(4).iterrows():
        chunks.append(
            RagChunk(
                source_type="similar_case",
                source_id=str(int(row["hadm_id"])),
                title=f"Similar admission {int(row['hadm_id'])}",
                content=(
                    f"Admission diagnosis: {row['admission_diagnosis']}. "
                    f"Similarity: {row['similarity']:.3f}. "
                    f"Match reasons: {row['match_reasons']}."
                ),
            )
        )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
    matrix = vectorizer.fit_transform([f"{chunk.title}. {chunk.content}" for chunk in chunks])
    return RagIndex(chunks=chunks, matrix=matrix, vectorizer=vectorizer)


def _expand_query(query: str) -> str:
    lowered = query.lower()
    expansions = [query]
    for key, value in QUERY_EXPANSIONS.items():
        if key in lowered:
            expansions.append(value)
    return " ".join(expansions)


def retrieve_rag_chunks(index: RagIndex, query: str, top_n: int = 4) -> list[tuple[RagChunk, float]]:
    if not query.strip():
        return []
    query_vector = index.vectorizer.transform([_expand_query(query)])
    scores = linear_kernel(query_vector, index.matrix).ravel()
    ranked = scores.argsort()[::-1]
    results: list[tuple[RagChunk, float]] = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append((index.chunks[idx], score))
        if len(results) >= top_n:
            break
    return results


def llm_rag_available() -> bool:
    _ensure_env_loaded()
    return bool(os.getenv("OPENAI_API_KEY"))


def _build_llm_prompt(query: str, evidence: list[tuple[RagChunk, float]]) -> str:
    context_blocks = []
    for chunk, score in evidence:
        context_blocks.append(
            f"Title: {chunk.title}\n"
            f"Source type: {chunk.source_type}\n"
            f"Source id: {chunk.source_id}\n"
            f"Retrieval score: {score:.3f}\n"
            f"Content: {chunk.content}"
        )
    joined_context = "\n\n".join(context_blocks)
    return (
        "You are a clinical workflow assistant. Answer only from the retrieved evidence. "
        "Do not invent facts. If evidence is insufficient, say so explicitly. "
        "Keep the answer concise and clinician-facing. "
        "Use this exact structure with short content under each heading: \n"
        "Clinical response\n"
        "Question: <repeat the user question>\n"
        "Answer: <1-3 sentences>\n"
        "Supporting evidence:\n"
        "- <point with citation like [Current triage rationale]>\n"
        "- <point with citation like [Current case overview]>\n"
        "Uncertainty: <short line, or 'No major uncertainty from retrieved evidence.'>\n"
        "Citations: [Title 1], [Title 2]\n\n"
        f"Question: {query}\n\n"
        f"Retrieved evidence:\n{joined_context}"
    )


def _answer_with_llm(index: RagIndex, query: str, evidence: list[tuple[RagChunk, float]]) -> dict[str, object]:
    _ensure_env_loaded()
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError, RateLimitError
    except ImportError:
        return {
            "answer": "LLM mode is configured in the app, but the `openai` package is not installed. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "LLM mode was requested, but OPENAI_API_KEY is not set. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        response = client.responses.create(
            model=model,
            input=_build_llm_prompt(query, evidence),
            temperature=0.2,
        )
    except RateLimitError:
        return {
            "answer": "The LLM provider rejected the request because the current API key has no remaining quota or is rate-limited. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    except APITimeoutError:
        return {
            "answer": "The LLM provider timed out while generating a grounded answer. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    except APIConnectionError:
        return {
            "answer": "The app could not reach the LLM provider. Check your network or provider base URL. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    except APIStatusError as exc:
        status_code = getattr(exc, "status_code", "unknown")
        return {
            "answer": f"The LLM provider returned an API error (status {status_code}). Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    except OpenAIError:
        return {
            "answer": "The LLM provider returned an unexpected OpenAI client error. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    except Exception:
        return {
            "answer": "An unexpected error occurred while generating an LLM-grounded answer. Falling back to local grounded mode.",
            "evidence": _evidence_payload(evidence),
            "mode": "fallback-local",
        }
    answer_text = getattr(response, "output_text", "").strip()
    if not answer_text:
        answer_text = "The LLM returned an empty response. Review the evidence below instead."
    return {
        "answer": answer_text,
        "evidence": _evidence_payload(evidence),
        "mode": "llm-grounded",
        "model": model,
    }


def answer_grounded_question(index: RagIndex, query: str, top_n: int = 4, use_llm: bool = False) -> dict[str, object]:
    evidence = retrieve_rag_chunks(index, query, top_n=top_n)
    if not evidence:
        return {
            "answer": "No grounded evidence was retrieved for that question. Try a more specific question about diagnoses, medications, labs, or note details.",
            "evidence": [],
            "mode": "no-evidence",
        }

    if use_llm:
        llm_result = _answer_with_llm(index, query, evidence)
        if llm_result.get("mode") == "llm-grounded":
            return llm_result
        fallback_lines = [llm_result["answer"], "", "Falling back to local grounded answer."]
        local_result = answer_grounded_question(index, query, top_n=top_n, use_llm=False)
        return {
            "answer": "\n".join(fallback_lines + ["", local_result["answer"]]),
            "evidence": local_result["evidence"],
            "mode": "fallback-local",
        }

    answer_lines = [
        _local_grounded_answer(query, evidence),
    ]
    return {
        "answer": "\n".join(answer_lines),
        "evidence": _evidence_payload(evidence),
        "mode": "local-grounded",
    }
