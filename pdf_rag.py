"""PDF retrieval plus an optional agent that may consult external search.

Answer generation receives the question and retrieved evidence only.
Gold answers and labeled pages are not arguments of this module.
"""

import base64
import re

from google.genai import types

import ai_engine
from ai_engine import (
    MODEL_NAME,
    PDF_SYSTEM_RULES,
    _remember_usage,
    ask_question_vision,
    generate_from_system,
    get_gemini_client,
)
from utils import chunk_text, embed_chunks, retrieve_with_scores
from web_search import search_web

PDF_ONLY = "pdf"
PDF_WEB = "pdf_web"
AGENT = "agent"
MODES = (PDF_ONLY, PDF_WEB, AGENT)

ABSTAIN_TEXT = (
    "I can't answer this from the available evidence. "
    "The retrieved PDF pages and external sources do not support a verified answer."
)
MAX_RETRIEVAL_ATTEMPTS = 2
SUFFICIENT_SCORE = 0.45
IMAGE_PAGE_CHARS = 100


def transcribe_page(page_image):
    """Read one rendered page. Used so sparse and scanned pages can be retrieved."""
    client = get_gemini_client()
    prompt = (
        "Transcribe this page into plain text. Preserve headings, numbers, "
        "and formulas. Do not add facts that are not visible."
    )
    parts = [
        types.Part.from_text(text=f"--- Page {page_image['page_num']} ---"),
        types.Part.from_bytes(
            data=base64.b64decode(page_image["b64"]),
            mime_type="image/jpeg",
        ),
        types.Part.from_text(text=prompt),
    ]
    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction="You transcribe document pages. Output only the transcription.",
            max_output_tokens=2000,
            temperature=0,
        ),
    ))
    return (response.text or "").strip()


def merge_transcripts(text_by_page, page_images, pages, transcripts):
    """Fill selected low-text pages without dropping text that is already real."""
    updated = dict(text_by_page)
    by_num = {image["page_num"]: image for image in page_images}
    for page_num in pages:
        if page_num in transcripts and str(transcripts[page_num]).strip():
            updated[page_num] = transcripts[page_num]
            continue
        image = by_num.get(page_num)
        if image is None:
            continue
        transcripts[page_num] = transcribe_page(image)
        if transcripts[page_num]:
            updated[page_num] = transcripts[page_num]
    return updated, transcripts


def chunks_from_pages(text_by_page, source="document"):
    """One chunk list with the PDF page kept on every piece."""
    rows = []
    for page_num in sorted(text_by_page):
        text = str(text_by_page.get(page_num) or "").strip()
        if not text:
            continue
        for chunk in chunk_text(text, chunk_size=600):
            rows.append({
                "text": f"--- Page {page_num} ---\n{chunk}",
                "page": int(page_num),
                "source": source,
            })
    return rows


def sparse_page_numbers(text_by_page, limit=IMAGE_PAGE_CHARS):
    return [
        page_num
        for page_num, text in text_by_page.items()
        if len(str(text or "").strip()) < limit
    ]


def evidence_sufficient(scores, threshold=SUFFICIENT_SCORE):
    numeric = [score for score in scores if isinstance(score, (int, float))]
    if not numeric:
        return bool(scores)
    return max(numeric) >= threshold


def should_abstain(mode, pdf_scores, web_results, attempts, threshold=SUFFICIENT_SCORE):
    """Agent abstains only after the retrieval retry, and only without evidence."""
    if mode != AGENT:
        return False
    if evidence_sufficient(pdf_scores, threshold):
        return False
    if web_results:
        return False
    return attempts >= MAX_RETRIEVAL_ATTEMPTS


def looks_like_abstention(text):
    lowered = (text or "").lower()
    cues = (
        "can't answer this from the available evidence",
        "cannot answer",
        "not in the supplied evidence",
        "do not support a verified answer",
        "does not contain",
        "do not contain",
        "no mention",
        "no information",
        "not available in the",
        "insufficient evidence",
    )
    return any(cue in lowered for cue in cues)


def claims_outside_context(claims, context):
    """Claims whose text does not appear in the retrieved context."""
    haystack = (context or "").lower()
    return [claim for claim in claims if claim.lower() not in haystack]


def matches_required_facts(answer, required_phrases):
    lowered = (answer or "").lower()
    return all(phrase.lower() in lowered for phrase in required_phrases)


def cited_pages(text):
    return [int(number) for number in re.findall(r"\bpage\s+(\d+)\b", text or "", re.I)]


def source_footer(pages, web_results):
    lines = ["", "Sources:"]
    unique_pages = []
    for page in pages:
        if page not in unique_pages:
            unique_pages.append(page)
    if unique_pages:
        lines.append("PDF pages: " + ", ".join(f"page {page}" for page in unique_pages))
    else:
        lines.append("PDF pages: none retrieved")
    if web_results:
        lines.append("External sources:")
        for item in web_results:
            lines.append(f"- {item.get('title') or 'source'}: {item.get('url')}")
    else:
        lines.append("External sources: none")
    return "\n".join(lines)


def build_evidence_prompt(pdf_chunks, web_results):
    """Evidence prompt. There is no parameter for a reference answer."""
    pdf_block = "\n\n".join(chunk["text"] for chunk in pdf_chunks) or "(no PDF excerpts)"
    if web_results:
        web_lines = []
        for item in web_results:
            web_lines.append(
                f"Title: {item.get('title')}\nURL: {item.get('url')}\nSnippet: {item.get('snippet')}"
            )
        web_block = "\n\n".join(web_lines)
    else:
        web_block = "(no external sources)"
    return f"""You are an expert document analyst.
{PDF_SYSTEM_RULES}

PDF EVIDENCE:
{pdf_block}

EXTERNAL EVIDENCE:
{web_block}"""


def prompt_contains_reference(prompt, reference):
    cleaned = (reference or "").strip()
    return bool(cleaned) and cleaned in (prompt or "")


def _retrieve(question, chunk_rows, embeddings, top_k):
    if not chunk_rows:
        return []
    texts = [row["text"] for row in chunk_rows]
    scored = retrieve_with_scores(question, texts, embeddings, top_k=top_k)
    by_text = {}
    for row in chunk_rows:
        by_text.setdefault(row["text"], row)
    hits = []
    for text, score in scored:
        row = by_text.get(text)
        if row is None:
            continue
        hits.append({**row, "score": score})
    return hits


def answer_from_index(question, chunk_rows, embeddings, page_images, scanned, mode, history=None):
    """Retrieve, optionally search, retry once, and abstain when the agent has no evidence."""
    if history is None:
        history = []
    mode = mode if mode in MODES else PDF_ONLY
    notes = []
    web_payload = {"available": False, "results": [], "reason": "not requested"}
    hits = []
    top_k = 5

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        hits = _retrieve(question, chunk_rows, embeddings, top_k=top_k)
        scores = [hit.get("score") for hit in hits]
        if evidence_sufficient(scores) or attempt == MAX_RETRIEVAL_ATTEMPTS:
            break
        top_k = 8
        notes.append("retried PDF retrieval once with a wider candidate set")

    scores = [hit.get("score") for hit in hits]
    need_web = mode in (PDF_WEB, AGENT) and not evidence_sufficient(scores)
    if need_web:
        web_payload = search_web(question, max_results=3)
        if not web_payload.get("available"):
            notes.append(web_payload.get("reason") or "web search unavailable")
        elif not web_payload.get("results"):
            notes.append(web_payload.get("reason") or "web search returned no results")

    web_results = web_payload.get("results") or []
    attempts = MAX_RETRIEVAL_ATTEMPTS if any(note.startswith("retried") for note in notes) else 1
    if hits and not evidence_sufficient(scores):
        attempts = MAX_RETRIEVAL_ATTEMPTS

    abstained = should_abstain(mode, scores, web_results, attempts)
    pages = []
    for hit in hits:
        if hit.get("page") not in pages:
            pages.append(hit["page"])

    if abstained:
        answer = ABSTAIN_TEXT + source_footer(pages, web_results)
        return _result(answer, pages, web_payload, notes, True, hits, scanned)

    if scanned and page_images:
        visible = pages[:5] if pages else None
        answer = ask_question_vision(
            question,
            page_images,
            chat_history=history,
            pages_to_send=visible,
        )
    else:
        system = build_evidence_prompt(hits, web_results)
        answer = generate_from_system(question, system, chat_history=history)

    answer = (answer or "").strip() + source_footer(pages, web_results)
    return _result(answer, pages, web_payload, notes, looks_like_abstention(answer), hits, scanned)


def _result(answer, pages, web_payload, notes, abstained, hits, scanned):
    return {
        "answer": answer,
        "pages": pages,
        "web": web_payload,
        "notes": notes,
        "abstained": abstained,
        "hits": hits,
        "usage": ai_engine.LAST_USAGE,
        "mode_label": "scanned" if scanned else "digital",
    }
