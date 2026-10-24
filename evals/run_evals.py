"""RAG evaluation runner for pdf-rag-system."""

import base64
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

load_dotenv(PROJECT / ".env")

from utils import (
    extract_text_from_pdf,
    convert_pdf_to_images,
    is_scanned_pdf,
    chunk_text,
    embed_chunks,
    retrieve_relevant_chunks,
)
import ai_engine
from ai_engine import (
    MODEL_NAME,
    _remember_usage,
    ask_question_text,
    ask_question_vision,
    get_gemini_client,
)
from pdf_rag import looks_like_abstention

JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL_NAME)
DOCS_DIR = ROOT / "docs"
# Retrieved pages sent to the vision model. Chosen by retrieval, not by
# taking the first pages of the file and not by the reference page number.
MAX_VISION_PAGES = 5
_TRANSIENT = (
    "429",
    "resource_exhausted",
    "unavailable",
    "503",
    "500",
    "deadline",
    "overloaded",
    "try again",
    "temporarily",
)

_DOCUMENTS = {}
_TRANSCRIPTS = {}
_judge_client = None
_vision_client = None
_next_api_at = 0.0
_SCAN_CACHE_PATH = Path(tempfile.gettempdir()) / "pdf-rag-system-scan-cache.json"
# Free tier allows 15 generateContent requests per minute.
_MIN_API_GAP = 4.5
# A page this short was not really extracted; screenshots and scans look like this.
IMAGE_PAGE_CHARS = 100


def _redact(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    key = os.getenv("GEMINI_API_KEY") or ""
    if key:
        text = text.replace(key, "[REDACTED]")
    return text


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in _TRANSIENT)


def _quota_delay(exc: Exception) -> float:
    """Seconds to wait after a rate limit. Extra retries before then count against the cap."""
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(exc), re.I)
    if match:
        return float(match.group(1)) + 1.0
    return 20.0


def _pace():
    global _next_api_at
    now = time.time()
    if now < _next_api_at:
        time.sleep(_next_api_at - now)
    _next_api_at = time.time() + _MIN_API_GAP


def _with_retries(fn):
    last_error = None
    for attempt in range(4):
        try:
            _pace()
            return fn()
        except Exception as exc:
            last_error = exc
            limited = _is_retryable(exc) or "closed" in str(exc).lower()
            if attempt == 3 or not limited:
                raise
            if "closed" in str(exc).lower():
                _reset_vision_client()
            delay = _quota_delay(exc) if _is_retryable(exc) else 2.0
            global _next_api_at
            _next_api_at = max(_next_api_at, time.time() + delay)
            print(f"  waiting {delay:.0f}s after API limit", flush=True)
    raise last_error


def get_judge_client():
    global _judge_client

    if _judge_client is None:
        load_dotenv(PROJECT / ".env")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your local .env file."
            )
        _judge_client = genai.Client(api_key=api_key)

    return _judge_client


def validate_dataset(questions) -> None:
    """Require every question to name a PDF that exists in evals/docs/."""
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    if len(questions) != 39:
        raise ValueError(f"Expected 39 questions, found {len(questions)}.")

    errors = []
    for index, item in enumerate(questions, 1):
        if not isinstance(item, dict):
            errors.append(f"Q{index}: question is not an object")
            continue
        for field in ("q", "answer", "answerable"):
            if field not in item:
                errors.append(f"Q{index}: missing {field}")
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            errors.append(f"Q{index}: missing source")
            continue
        if Path(source).name != source:
            errors.append(f"Q{index}: source must be a PDF file name")
        pdf_path = DOCS_DIR / source
        if not pdf_path.is_file():
            errors.append(f"Q{index}: PDF not found in evals/docs/: {source}")

    if errors:
        raise ValueError("Invalid evaluation dataset:\n" + "\n".join(errors))


def _pages_from_chunks(chunks, limit=MAX_VISION_PAGES) -> list[int]:
    pages = []
    for chunk in chunks:
        page = chunk.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
        if len(pages) >= limit:
            break
    return pages


def _parse_page_list(text: str, allowed: set[int]) -> list[int]:
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("pages", [])
        if isinstance(data, list):
            return [page for page in data if isinstance(page, int) and page in allowed]
    return []


def _reset_vision_client():
    global _vision_client
    _vision_client = None


def _get_vision_client():
    """Keep one Gemini client alive. A temporary client is closed before the request."""
    global _vision_client
    if _vision_client is None:
        _vision_client = get_gemini_client()
    return _vision_client


def _image_page_numbers(text_by_page: dict) -> list[int]:
    pages = []
    for page_num, page_text in text_by_page.items():
        if len(str(page_text).strip()) < IMAGE_PAGE_CHARS:
            pages.append(page_num)
    return pages


def _render_pages(pdf_bytes: bytes, page_numbers: list[int]) -> list[dict]:
    """Render selected pages the same way utils.convert_pdf_to_images does."""
    import fitz

    wanted = set(page_numbers)
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for index in range(len(document)):
            page_num = index + 1
            if page_num not in wanted:
                continue
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            images.append({
                "page_num": page_num,
                "b64": base64.b64encode(pixmap.tobytes("jpeg")).decode("utf-8"),
            })
    finally:
        document.close()
    return images


def _vision_request(
    prompt: str,
    page_images: list[dict],
    pages: list[int],
    system: str,
    max_output_tokens: int = 2000,
) -> str:
    """Gemini vision call used to index or select scanned pages."""
    selected = [image for image in page_images if image["page_num"] in pages]
    parts = []
    for image in selected:
        parts.append(types.Part.from_text(text=f"--- Page {image['page_num']} ---"))
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(image["b64"]),
            mime_type="image/jpeg",
        ))
    parts.append(types.Part.from_text(text=prompt))

    def _send():
        client = _get_vision_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
                temperature=0,
            ),
        )
        return (response.text or "").strip()

    return _with_retries(_send)


def _transcribe_page(page_images: list[dict], page_num: int) -> str:
    prompt = (
        "Transcribe this page into plain text. Preserve headings, numbered "
        "lists, formulas, and examples. Do not summarize and do not add "
        "facts that are not visible."
    )
    return _vision_request(
        prompt,
        page_images,
        [page_num],
        "You transcribe document pages. Output only the transcription.",
    )


def _index_pages(source: str, text_by_page: dict) -> list[dict]:
    chunks = []
    for page_num, page_text in text_by_page.items():
        if not str(page_text).strip():
            continue
        for chunk in chunk_text(str(page_text), chunk_size=600):
            chunks.append({
                "text": f"--- Page {page_num} ---\n{chunk}",
                "page": page_num,
                "source": Path(source).name,
            })
    return chunks


def _scan_cache_key(source: str) -> str:
    pdf_path = DOCS_DIR / Path(source).name
    stat = pdf_path.stat()
    return f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _read_scan_cache() -> dict:
    if not _SCAN_CACHE_PATH.is_file():
        return {}
    try:
        data = json.loads(_SCAN_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_scan_cache(cache: dict) -> None:
    _SCAN_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def _transcribe_scanned(source: str, page_images: list[dict], label: str = "scanned") -> dict:
    print(
        f"Reading {label} pages for retrieval: {source} "
        f"({len(page_images)} pages)",
        flush=True,
    )
    text_by_page = _TRANSCRIPTS.setdefault(source, {})
    cache = _read_scan_cache()
    key = _scan_cache_key(source)
    saved = cache.get(key, {})
    if not isinstance(saved, dict):
        saved = {}
    for page_str, text in saved.items():
        if str(page_str).isdigit():
            text_by_page.setdefault(int(page_str), text)

    for image in page_images:
        page_num = image["page_num"]
        if page_num in text_by_page and str(text_by_page[page_num]).strip():
            continue
        started = time.time()
        text_by_page[page_num] = _transcribe_page(page_images, page_num)
        saved[str(page_num)] = text_by_page[page_num]
        cache[key] = saved
        _write_scan_cache(cache)
        print(
            f"  transcribed page {page_num}/{len(page_images)} "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )
    print(
        f"  scanned transcripts ready "
        f"({sum(1 for text in text_by_page.values() if str(text).strip())}"
        f"/{len(page_images)})",
        flush=True,
    )
    return dict(text_by_page)


def select_relevant_scanned_pages(question: str, doc: dict) -> list[int]:
    """Pick scanned pages by content. Never defaults to only the first four."""
    page_images = doc["page_images"]
    if not page_images:
        raise RuntimeError(
            "Scanned PDF has no page images. Check that pymupdf is installed."
        )

    available = [image["page_num"] for image in page_images]
    selected = []
    batch_size = 4
    for start in range(0, len(available), batch_size):
        batch = available[start:start + batch_size]
        prompt = (
            f"Question: {question}\n"
            f"The attached images are PDF pages {batch}, in that order. "
            "Which of these pages contain information that would help answer "
            "the question? Reply with JSON only, using page numbers from this "
            'batch: {"pages": []}. Use an empty list if none of them help.'
        )
        raw = _vision_request(
            prompt,
            page_images,
            batch,
            "You choose relevant document pages and reply with JSON only.",
            max_output_tokens=200,
        )
        for page in _parse_page_list(raw, set(batch)):
            if page not in selected:
                selected.append(page)

    if selected:
        return selected[:MAX_VISION_PAGES]

    # Relevance check found nothing. Cover the whole document instead of
    # sending only the opening pages.
    if len(available) <= MAX_VISION_PAGES:
        return available
    indexes = [
        round(i * (len(available) - 1) / (MAX_VISION_PAGES - 1))
        for i in range(MAX_VISION_PAGES)
    ]
    spread = []
    for index in indexes:
        page = available[index]
        if page not in spread:
            spread.append(page)
    return spread


def load_document(source: str) -> dict:
    if source in _DOCUMENTS:
        return _DOCUMENTS[source]

    pdf_path = DOCS_DIR / Path(source).name
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Put the PDF in evals/docs/."
        )

    pdf_bytes = pdf_path.read_bytes()
    text_by_page, total_pages = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    scanned = is_scanned_pdf(text_by_page)

    page_images = []
    retrieval_text = dict(text_by_page)
    if scanned:
        page_images = convert_pdf_to_images(io.BytesIO(pdf_bytes))
        if not page_images:
            raise RuntimeError(
                f"Could not render pages for scanned PDF {pdf_path.name}."
            )
        retrieval_text = _transcribe_scanned(source, page_images, label="scanned")
    else:
        # Digital PDFs can still contain screenshot pages with no real text.
        sparse_pages = _image_page_numbers(text_by_page)
        if sparse_pages:
            page_images = _render_pages(pdf_bytes, sparse_pages)
            if not page_images:
                raise RuntimeError(
                    f"Could not render image pages in {pdf_path.name}."
                )
            transcripts = _transcribe_scanned(
                source, page_images, label="image"
            )
            for page_num, transcript in transcripts.items():
                if str(transcript).strip():
                    retrieval_text[page_num] = transcript

    chunks = _index_pages(source, retrieval_text)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Indexing {Path(source).name} ({len(chunks)} chunks)", flush=True)
    embeddings = embed_chunks(texts) if texts else None

    doc = {
        "text_by_page": text_by_page,
        "retrieval_text": retrieval_text,
        "page_images": page_images,
        "total_pages": total_pages,
        "scanned": scanned,
        "chunks": chunks,
        "embeddings": embeddings,
    }
    _DOCUMENTS[source] = doc
    return doc


def retrieve(question: str, item: dict) -> list[dict]:
    doc = load_document(item["source"])
    if not doc["chunks"]:
        return []

    selected = retrieve_relevant_chunks(
        question,
        [chunk["text"] for chunk in doc["chunks"]],
        doc["embeddings"],
        top_k=5,
    )
    by_text = {chunk["text"]: chunk for chunk in doc["chunks"]}
    return [by_text[text] for text in selected if text in by_text]


def generate_answer(question: str, item: dict, chunks: list[dict]) -> tuple[str, list]:
    doc = load_document(item["source"])

    if doc["scanned"]:
        pages = _pages_from_chunks(chunks)
        if not pages:
            print("  selecting scanned pages by content", flush=True)
            pages = select_relevant_scanned_pages(question, doc)
        if not pages:
            raise RuntimeError("No scanned pages were selected.")
        answer = _with_retries(lambda: ask_question_vision(
            question,
            doc["page_images"],
            chat_history=[],
            pages_to_send=pages,
        ))
        return (answer or "").strip(), pages

    selected_texts = [chunk["text"] for chunk in chunks]
    selected_embeddings = embed_chunks(selected_texts) if selected_texts else None
    answer = _with_retries(lambda: ask_question_text(
        question,
        doc["text_by_page"],
        chat_history=[],
        chunks=selected_texts,
        chunk_embeddings=selected_embeddings,
    ))
    return (answer or "").strip(), [chunk.get("page") for chunk in chunks]


def is_hit(item: dict, chunks: list[dict]):
    # `page` in the dataset is the 1-based PDF page from extraction.
    if not item.get("answerable") or item.get("page") is None:
        return None

    for chunk in chunks:
        page = chunk.get("page")
        if page is None or page != item["page"]:
            continue
        source = chunk.get("source")
        if source and Path(str(source)).name != Path(item["source"]).name:
            continue
        return True
    return False


def judge(item: dict, model_answer: str, chunks: list[dict]) -> dict:
    # Keep each retrieved chunk intact. A short per-chunk cutoff drops
    # tables and citations that sit later on the same page, so the judge
    # then marks supported answers as ungrounded.
    context_parts = []
    remaining = 30000
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        if not text.strip() or remaining <= 0:
            continue
        if len(text) > remaining:
            text = text[:remaining]
        context_parts.append(text)
        remaining -= len(text)
    context = "\n---\n".join(context_parts)

    if item["answerable"]:
        rule = (
            f"Reference answer: {item['answer']}\n"
            "Mark correct=true if the model answer contains the same "
            "important facts. Wording can differ. Mark false if key "
            "facts are incorrect or missing."
        )
    else:
        rule = (
            "The requested information is not available in the document. "
            "Mark correct=true only if the answer appropriately states "
            "that the information cannot be found. Mark false if it "
            "invents an answer."
        )

    prompt = (
        f"Question: {item['q']}\n\n"
        f"{rule}\n\n"
        f"Model answer to evaluate:\n{model_answer}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Evaluate groundedness separately from correctness. "
        "grounded=true only if every factual claim is supported by the "
        "retrieved context, or the answer says the information is unavailable. "
        "Extra background from general knowledge is not grounded. "
        "A correct reference fact does not excuse an unsupported extra claim. "
        "Explain your decision briefly."
    )

    def _send_judge():
        return get_judge_client().models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "correct": {"type": "BOOLEAN"},
                        "grounded": {"type": "BOOLEAN"},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["correct", "grounded", "reason"],
                },
            ),
        )

    response = _remember_usage(_with_retries(_send_judge))
    judge_usage = ai_engine.LAST_USAGE

    try:
        data = json.loads(response.text or "")
        if "correct" not in data or "grounded" not in data:
            raise KeyError("missing fields")
        return {
            "correct": data["correct"] is True,
            "grounded": data["grounded"] is True,
            "reason": str(data.get("reason", "")),
            "usage": judge_usage,
        }
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return {
            "correct": None,
            "grounded": None,
            "reason": "Not scored automatically: Gemini judge returned invalid or incomplete JSON.",
            "usage": judge_usage,
        }


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def _add_usage(total, usage):
    if not usage:
        return total
    for key in ("prompt_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
    return total


def main(output_name="results_improved.json"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    questions_path = ROOT / "questions.json"
    with questions_path.open(encoding="utf-8") as file:
        questions = json.load(file)
    validate_dataset(questions)
    for source in dict.fromkeys(item["source"] for item in questions):
        load_document(source)

    results = []
    token_totals = {}
    output_path = ROOT / output_name

    for i, item in enumerate(questions, 1):
        record = dict(item)
        try:
            start = time.time()
            chunks = retrieve(item["q"], item)
            model_answer, pages_sent = generate_answer(item["q"], item, chunks)
            answer_usage = ai_engine.LAST_USAGE

            record["secs"] = round(time.time() - start, 2)
            record["hit"] = is_hit(item, chunks)
            record["answer_given"] = model_answer
            record["retrieved_pages"] = [chunk.get("page") for chunk in chunks]
            record["pages_sent"] = pages_sent
            record["mode"] = "scanned" if load_document(item["source"])["scanned"] else "digital"
            record["abstained"] = looks_like_abstention(model_answer)
            record["answer_usage"] = answer_usage
            _add_usage(token_totals, answer_usage)

            if not chunks:
                record["correct"] = None
                record["grounded"] = None
                record["reason"] = (
                    "Not scored automatically: no retrieved text context."
                )
            else:
                verdict = judge(item, model_answer, chunks)
                record["correct"] = verdict["correct"]
                record["grounded"] = verdict["grounded"]
                record["reason"] = verdict["reason"]
                record["judge_usage"] = verdict.get("usage")
                _add_usage(token_totals, verdict.get("usage"))

        except Exception as exc:
            record["error"] = _redact(exc)
            record["correct"] = None
            record["grounded"] = None
            record["hit"] = None

        results.append(record)
        if record.get("error"):
            status = "ERROR"
        elif record.get("correct") is True:
            status = "OK"
        elif record.get("correct") is False:
            status = "MISS"
        else:
            status = "UNSCORED"
        print(f"[{i:02d}/{len(questions)}] {status} {item['q'][:70]}", flush=True)
        if record.get("error"):
            print(f"    {record['error'][:400]}", flush=True)

        output_path.write_text(
            _redact(json.dumps(results, indent=2, ensure_ascii=False)),
            encoding="utf-8",
        )

    scored = [row for row in results if row.get("correct") is not None]
    answerable = [row for row in scored if row["answerable"]]
    unanswerable = [row for row in scored if not row["answerable"]]
    checked_hits = [row for row in results if row.get("hit") is not None]
    times = [row["secs"] for row in results if "secs" in row]
    errors = [row for row in results if row.get("error")]
    unscored = [
        (index, row)
        for index, row in enumerate(results, 1)
        if row.get("correct") is None and not row.get("error")
    ]
    grounded = [row for row in scored if row.get("grounded") is not None]

    print("\n========== SUMMARY ==========")
    print(f"Questions:             {len(results)}")
    print(f"Errors:                {len(errors)}")
    print(f"Unscored:              {len(unscored)}")
    print(
        f"Retrieval hit rate:    "
        f"{pct(sum(row['hit'] is True for row in checked_hits), len(checked_hits))}"
        f" ({sum(row['hit'] is True for row in checked_hits)}/{len(checked_hits)})"
    )
    print(
        f"Answer accuracy:       "
        f"{pct(sum(row['correct'] is True for row in scored), len(scored))}"
        f" ({sum(row['correct'] is True for row in scored)}/{len(scored)})"
    )
    print(
        f"  Answerable:          "
        f"{pct(sum(row['correct'] is True for row in answerable), len(answerable))}"
        f" ({sum(row['correct'] is True for row in answerable)}/{len(answerable)})"
    )
    print(
        f"  Unanswerable:        "
        f"{pct(sum(row['correct'] is True for row in unanswerable), len(unanswerable))}"
        f" ({sum(row['correct'] is True for row in unanswerable)}/{len(unanswerable)})"
    )
    print(
        f"Groundedness:          "
        f"{pct(sum(row['grounded'] is True for row in grounded), len(grounded))}"
        f" ({sum(row['grounded'] is True for row in grounded)}/{len(grounded)})"
    )
    print(
        f"Avg seconds/question:  {sum(times) / len(times):.2f}"
        if times else "Avg seconds/question:  n/a"
    )
    abstained = [row for row in results if row.get("abstained") is True]
    unanswerable_rows = [row for row in results if row.get("answerable") is False and not row.get("error")]
    good_abstentions = [
        row for row in unanswerable_rows
        if row.get("abstained") and row.get("correct") is True
    ]
    false_abstentions = [
        row for row in results
        if row.get("answerable") and row.get("abstained") and not row.get("error")
    ]
    print(
        f"Abstention quality:    "
        f"{pct(len(good_abstentions), len(unanswerable_rows))}"
        f" ({len(good_abstentions)}/{len(unanswerable_rows)} unanswerable refusals)"
    )
    print(f"False abstentions:     {len(false_abstentions)}")
    if token_totals:
        print(
            "Token usage:           "
            f"prompt {token_totals.get('prompt_tokens', 0)}, "
            f"output {token_totals.get('output_tokens', 0)}, "
            f"total {token_totals.get('total_tokens', 0)}"
        )
    else:
        print("Token usage:           unavailable")
    print(f"Abstentions flagged:   {len(abstained)}")
    if unscored:
        print("Unscored questions:")
        for index, row in unscored:
            print(f"  [{index:02d}] {row['q'][:70]} — {row.get('reason', '')}")
    if errors:
        print("Errors:")
        for index, row in enumerate(results, 1):
            if row.get("error"):
                print(f"  [{index:02d}] {_redact(row['error'])[:300]}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PDF RAG evaluation.")
    parser.add_argument(
        "--output",
        default="results_improved.json",
        help="File name inside evals/. Does not replace results.json unless you pass that name.",
    )
    args = parser.parse_args()
    main(args.output)
