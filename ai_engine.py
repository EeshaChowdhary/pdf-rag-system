import base64
from google import genai
from google.genai import types
from utils import chunk_text, embed_chunks, retrieve_relevant_chunks

# gemini-3.1-flash-lite: stable (since May 2026), low-cost, generous free tier,
# no scheduled shutdown as of writing. Change this one line to switch models later.
MODEL_NAME = "gemini-3.1-flash-lite"
LAST_USAGE = None

PDF_SYSTEM_RULES = """Answer only from the evidence supplied in this request.
- Cite the PDF page number for every fact taken from the document.
- Cite the external title and URL for every fact taken from web results.
- Keep PDF evidence and external evidence separate.
- If the PDF and an external source disagree, state both sides and do not declare a winner.
- If the evidence does not contain the answer, say that it is not in the supplied evidence.
- Do not add facts from general knowledge and do not present them as verified."""


def usage_from_response(response):
    """Token counts when the API returns them. Never includes credentials."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "total_tokens": getattr(meta, "total_token_count", None),
    }


def _remember_usage(response):
    global LAST_USAGE
    LAST_USAGE = usage_from_response(response)
    return response


def get_gemini_client():
    """Initialize Gemini client"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    return genai.Client(api_key=api_key)


def _history_to_contents(chat_history):
    """Convert {'role': 'user'/'assistant', 'content': ...} history into
    Gemini's Content objects. Gemini uses 'model' where we used 'assistant'."""
    contents = []
    for msg in chat_history[-6:]:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    return contents


def ask_question_text(question, text_by_page, chat_history=None, chunks=None, chunk_embeddings=None):
    """For typed/digital PDFs — retrieves the chunks most relevant to the
    question via embedding similarity, instead of always sending the first N chunks."""
    if chat_history is None:
        chat_history = []
    client = get_gemini_client()

    if chunks is None:
        full_context = ""
        for page_num, text in text_by_page.items():
            if text.strip():
                full_context += f"\n--- Page {page_num} ---\n{text}"
        chunks = chunk_text(full_context, chunk_size=600)

    if chunk_embeddings is None and chunks:
        chunk_embeddings = embed_chunks(chunks)

    relevant_chunks = retrieve_relevant_chunks(question, chunks, chunk_embeddings, top_k=5)
    context = "\n\n".join(relevant_chunks)

    system = f"""You are an expert AI tutor and document analyst.
{PDF_SYSTEM_RULES}

PDF EXCERPTS:
{context}"""

    contents = _history_to_contents(chat_history)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1500),
    ))
    return response.text


def spread_page_images(page_images, limit=5):
    """Pages spread across the file, used only when retrieval did not choose any."""
    if not page_images or limit <= 0:
        return []
    if len(page_images) <= limit:
        return list(page_images)
    indexes = [
        round(i * (len(page_images) - 1) / (limit - 1))
        for i in range(limit)
    ]
    selected = []
    for index in indexes:
        image = page_images[index]
        if image not in selected:
            selected.append(image)
    return selected


def ask_question_vision(question, page_images, chat_history=None, pages_to_send=None):
    """For scanned/handwritten PDFs — uses Gemini's native vision to read images.

    pages_to_send must be the pages retrieval selected. When it is omitted,
    pages are spread across the document instead of taking only the first four.
    """
    if chat_history is None:
        chat_history = []
    client = get_gemini_client()

    if pages_to_send:
        selected = [p for p in page_images if p["page_num"] in pages_to_send]
    else:
        selected = spread_page_images(page_images, limit=5)

    parts = []
    for pg in selected:
        parts.append(types.Part.from_text(text=f"--- Page {pg['page_num']} ---"))
        parts.append(types.Part.from_bytes(data=base64.b64decode(pg["b64"]), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=question))

    system = f"""You are an expert AI tutor. You can read handwritten notes,
typed documents, diagrams, tables, and math formulas from images.
{PDF_SYSTEM_RULES}"""

    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1500),
    ))
    return response.text


def generate_summary_vision(page_images, pages_to_send=None):
    """Generate a document summary using Vision on selected pages."""
    client = get_gemini_client()
    if pages_to_send:
        selected = [p for p in page_images if p["page_num"] in pages_to_send]
    else:
        selected = spread_page_images(page_images, limit=5)
    parts = []
    for pg in selected:
        parts.append(types.Part.from_text(text=f"--- Page {pg['page_num']} ---"))
        parts.append(types.Part.from_bytes(data=base64.b64decode(pg["b64"]), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(
        text="List all questions and topics in this document. For each one give a brief 2-line summary."
    ))

    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction="You are an expert document analyst. Summarize only what is visible on the supplied pages. Cite page numbers.",
            max_output_tokens=2000,
        ),
    ))
    return response.text


def generate_from_system(question, system, chat_history=None):
    """Answer from a caller-built evidence prompt. The prompt must not include a gold answer."""
    if chat_history is None:
        chat_history = []
    client = get_gemini_client()
    contents = _history_to_contents(chat_history)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1500),
    ))
    return response.text


def generate_summary_text(text_by_page):
    """Generate summary for typed PDFs"""
    client = get_gemini_client()
    full_text = ""
    for page_num, text in text_by_page.items():
        if text.strip():
            full_text += f"\nPage {page_num}: {text}"
    full_text = full_text[:6000]

    response = _remember_usage(client.models.generate_content(
        model=MODEL_NAME,
        contents=f"Summarize this document:\n{full_text}",
        config=types.GenerateContentConfig(
            system_instruction="You are an expert document analyst. Summarize only the supplied text and cite page numbers. Use ## headers and bullet points.",
            max_output_tokens=1500,
        ),
    ))
    return response.text
