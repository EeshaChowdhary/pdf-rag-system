import PyPDF2
import io
from PIL import Image
import base64
import numpy as np
from fastembed import TextEmbedding

_embedder = None


def get_embedder():
    """Lazy-load the embedding model once per process (CPU-only, no API key needed)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file - works for typed/digital PDFs"""
    text_by_page = {}
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_by_page[page_num + 1] = text if text.strip() else ""
        return text_by_page, total_pages
    except Exception:
        return {}, 0


def convert_pdf_to_images(uploaded_file):
    """Convert each PDF page to base64 image for Vision AI"""
    try:
        import fitz  # PyMuPDF
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpeg")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append({"page_num": page_num + 1, "b64": b64})
        doc.close()
        return images
    except ImportError:
        return []
    except Exception:
        return []


def is_scanned_pdf(text_by_page):
    """Check if PDF is scanned/handwritten (no extractable text)"""
    total_text = "".join(text_by_page.values())
    return len(total_text.strip()) < 100


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for retrieval"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks):
    """Embed a list of text chunks into vectors for semantic search.
    Returns an (n_chunks, dim) numpy array, or an empty array if no chunks."""
    if not chunks:
        return np.array([])
    embedder = get_embedder()
    return np.array(list(embedder.embed(chunks)))


def retrieve_relevant_chunks(query, chunks, chunk_embeddings, top_k=5):
    """Return the top_k chunks most semantically similar to the query,
    ranked by cosine similarity — this is the actual 'R' in RAG."""
    if not chunks:
        return []
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return chunks[:top_k]

    embedder = get_embedder()
    query_vec = np.array(list(embedder.embed([query])))[0]

    denom = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_vec)
    denom[denom == 0] = 1e-10
    scores = (chunk_embeddings @ query_vec) / denom

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def get_file_info(uploaded_file):
    """Get basic info about the uploaded file"""
    return {
        "name": uploaded_file.name,
        "size": f"{uploaded_file.size / 1024:.1f} KB"
    }
