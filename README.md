# ≋ PDF Reader AI — Vision-Based RAG System

An AI-powered PDF assistant that reads **typed, scanned, and handwritten** documents — including math formulas, diagrams, and tables — and answers questions about them in a chat interface, built with Streamlit and Gemini.

Most "PDF RAG" projects only handle clean, digital text. This one is built to also handle the messier real-world case: scanned pages and handwritten notes, where standard text extraction fails completely.

## Features

- **Dual-mode document understanding**
  - **Digital PDFs** → text is extracted and chunked, then the most relevant chunks are retrieved for each question using embedding-based semantic search — not just the first few chunks in the document.
  - **Scanned / handwritten PDFs** → pages are rendered to images (PyMuPDF) and read directly by a vision-capable LLM, so handwriting, diagrams, and math formulas that have no extractable text are still understood.

- **Real retrieval, not context-stuffing** — document chunks are embedded locally with [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`, CPU-only, no API key required) and ranked by cosine similarity to the question before being sent to the model.

- **Chat interface** with conversation history, quick-prompt suggestions, and a per-page preview tab.

- **Auto-generated document summaries** for both digital and scanned PDFs.

- **Local-first** — runs entirely on your machine via Streamlit; only the question/document text and image content are sent to the LLM API.

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 (text), PyMuPDF (page → image rendering) |
| Retrieval | FastEmbed (local embeddings) + cosine similarity |
| LLM | Google Gemini (`gemini-3.1-flash-lite`) — text + native vision |
| Config | python-dotenv |

## How It Works

1. **Upload** a PDF. The app checks whether it contains extractable text.
2. **Digital PDFs**: text is split into overlapping chunks, embedded once with FastEmbed, and indexed in memory.
3. **Scanned/handwritten PDFs**: each page is rendered to a JPEG image via PyMuPDF at 2x zoom for clarity.
4. **Ask a question**:
   - Digital path → the question is embedded, compared against every chunk's embedding, and the top-5 most relevant chunks are sent to the model as context.
   - Scanned path → the relevant page images are sent directly to Gemini's vision model, which reads the handwriting/diagrams/formulas natively.
5. The model responds with an answer that cites the page(s) it drew from.

## Setup

```bash
git clone <your-repo-url>
cd pdf-rag-system
python -m pip install -r requirements.txt