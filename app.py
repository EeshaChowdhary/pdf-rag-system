import streamlit as st
import os
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdf, convert_pdf_to_images, is_scanned_pdf,
    get_file_info, chunk_text, embed_chunks
)
from ai_engine import ask_question_vision, ask_question_text, generate_summary_vision, generate_summary_text

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Reader AI",
    page_icon="≋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS — Ocean theme ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background — deep tide */
    .stApp { background-color: #061826; color: #dff5f1; }

    /* Sidebar — darker trench */
    [data-testid="stSidebar"] { background-color: #051520; border-right: 1px solid #123444; }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #0a2436;
        color: #dff5f1;
        border: 1px solid #123444;
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #14919b, #22b8b0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0e7a8c, #1aa39c);
        border: none;
    }

    /* Chat messages */
    .user-msg {
        background: #0d2f3f;
        border: 1px solid #22b8b044;
        border-radius: 14px 14px 4px 14px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        color: #dff5f1;
    }
    .ai-msg {
        background: #0a2130;
        border: 1px solid #123444;
        border-radius: 14px 14px 14px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 10%;
        color: #dff5f1;
    }
    .ai-label {
        font-size: 11px;
        color: #6a8fa3;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }

    /* Cards */
    .info-card {
        background: #0a2436;
        border: 1px solid #123444;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Page badge */
    .page-badge {
        display: inline-block;
        background: #22b8b022;
        color: #5fd4c5;
        border: 1px solid #22b8b044;
        border-radius: 10px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 6px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
if "text_by_page"     not in st.session_state: st.session_state.text_by_page     = {}
if "page_images"      not in st.session_state: st.session_state.page_images      = []
if "is_scanned"       not in st.session_state: st.session_state.is_scanned       = False
if "doc_name"         not in st.session_state: st.session_state.doc_name         = None
if "total_pages"      not in st.session_state: st.session_state.total_pages      = 0
if "summary"          not in st.session_state: st.session_state.summary          = None
if "active_tab"       not in st.session_state: st.session_state.active_tab       = "chat"
if "chunks"           not in st.session_state: st.session_state.chunks           = []
if "chunk_embeddings" not in st.session_state: st.session_state.chunk_embeddings = None

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ≋ PDF Reader AI")
    st.markdown("*Reads handwriting, diagrams & formulas*")
    st.divider()

    # Upload
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Works with handwritten, scanned or typed PDFs"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Processing PDF..."):
                # Extract text
                text_by_page, total_pages = extract_text_from_pdf(uploaded_file)
                uploaded_file.seek(0)

                # Check if scanned/handwritten
                scanned = is_scanned_pdf(text_by_page)

                # Convert to images for vision
                page_images = convert_pdf_to_images(uploaded_file)

                # Build the retrieval index once per document, not per question
                chunks = []
                if not scanned:
                    full_context = ""
                    for page_num, text in text_by_page.items():
                        if text.strip():
                            full_context += f"\n--- Page {page_num} ---\n{text}"
                    chunks = chunk_text(full_context, chunk_size=600)

                # Save to session
                st.session_state.text_by_page = text_by_page
                st.session_state.page_images  = page_images
                st.session_state.is_scanned   = scanned
                st.session_state.doc_name     = uploaded_file.name
                st.session_state.total_pages  = total_pages
                st.session_state.chat_history = []
                st.session_state.summary      = None
                st.session_state.chunks       = chunks

            if chunks:
                with st.spinner("Indexing document for retrieval..."):
                    st.session_state.chunk_embeddings = embed_chunks(chunks)
            else:
                st.session_state.chunk_embeddings = None

            st.success(f"✓ Loaded successfully!")

    # Doc info
    if st.session_state.doc_name:
        st.markdown("### 📄 Document")
        st.markdown(f"""
        <div class="info-card">
            <b style="color:#5fd4c5">{st.session_state.doc_name}</b><br/>
            <span style="color:#6a8fa3;font-size:12px">
                {st.session_state.total_pages} pages · 
                {"🖊 Handwritten/Scanned" if st.session_state.is_scanned else "📝 Digital Text"}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear & Upload New"):
            st.session_state.chat_history     = []
            st.session_state.text_by_page     = {}
            st.session_state.page_images      = []
            st.session_state.is_scanned       = False
            st.session_state.doc_name         = None
            st.session_state.total_pages      = 0
            st.session_state.summary          = None
            st.session_state.chunks           = []
            st.session_state.chunk_embeddings = None
            st.rerun()

    st.divider()

    # Quick prompts
    if st.session_state.doc_name:
        st.markdown("### 💡 Quick Questions")
        quick_prompts = [
            "Explain question 1 clearly",
            "Explain question 2 clearly",
            "Summarize all topics",
            "Explain the example step by step",
            "What are the key concepts?",
        ]
        for prompt in quick_prompts:
            if st.button(prompt, key=f"quick_{prompt}"):
                st.session_state.quick_input = prompt

    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#6a8fa3;line-height:1.8">
        ✓ Handwritten notes<br/>
        ✓ Scanned documents<br/>
        ✓ Math formulas<br/>
        ✓ Diagrams & tables<br/>
        ✓ Typed PDFs
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ─────────────────────────────────────────────────────────────
if not st.session_state.doc_name:
    # Landing page
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:64px;margin-bottom:16px;color:#22b8b0">≋</div>
        <h1 style="color:#5fd4c5;font-size:36px;margin-bottom:8px">PDF Reader AI</h1>
        <p style="color:#6a8fa3;font-size:16px;margin-bottom:40px">
            Upload any PDF — handwritten, scanned or typed.<br/>
            Ask questions and get instant AI-powered answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="info-card" style="text-align:center">
            <div style="font-size:28px">🖊</div>
            <b style="color:#5fd4c5">Handwriting</b>
            <p style="color:#6a8fa3;font-size:12px">Reads your written notes</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card" style="text-align:center">
            <div style="font-size:28px">📐</div>
            <b style="color:#5fd4c5">Diagrams</b>
            <p style="color:#6a8fa3;font-size:12px">Understands drawings & charts</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card" style="text-align:center">
            <div style="font-size:28px">∑</div>
            <b style="color:#5fd4c5">Math</b>
            <p style="color:#6a8fa3;font-size:12px">Reads formulas & equations</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="info-card" style="text-align:center">
            <div style="font-size:28px">📋</div>
            <b style="color:#5fd4c5">Tables</b>
            <p style="color:#6a8fa3;font-size:12px">Extracts tabular data</p>
        </div>""", unsafe_allow_html=True)

else:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Pages", "✦ Summary"])

    # ── TAB 1 CHAT ──
    with tab1:
        st.markdown(f"### Asking about: *{st.session_state.doc_name}*")

        # Chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-msg">
                    <div class="ai-label">≋ AI Tutor</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Input — a form so questions only fire on explicit submit, not on every rerun
        with st.form(key="ask_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                default_val = st.session_state.get("quick_input", "")
                user_input = st.text_input(
                    "Ask anything about your PDF",
                    value=default_val,
                    placeholder="e.g. Explain question 1 clearly...",
                    label_visibility="collapsed"
                )
            with col2:
                ask_btn = st.form_submit_button("Ask →", use_container_width=True)
            if "quick_input" in st.session_state:
                del st.session_state.quick_input

        if ask_btn and user_input.strip():
            with st.spinner("Reading your PDF..."):
                try:
                    if st.session_state.is_scanned or st.session_state.page_images:
                        answer = ask_question_vision(
                            user_input,
                            st.session_state.page_images,
                            st.session_state.chat_history
                        )
                    else:
                        answer = ask_question_text(
                            user_input,
                            st.session_state.text_by_page,
                            st.session_state.chat_history,
                            chunks=st.session_state.chunks,
                            chunk_embeddings=st.session_state.chunk_embeddings
                        )

                    st.session_state.chat_history.append({"role":"user",    "content": user_input})
                    st.session_state.chat_history.append({"role":"assistant","content": answer})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "API" in str(e) or "key" in str(e).lower():
                        st.info("Check your GEMINI_API_KEY in the .env file")

    # ── TAB 2 PAGES ──
    with tab2:
        st.markdown("### 📄 Document Pages")
        if st.session_state.page_images:
            cols_per_row = 3
            images = st.session_state.page_images
            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(images):
                        pg = images[i + j]
                        with col:
                            img_html = f'<img src="data:image/jpeg;base64,{pg["b64"]}" style="width:100%;border-radius:8px;border:1px solid #123444;"/>'
                            st.markdown(img_html, unsafe_allow_html=True)
                            st.caption(f"Page {pg['page_num']}")
                            if st.button(f"Ask about page {pg['page_num']}", key=f"pg_{pg['page_num']}"):
                                st.session_state.quick_input = f"Explain everything on page {pg['page_num']} in detail"
                                st.session_state.active_tab = "chat"
                                st.rerun()
        else:
            st.info("No page previews available — this PDF may not need PyMuPDF, or PyMuPDF failed to load. Check that `pymupdf` is installed.")

    # ── TAB 3 SUMMARY ──
    with tab3:
        st.markdown("### ✦ Document Summary")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            if st.button("↺ Regenerate Summary"):
                st.session_state.summary = None
                st.rerun()
        else:
            st.markdown("Click below to generate an AI summary of your document.")
            if st.button("✦ Generate Summary", use_container_width=True):
                with st.spinner("Reading and summarizing..."):
                    try:
                        if st.session_state.is_scanned or st.session_state.page_images:
                            summary = generate_summary_vision(st.session_state.page_images)
                        else:
                            summary = generate_summary_text(st.session_state.text_by_page)
                        st.session_state.summary = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")