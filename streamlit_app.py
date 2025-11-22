# streamlit_app.py
import streamlit as st
import time
import os
from typing import Optional, List

from rag_pipeline import RAGEngine
from ingestion import extract_text_from_file

st.set_page_config(page_title="RAG Engine UI", layout="wide")
st.title("Ask the RAG engine")

# -------------------------------------------
# Engine
# -------------------------------------------
def init_engine(api_key: Optional[str] = None):
    return RAGEngine(api_key=api_key)

# Sidebar options
st.sidebar.header("Options")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key (optional)",
    value=os.getenv("OPENAI_API_KEY") or "",
    type="password"
)
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

use_openai_extract = st.sidebar.checkbox("Use OpenAI for file extraction", value=True)
auto_ingest_uploaded = st.sidebar.checkbox("Auto-ingest uploaded files", value=True)

st.sidebar.write("Key provided:", "Yes" if (api_key_input or os.getenv("OPENAI_API_KEY")) else "No")

# init engine in session state
if "engine" not in st.session_state or st.session_state.get("engine_api_key") != api_key_input:
    st.session_state["engine"] = init_engine(api_key_input or None)
    st.session_state["engine_api_key"] = api_key_input or None

engine: RAGEngine = st.session_state["engine"]

# -------------------------------------------
# File uploader (uploads only; no default dataset)
# -------------------------------------------
st.sidebar.markdown("### Ingest files (uploads only)")
uploaded_files = st.sidebar.file_uploader(
    "Upload files to ingest (TXT, PDF, DOCX, XLSX)",
    type=["txt", "pdf", "docx", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.info(f"{len(uploaded_files)} file(s) selected")
    if st.sidebar.button("Ingest uploaded files now"):
        tmp_dir = "uploaded_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        extracted_texts: List[str] = []
        for f in uploaded_files:
            save_path = os.path.join(tmp_dir, f.name)
            with open(save_path, "wb") as fh:
                fh.write(f.getbuffer())

            # Extract text (OpenAI or local)
            try:
                text = extract_text_from_file(save_path, use_openai=use_openai_extract)
                if text and text.strip():
                    extracted_texts.append(text)
                else:
                    # show helpful debug info
                    st.sidebar.warning(f"No text extracted from {f.name}")
                    try:
                        size = os.path.getsize(save_path)
                        st.sidebar.caption(f"File size: {size} bytes")
                        with open(save_path, "rb") as fh:
                            preview = fh.read(200)
                        st.sidebar.text(f"File preview (bytes): {preview!r}")
                    except Exception:
                        pass
            except Exception as e:
                st.sidebar.error(f"Failed extracting {f.name}: {e}")

        if extracted_texts:
            try:
                engine.ingest(extracted_texts)
                doc_count = getattr(engine, "doc_count", None)
                if doc_count is not None:
                    st.sidebar.success(f"Ingested {len(extracted_texts)} file(s) — created ~{doc_count} chunks")
                else:
                    st.sidebar.success(f"Ingested {len(extracted_texts)} file(s).")
            except Exception as e:
                st.sidebar.error(f"Ingest failed: {e}")

# -------------------------------------------
# Query UI
# -------------------------------------------
st.header("Query the knowledge base")
query = st.text_area("Enter your question here", height=120)

if st.button("Get Answer"):
    if not query.strip():
        st.error("Please enter a question")
    else:
        start = time.time()
        try:
            answer, used_fallback, assembled_prompt, error = engine.query(query)
        except Exception as e:
            answer, used_fallback, assembled_prompt, error = None, True, None, repr(e)

        elapsed = time.time() - start

        st.subheader("Answer (final)")
        q_display = f"Question: {query}"

        if used_fallback and (not answer or (isinstance(answer, str) and answer.strip().startswith("Context:"))):
            a_display = "Answer: No generation from LLM (showing retrieved context)"
            if error:
                a_display += f" (Hint: {str(error)[:160]})"
        else:
            a_display = f"Answer: {answer}"

        st.code(f"{q_display}\n{a_display}")
        st.write(f"Response time: {elapsed:.2f}s")

        # Context expander
        with st.expander("Context (expand to view)", expanded=False):
            try:
                docs = engine.get_top_docs(query, k=5)
                if docs:
                    for i, (doc_text, score) in enumerate(docs, start=1):
                        st.markdown(f"--- doc {i} (score={score:.4f}) ---")
                        st.text(doc_text[:2000])
                else:
                    st.write("No context available. Upload and ingest files to populate the KB.")
            except Exception as e:
                st.write("Context not available. Error:", str(e))

# -------------------------------------------
# Engine reload
# -------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Reload engine (clear & re-init)"):
    try:
        del st.session_state["engine"]
    except Exception:
        pass
    st.session_state["engine"] = init_engine(api_key_input or None)
    st.session_state["engine_api_key"] = api_key_input or None
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

st.caption("Note: This UI ingests only uploaded files. No default dataset is auto-ingested.")
