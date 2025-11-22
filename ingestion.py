# ingestion.py
"""
Robust ingestion helpers.

Behavior:
 - Prefer OpenAI Responses file extraction when OPENAI_API_KEY is present and use_openai=True.
 - Fallback to local extractors:
     - PDF: PyMuPDF (fitz) primary, pdfminer.six fallback
     - DOCX: dependency-free XML extractor (no python-docx required)
     - XLSX: pandas + openpyxl fallback
     - TXT: plain read
 - Provides extract_text_from_file(path, use_openai=True) and load_texts_from_folder(folder, ext=None, use_openai=True)
"""

import os
from typing import List, Optional

# --- Optional third-party libs (lazy imports) ---
_pymupdf_available = False
_pdfminer_available = False
_pandas_available = False
_openpyxl_available = False

try:
    import fitz  # PyMuPDF
    _pymupdf_available = True
except Exception:
    fitz = None

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract_text
    _pdfminer_available = True
except Exception:
    _pdfminer_extract_text = None

try:
    import pandas as pd
    _pandas_available = True
except Exception:
    pd = None

try:
    import openpyxl  # noqa: F401
    _openpyxl_available = True
except Exception:
    openpyxl = None

# OpenAI client (optional)
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAIClient = None

# -------------------------
# OpenAI extraction helper
# -------------------------
def _extract_via_openai(path: str, model: str = "gpt-4o-mini") -> str:
    """
    Upload file to OpenAI (purpose='assistants') and ask the Responses API to extract text.
    Returns extracted text or raises RuntimeError on unrecoverable failure.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI client not installed (`openai` package).")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAIClient(api_key=api_key)

    # Upload file (use allowed purpose "assistants")
    try:
        with open(path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="assistants")
    except Exception as e:
        raise RuntimeError(f"OpenAI file upload failed: {e}")

    # get file id robustly
    file_id = getattr(uploaded, "id", None)
    if not file_id and isinstance(uploaded, dict):
        file_id = uploaded.get("id")

    if not file_id:
        # best effort: try other attr names
        try:
            file_id = uploaded["id"]
        except Exception:
            file_id = None

    if not file_id:
        raise RuntimeError("OpenAI file upload returned no file id.")

    # Call Responses API to extract text
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract all readable text and tables from the provided file. Return plain text only."},
                        {"type": "input_file", "file_id": file_id}
                    ],
                }
            ],
        )
    except Exception as e:
        # Try cleanup then bubble up
        try:
            if hasattr(client.files, "delete"):
                try:
                    client.files.delete(id=file_id)
                except Exception:
                    pass
        except Exception:
            pass
        raise RuntimeError(f"OpenAI responses.create failed: {e}")

    # Parse common SDK shapes to obtain plain text
    extracted = None
    try:
        extracted = getattr(resp, "output_text", None)
    except Exception:
        extracted = None

    if not extracted:
        out = None
        try:
            out = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
        except Exception:
            out = None

        if isinstance(out, list) and out:
            first = out[0]
            content = None
            if isinstance(first, dict):
                content = first.get("content")
            else:
                content = getattr(first, "content", None)
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        parts.append(c["text"])
                    elif hasattr(c, "text"):
                        parts.append(getattr(c, "text"))
                if parts:
                    extracted = "\n".join(parts)

    if not extracted:
        try:
            extracted = str(resp)
        except Exception:
            extracted = ""

    # Best-effort cleanup of uploaded file on OpenAI side
    try:
        if hasattr(client.files, "delete"):
            try:
                client.files.delete(id=file_id)
            except Exception:
                pass
    except Exception:
        pass

    return extracted or ""

# -------------------------
# Local extractors
# -------------------------
def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:
        try:
            with open(path, "rb") as fh:
                return fh.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

def extract_text_from_pdf_local(path: str) -> str:
    # PyMuPDF primary
    if _pymupdf_available:
        try:
            doc = fitz.open(path)
            parts = []
            for page in doc:
                txt = page.get_text("text")
                if txt and txt.strip():
                    parts.append(txt)
            doc.close()
            full = "\n".join(parts)
            if full and full.strip():
                return full
        except Exception:
            pass

    # pdfminer fallback
    if _pdfminer_available:
        try:
            txt = _pdfminer_extract_text(path)
            if txt and txt.strip():
                return txt
        except Exception:
            pass

    return ""

def extract_text_from_docx_local(path: str) -> str:
    """
    Dependency-free docx extractor:
    - opens the .docx zip and reads word/document.xml
    - extracts <w:t> nodes and joins them.
    Returns '' on any failure.
    """
    try:
        import zipfile
        import xml.etree.ElementTree as ET
    except Exception:
        return ""

    try:
        with zipfile.ZipFile(path, 'r') as z:
            if "word/document.xml" not in z.namelist():
                return ""
            raw = z.read("word/document.xml")
            root = ET.fromstring(raw)
            # Word namespace
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            parts = []
            for node in root.findall(".//w:t", ns):
                if node.text:
                    parts.append(node.text)
            text = " ".join(parts)
            text = " ".join(text.split())
            return text
    except Exception:
        return ""

def extract_text_from_xlsx_local(path: str) -> str:
    if not (_pandas_available and _openpyxl_available):
        raise RuntimeError("pandas/openpyxl not available for xlsx extraction.")
    out = []
    try:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        for sheet, df in xls.items():
            out.append(f"Sheet: {sheet}")
            for row in df.fillna("").astype(str).values:
                out.append(" | ".join(cell for cell in row))
        return "\n".join(out)
    except Exception as e:
        raise RuntimeError(f"xlsx extraction error: {e}")

# -------------------------
# Top-level convenience
# -------------------------
def extract_text_from_file(path: str, use_openai: bool = True, openai_model: str = "gpt-4o-mini") -> str:
    """
    Extract text from a local file path.
    - use_openai: if True and OPENAI_API_KEY present + openai package available, will attempt OpenAI extraction first.
    """
    if use_openai and _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            txt = _extract_via_openai(path, model=openai_model)
            if txt and txt.strip():
                return txt
        except Exception as e:
            # log to stdout and fall back to local extraction
            print(f"[ingestion] OpenAI extraction failed for {path}: {e}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    if ext == ".pdf":
        try:
            return extract_text_from_pdf_local(path)
        except Exception as e:
            print(f"[ingestion] pdf extraction failed: {e}")
            return ""
    if ext == ".docx":
        try:
            return extract_text_from_docx_local(path)
        except Exception as e:
            print(f"[ingestion] docx extraction failed: {e}")
            return ""
    if ext in (".xlsx", ".xls"):
        try:
            return extract_text_from_xlsx_local(path)
        except Exception as e:
            print(f"[ingestion] xlsx extraction failed: {e}")
            return ""

    # unknown: best-effort binary->text decode
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def load_texts_from_folder(folder: str, ext: Optional[str] = None, use_openai: bool = True) -> List[str]:
    """
    Walk folder and extract texts. Optional ext filter to keep backward compatibility.
    """
    texts: List[str] = []
    if not os.path.exists(folder):
        return texts
    for root, _, files in os.walk(folder):
        for fn in files:
            if ext and not fn.lower().endswith(ext.lower()):
                continue
            path = os.path.join(root, fn)
            try:
                txt = extract_text_from_file(path, use_openai=use_openai)
                if txt and txt.strip():
                    texts.append(txt)
            except Exception as e:
                print(f"[ingestion] failed to extract {path}: {e}")
    return texts
