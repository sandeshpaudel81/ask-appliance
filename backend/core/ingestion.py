"""
Ingestion pipeline (mirrors ingestion.ipynb):
  1. Extract text (pymupdf) + tables (pdfplumber)
  2. Clean pages + section-aware sliding-window chunking
  3. Embed with Gemini gemini-embedding-001
  4. Store per-appliance FAISS index
"""
import os
import re
import json
import tempfile
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
import faiss
import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072

CHUNK_SIZE = 500        # target words per chunk
CHUNK_OVERLAP = 60      # words of overlap between consecutive chunks
MIN_CHUNK_WORDS = 25    # discard chunks shorter than this
EMBED_BATCH_SIZE = 20
INDEXES_DIR = Path("indexes")

_client = None


def get_client() -> genai.Client:
    """Lazily create the Gemini client so .env is loaded first."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


# Regex to detect section headings
_HEADING_RE = re.compile(
    r"(?m)^("
    r"\d+[\.\d]*\s+[A-Z][A-Za-z]"                        # "1.2 Safety Instructions"
    r"|[A-Z]{4,}[\sA-Z]*[!:.]?\s*$"                      # "MAINTENANCE" / "MAINTENANCE:"
    r"|(?:WARNING|CAUTION|NOTE|IMPORTANT|SAFETY)[!:.]?"  # "WARNING!" / "IMPORTANT:"
    r")",
)


def _extract_text_pymupdf(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text", sort=True)
        if len(raw.strip()) < 30:   # skip blank pages
            continue
        pages.append({"page": page_num, "text": raw})
    doc.close()
    return pages


def _extract_tables_pdfplumber(pdf_bytes: bytes) -> list[dict]:
    tables = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for tbl in page.extract_tables():
                    if not tbl:
                        continue
                    rows = []
                    for row in tbl:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        row_text = " | ".join(cleaned)
                        if row_text.strip(" |"):
                            rows.append(row_text)
                    if not rows:
                        continue
                    table_text = f"[TABLE - page {page_num}]\n" + "\n".join(rows)
                    tables.append({"page": page_num, "text": table_text})
    finally:
        os.unlink(tmp_path)

    return tables


def _clean_page(raw_text: str) -> tuple[str, str]:
    kept = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d{1,3}", stripped):   # page numbers
            continue
        if len(stripped) < 4:
            continue
        kept.append(stripped)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    section = "General"
    for match in _HEADING_RE.finditer(cleaned):
        section = match.group(0).strip()[:60]
    return cleaned, section


def _build_chunks(pages: list[dict], tables: list[dict], appliance_name: str) -> list[dict]:
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_data in pages:
        cleaned, section = _clean_page(page_data["text"])
        words = cleaned.split()
        if len(words) < MIN_CHUNK_WORDS:
            continue
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if len(chunk_words) < MIN_CHUNK_WORDS:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_data["page"],
                "section": section,
                "type": "text",
                "appliance": appliance_name,
            })

    for table_data in tables:
        chunks.append({
            "text": table_data["text"],
            "page": table_data["page"],
            "section": "Tables / Specifications",
            "type": "table",
            "appliance": appliance_name,
        })

    return chunks


async def _embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    client = get_client()
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_vecs.extend(e.values for e in result.embeddings)
    return np.array(all_vecs, dtype="float32")


async def ingest_pdf(
    pdf_bytes: bytes,
    appliance_id: int,
    appliance_name: str,
) -> dict:
    """
    Full ingestion pipeline.
    Returns {"chunk_count": int, "index_path": str}
    """
    pages = _extract_text_pymupdf(pdf_bytes)
    tables = _extract_tables_pdfplumber(pdf_bytes)
    all_chunks = _build_chunks(pages, tables, appliance_name)

    if not all_chunks:
        raise ValueError("No extractable content found in PDF.")

    embeddings = await _embed_texts([c["text"] for c in all_chunks], "RETRIEVAL_DOCUMENT")

    # Normalise → cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    index_dir = INDEXES_DIR / str(appliance_id)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return {
        "chunk_count": index.ntotal,
        "index_path": str(index_dir),
    }


async def query_index(
    appliance_id: int,
    query: str,
    top_k: int = 4,
) -> list[dict]:
    """
    Retrieve top-k chunks for a query from an appliance's FAISS index.
    Returns list of {text, page, section, score}.
    """
    index_dir = INDEXES_DIR / str(appliance_id)
    if not index_dir.exists():
        raise FileNotFoundError(f"No index for appliance {appliance_id}")

    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    q_vec = await _embed_texts([query], "RETRIEVAL_QUERY")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results
