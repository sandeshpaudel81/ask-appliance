"""
Ingestion pipeline:
  1. Extract text (pymupdf) + tables (pdfplumber)
  2. Chunk with LlamaIndex SentenceSplitter — section-aware
  3. Embed with OpenAI text-embedding-3-small
  4. Store per-appliance FAISS index
"""
import os
import re
import json
import tempfile
from pathlib import Path
from typing import IO

import fitz  # pymupdf
import pdfplumber
from openai import AsyncOpenAI
import faiss
import numpy as np

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
INDEXES_DIR = Path("indexes")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Section header detection (matches numbered headings + ALL CAPS headings) ──
_SECTION_RE = re.compile(
    r"^(\d+[\.\d]*\s+[A-Z]|[A-Z]{4,}[\s\w]*$|(?:SAFETY|WARNING|CAUTION|IMPORTANT|NOTE)\b)",
    re.MULTILINE,
)


def _extract_text_pymupdf(pdf_bytes: bytes) -> list[dict]:
    """Extract text blocks with page numbers."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def _extract_tables_pdfplumber(pdf_bytes: bytes) -> list[dict]:
    """Extract tables as markdown-like strings with page numbers."""
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
                        cleaned = [str(c).strip() if c else "" for c in row]
                        rows.append(" | ".join(cleaned))
                    table_text = "\n".join(rows)
                    tables.append({"page": page_num, "text": f"[TABLE]\n{table_text}\n[/TABLE]"})
    finally:
        os.unlink(tmp_path)

    return tables


def _chunk_text(text: str, page: int, appliance_name: str) -> list[dict]:
    """
    Split text into overlapping chunks, preserving section context.
    Returns list of {text, page, section} dicts.
    """
    chunks = []
    words = text.split()
    current_section = "General"

    # Detect section boundaries
    for match in _SECTION_RE.finditer(text):
        current_section = match.group(0).strip()[:60]

    # Simple sliding window — production would use LlamaIndex SentenceSplitter
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i : i + CHUNK_SIZE]
        if len(chunk_words) < 20:  # skip tiny tail chunks
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "page": page,
            "section": current_section,
            "appliance": appliance_name,
        })
    return chunks


async def _embed_texts(texts: list[str]) -> np.ndarray:
    """Batch embed texts, respecting rate limits with batching."""
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [e.embedding for e in response.data]
        all_embeddings.extend(vecs)
    return np.array(all_embeddings, dtype="float32")


async def ingest_pdf(
    pdf_bytes: bytes,
    appliance_id: int,
    appliance_name: str,
) -> dict:
    """
    Full ingestion pipeline.
    Returns {"chunk_count": int, "index_path": str}
    """
    # 1. Extract
    pages = _extract_text_pymupdf(pdf_bytes)
    tables = _extract_tables_pdfplumber(pdf_bytes)

    # 2. Merge & chunk
    all_chunks: list[dict] = []
    for page_data in pages:
        page_chunks = _chunk_text(page_data["text"], page_data["page"], appliance_name)
        all_chunks.extend(page_chunks)
    for table_data in tables:
        all_chunks.append({
            "text": table_data["text"],
            "page": table_data["page"],
            "section": "Tables",
            "appliance": appliance_name,
        })

    if not all_chunks:
        raise ValueError("No extractable content found in PDF.")

    # 3. Embed
    texts = [c["text"] for c in all_chunks]
    embeddings = await _embed_texts(texts)

    # 4. Build FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product (cosine after normalisation)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 5. Persist index + metadata
    INDEXES_DIR.mkdir(exist_ok=True)
    index_dir = INDEXES_DIR / str(appliance_id)
    index_dir.mkdir(exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    return {
        "chunk_count": len(all_chunks),
        "index_path": str(index_dir),
    }


async def query_index(
    appliance_id: int,
    query: str,
    top_k: int = 6,
) -> list[dict]:
    """
    Retrieve top-k chunks for a query from an appliance's FAISS index.
    Returns list of {text, page, section, score}.
    """
    index_dir = INDEXES_DIR / str(appliance_id)
    if not index_dir.exists():
        raise FileNotFoundError(f"No index for appliance {appliance_id}")

    # Load
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)

    # Embed query
    response = await client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_vec)

    # Search
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append({**chunk, "score": float(score)})

    return results
