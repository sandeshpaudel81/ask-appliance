"""
Ingestion pipeline (LlamaIndex):
  1. Extract text (pymupdf) + tables (pdfplumber)
  2. Clean pages, detect section headings
  3. Chunk with LlamaIndex SentenceSplitter (tables kept whole)
  4. Embed with Gemini gemini-embedding-001
  5. Persist a per-appliance LlamaIndex VectorStoreIndex
"""
import asyncio
import os
import re
import tempfile
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

EMBED_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 64      # tokens of overlap between consecutive chunks
MIN_CHUNK_WORDS = 25    # discard chunks shorter than this
EMBED_BATCH_SIZE = 20
INDEXES_DIR = Path("indexes")

# Metadata kept on every node but hidden from the embedding / LLM text
_HIDDEN_METADATA = ["type", "appliance"]

_embed_model = None


def get_embed_model() -> GoogleGenAIEmbedding:
    """Lazily create the embedding model so .env is loaded first."""
    global _embed_model
    if _embed_model is None:
        _embed_model = GoogleGenAIEmbedding(
            model_name=EMBED_MODEL,
            api_key=os.getenv("GEMINI_API_KEY"),
            embed_batch_size=EMBED_BATCH_SIZE,
        )
    return _embed_model


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


def _build_nodes(pages: list[dict], tables: list[dict], appliance_name: str) -> list[TextNode]:
    # Text pages → Documents → SentenceSplitter nodes
    documents = []
    for page_data in pages:
        cleaned, section = _clean_page(page_data["text"])
        if len(cleaned.split()) < MIN_CHUNK_WORDS:
            continue
        documents.append(Document(
            text=cleaned,
            metadata={
                "page": page_data["page"],
                "section": section,
                "type": "text",
                "appliance": appliance_name,
            },
            excluded_embed_metadata_keys=_HIDDEN_METADATA,
            excluded_llm_metadata_keys=_HIDDEN_METADATA,
        ))

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = [
        n for n in splitter.get_nodes_from_documents(documents)
        if len(n.get_content().split()) >= MIN_CHUNK_WORDS
    ]

    # Tables are kept whole so rows aren't split apart
    for table_data in tables:
        nodes.append(TextNode(
            text=table_data["text"],
            metadata={
                "page": table_data["page"],
                "section": "Tables / Specifications",
                "type": "table",
                "appliance": appliance_name,
            },
            excluded_embed_metadata_keys=_HIDDEN_METADATA,
            excluded_llm_metadata_keys=_HIDDEN_METADATA,
        ))

    return nodes


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
    nodes = _build_nodes(pages, tables, appliance_name)

    if not nodes:
        raise ValueError("No extractable content found in PDF.")

    # Embed asynchronously, then build the index from pre-embedded nodes
    embed_model = get_embed_model()
    texts = [n.get_content(metadata_mode="embed") for n in nodes]
    embeddings = await embed_model.aget_text_embedding_batch(texts)
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb

    index = VectorStoreIndex(nodes, embed_model=embed_model)

    index_dir = INDEXES_DIR / str(appliance_id)
    index_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(index.storage_context.persist, persist_dir=str(index_dir))

    return {
        "chunk_count": len(nodes),
        "index_path": str(index_dir),
    }


async def load_index(appliance_id: int) -> VectorStoreIndex:
    """Load an appliance's persisted LlamaIndex index."""
    index_dir = INDEXES_DIR / str(appliance_id)
    if not (index_dir / "docstore.json").exists():
        raise FileNotFoundError(f"No index for appliance {appliance_id}")

    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    return await asyncio.to_thread(
        load_index_from_storage, storage_context, embed_model=get_embed_model()
    )
