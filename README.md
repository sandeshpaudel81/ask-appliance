# ApplianceAI

A domain-specific AI assistant that answers questions about home appliances by reasoning over their official PDF manuals.

## Stack

| Layer | Tool |
|-------|------|
| LLM | GPT-4o Mini |
| Embeddings | OpenAI text-embedding-3-small |
| PDF text | PyMuPDF (fitz) |
| PDF tables | pdfplumber |
| Vector search | FAISS (one index per appliance) |
| Reranker | FlashRank ms-marco-MiniLM-L-12-v2 |
| Backend | FastAPI + SQLAlchemy async |
| Database | SQLite (aiosqlite) |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |

## Project structure

```
appliance-ai/
├── main.py                        # FastAPI entry point
├── requirements.txt
├── .env.example
├── backend/
│   ├── core/
│   │   ├── ingestion.py           # PDF parse → chunk → embed → FAISS
│   │   └── chat_engine.py         # RAG query → rerank → GPT-4o Mini
│   ├── models/
│   │   └── database.py            # SQLAlchemy models (Appliance, ChatMessage)
│   └── routers/
│       ├── appliances.py          # POST /api/appliances/, GET, DELETE
│       └── chat.py                # POST /api/chat/, GET history, DELETE history
├── frontend/
│   └── index.html                 # Full single-page app
├── indexes/                       # FAISS indexes (auto-created, gitignore)
│   └── {appliance_id}/
│       ├── index.faiss
│       └── chunks.json
└── data/                          # SQLite DB (auto-created, gitignore)
    └── appliance_ai.db
```

## Setup

```bash
# 1. Clone / enter directory
cd appliance-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your key

# 5. Run
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/appliances/ | List all appliances |
| POST | /api/appliances/ | Upload manual + ingest |
| DELETE | /api/appliances/{id} | Remove appliance + index |
| POST | /api/chat/ | Send a chat message |
| GET | /api/chat/history/{id} | Load chat history |
| DELETE | /api/chat/history/{id} | Clear chat history |
| GET | /api/health | Health check |

## Ingestion pipeline

```
PDF bytes
  ↓ PyMuPDF       → text per page
  ↓ pdfplumber    → tables as markdown
  ↓ Chunker       → sliding window (512 tokens, 64 overlap), section-aware
  ↓ OpenAI embed  → 1536-dim vectors (text-embedding-3-small)
  ↓ FAISS         → IndexFlatIP (cosine after L2 normalisation)
  ↓ Persist       → indexes/{id}/index.faiss + chunks.json
```

## RAG chat pipeline

```
User query
  ↓ Embed query (same model)
  ↓ FAISS search  → top 8 chunks
  ↓ FlashRank     → rerank to top 4
  ↓ Build prompt  → system + manual excerpts + last 6 turns
  ↓ GPT-4o Mini   → grounded answer
  ↓ Return        → answer + source references (page, section, preview)
```

## Key design decisions

- **One FAISS index per appliance** — allows clean delete/update without rebuilding everything
- **Section-aware chunking** — detects numbered headings, WARNING/SAFETY blocks
- **FlashRank reranker** — free local model, dramatically improves precision over raw cosine similarity
- **Source chips in UI** — every answer shows which page/section it came from; hover for preview
- **SQLite** — zero-config, perfect for single-user local tool; swap to Postgres + pgvector for multi-user

## Portfolio notes

Things that make this stand out:
1. Real structured reasoning — not just a chatbot, it cites manual sections
2. Per-appliance isolation — correct multi-document RAG pattern
3. Table extraction — handles spec tables that break most PDF parsers
4. Reranking — shows awareness of production RAG patterns
5. Source attribution UI — builds trust and verifiability
