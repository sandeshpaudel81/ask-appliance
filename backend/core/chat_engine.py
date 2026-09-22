"""
Chat engine: retrieves context → (optionally reranks) → calls GPT-4o Mini.
"""
import os
import json
from openai import AsyncOpenAI
from .ingestion import query_index

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert home appliance assistant. You help users with:
- How to USE their appliance correctly (modes, settings, temperatures, programs)
- TROUBLESHOOTING problems step by step
- SAFETY tips and warnings
- Maintenance and care advice
- Understanding technical specifications

You ONLY answer based on the provided manual excerpts. If the answer isn't in the excerpts, say:
"I couldn't find that in your {appliance} manual — please check the manual directly or contact the manufacturer."

Always be specific: mention exact settings, temperatures, spin speeds, or program names from the manual.
Format your answers clearly with numbered steps when describing a process.
Highlight safety warnings prominently."""

# Try to import flashrank for reranking; gracefully degrade if unavailable
try:
    from flashrank import Ranker, RerankRequest
    _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".cache/flashrank")
    RERANK_AVAILABLE = True
except Exception:
    RERANK_AVAILABLE = False


def _rerank(query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
    """Rerank retrieved chunks with FlashRank. Falls back to score order."""
    if not RERANK_AVAILABLE or not chunks:
        return chunks[:top_n]
    try:
        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
        request = RerankRequest(query=query, passages=passages)
        results = _ranker.rerank(request)
        top_ids = [r["id"] for r in results[:top_n]]
        return [chunks[i] for i in top_ids]
    except Exception:
        return chunks[:top_n]


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i} — Page {c.get('page','?')}, Section: {c.get('section','General')}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def chat(
    appliance_id: int,
    appliance_name: str,
    user_message: str,
    history: list[dict],
) -> dict:
    """
    Full RAG chat turn.
    Returns {"answer": str, "sources": list[dict]}
    """
    # 1. Retrieve
    raw_chunks = await query_index(appliance_id, user_message, top_k=8)

    # 2. Rerank
    top_chunks = _rerank(user_message, raw_chunks, top_n=4)

    # 3. Build messages
    context = _build_context(top_chunks)
    system = SYSTEM_PROMPT.replace("{appliance}", appliance_name)

    messages = [
        {"role": "system", "content": system},
        {
            "role": "system",
            "content": f"MANUAL EXCERPTS FOR {appliance_name.upper()}:\n\n{context}",
        },
    ]
    # Include last 6 turns of history for continuity
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # 4. Call LLM
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )
    answer = response.choices[0].message.content

    # 5. Build source references for UI
    sources = [
        {
            "page": c.get("page"),
            "section": c.get("section"),
            "preview": c["text"][:160] + "…",
            "score": round(c.get("score", 0), 3),
        }
        for c in top_chunks
    ]

    return {"answer": answer, "sources": sources}
