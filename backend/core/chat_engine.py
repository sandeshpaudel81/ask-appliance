"""
Chat engine: retrieves context → calls Gemini.
"""
from google.genai import types
from .ingestion import query_index, get_client

LLM_MODEL = "gemini-3.6-flash"

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
    # 1. Retrieve (same as notebook: top 4 by cosine similarity)
    top_chunks = await query_index(appliance_id, user_message, top_k=4)

    # 2. Build prompt
    context = _build_context(top_chunks)
    system = SYSTEM_PROMPT.replace("{appliance}", appliance_name)

    system_instruction = (
        f"{system}\n\nMANUAL EXCERPTS FOR {appliance_name.upper()}:\n\n{context}"
    )

    # Include last 6 turns of history for continuity (Gemini uses "model" for assistant)
    contents = []
    for turn in history[-6:]:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    # 3. Call LLM
    response = await get_client().aio.models.generate_content(
        model=LLM_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    answer = response.text or "I couldn't generate an answer. Please try again."

    # 4. Build source references for UI
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
