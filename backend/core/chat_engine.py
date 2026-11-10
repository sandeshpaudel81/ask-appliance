"""
Chat engine (LlamaIndex):
  condense follow-up question → retrieve top chunks → Gemini answers from manual excerpts.
"""
import os

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.google_genai import GoogleGenAI

from .ingestion import load_index

LLM_MODEL = "gemini-3.6-flash"
TOP_K = 4

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

# {context_str} is filled in by LlamaIndex with the retrieved chunks (incl. page/section)
CONTEXT_PROMPT = """MANUAL EXCERPTS FOR {appliance}:

{{context_str}}

Use ONLY the excerpts above (and the previous chat history) to answer the user."""

_llm = None


def get_llm() -> GoogleGenAI:
    """Lazily create the LLM so .env is loaded first."""
    global _llm
    if _llm is None:
        _llm = GoogleGenAI(
            model=LLM_MODEL,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )
    return _llm


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
    index = await load_index(appliance_id)

    # Braces in the appliance name would break the prompt template
    safe_name = appliance_name.replace("{", "(").replace("}", ")")

    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=TOP_K),
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT.replace("{appliance}", safe_name),
        context_prompt=CONTEXT_PROMPT.format(appliance=safe_name.upper()),
    )

    # Last 6 turns of history for continuity
    chat_history = [
        ChatMessage(
            role=MessageRole.ASSISTANT if turn["role"] == "assistant" else MessageRole.USER,
            content=turn["content"],
        )
        for turn in history[-6:]
    ]

    response = await engine.achat(user_message, chat_history=chat_history)
    answer = response.response or "I couldn't generate an answer. Please try again."

    # Source references for UI
    sources = [
        {
            "page": n.node.metadata.get("page"),
            "section": n.node.metadata.get("section"),
            "preview": n.node.get_content()[:160] + "…",
            "score": round(n.score or 0, 3),
        }
        for n in response.source_nodes
    ]

    return {"answer": answer, "sources": sources}
