from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import json

from ..models.database import get_db, Appliance, ChatMessage
from ..core.chat_engine import chat

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    appliance_id: int
    message: str


class SourceRef(BaseModel):
    page: Optional[int]
    section: Optional[str]
    preview: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Load appliance
    result = await db.execute(select(Appliance).where(Appliance.id == req.appliance_id))
    appliance = result.scalar_one_or_none()
    if not appliance:
        raise HTTPException(404, "Appliance not found.")
    if not appliance.index_path:
        raise HTTPException(400, "No manual indexed for this appliance yet.")

    # Load recent chat history
    hist_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.appliance_id == req.appliance_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(12)
    )
    raw_history = list(reversed(hist_result.scalars().all()))
    history = [{"role": m.role, "content": m.content} for m in raw_history]

    # Call RAG engine
    try:
        response = await chat(appliance.id, appliance.name, req.message, history)
    except FileNotFoundError:
        raise HTTPException(500, "Index not found. Please re-upload the manual.")
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")

    # Persist messages
    user_msg = ChatMessage(appliance_id=req.appliance_id, role="user", content=req.message)
    asst_msg = ChatMessage(
        appliance_id=req.appliance_id,
        role="assistant",
        content=response["answer"],
        sources=json.dumps(response["sources"]),
    )
    db.add(user_msg)
    db.add(asst_msg)
    await db.commit()

    return ChatResponse(answer=response["answer"], sources=response["sources"])


@router.get("/history/{appliance_id}")
async def get_history(appliance_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.appliance_id == appliance_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    return [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]


@router.delete("/history/{appliance_id}")
async def clear_history(appliance_id: int, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    await db.execute(delete(ChatMessage).where(ChatMessage.appliance_id == appliance_id))
    await db.commit()
    return {"cleared": appliance_id}
