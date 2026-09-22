from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional
import shutil, os
from pathlib import Path

from ..models.database import get_db, Appliance
from ..core.ingestion import ingest_pdf

router = APIRouter(prefix="/appliances", tags=["appliances"])


class ApplianceOut(BaseModel):
    id: int
    name: str
    brand: Optional[str]
    category: Optional[str]
    manual_filename: Optional[str]
    chunk_count: int

    class Config:
        from_attributes = True


@router.get("/", response_model=list[ApplianceOut])
async def list_appliances(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Appliance).order_by(Appliance.created_at.desc()))
    return result.scalars().all()


@router.post("/", response_model=ApplianceOut)
async def create_appliance(
    name: str = Form(...),
    brand: str = Form(""),
    category: str = Form(""),
    manual: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not manual.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF manuals are supported.")

    # Save appliance record first to get ID
    appliance = Appliance(
        name=name,
        brand=brand or None,
        category=category or None,
        manual_filename=manual.filename,
    )
    db.add(appliance)
    await db.commit()
    await db.refresh(appliance)

    # Read PDF bytes
    pdf_bytes = await manual.read()

    try:
        result = await ingest_pdf(pdf_bytes, appliance.id, name)
    except Exception as e:
        await db.delete(appliance)
        await db.commit()
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    appliance.chunk_count = result["chunk_count"]
    appliance.index_path = result["index_path"]
    await db.commit()
    await db.refresh(appliance)
    return appliance


@router.delete("/{appliance_id}")
async def delete_appliance(appliance_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Appliance).where(Appliance.id == appliance_id))
    appliance = result.scalar_one_or_none()
    if not appliance:
        raise HTTPException(404, "Appliance not found.")

    # Remove FAISS index
    index_dir = Path("indexes") / str(appliance_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)

    await db.delete(appliance)
    await db.commit()
    return {"deleted": appliance_id}
