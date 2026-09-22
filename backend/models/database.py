from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text, Integer, ForeignKey
from datetime import datetime, timezone
import os

DATABASE_URL = "sqlite+aiosqlite:///./data/appliance_ai.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Appliance(Base):
    __tablename__ = "appliances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    brand: Mapped[str] = mapped_column(String(80), nullable=True)
    category: Mapped[str] = mapped_column(String(80), nullable=True)
    manual_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    index_path: Mapped[str] = mapped_column(String(255), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    appliance_id: Mapped[int] = mapped_column(Integer, ForeignKey("appliances.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str] = mapped_column(Text, nullable=True)  # JSON list of source snippets
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


async def init_db():
    os.makedirs("data", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
