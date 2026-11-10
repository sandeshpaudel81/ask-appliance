from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from backend.models.database import init_db
from backend.routers.appliances import router as appliances_router
from backend.routers.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AskAppliance",
    description="Domain-specific AI assistant for home appliance manuals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(appliances_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

# Serve frontend
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AskAppliance"}
