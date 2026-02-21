"""AyurYukti FastAPI application with REST API for all modules."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import LoginRequest, TokenResponse, authenticate_user
from src.api.routers import prakritimitra, rogaradar, vaksetu, yuktishaala

ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    yield


app = FastAPI(
    title="AyurYukti API",
    description="REST API for AyurYukti — Rational Intelligence for Ayurveda",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vaksetu.router, prefix="/api/v1/vaksetu", tags=["VakSetu"])
app.include_router(prakritimitra.router, prefix="/api/v1/prakritimitra", tags=["PrakritiMitra"])
app.include_router(rogaradar.router, prefix="/api/v1/rogaradar", tags=["RogaRadar"])
app.include_router(yuktishaala.router, prefix="/api/v1/yuktishaala", tags=["YuktiShaala"])


@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    return {
        "service": "AyurYukti API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(request: LoginRequest) -> TokenResponse:
    """Authenticate and receive JWT token."""
    result = authenticate_user(request.username, request.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return result


@app.get("/auth/demo-credentials", tags=["Auth"])
async def demo_credentials() -> Dict[str, str]:
    """Return demo credentials for testing."""
    return {
        "doctor": "doctor / doctor123",
        "official": "official / official123",
        "admin": "admin / admin123",
    }
