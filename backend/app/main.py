"""
Hea Early Health Risk Detector — FastAPI Application Entrypoint

An AI/ML prototype that detects weak health risk signals from self-reported
everyday inputs, enabling early health risk insights without medical records.

Built for https://tryhea.com
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.middleware.security import SecurityMiddleware
from app.routers import user, health_input, inference, feedback, privacy, ai_insights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events — startup and shutdown."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down Hea backend")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI/ML prototype that detects weak health risk signals from self-reported "
        "everyday inputs on https://tryhea.com. Provides early health risk insights "
        "using NLP, time-series analysis, and ensemble fusion classification."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ────────────────────────────────────────────

# CORS (allow frontend origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers & rate limiting
app.add_middleware(SecurityMiddleware)


# ─── Routers ──────────────────────────────────────────────

app.include_router(user.router, prefix=settings.API_V1_PREFIX)
app.include_router(health_input.router, prefix=settings.API_V1_PREFIX)
app.include_router(inference.router, prefix=settings.API_V1_PREFIX)
app.include_router(feedback.router, prefix=settings.API_V1_PREFIX)
app.include_router(privacy.router, prefix=settings.API_V1_PREFIX)
app.include_router(ai_insights.router, prefix=settings.API_V1_PREFIX)


# ─── Root Endpoints ───────────────────────────────────────

@app.get("/", tags=["Health Check"])
def root():
    """API root — health check."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "database": "connected",
        "ml_models": "ready",
    }
