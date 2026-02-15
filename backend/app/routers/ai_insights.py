"""
AI Insights Router — Endpoints for Gemini-powered health insights.

Uses Gemini Pro for deep pattern analysis and Gemini Flash for quick daily tips.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.database import get_db
from app.models.health_input import HealthInput
from app.models.risk_assessment import RiskAssessment
from app.services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["AI Insights"])


# ─── Request / Response Schemas ───────────────────────

class AnalysisRequest(BaseModel):
    user_id: str
    include_days: int = 7
    context: Optional[str] = None  # Optional user context/question


class QuickTipRequest(BaseModel):
    free_text_input: Optional[str] = ""
    emoji_input: Optional[str] = ""
    selected_symptoms: list[str] = []
    daily_metrics: dict = {}


# ─── Endpoints ────────────────────────────────────────

@router.get("/status")
def ai_status():
    """Check if Gemini AI service is available."""
    available = gemini_service.is_available()
    return {
        "ai_available": available,
        "models": {
            "pro": gemini_service.client is not None,
            "flash": gemini_service.client is not None,
        },
        "message": "Gemini AI is ready ✨" if available else "AI unavailable — using rule-based fallbacks",
    }


@router.post("/analyze")
async def analyze_patterns(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Deep pattern analysis using Gemini Pro.
    Analyzes recent health inputs and risk assessments to find patterns.
    """
    # Fetch recent health inputs
    inputs = (
        db.query(HealthInput)
        .filter(HealthInput.user_id == request.user_id)
        .order_by(HealthInput.created_at.desc())
        .limit(request.include_days)
        .all()
    )

    # Fetch latest assessment
    latest_assessment = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.user_id == request.user_id)
        .order_by(RiskAssessment.created_at.desc())
        .first()
    )

    # Build health data payload
    health_data = {
        "inputs": [
            {
                "free_text_input": inp.free_text_input,
                "selected_symptoms": inp.selected_symptoms or [],
                "daily_metrics": inp.daily_metrics or {},
                "created_at": str(inp.created_at),
            }
            for inp in inputs
        ],
    }

    if latest_assessment:
        health_data["assessment"] = {
            "risk_level": latest_assessment.risk_level,
            "confidence_score": latest_assessment.confidence_score,
            "signal_details": latest_assessment.signal_details or {},
        }

    # Call Gemini Pro
    result = await gemini_service.analyze_health_patterns(
        health_data=health_data,
        user_context=request.context,
    )

    return {
        "status": "success",
        "data": result,
    }


@router.post("/quick-tip")
async def quick_tip(request: QuickTipRequest):
    """
    Quick wellness tip using Gemini Flash.
    Instant feedback on a daily log entry — no database needed.
    """
    daily_log = {
        "free_text_input": request.free_text_input,
        "emoji_input": request.emoji_input,
        "selected_symptoms": request.selected_symptoms,
        "daily_metrics": request.daily_metrics,
    }

    result = await gemini_service.get_quick_tip(daily_log)

    return {
        "status": "success",
        "data": result,
    }
