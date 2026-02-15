"""
Inference Router — triggers risk assessment from ML pipeline.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.health_input import HealthInput
from app.models.risk_assessment import RiskAssessment
from app.schemas.schemas import AssessmentRequest, RiskAssessmentResponse
from app.services.inference_service import inference_service

router = APIRouter(prefix="/assess", tags=["Risk Assessment"])


@router.post("/", response_model=RiskAssessmentResponse, status_code=201)
def run_assessment(request: AssessmentRequest, db: Session = Depends(get_db)):
    """
    Trigger ML risk assessment for a user.
    Analyzes recent health inputs using NLP + time-series + fusion models.
    """
    # Verify user
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.consent_ml_usage:
        raise HTTPException(status_code=403, detail="ML usage consent required for risk assessment")

    # Fetch recent health inputs
    recent_inputs = (
        db.query(HealthInput)
        .filter(HealthInput.user_id == request.user_id)
        .order_by(HealthInput.created_at.desc())
        .limit(request.include_history_days)
        .all()
    )
    # coded by ritik raj
    if not recent_inputs:
        raise HTTPException(status_code=400, detail="No health inputs found. Submit daily logs first.")

    # Get latest input for primary analysis
    latest = recent_inputs[0]

    # Build historical metrics for trend analysis
    historical_metrics = [
        {
            "sleep_hours": inp.sleep_hours,
            "mood_score": inp.mood_score,
            "energy_level": inp.energy_level,
            "stress_level": inp.stress_level,
            "steps_count": inp.steps_count,
        }
        for inp in reversed(recent_inputs)
    ]

    # Build daily metrics dict
    daily_metrics = {
        "sleep_hours": latest.sleep_hours,
        "mood_score": latest.mood_score,
        "energy_level": latest.energy_level,
        "stress_level": latest.stress_level,
        "steps_count": latest.steps_count,
    }

    # Run ML inference
    result = inference_service.assess_risk(
        symptom_text=latest.symptom_text,
        emoji_inputs=latest.emoji_inputs or [],
        checkbox_selections=latest.checkbox_selections or [],
        daily_metrics=daily_metrics,
        historical_metrics=historical_metrics,
    )

    # Store assessment result
    assessment = RiskAssessment(
        user_id=request.user_id,
        risk_level=result["risk_level"],
        confidence_score=result["confidence_score"],
        explanation_text=result["explanation_text"],
        signal_details=result["signal_details"],
        model_version=result["model_version"],
        inference_time_ms=result["inference_time_ms"],
        input_ids=[inp.id for inp in recent_inputs],
    )

    db.add(assessment)

    # Mark inputs as processed
    for inp in recent_inputs:
        inp.is_processed = "processed"

    db.commit()
    db.refresh(assessment)
    return assessment


@router.get("/history", response_model=list[RiskAssessmentResponse])
def get_assessment_history(user_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """Get risk assessment history for a user."""
    assessments = (
        db.query(RiskAssessment)
        .filter(RiskAssessment.user_id == user_id)
        .order_by(RiskAssessment.created_at.desc())
        .limit(limit)
        .all()
    )
    return assessments


@router.get("/{assessment_id}", response_model=RiskAssessmentResponse)
def get_assessment(assessment_id: str, db: Session = Depends(get_db)):
    """Get a specific risk assessment by ID."""
    assessment = db.query(RiskAssessment).filter(RiskAssessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
