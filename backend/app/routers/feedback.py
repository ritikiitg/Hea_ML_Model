"""
Feedback Router — user signal confirmation/rejection endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.risk_assessment import RiskAssessment
from app.schemas.schemas import FeedbackCreate, FeedbackResponse, APIResponse
from app.services.feedback_service import feedback_service

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/", response_model=FeedbackResponse, status_code=201)
def submit_feedback(user_id: str, feedback_data: FeedbackCreate, db: Session = Depends(get_db)):
    """
    Submit feedback on a risk assessment.
    Supports: confirm (signal was accurate), reject (false alarm), adjust (different risk level).
    """
    # Verify user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify assessment exists and belongs to user
    assessment = db.query(RiskAssessment).filter(
        RiskAssessment.id == feedback_data.assessment_id,
        RiskAssessment.user_id == user_id,
    ).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found for this user")

    # Process feedback
    feedback = feedback_service.process_feedback(
        db=db,
        user_id=user_id,
        assessment_id=feedback_data.assessment_id,
        feedback_type=feedback_data.feedback_type.value,
        relevance_score=feedback_data.relevance_score,
        adjusted_risk_level=feedback_data.adjusted_risk_level.value if feedback_data.adjusted_risk_level else None,
        comment=feedback_data.comment,
    )

    return feedback


@router.get("/stats", response_model=dict)
def get_feedback_stats(user_id: str, db: Session = Depends(get_db)):
    """Get feedback statistics for a user — used for transparency."""
    stats = feedback_service.get_user_feedback_stats(db, user_id)
    return stats
