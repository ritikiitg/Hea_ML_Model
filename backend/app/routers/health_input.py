"""
Health Input Router — daily health data submission endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.health_input import HealthInput
from app.schemas.schemas import HealthInputCreate, HealthInputResponse
from app.services.sanitizer import InputSanitizer

router = APIRouter(prefix="/inputs", tags=["Health Inputs"])


@router.post("/", response_model=HealthInputResponse, status_code=201)
def submit_health_input(user_id: str, input_data: HealthInputCreate, db: Session = Depends(get_db)):
    """
    Submit daily health input (symptoms, metrics, emoji).
    All inputs are sanitized for security and normalized for ML processing.
    """
    # Verify user exists and has consent
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.consent_data_storage:
        raise HTTPException(status_code=403, detail="Data storage consent required")

    # Validate input length
    if not InputSanitizer.validate_input_length(input_data.symptom_text):
        raise HTTPException(status_code=400, detail="Input text exceeds maximum length")

    # Sanitize all text inputs
    sanitized = InputSanitizer.sanitize_health_input(
        input_data.symptom_text,
        input_data.emoji_inputs,
        input_data.checkbox_selections,
    )

    # Create health input record
    health_input = HealthInput(
        user_id=user_id,
        symptom_text=sanitized["symptom_text"],
        emoji_inputs=sanitized["emoji_inputs"],
        checkbox_selections=sanitized["checkbox_selections"],
        sleep_hours=input_data.daily_metrics.sleep_hours if input_data.daily_metrics else None,
        mood_score=input_data.daily_metrics.mood_score if input_data.daily_metrics else None,
        energy_level=input_data.daily_metrics.energy_level if input_data.daily_metrics else None,
        stress_level=input_data.daily_metrics.stress_level if input_data.daily_metrics else None,
        steps_count=input_data.daily_metrics.steps_count if input_data.daily_metrics else None,
        water_intake_ml=input_data.daily_metrics.water_intake_ml if input_data.daily_metrics else None,
        wearable_data=input_data.wearable_data,
        input_source=input_data.input_source.value,
    )

    db.add(health_input)
    db.commit()
    db.refresh(health_input)
    return health_input


@router.get("/", response_model=list[HealthInputResponse])
def get_user_inputs(user_id: str, limit: int = 30, db: Session = Depends(get_db)):
    """Get user's health input history."""
    inputs = (
        db.query(HealthInput)
        .filter(HealthInput.user_id == user_id)
        .order_by(HealthInput.created_at.desc())
        .limit(limit)
        .all()
    )
    return inputs


@router.get("/{input_id}", response_model=HealthInputResponse)
def get_health_input(input_id: str, db: Session = Depends(get_db)):
    """Get a specific health input by ID."""
    health_input = db.query(HealthInput).filter(HealthInput.id == input_id).first()
    if not health_input:
        raise HTTPException(status_code=404, detail="Health input not found")
    return health_input
