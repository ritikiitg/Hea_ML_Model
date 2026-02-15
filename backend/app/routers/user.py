"""
User Router — onboarding, user creation, and profile management.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.schemas import UserCreate, UserResponse, APIResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserResponse, status_code=201)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with consent preferences (onboarding step 1)."""
    if not user_data.consent.consent_data_storage:
        raise HTTPException(status_code=400, detail="Data storage consent is required to use Hea")

    user = User(
        anonymous_id=f"hea_{uuid.uuid4().hex[:12]}",
        consent_data_storage=user_data.consent.consent_data_storage,
        consent_ml_usage=user_data.consent.consent_ml_usage,
        consent_anonymized_research=user_data.consent.consent_anonymized_research,
        consent_wearable_data=user_data.consent.consent_wearable_data,
        consent_given_at=datetime.utcnow(),
        notification_preferences=user_data.notification_preferences or {},
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get user profile."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
# coded by ritik raj

@router.put("/{user_id}/complete-onboarding", response_model=APIResponse)
def complete_onboarding(user_id: str, db: Session = Depends(get_db)):
    """Mark user onboarding as complete."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.onboarding_completed = True
    user.updated_at = datetime.utcnow()
    db.commit()

    return APIResponse(success=True, message="Onboarding completed successfully")
