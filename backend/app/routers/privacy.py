"""
Privacy Router — consent management, data export, and deletion.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.schemas import PrivacySettings, APIResponse
from app.services.privacy_service import privacy_service

router = APIRouter(prefix="/privacy", tags=["Privacy & GDPR"])


@router.get("/{user_id}", response_model=PrivacySettings)
def get_privacy_settings(user_id: str, db: Session = Depends(get_db)):
    """Get current privacy/consent settings for a user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return PrivacySettings(
        consent_data_storage=user.consent_data_storage,
        consent_ml_usage=user.consent_ml_usage,
        consent_anonymized_research=user.consent_anonymized_research,
        consent_wearable_data=user.consent_wearable_data,
        data_retention_days=int(user.data_retention_days),
    )


@router.put("/{user_id}", response_model=APIResponse)
def update_privacy_settings(user_id: str, settings: PrivacySettings, db: Session = Depends(get_db)):
    """Update privacy/consent settings."""
    try:
        privacy_service.update_consent(db, user_id, settings.model_dump())
        return APIResponse(success=True, message="Privacy settings updated successfully")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{user_id}/export", response_model=dict)
def export_user_data(user_id: str, db: Session = Depends(get_db)):
    """Export all user data (GDPR right of access)."""
    try:
        return privacy_service.export_user_data(db, user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# coded by ritik raj
@router.delete("/{user_id}", response_model=APIResponse)
def delete_user_data(user_id: str, confirm: bool = False, db: Session = Depends(get_db)):
    """Delete all user data (GDPR right to erasure). Requires confirmation."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to permanently delete all data")

    try:
        result = privacy_service.delete_user_data(db, user_id)
        return APIResponse(success=True, message="All user data deleted", data=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
