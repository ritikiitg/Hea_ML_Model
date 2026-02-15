"""
User data model — stores anonymized user profile with consent flags.
No PHI/medical records stored.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, JSON
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    anonymous_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Consent flags (GDPR compliant)
    consent_data_storage = Column(Boolean, default=False)
    consent_ml_usage = Column(Boolean, default=False)
    consent_anonymized_research = Column(Boolean, default=False)
    consent_wearable_data = Column(Boolean, default=False)
    consent_given_at = Column(DateTime, nullable=True)

    # User preferences
    notification_preferences = Column(JSON, default=dict)
    data_retention_days = Column(String, default="365")

    # Status
    is_active = Column(Boolean, default=True)
    onboarding_completed = Column(Boolean, default=False)
