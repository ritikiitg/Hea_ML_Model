"""
Feedback model — stores user confirmation/rejection of detected health signals.
Used for adaptive model improvement. 
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
from app.database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    assessment_id = Column(String, ForeignKey("risk_assessments.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Feedback type
    feedback_type = Column(String, nullable=False)  # confirm, reject, adjust

    # User's assessment of accuracy (1-5)
    relevance_score = Column(Integer, nullable=True)

    # User's adjusted risk level (if they disagree)
    adjusted_risk_level = Column(String, nullable=True)  # LOW, WEAK, MODERATE, HIGH

    # Free text feedback
    comment = Column(Text, nullable=True)

    # Used for model retraining
    used_for_training = Column(String, default="pending")  # pending, used, skipped
    confidence_adjustment = Column(Float, default=0.0)  # delta applied to future scores
