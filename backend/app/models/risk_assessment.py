"""
Risk Assessment model — stores ML-generated risk evaluation results.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Text
from app.database import Base


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Risk output
    risk_level = Column(String, nullable=False)  # LOW, WEAK, MODERATE, HIGH
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0

    # Human-readable explanation
    explanation_text = Column(Text, nullable=False)

    # Detailed signal breakdown
    signal_details = Column(JSON, default=dict)
    # Example: {
    #   "nlp_signals": [{"signal": "fatigue mentions increased", "weight": 0.6}],
    #   "timeseries_signals": [{"signal": "sleep_drop_detected", "weight": 0.8}],
    #   "combined_score": 0.72
    # }

    # Model metadata
    model_version = Column(String, default="v0.1.0")
    inference_time_ms = Column(Float, nullable=True)

    # Input references (which health inputs contributed)
    input_ids = Column(JSON, default=list)

    # Feedback status
    feedback_received = Column(String, default="none")  # none, confirmed, rejected, adjusted
