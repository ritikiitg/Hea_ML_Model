"""
Health Input model — stores daily user-submitted health data.
Supports free text, emoji, checkboxes, and daily metrics.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Text
from app.database import Base


class HealthInput(Base):
    __tablename__ = "health_inputs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Free text symptom description (sanitized)
    symptom_text = Column(Text, nullable=True)

    # Emoji inputs (stored as descriptive tokens)
    emoji_inputs = Column(JSON, default=list)

    # Checkbox selections (predefined symptom categories)
    checkbox_selections = Column(JSON, default=list)

    # Daily metrics
    sleep_hours = Column(Float, nullable=True)
    mood_score = Column(Integer, nullable=True)  # 1-10 scale
    energy_level = Column(Integer, nullable=True)  # 1-10 scale
    stress_level = Column(Integer, nullable=True)  # 1-10 scale
    steps_count = Column(Integer, nullable=True)
    water_intake_ml = Column(Integer, nullable=True)

    # Optional wearable summary
    wearable_data = Column(JSON, nullable=True)

    # Processing status
    is_processed = Column(String, default="pending")  # pending, processed, failed

    # Source metadata
    input_source = Column(String, default="web")  # web, whatsapp, voice
