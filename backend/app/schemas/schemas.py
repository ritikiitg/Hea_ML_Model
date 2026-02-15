"""
Pydantic schemas for all API request/response models.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class FeedbackType(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    ADJUST = "adjust"


class InputSource(str, Enum):
    WEB = "web"
    WHATSAPP = "whatsapp"
    VOICE = "voice"


# ─── User Schemas ─────────────────────────────────────────

class ConsentRequest(BaseModel):
    consent_data_storage: bool = False
    consent_ml_usage: bool = False
    consent_anonymized_research: bool = False
    consent_wearable_data: bool = False


class UserCreate(BaseModel):
    consent: ConsentRequest
    notification_preferences: Optional[dict] = {}


class UserResponse(BaseModel):
    id: str
    anonymous_id: str
    created_at: datetime
    onboarding_completed: bool
    consent_data_storage: bool
    consent_ml_usage: bool

    class Config:
        from_attributes = True


# ─── Health Input Schemas ─────────────────────────────────

class DailyMetrics(BaseModel):
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours of sleep")
    mood_score: Optional[int] = Field(None, ge=1, le=10, description="Mood score 1-10")
    energy_level: Optional[int] = Field(None, ge=1, le=10, description="Energy level 1-10")
    stress_level: Optional[int] = Field(None, ge=1, le=10, description="Stress level 1-10")
    steps_count: Optional[int] = Field(None, ge=0, description="Step count")
    water_intake_ml: Optional[int] = Field(None, ge=0, description="Water intake in ml")


class HealthInputCreate(BaseModel):
    symptom_text: Optional[str] = Field(None, max_length=5000, description="Free text symptom description")
    emoji_inputs: Optional[List[str]] = Field(default=[], description="Emoji health indicators")
    checkbox_selections: Optional[List[str]] = Field(default=[], description="Selected symptom categories")
    daily_metrics: Optional[DailyMetrics] = None
    wearable_data: Optional[dict] = None
    input_source: InputSource = InputSource.WEB

    @validator("symptom_text")
    def validate_symptom_text(cls, v):
        if v and len(v.strip()) == 0:
            return None
        return v

# coded by Ritik Raj
class HealthInputResponse(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    symptom_text: Optional[str]
    emoji_inputs: list
    checkbox_selections: list
    sleep_hours: Optional[float]
    mood_score: Optional[int]
    energy_level: Optional[int]
    stress_level: Optional[int]
    steps_count: Optional[int]
    is_processed: str

    class Config:
        from_attributes = True


# ─── Risk Assessment Schemas ──────────────────────────────

class SignalDetail(BaseModel):
    signal: str
    weight: float
    category: str  # nlp, timeseries, behavioral


class RiskAssessmentResponse(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    risk_level: RiskLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation_text: str
    signal_details: dict
    model_version: str
    inference_time_ms: Optional[float]
    feedback_received: str

    class Config:
        from_attributes = True


class AssessmentRequest(BaseModel):
    user_id: str
    include_history_days: int = Field(default=7, ge=1, le=90, description="Days of history to analyze")


# ─── Feedback Schemas ─────────────────────────────────────

class FeedbackCreate(BaseModel):
    assessment_id: str
    feedback_type: FeedbackType
    relevance_score: Optional[int] = Field(None, ge=1, le=5)
    adjusted_risk_level: Optional[RiskLevel] = None
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    assessment_id: str
    feedback_type: str
    relevance_score: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Privacy Schemas ──────────────────────────────────────

class PrivacySettings(BaseModel):
    consent_data_storage: bool
    consent_ml_usage: bool
    consent_anonymized_research: bool
    consent_wearable_data: bool
    data_retention_days: int = Field(default=365, ge=30, le=730)


class DataExportResponse(BaseModel):
    export_id: str
    status: str
    download_url: Optional[str] = None
    created_at: datetime


# ─── Generic Response ─────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
