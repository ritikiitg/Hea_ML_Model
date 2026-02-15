"""
Hea Backend Configuration
Environment-based settings for database, AWS, ML models, and privacy policies.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Hea Early Health Risk Detector"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "https://tryhea.com"]

    # Database
    DATABASE_URL: str = "sqlite:///./hea_dev.db"

    # AWS Configuration
    AWS_REGION: str = "eu-west-2"  # London (UK data residency)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_LOGS: str = "hea-logs"
    S3_BUCKET_FEATURES: str = "hea-features"
    S3_BUCKET_MODELS: str = "hea-model-artifacts"

    # ML Model Paths
    NLP_MODEL_PATH: str = "models/weak_signal_nlp"
    TIMESERIES_MODEL_PATH: str = "models/timeseries_detector"
    FUSION_MODEL_PATH: str = "models/fusion_classifier"

    # Privacy & GDPR
    DATA_RETENTION_DAYS: int = 365
    ANONYMIZATION_ENABLED: bool = True
    CONSENT_REQUIRED: bool = True

    # Security
    SECRET_KEY: str = "hea-dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    MAX_INPUT_LENGTH: int = 5000
    RATE_LIMIT_PER_MINUTE: int = 60

    # Inference
    INFERENCE_TIMEOUT_MS: int = 500

    # Gemini AI
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro-preview-05-06"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash-preview-05-20"
    BATCH_PROCESSING_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
