"""
Privacy Service — GDPR-compliant data management.
Handles consent management, data export, anonymization, and retention.
"""

import uuid
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.health_input import HealthInput
from app.models.risk_assessment import RiskAssessment
from app.models.feedback import Feedback
from app.config import settings

logger = logging.getLogger(__name__)


class PrivacyService:
    """Manages user privacy, GDPR compliance, and data lifecycle."""

    def update_consent(self, db: Session, user_id: str, consent_data: dict) -> User:
        """Update user consent preferences."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        user.consent_data_storage = consent_data.get("consent_data_storage", user.consent_data_storage)
        user.consent_ml_usage = consent_data.get("consent_ml_usage", user.consent_ml_usage)
        user.consent_anonymized_research = consent_data.get("consent_anonymized_research", user.consent_anonymized_research)
        user.consent_wearable_data = consent_data.get("consent_wearable_data", user.consent_wearable_data)
        user.consent_given_at = datetime.utcnow()

        if "data_retention_days" in consent_data:
            user.data_retention_days = str(consent_data["data_retention_days"])

        db.commit()
        db.refresh(user)
        logger.info(f"Consent updated for user {user_id}")
        return user

    def export_user_data(self, db: Session, user_id: str) -> dict:
        """Export all user data (GDPR right of access / data portability)."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        health_inputs = db.query(HealthInput).filter(HealthInput.user_id == user_id).all()
        assessments = db.query(RiskAssessment).filter(RiskAssessment.user_id == user_id).all()
        feedbacks = db.query(Feedback).filter(Feedback.user_id == user_id).all()

        export_data = {
            "export_id": str(uuid.uuid4()),
            "exported_at": datetime.utcnow().isoformat(),
            "user": {
                "id": user.id,
                "anonymous_id": user.anonymous_id,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "consent_data_storage": user.consent_data_storage,
                "consent_ml_usage": user.consent_ml_usage,
                "onboarding_completed": user.onboarding_completed,
            },
            "health_inputs": [
                {
                    "id": h.id,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                    "symptom_text": h.symptom_text,
                    "emoji_inputs": h.emoji_inputs,
                    "checkbox_selections": h.checkbox_selections,
                    "sleep_hours": h.sleep_hours,
                    "mood_score": h.mood_score,
                    "energy_level": h.energy_level,
                    "stress_level": h.stress_level,
                    "steps_count": h.steps_count,
                }
                for h in health_inputs
            ],
            "risk_assessments": [
                {
                    "id": a.id,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "risk_level": a.risk_level,
                    "confidence_score": a.confidence_score,
                    "explanation_text": a.explanation_text,
                }
                for a in assessments
            ],
            "feedback": [
                {
                    "id": f.id,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                    "feedback_type": f.feedback_type,
                    "relevance_score": f.relevance_score,
                    "comment": f.comment,
                }
                for f in feedbacks
            ],
        }

        logger.info(f"Data exported for user {user_id}: {len(health_inputs)} inputs, {len(assessments)} assessments")
        return export_data

    def delete_user_data(self, db: Session, user_id: str) -> dict:
        """Delete all user data (GDPR right to erasure)."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Delete all related data
        feedback_count = db.query(Feedback).filter(Feedback.user_id == user_id).delete()
        assessment_count = db.query(RiskAssessment).filter(RiskAssessment.user_id == user_id).delete()
        input_count = db.query(HealthInput).filter(HealthInput.user_id == user_id).delete()
        db.delete(user)
        db.commit()

        logger.info(f"Data deleted for user {user_id}: {input_count} inputs, {assessment_count} assessments, {feedback_count} feedbacks")
        return {
            "deleted_inputs": input_count,
            "deleted_assessments": assessment_count,
            "deleted_feedbacks": feedback_count,
            "user_deleted": True,
        }

    def enforce_retention_policy(self, db: Session) -> dict:
        """Purge data older than retention period (batch job)."""
        cutoff = datetime.utcnow() - timedelta(days=settings.DATA_RETENTION_DAYS)

        old_inputs = db.query(HealthInput).filter(HealthInput.created_at < cutoff).delete()
        old_assessments = db.query(RiskAssessment).filter(RiskAssessment.created_at < cutoff).delete()
        old_feedback = db.query(Feedback).filter(Feedback.created_at < cutoff).delete()

        db.commit()
        logger.info(f"Retention policy enforced: purged {old_inputs} inputs, {old_assessments} assessments, {old_feedback} feedbacks")
        return {
            "purged_inputs": old_inputs,
            "purged_assessments": old_assessments,
            "purged_feedbacks": old_feedback,
            "cutoff_date": cutoff.isoformat(),
        }

    def anonymize_for_research(self, db: Session, user_id: str) -> dict:
        """Generate de-identified summary for research purposes."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.consent_anonymized_research:
            return {"error": "User not found or research consent not given"}

        inputs = db.query(HealthInput).filter(HealthInput.user_id == user_id).all()
        assessments = db.query(RiskAssessment).filter(RiskAssessment.user_id == user_id).all()

        # De-identified summary (no PII)
        return {
            "anonymous_id": user.anonymous_id,
            "data_points": len(inputs),
            "assessment_count": len(assessments),
            "avg_mood": round(sum(i.mood_score for i in inputs if i.mood_score) / max(1, sum(1 for i in inputs if i.mood_score)), 2),
            "avg_sleep": round(sum(i.sleep_hours for i in inputs if i.sleep_hours) / max(1, sum(1 for i in inputs if i.sleep_hours)), 2),
            "risk_distribution": {
                level: sum(1 for a in assessments if a.risk_level == level)
                for level in ["LOW", "WEAK", "MODERATE", "HIGH"]
            },
        }


# Singleton instance
privacy_service = PrivacyService()
