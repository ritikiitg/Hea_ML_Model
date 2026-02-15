"""
Feedback Service — adaptive confidence adjustment based on user feedback.
Supports the feedback loop for model self-calibration.
"""

import logging
from sqlalchemy.orm import Session
from app.models.feedback import Feedback
from app.models.risk_assessment import RiskAssessment

logger = logging.getLogger(__name__)


class FeedbackService:
    """Manages user feedback on risk assessments and adjusts model confidence."""

    # Confidence adjustment factors
    CONFIRM_BOOST = 0.05       # Small positive adjustment when user confirms
    REJECT_PENALTY = -0.10     # Larger negative adjustment when user rejects
    ADJUST_FACTOR = 0.03       # Per-level difference adjustment

    def process_feedback(self, db: Session, user_id: str, assessment_id: str,
                         feedback_type: str, relevance_score: int | None = None,
                         adjusted_risk_level: str | None = None,
                         comment: str | None = None) -> Feedback:
        """
        Process user feedback on a risk assessment.
        Updates confidence calibration for future assessments.
        """
        # Create feedback record
        feedback = Feedback(
            user_id=user_id,
            assessment_id=assessment_id,
            feedback_type=feedback_type,
            relevance_score=relevance_score,
            adjusted_risk_level=adjusted_risk_level,
            comment=comment,
        )

        # Calculate confidence adjustment
        adjustment = self._calculate_adjustment(feedback_type, relevance_score, adjusted_risk_level, assessment_id, db)
        feedback.confidence_adjustment = adjustment

        db.add(feedback)

        # Update the assessment's feedback status
        assessment = db.query(RiskAssessment).filter(RiskAssessment.id == assessment_id).first()
        if assessment:
            assessment.feedback_received = feedback_type

        db.commit()
        db.refresh(feedback)

        logger.info(
            f"Feedback processed: user={user_id}, assessment={assessment_id}, "
            f"type={feedback_type}, adjustment={adjustment:.3f}"
        )

        return feedback

    def _calculate_adjustment(self, feedback_type: str, relevance_score: int | None,
                               adjusted_risk_level: str | None, assessment_id: str,
                               db: Session) -> float:
        """Calculate confidence score adjustment based on feedback."""
        if feedback_type == "confirm":
            # User confirms — boost confidence
            base = self.CONFIRM_BOOST
            if relevance_score:
                base *= (relevance_score / 5.0)  # Scale by relevance
            return round(base, 4)

        elif feedback_type == "reject":
            # User rejects — reduce confidence
            base = self.REJECT_PENALTY
            if relevance_score:
                base *= (1 - relevance_score / 5.0)  # Lower relevance = bigger penalty
            return round(base, 4)

        elif feedback_type == "adjust" and adjusted_risk_level:
            # User adjusts risk level — proportional correction
            risk_levels = {"LOW": 0, "WEAK": 1, "MODERATE": 2, "HIGH": 3}
            assessment = db.query(RiskAssessment).filter(RiskAssessment.id == assessment_id).first()
            if assessment:
                original_level = risk_levels.get(assessment.risk_level, 0)
                adjusted_level = risk_levels.get(adjusted_risk_level, 0)
                diff = adjusted_level - original_level
                return round(diff * self.ADJUST_FACTOR, 4)

        return 0.0

    def get_user_feedback_stats(self, db: Session, user_id: str) -> dict:
        """Get feedback statistics for a user — used for model calibration."""
        feedbacks = db.query(Feedback).filter(Feedback.user_id == user_id).all()

        if not feedbacks:
            return {"total": 0, "confirm_rate": 0.0, "avg_relevance": 0.0, "net_adjustment": 0.0}

        total = len(feedbacks)
        confirms = sum(1 for f in feedbacks if f.feedback_type == "confirm")
        relevance_scores = [f.relevance_score for f in feedbacks if f.relevance_score]
        adjustments = [f.confidence_adjustment for f in feedbacks if f.confidence_adjustment]

        return {
            "total": total,
            "confirm_rate": round(confirms / total, 3) if total else 0.0,
            "avg_relevance": round(sum(relevance_scores) / len(relevance_scores), 2) if relevance_scores else 0.0,
            "net_adjustment": round(sum(adjustments), 4) if adjustments else 0.0,
        }


# Singleton instance
feedback_service = FeedbackService()
