"""
Hea ML Explainability Module
Attention scoring + rule-based signal aggregation + SHAP-based feature importance
for human-readable explanations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for risk assessments.
    Combines attention-based token importance with rule-based signal aggregation.
    """

    # Signal category descriptions (non-clinical language)
    SIGNAL_DESCRIPTIONS = {
        "fatigue_signal": "You mentioned feeling tired or low on energy",
        "pain_signal": "Pain-related symptoms were detected in your description",
        "mood_signal": "Your mood appears lower than usual",
        "sleep_signal": "Sleep disruption patterns were noticed",
        "digestive_signal": "Digestive or stomach-related concerns were noted",
        "respiratory_signal": "Breathing or respiratory changes detected",
        "cardiovascular_signal": "Heart or circulation-related signals were flagged",
        "neurological_signal": "Head, vision, or cognitive changes were noted",
    }

    # Metric change explanations
    METRIC_EXPLANATIONS = {
        "sleep_hours": {
            "low": "Your sleep has been shorter than usual",
            "high": "You've been sleeping more than usual, which can sometimes indicate fatigue",
            "declining": "Your sleep duration has been trending downward recently",
        },
        "mood_score": {
            "low": "Your mood has been below your usual range",
            "declining": "We've noticed a gradual decline in your mood scores",
        },
        "energy_level": {
            "low": "Your energy levels have been consistently low",
            "declining": "Your energy appears to be dropping over recent days",
        },
        "stress_level": {
            "high": "Your stress levels have been elevated",
            "increasing": "Stress appears to be building up over time",
        },
    }

    # Next steps suggestions (non-medical advice)
    NEXT_STEPS = {
        "LOW": [
            "Keep tracking your daily health — consistency improves detection accuracy",
            "Consider setting up regular check-in reminders",
        ],
        "WEAK": [
            "Continue monitoring these patterns over the next few days",
            "Try to identify any recent lifestyle changes that might be contributing",
            "Ensure you're staying hydrated and getting enough rest",
        ],
        "MODERATE": [
            "Consider discussing these patterns with your GP at your next visit",
            "Try to maintain consistent sleep and meal schedules",
            "If symptoms persist for more than a week, seek professional guidance",
            "Use the feedback feature to help us refine our detection for you",
        ],
        "HIGH": [
            "We strongly recommend speaking with a healthcare professional soon",
            "If you're experiencing acute symptoms, please contact your doctor or NHS 111",
            "Keep logging daily so we can track any changes",
            "Remember: Hea is a wellness companion, not a medical diagnostic tool",
        ],
    }

    @classmethod
    def generate_explanation(
        cls,
        risk_level: str,
        confidence_score: float,
        nlp_signals: list[dict],
        ts_signals: list[dict],
        attention_scores: Optional[list[dict]] = None,
    ) -> dict:
        """
        Generate a comprehensive, user-friendly explanation.

        Returns:
            dict with summary, signal_explanations, next_steps, attention_highlights
        """
        # Main summary
        summary = cls._generate_summary(risk_level, confidence_score)

        # Signal-by-signal explanations
        signal_explanations = cls._explain_signals(nlp_signals, ts_signals)

        # Next steps
        next_steps = cls.NEXT_STEPS.get(risk_level, cls.NEXT_STEPS["LOW"])

        # Attention highlights (which words/inputs mattered most)
        attention_highlights = []
        if attention_scores:
            top_tokens = sorted(attention_scores, key=lambda x: x.get("attention_weight", 0), reverse=True)[:5]
            attention_highlights = [
                {
                    "term": t["token"],
                    "importance": t["attention_weight"],
                    "reason": "health-related keyword" if t.get("is_health_keyword") else "contextual signal",
                }
                for t in top_tokens
                if t.get("attention_weight", 0) > 0.3
            ]

        return {
            "summary": summary,
            "confidence_note": cls._confidence_note(confidence_score),
            "signal_explanations": signal_explanations,
            "next_steps": next_steps,
            "attention_highlights": attention_highlights,
            "disclaimer": (
                "Hea is a wellness monitoring tool and does not provide medical diagnosis, "
                "advice, or treatment. Always consult a qualified healthcare professional "
                "for medical concerns."
            ),
        }

    @classmethod
    def _generate_summary(cls, risk_level: str, confidence: float) -> str:
        """Generate the main explanation summary."""
        summaries = {
            "LOW": "Your recent health inputs look stable and within normal patterns.",
            "WEAK": "We noticed a few subtle signals that are worth keeping an eye on, but nothing that requires immediate attention.",
            "MODERATE": "We've detected some noteworthy patterns in your recent health data that suggest you should be more mindful of your wellbeing.",
            "HIGH": "Several patterns in your recent health data are raising concern. We recommend taking action and consulting with a healthcare professional.",
        }
        return summaries.get(risk_level, summaries["LOW"])

    @classmethod
    def _explain_signals(cls, nlp_signals: list[dict], ts_signals: list[dict]) -> list[dict]:
        """Generate individual signal explanations."""
        explanations = []

        for signal in nlp_signals:
            if signal.get("weight", 0) >= 0.3:
                explanations.append({
                    "source": "Your symptom description",
                    "finding": signal.get("signal", "Pattern detected"),
                    "importance": "high" if signal["weight"] >= 0.7 else "moderate" if signal["weight"] >= 0.5 else "low",
                    "icon": "📝",
                })

        for signal in ts_signals:
            if signal.get("weight", 0) >= 0.3:
                explanations.append({
                    "source": "Your daily metrics",
                    "finding": signal.get("signal", "Change detected"),
                    "importance": "high" if signal["weight"] >= 0.7 else "moderate" if signal["weight"] >= 0.5 else "low",
                    "icon": "📊",
                })

        # Sort by importance
        importance_order = {"high": 0, "moderate": 1, "low": 2}
        explanations.sort(key=lambda x: importance_order.get(x["importance"], 3))

        return explanations[:8]  # Top 8 explanations

    @classmethod
    def _confidence_note(cls, confidence: float) -> str:
        """Human-readable confidence explanation."""
        if confidence >= 0.8:
            return "We have high confidence in this assessment based on consistent signals across your inputs."
        elif confidence >= 0.6:
            return "We have moderate confidence in this assessment. More data points will improve accuracy."
        elif confidence >= 0.4:
            return "Our confidence is limited. Continue logging daily to help us build a clearer picture."
        else:
            return "This is a preliminary assessment. We need more data to provide more reliable insights."

    @classmethod
    def explain_with_shap(cls, risk_factors: list[dict], risk_level: str,
                          confidence_score: float) -> dict:
        """
        Generate explanation from SHAP-based risk factors (from prediction_service).

        Args:
            risk_factors: List from HealthRiskPredictor._explain() with
                          feature, label, shap_value, direction, value, importance.
            risk_level: Predicted risk level (LOW/WEAK/MODERATE/HIGH).
            confidence_score: Model confidence.

        Returns:
            dict with summary, shap_explanations, next_steps, disclaimer.
        """
        summary = cls._generate_summary(risk_level, confidence_score)

        # Convert SHAP factors to human-readable explanations
        shap_explanations = []
        for factor in risk_factors[:8]:
            icon = "⬆️" if factor.get("direction") == "increases risk" else "⬇️"
            shap_explanations.append({
                "source": "Longitudinal health model",
                "finding": f"{factor['label']}: {factor.get('direction', 'relevant')}",
                "feature": factor["feature"],
                "value": factor.get("value"),
                "importance": (
                    "high" if factor.get("importance", 0) > 0.1
                    else "moderate" if factor.get("importance", 0) > 0.05
                    else "low"
                ),
                "icon": icon,
            })

        return {
            "summary": summary,
            "confidence_note": cls._confidence_note(confidence_score),
            "signal_explanations": shap_explanations,
            "next_steps": cls.NEXT_STEPS.get(risk_level, cls.NEXT_STEPS["LOW"]),
            "model_type": "ensemble_lgbm_xgb_shap",
            "disclaimer": (
                "Hea is a wellness monitoring tool and does not provide medical diagnosis, "
                "advice, or treatment. Always consult a qualified healthcare professional "
                "for medical concerns."
            ),
        }
