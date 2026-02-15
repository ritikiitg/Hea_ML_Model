"""
FusionClassifier — Ensemble model that combines NLP signal scores and
time-series anomaly scores to produce final risk assessments.

Architecture:
  [NLP risk_embedding (128)] + [TS ts_embedding (64)] → MLP → risk_level + confidence

Purpose: Aggregate weak signals from multiple detection models into
calibrated risk levels with confidence scores and explanations.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class FusionClassifierModel(nn.Module):
        """
        Neural fusion classifier combining NLP and time-series embeddings.

        Input: concatenated embeddings [nlp_embedding(128) + ts_embedding(64)] = 192
        Output: {
            risk_probabilities: (batch, 4) — probability for each risk level
            confidence: (batch, 1) — calibrated confidence score
        }
        """

        RISK_LEVELS = ["LOW", "WEAK", "MODERATE", "HIGH"]

        def __init__(self, nlp_dim: int = 128, ts_dim: int = 64, dropout: float = 0.3):
            super().__init__()

            combined_dim = nlp_dim + ts_dim  # 192

            # Fusion layers
            self.fusion_network = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Risk classification head
            self.risk_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(self.RISK_LEVELS)),
            )

            # Confidence head
            self.confidence_head = nn.Sequential(
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, nlp_embedding, ts_embedding):
            # Concatenate embeddings
            combined = torch.cat([nlp_embedding, ts_embedding], dim=-1)

            # Fusion
            fused = self.fusion_network(combined)

            # Risk probabilities
            risk_logits = self.risk_head(fused)
            risk_probs = torch.softmax(risk_logits, dim=-1)

            # Confidence score
            confidence = self.confidence_head(fused)

            return {
                "risk_probabilities": risk_probs,
                "risk_logits": risk_logits,
                "confidence": confidence,
            }


class WeightedFusionClassifier:
    """
    Rule-based fusion classifier for prototype deployment.
    Combines NLP and time-series scores using learned-like weights.
    """

    RISK_LEVELS = ["LOW", "WEAK", "MODERATE", "HIGH"]
    RISK_THRESHOLDS = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]

    # Fusion weights (tunable)
    NLP_WEIGHT = 0.55
    TS_WEIGHT = 0.45

    def classify(
        self,
        nlp_score: float,
        ts_score: float,
        nlp_signals: list[dict] = None,
        ts_signals: list[dict] = None,
        feedback_adjustment: float = 0.0,
    ) -> dict:
        """
        Fuse NLP and time-series scores into a risk classification.

        Args:
            nlp_score: NLP signal detection score (0-1)
            ts_score: Time-series anomaly score (0-1)
            nlp_signals: List of NLP signal details
            ts_signals: List of time-series signal details
            feedback_adjustment: Cumulative adjustment from user feedback

        Returns:
            dict with risk_level, confidence_score, explanation, signal_breakdown
        """
        # Weighted combination
        raw_score = (nlp_score * self.NLP_WEIGHT) + (ts_score * self.TS_WEIGHT)

        # Apply feedback adjustment (self-calibrating)
        adjusted_score = max(0.0, min(1.0, raw_score + feedback_adjustment))

        # Determine risk level
        risk_level = self.RISK_LEVELS[0]
        for i, (low, high) in enumerate(self.RISK_THRESHOLDS):
            if low <= adjusted_score < high:
                risk_level = self.RISK_LEVELS[i]
                break
        if adjusted_score >= 1.0:
            risk_level = "HIGH"

        # Confidence — based on signal agreement and data richness
        signal_agreement = 1.0 - abs(nlp_score - ts_score)
        data_richness = min(1.0, (len(nlp_signals or []) + len(ts_signals or [])) / 10)
        confidence = round(0.5 * signal_agreement + 0.3 * data_richness + 0.2 * adjusted_score, 3)

        # Build explanation
        explanation = self._build_explanation(
            risk_level, adjusted_score, nlp_score, ts_score,
            nlp_signals or [], ts_signals or [],
        )

        # Signal breakdown
        all_signals = sorted(
            (nlp_signals or []) + (ts_signals or []),
            key=lambda s: s.get("weight", 0),
            reverse=True,
        )

        return {
            "risk_level": risk_level,
            "confidence_score": confidence,
            "raw_score": round(raw_score, 3),
            "adjusted_score": round(adjusted_score, 3),
            "explanation_text": explanation,
            "signal_breakdown": {
                "nlp_contribution": round(nlp_score * self.NLP_WEIGHT, 3),
                "ts_contribution": round(ts_score * self.TS_WEIGHT, 3),
                "feedback_adjustment": round(feedback_adjustment, 4),
                "signal_agreement": round(signal_agreement, 3),
                "top_signals": all_signals[:5],
            },
        }

    def _build_explanation(
        self, risk_level: str, score: float,
        nlp_score: float, ts_score: float,
        nlp_signals: list, ts_signals: list,
    ) -> str:
        """Generate human-readable, non-clinical explanation."""
        templates = {
            "LOW": (
                "Everything looks consistent with your normal patterns. "
                "No notable changes detected in your recent inputs. Keep logging — consistency helps us learn your baseline!"
            ),
            "WEAK": (
                "We picked up a few subtle signals worth noting. "
                "These are early-stage patterns that we'll continue to monitor. "
                "No action needed right now, but keep tracking."
            ),
            "MODERATE": (
                "We've noticed some patterns that suggest a shift from your usual baseline. "
                "It might be worth paying extra attention to how you're feeling. "
                "If these patterns continue, consider chatting with your GP."
            ),
            "HIGH": (
                "Several indicators suggest a significant change in your recent health patterns. "
                "We recommend reaching out to a healthcare professional for a check-in. "
                "Remember: Hea helps you stay aware — it's not a medical diagnosis."
            ),
        }

        explanation = templates.get(risk_level, templates["LOW"])

        # Add specific context
        notable_nlp = [s for s in nlp_signals if s.get("weight", 0) >= 0.4]
        notable_ts = [s for s in ts_signals if s.get("weight", 0) >= 0.4]

        context_parts = []
        if notable_nlp:
            context_parts.append(f"We identified {len(notable_nlp)} notable pattern(s) in your symptom descriptions")
        if notable_ts:
            context_parts.append(f"{len(notable_ts)} change(s) in your daily metrics")

        if context_parts:
            explanation += " Specifically: " + ", and ".join(context_parts) + "."

        return explanation


# Factory function
def create_fusion_classifier(use_neural: bool = True):
    """Create the appropriate fusion classifier."""
    if HAS_TORCH and use_neural:
        try:
            model = FusionClassifierModel()
            logger.info("Created FusionClassifierModel (neural)")
            return model
        except Exception as e:
            logger.warning(f"Could not create neural fusion: {e}")

    logger.info("Created WeightedFusionClassifier (rule-based)")
    return WeightedFusionClassifier()
