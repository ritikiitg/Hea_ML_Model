"""
Hea ML Feature Extraction
NLP embeddings, temporal features, and behavior summaries for health signal detection.
"""

import logging
import numpy as np
from typing import Optional
from ml.preprocessing import TextPreprocessor, MetricsPreprocessor

logger = logging.getLogger(__name__)

# Try to import transformer models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not available, using lightweight embeddings")


class NLPFeatureExtractor:
    """Extracts NLP features from health-related text inputs."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load DistilBERT model."""
        if HAS_TRANSFORMERS and not self._loaded:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                self._loaded = True
                logger.info(f"Loaded NLP model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}. Using fallback.")

    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Extract 768-dim embedding from text using DistilBERT."""
        # Normalize text first
        normalized = TextPreprocessor.normalize_text(text)
        if not normalized:
            return np.zeros(768)

        self._load_model()

        if self._loaded and self._tokenizer and self._model:
            try:
                inputs = self._tokenizer(
                    normalized,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)
                # Mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return embedding
            except Exception as e:
                logger.warning(f"Transformer embedding failed: {e}")

        # Fallback: simple TF-IDF-like embedding
        return self._fallback_embedding(normalized)

    def extract_attention_scores(self, text: str) -> list[dict]:
        """Extract attention-weighted token scores for explainability."""
        normalized = TextPreprocessor.normalize_text(text)
        if not normalized:
            return []

        keywords = TextPreprocessor.extract_symptom_keywords(normalized)
        tokens = normalized.split()

        # Simulate attention scores based on keyword presence
        scores = []
        for token in tokens:
            weight = 0.1  # baseline
            if any(kw in token for kw in keywords):
                weight = 0.7 + (0.3 * (token in keywords))
            scores.append({
                "token": token,
                "attention_weight": round(weight, 3),
                "is_health_keyword": any(kw in token for kw in keywords),
            })

        # Sort by attention weight
        scores.sort(key=lambda x: x["attention_weight"], reverse=True)
        return scores[:20]  # Top 20 tokens

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Lightweight fallback embedding using keyword vectors."""
        # Create a 768-dim vector based on symptom keyword presence
        keywords = TextPreprocessor.extract_symptom_keywords(text)
        embedding = np.zeros(768)

        # Seed different dimensions based on keywords
        for i, kw in enumerate(keywords):
            hash_val = hash(kw) % 768
            embedding[hash_val] = 1.0
            # Spread activation to nearby dimensions
            for offset in [-1, 1]:
                idx = (hash_val + offset) % 768
                embedding[idx] = 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class TemporalFeatureExtractor:
    """Extracts temporal features from historical daily metrics."""

    @classmethod
    def extract_temporal_features(cls, historical_metrics: list[dict]) -> dict:
        """
        Extract time-series features from historical data.
        
        Features:
        - Rolling averages (3-day, 7-day windows)
        - Variance / volatility
        - Trend slopes
        - Change-point indicators
        """
        if not historical_metrics:
            return {"has_history": False}

        features = {"has_history": True, "history_length": len(historical_metrics)}

        # Compute rolling features
        rolling = MetricsPreprocessor.compute_rolling_features(historical_metrics, window=3)
        features.update(rolling)

        # Additional temporal features
        for metric in ["sleep_hours", "mood_score", "energy_level", "stress_level"]:
            values = [h.get(metric) for h in historical_metrics if h.get(metric) is not None]

            if len(values) >= 2:
                features[f"{metric}_min"] = min(values)
                features[f"{metric}_max"] = max(values)
                features[f"{metric}_range"] = max(values) - min(values)
                features[f"{metric}_latest"] = values[-1]

                # Day-over-day change
                features[f"{metric}_day_change"] = round(values[-1] - values[-2], 2)

                # Volatility (std of differences)
                if len(values) >= 3:
                    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
                    features[f"{metric}_volatility"] = round(
                        (sum(d ** 2 for d in diffs) / len(diffs)) ** 0.5, 3
                    )

        return features


class BehaviorSummaryExtractor:
    """Generates behavior summaries from combined inputs."""

    @classmethod
    def extract_behavior_summary(
        cls,
        text_features: dict,
        temporal_features: dict,
        checkbox_selections: list,
        emoji_tokens: list,
    ) -> dict:
        """
        Create a combined behavior summary for the fusion classifier.
        """
        summary = {
            "input_richness": 0,  # How much data the user provided
            "concern_level": 0.0,  # Overall concern indicators
            "consistency_score": 0.0,  # How consistent signals are
            "categories": [],
        }

        # Input richness
        richness = 0
        if text_features.get("has_text"):
            richness += 2
        if checkbox_selections:
            richness += len(checkbox_selections)
        if emoji_tokens:
            richness += len(emoji_tokens)
        if temporal_features.get("has_history"):
            richness += temporal_features.get("history_length", 0)
        summary["input_richness"] = richness

        # Concern categorization
        high_concern_symptoms = {"chest_pain", "shortness_of_breath", "heart_palpitations", "numbness"}
        moderate_concern_symptoms = {"headache", "fatigue", "insomnia", "anxiety", "dizziness"}

        critical_count = len(set(checkbox_selections) & high_concern_symptoms)
        moderate_count = len(set(checkbox_selections) & moderate_concern_symptoms)

        summary["concern_level"] = round(min(1.0, critical_count * 0.3 + moderate_count * 0.15), 3)

        # Categories
        if critical_count > 0:
            summary["categories"].append("critical_symptoms")
        if moderate_count > 0:
            summary["categories"].append("moderate_symptoms")

        # Check for declining trends in temporal data
        declining_metrics = []
        for metric in ["mood_score", "sleep_hours", "energy_level"]:
            trend = temporal_features.get(f"{metric}_trend_dir")
            if trend == "declining":
                declining_metrics.append(metric)

        if declining_metrics:
            summary["categories"].append("declining_trends")
            summary["declining_metrics"] = declining_metrics
            summary["concern_level"] = min(1.0, summary["concern_level"] + 0.1 * len(declining_metrics))

        # Consistency: if NLP and time-series signals align
        summary["consistency_score"] = round(summary["concern_level"], 3)

        return summary
