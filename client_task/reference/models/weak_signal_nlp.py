"""
WeakSignalDetector_NLP — DistilBERT-based model for extracting weak symptom
language patterns from free-text health inputs.

Architecture:
  DistilBERT encoder → attention pooling → classification head → risk signals

Purpose: Detect subtle symptom mentions, severity language, frequency indicators,
and emotional distress signals in natural language health descriptions.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel, DistilBertConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch/transformers not available. WeakSignalDetector will use rule-based fallback.")


if HAS_TORCH:
    class WeakSignalNLPModel(nn.Module):
        """
        DistilBERT-based weak signal detector.
        
        Input: tokenized text (max 256 tokens)
        Output: {
            signal_scores: (batch, num_signals) — presence scores for each signal type
            attention_weights: (batch, seq_len) — token-level attention for explainability
            risk_embedding: (batch, 128) — dense risk representation for fusion
        }
        """

        NUM_SIGNAL_TYPES = 8  # fatigue, pain, mood, sleep, digestive, respiratory, cardiovascular, neurological

        def __init__(self, pretrained_model: str = "distilbert-base-uncased", dropout: float = 0.3):
            super().__init__()

            self.bert = DistilBertModel.from_pretrained(pretrained_model)

            # Freeze early layers, fine-tune later ones
            for param in list(self.bert.parameters())[:-30]:
                param.requires_grad = False

            hidden_size = self.bert.config.hidden_size  # 768

            # Attention pooling layer
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

            # Signal classification head
            self.signal_classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, self.NUM_SIGNAL_TYPES),
                nn.Sigmoid(),
            )

            # Risk embedding head (for fusion)
            self.risk_encoder = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
            )

        def forward(self, input_ids, attention_mask=None):
            # Get BERT hidden states
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

            # Attention pooling
            attn_scores = self.attention_layer(hidden_states).squeeze(-1)  # (batch, seq_len)
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)

            # Weighted sum of hidden states
            pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, 768)

            # Signal scores
            signal_scores = self.signal_classifier(pooled)  # (batch, num_signals)

            # Risk embedding
            risk_embedding = self.risk_encoder(pooled)  # (batch, 128)

            return {
                "signal_scores": signal_scores,
                "attention_weights": attn_weights,
                "risk_embedding": risk_embedding,
            }

        def get_signal_names(self):
            return [
                "fatigue_signal", "pain_signal", "mood_signal", "sleep_signal",
                "digestive_signal", "respiratory_signal", "cardiovascular_signal", "neurological_signal",
            ]


class WeakSignalDetectorFallback:
    """
    Rule-based fallback for environments without PyTorch.
    Uses keyword matching and scoring heuristics.
    """

    SIGNAL_KEYWORDS = {
        "fatigue_signal": ["tired", "exhausted", "fatigue", "no energy", "sleepy", "lethargic", "drained"],
        "pain_signal": ["pain", "ache", "sore", "cramp", "sharp", "throbbing", "burning"],
        "mood_signal": ["sad", "depressed", "anxious", "worry", "panic", "irritable", "mood", "crying"],
        "sleep_signal": ["insomnia", "can't sleep", "waking up", "nightmares", "restless", "sleep"],
        "digestive_signal": ["nausea", "vomiting", "stomach", "bloating", "diarrhea", "constipation", "appetite"],
        "respiratory_signal": ["cough", "breathing", "wheeze", "congestion", "shortness of breath", "chest"],
        "cardiovascular_signal": ["heart", "palpitations", "chest pain", "blood pressure", "dizzy", "faint"],
        "neurological_signal": ["headache", "migraine", "numbness", "tingling", "vision", "memory", "confusion"],
    }

    def predict(self, text: str) -> dict:
        """Run rule-based signal detection."""
        text_lower = text.lower() if text else ""

        signal_scores = {}
        for signal_name, keywords in self.SIGNAL_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = min(1.0, matches * 0.25)
            signal_scores[signal_name] = round(score, 3)

        # Generate pseudo attention weights
        words = text_lower.split() if text_lower else []
        all_keywords = [kw for kws in self.SIGNAL_KEYWORDS.values() for kw in kws]
        attention = [0.8 if any(kw in word for kw in all_keywords) else 0.1 for word in words]

        # Risk embedding (simplified 128-dim)
        risk_embedding = np.zeros(128)
        for i, (name, score) in enumerate(signal_scores.items()):
            risk_embedding[i * 16:(i + 1) * 16] = score

        return {
            "signal_scores": signal_scores,
            "attention_weights": attention[:50],
            "risk_embedding": risk_embedding.tolist(),
        }


# Factory function
def create_weak_signal_detector(use_transformer: bool = True):
    """Create the appropriate detector based on available dependencies."""
    if HAS_TORCH and use_transformer:
        try:
            model = WeakSignalNLPModel()
            logger.info("Created WeakSignalNLPModel (DistilBERT)")
            return model
        except Exception as e:
            logger.warning(f"Could not create transformer model: {e}")

    logger.info("Created WeakSignalDetectorFallback (rule-based)")
    return WeakSignalDetectorFallback()
