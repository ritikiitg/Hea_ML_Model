"""
Hea ML Preprocessing Pipeline
Text normalization, emoji tokenization, and spell correction for health inputs.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import optional dependencies gracefully
try:
    import emoji as emoji_lib
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    logger.warning("emoji library not available, using basic emoji processing")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logger.warning("textblob not available, spell correction disabled")


class TextPreprocessor:
    """Normalizes and preprocesses text inputs for ML feature extraction."""

    # Common health-related abbreviations
    HEALTH_ABBREVIATIONS = {
        "headache": "headache",
        "ha": "headache",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "n/v": "nausea and vomiting",
        "abd": "abdominal",
        "bp": "blood pressure",
        "hr": "heart rate",
        "temp": "temperature",
        "wt": "weight",
        "bmi": "body mass index",
        "lbp": "lower back pain",
        "gerd": "acid reflux",
        "ibs": "irritable bowel syndrome",
        "uti": "urinary tract infection",
        "rls": "restless leg syndrome",
    }

    # Common symptom misspellings
    SYMPTOM_CORRECTIONS = {
        "headake": "headache",
        "hedache": "headache",
        "stomache": "stomach ache",
        "diarhea": "diarrhea",
        "diarrhoea": "diarrhea",
        "nausia": "nausea",
        "dizzyness": "dizziness",
        "tierd": "tired",
        "anxiuos": "anxious",
        "insomina": "insomnia",
        "fatique": "fatigue",
        "migriane": "migraine",
        "brething": "breathing",
        "palpatations": "palpitations",
        "congestion": "congestion",
    }

    @classmethod
    def normalize_text(cls, text: Optional[str]) -> Optional[str]:
        """Full text normalization pipeline."""
        if not text:
            return None

        # Step 1: Lowercase
        text = text.lower().strip()

        # Step 2: Expand abbreviations
        text = cls._expand_abbreviations(text)

        # Step 3: Correct common misspellings
        text = cls._correct_spelling(text)

        # Step 4: Normalize whitespace & punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)
        text = text.strip()

        return text if text else None

    @classmethod
    def tokenize_emojis(cls, emoji_list: list[str]) -> list[str]:
        """Convert emojis to descriptive text tokens."""
        tokens = []
        for item in emoji_list:
            if HAS_EMOJI and emoji_lib.is_emoji(item):
                desc = emoji_lib.demojize(item, delimiters=("", ""))
                desc = desc.replace("_", " ").replace(":", "").strip()
                tokens.append(desc)
            else:
                # Already text, just normalize
                normalized = cls.normalize_text(item)
                if normalized:
                    tokens.append(normalized)
        return tokens

    @classmethod
    def extract_symptom_keywords(cls, text: Optional[str]) -> list[str]:
        """Extract health-related keywords from text."""
        if not text:
            return []

        SYMPTOM_KEYWORDS = [
            "headache", "migraine", "fatigue", "tired", "exhausted",
            "nausea", "vomiting", "dizziness", "dizzy", "insomnia",
            "anxiety", "anxious", "depressed", "depression", "stress",
            "pain", "ache", "sore", "cramp", "inflammation",
            "fever", "cough", "cold", "flu", "congestion",
            "shortness of breath", "breathing", "palpitations",
            "chest pain", "numbness", "tingling", "weakness",
            "appetite", "weight loss", "weight gain", "bloating",
            "constipation", "diarrhea", "rash", "itch", "swelling",
            "blurry vision", "tinnitus", "hearing", "memory",
            "concentration", "brain fog", "restless", "sweating",
        ]

        text_lower = text.lower()
        found = [kw for kw in SYMPTOM_KEYWORDS if kw in text_lower]
        return found

    @classmethod
    def _expand_abbreviations(cls, text: str) -> str:
        """Replace health abbreviations with full terms."""
        words = text.split()
        expanded = []
        for word in words:
            clean = word.strip(".,!?")
            if clean in cls.HEALTH_ABBREVIATIONS:
                expanded.append(cls.HEALTH_ABBREVIATIONS[clean])
            else:
                expanded.append(word)
        return " ".join(expanded)

    @classmethod
    def _correct_spelling(cls, text: str) -> str:
        """Correct common health-related misspellings."""
        # First, apply known corrections
        for wrong, right in cls.SYMPTOM_CORRECTIONS.items():
            text = text.replace(wrong, right)

        # Optionally use TextBlob for general spelling correction
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                corrected = str(blob.correct())
                # Only use correction if it's not too different (avoid over-correction)
                if len(corrected) < len(text) * 2:
                    return corrected
            except Exception:
                pass

        return text


class MetricsPreprocessor:
    """Preprocesses daily metrics for time-series analysis."""

    # Normal ranges for z-score calculation
    NORMAL_RANGES = {
        "sleep_hours": {"mean": 7.5, "std": 1.5},
        "mood_score": {"mean": 6.0, "std": 2.0},
        "energy_level": {"mean": 6.0, "std": 2.0},
        "stress_level": {"mean": 4.0, "std": 2.0},
        "steps_count": {"mean": 8000, "std": 3000},
        "water_intake_ml": {"mean": 2000, "std": 500},
    }

    @classmethod
    def normalize_metrics(cls, metrics: dict) -> dict:
        """Normalize daily metrics using z-score standardization."""
        normalized = {}
        for key, value in metrics.items():
            if value is not None and key in cls.NORMAL_RANGES:
                mean = cls.NORMAL_RANGES[key]["mean"]
                std = cls.NORMAL_RANGES[key]["std"]
                normalized[key] = round((value - mean) / std, 4) if std else 0.0
                normalized[f"{key}_raw"] = value
            else:
                normalized[key] = value
        return normalized

    @classmethod
    def compute_rolling_features(cls, historical: list[dict], window: int = 3) -> dict:
        """Compute rolling statistics from historical metrics."""
        features = {}

        for metric in ["sleep_hours", "mood_score", "energy_level", "stress_level"]:
            values = [h.get(metric) for h in historical if h.get(metric) is not None]

            if len(values) >= window:
                recent = values[-window:]
                features[f"{metric}_rolling_mean"] = round(sum(recent) / len(recent), 2)
                features[f"{metric}_rolling_std"] = round(
                    (sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)) ** 0.5, 2
                )

                # Trend direction
                if len(values) >= window * 2:
                    older = values[-window * 2 : -window]
                    older_avg = sum(older) / len(older)
                    recent_avg = sum(recent) / len(recent)
                    features[f"{metric}_trend"] = round(recent_avg - older_avg, 2)
                    features[f"{metric}_trend_dir"] = "improving" if recent_avg > older_avg else "declining"

        return features
