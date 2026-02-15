"""
Input Sanitizer — OWASP-compliant input sanitization for all user inputs.
Handles XSS prevention, SQL injection prevention, emoji normalization.
"""

import re
import bleach
import emoji


class InputSanitizer:
    """Sanitizes user inputs for security and consistency."""

    # Allowed HTML tags (none for health inputs)
    ALLOWED_TAGS: list = []
    ALLOWED_ATTRIBUTES: dict = {}

    # Suspicious patterns for SQL injection
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|;|/\*|\*/|xp_|sp_)",
        r"('|\"|\\)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"document\.(cookie|location|write)",
    ]

    @classmethod
    def sanitize_text(cls, text: str | None) -> str | None:
        """Sanitize free-text input: strip HTML, check for injection patterns."""
        if text is None:
            return None

        # Strip HTML tags
        text = bleach.clean(text, tags=cls.ALLOWED_TAGS, attributes=cls.ALLOWED_ATTRIBUTES, strip=True)

        # Remove potential SQL injection patterns
        for pattern in cls.SQL_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove XSS patterns
        for pattern in cls.XSS_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text if text else None

    @classmethod
    def sanitize_emoji_inputs(cls, emojis: list[str]) -> list[str]:
        """Convert emojis to descriptive text tokens for ML processing."""
        sanitized = []
        for e in emojis:
            if emoji.is_emoji(e):
                # Convert emoji to descriptive text: 😴 → "sleeping_face"
                demojized = emoji.demojize(e, delimiters=("", ""))
                sanitized.append(demojized.replace(":", "").replace("_", " ").strip())
            else:
                # Sanitize as text
                clean = cls.sanitize_text(e)
                if clean:
                    sanitized.append(clean)
        return sanitized

    @classmethod
    def sanitize_checkbox_selections(cls, selections: list[str]) -> list[str]:
        """Validate checkbox selections against allowed categories."""
        ALLOWED_CATEGORIES = [
            "headache", "fatigue", "nausea", "dizziness", "insomnia",
            "anxiety", "joint_pain", "muscle_ache", "shortness_of_breath",
            "chest_pain", "stomach_pain", "back_pain", "fever", "cough",
            "sore_throat", "congestion", "appetite_change", "weight_change",
            "skin_changes", "vision_changes", "mood_changes", "concentration",
            "memory", "digestive_issues", "heart_palpitations", "sweating",
            "numbness", "tingling", "other"
        ]
        return [s.lower().strip() for s in selections if s.lower().strip() in ALLOWED_CATEGORIES]

    @classmethod
    def validate_input_length(cls, text: str | None, max_length: int = 5000) -> bool:
        """Check input doesn't exceed maximum length."""
        if text is None:
            return True
        return len(text) <= max_length

    @classmethod
    def sanitize_health_input(cls, symptom_text: str | None, emoji_inputs: list,
                               checkbox_selections: list) -> dict:
        """Full sanitization pipeline for a health input submission."""
        return {
            "symptom_text": cls.sanitize_text(symptom_text),
            "emoji_inputs": cls.sanitize_emoji_inputs(emoji_inputs or []),
            "checkbox_selections": cls.sanitize_checkbox_selections(checkbox_selections or []),
        }
