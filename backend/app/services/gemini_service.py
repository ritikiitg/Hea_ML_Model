"""
Hea Gemini AI Service — Uses Gemini Pro for deep health pattern analysis
and Gemini Flash for quick wellness tips and daily summaries.

Pro Model:  Deep analysis of symptom patterns, risk explanations, lifestyle correlations
Flash Model: Quick daily tips, mood-based suggestions, instant feedback
"""

import logging
import json
from typing import Optional
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

# ─── System Prompts ───────────────────────────────────

SYSTEM_PROMPT_PRO = """You are Hea's AI Health Insight Engine — a wellness companion that analyzes 
self-reported health data to detect subtle patterns. You are NOT a doctor and NEVER diagnose.

Your role:
1. Analyze patterns in sleep, mood, energy, stress, and symptom data
2. Identify subtle correlations the user might miss (e.g., sleep dip → mood drop → fatigue)
3. Provide clear, non-clinical, empathetic explanations
4. Suggest actionable wellness steps (hydration, sleep hygiene, stress management)
5. Flag when patterns suggest the user should consult a professional

Tone: Warm, supportive, evidence-aware. Like a knowledgeable friend, not a clinician.

IMPORTANT RULES:
- Never use medical terminology or suggest diagnoses
- Always include a disclaimer that you're a wellness tool, not medical advice
- Focus on patterns and lifestyle adjustments
- Be encouraging, not alarming
- Use plain language and emojis where appropriate for warmth

Respond in JSON format with these fields:
{
  "summary": "1-2 sentence overview of what you noticed",
  "patterns": ["list of 2-4 detected patterns"],
  "recommendations": ["list of 2-4 actionable suggestions"],
  "risk_note": "brief note on risk level (reassuring or cautionary)",
  "wellness_score_reasoning": "why you'd rate their current wellness trend",
  "disclaimer": "standard wellness disclaimer"
}"""

SYSTEM_PROMPT_FLASH = """You are Hea's quick wellness assistant. Given a user's daily health log, 
provide a brief, encouraging response with 1-2 quick tips. Keep it warm, short, and actionable.
Never diagnose. Use emojis. Respond in JSON:
{
  "greeting": "personalized 1-line greeting based on their data",
  "quick_tip": "one actionable wellness tip",
  "encouragement": "brief motivational note"
}"""


class GeminiService:
    """Manages Gemini Pro and Flash API calls for health insights."""

    def __init__(self):
        self.client = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the Gemini client."""
        if self._initialized:
            return

        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — AI insights will be unavailable")
            return

        try:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._initialized = True
            logger.info("Gemini AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def is_available(self) -> bool:
        """Check if Gemini service is configured and available."""
        self._ensure_initialized()
        return self.client is not None

    async def analyze_health_patterns(
        self,
        health_data: dict,
        user_context: Optional[str] = None,
    ) -> dict:
        """
        Deep analysis using Gemini Pro.
        Analyzes multi-day health data for pattern detection.
        """
        self._ensure_initialized()
        if not self.client:
            return self._fallback_analysis(health_data)

        prompt = self._build_analysis_prompt(health_data, user_context)

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_PRO_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_PRO,
                    temperature=0.7,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)
            logger.info("Gemini Pro analysis completed successfully")
            return {
                "source": "gemini-pro",
                "model": settings.GEMINI_PRO_MODEL,
                "analysis": result,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Gemini Pro analysis failed: {e}")
            return self._fallback_analysis(health_data)

    async def get_quick_tip(self, daily_log: dict) -> dict:
        """
        Quick response using Gemini Flash.
        Provides instant feedback on a single daily log entry.
        """
        self._ensure_initialized()
        if not self.client:
            return self._fallback_tip(daily_log)

        prompt = self._build_tip_prompt(daily_log)

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_FLASH_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_FLASH,
                    temperature=0.8,
                    max_output_tokens=300,
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)
            logger.info("Gemini Flash tip generated successfully")
            return {
                "source": "gemini-flash",
                "model": settings.GEMINI_FLASH_MODEL,
                "tip": result,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Gemini Flash tip failed: {e}")
            return self._fallback_tip(daily_log)

    # ─── Prompt Builders ──────────────────────────────

    def _build_analysis_prompt(self, health_data: dict, user_context: Optional[str]) -> str:
        """Build a structured prompt for Gemini Pro pattern analysis."""
        parts = ["Here is the user's recent health data:\n"]

        if "inputs" in health_data:
            parts.append("## Daily Logs")
            for entry in health_data["inputs"][-7:]:  # Last 7 days
                parts.append(f"- Date: {entry.get('created_at', 'unknown')}")
                if entry.get("free_text_input"):
                    parts.append(f"  Text: \"{entry['free_text_input']}\"")
                if entry.get("selected_symptoms"):
                    parts.append(f"  Symptoms: {', '.join(entry['selected_symptoms'])}")
                if entry.get("daily_metrics"):
                    metrics = entry["daily_metrics"]
                    parts.append(f"  Metrics: sleep={metrics.get('sleep_hours', '?')}h, mood={metrics.get('mood_score', '?')}/10, energy={metrics.get('energy_level', '?')}/10, stress={metrics.get('stress_level', '?')}/10")
                parts.append("")

        if "assessment" in health_data:
            a = health_data["assessment"]
            parts.append(f"\n## Latest Risk Assessment")
            parts.append(f"Risk Level: {a.get('risk_level', 'unknown')}")
            parts.append(f"Confidence: {a.get('confidence_score', 0):.0%}")
            if a.get("signal_details"):
                parts.append("Signals detected:")
                for sig in a["signal_details"].get("signals", [])[:5]:
                    parts.append(f"  - {sig.get('signal', '')}")

        if user_context:
            parts.append(f"\n## Additional Context\n{user_context}")

        parts.append("\nPlease analyze these patterns and provide insights.")
        return "\n".join(parts)

    def _build_tip_prompt(self, daily_log: dict) -> str:
        """Build a prompt for Gemini Flash quick tip."""
        parts = ["Today's health log:\n"]

        if daily_log.get("free_text_input"):
            parts.append(f"How they feel: \"{daily_log['free_text_input']}\"")
        if daily_log.get("emoji_input"):
            parts.append(f"Mood emoji: {daily_log['emoji_input']}")
        if daily_log.get("selected_symptoms"):
            parts.append(f"Symptoms: {', '.join(daily_log['selected_symptoms'])}")
        if daily_log.get("daily_metrics"):
            m = daily_log["daily_metrics"]
            parts.append(f"Sleep: {m.get('sleep_hours', '?')}h, Mood: {m.get('mood_score', '?')}/10, Energy: {m.get('energy_level', '?')}/10, Stress: {m.get('stress_level', '?')}/10")

        parts.append("\nGive a quick, encouraging wellness tip.")
        return "\n".join(parts)

    # ─── Fallbacks ────────────────────────────────────

    def _fallback_analysis(self, health_data: dict) -> dict:
        """Rule-based fallback when Gemini is unavailable."""
        return {
            "source": "rule-based-fallback",
            "analysis": {
                "summary": "Based on your recent logs, your patterns are being tracked. Keep logging daily for better insights!",
                "patterns": [
                    "Regular logging detected — this helps improve accuracy",
                    "Your data is building a personal baseline",
                ],
                "recommendations": [
                    "Continue logging daily for at least 7 days for pattern detection",
                    "Try to log at a consistent time each day",
                    "Include both text descriptions and metric sliders for best results",
                ],
                "risk_note": "Not enough AI-analyzed data yet. Keep going! 💪",
                "wellness_score_reasoning": "More data needed for accurate trend assessment",
                "disclaimer": "Hea is a wellness tool and does not provide medical advice. Consult a healthcare professional for medical concerns.",
            },
            "success": False,
        }

    def _fallback_tip(self, daily_log: dict) -> dict:
        """Rule-based fallback for quick tips."""
        mood = daily_log.get("daily_metrics", {}).get("mood_score", 5)
        if mood >= 7:
            greeting = "Great to see you're feeling good today! 🌟"
            tip = "Keep the positive momentum — maybe share your energy with someone who needs it."
        elif mood >= 4:
            greeting = "Thanks for checking in today! 👋"
            tip = "A 10-minute walk can boost your mood and energy — try stepping outside."
        else:
            greeting = "We're glad you're here. Taking a moment to log is self-care. 💙"
            tip = "Be gentle with yourself today. Try deep breathing: 4 seconds in, 7 hold, 8 out."

        return {
            "source": "rule-based-fallback",
            "tip": {
                "greeting": greeting,
                "quick_tip": tip,
                "encouragement": "Every log helps Hea understand you better. You're doing great! ✨",
            },
            "success": False,
        }


# Singleton instance
gemini_service = GeminiService()
