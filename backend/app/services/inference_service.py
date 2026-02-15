"""
Inference Service — orchestrates ML models for risk assessment.
Combines NLP weak signal detection, time-series anomaly detection,
and a trained LightGBM+XGBoost ensemble from the RAND HRS dataset.
"""

import os
import sys
import time
import uuid
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Ensure the ml/ directory is on sys.path so we can import prediction_service
_ml_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml")
_ml_dir = os.path.abspath(_ml_dir)
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)


class InferenceService:
    """
    Orchestrates the ML inference pipeline:
    1. WeakSignalDetector_NLP → extract symptom language patterns
    2. TimeSeriesAnomalyDetector → detect behavioral anomalies
    3. _map_to_hrs_features → bridge user data → trained model features
    4. Trained ensemble (LightGBM + XGBoost) → predict risk
    5. Fusion → weighted combination of rule-based + trained model
    """

    RISK_THRESHOLDS = {
        "LOW": (0.0, 0.25),
        "WEAK": (0.25, 0.50),
        "MODERATE": (0.50, 0.75),
        "HIGH": (0.75, 1.0),
    }

    HIGH_CONCERN_KEYWORDS = [
        "chest pain", "shortness of breath", "severe headache",
        "vision loss", "numbness", "fainting", "blood",
        "heart palpitations", "difficulty breathing", "confusion"
    ]

    MODERATE_CONCERN_KEYWORDS = [
        "persistent pain", "recurring", "worsening", "chronic",
        "can't sleep", "losing weight", "always tired", "anxious",
        "depressed", "dizzy", "nausea", "fever"
    ]

    def __init__(self):
        self._trained_predictor = None
        self._loaded = False

    # ── Model Loading ─────────────────────────────────────
    def _ensure_models_loaded(self):
        """Lazy-load the trained ensemble model on first call."""
        if not self._loaded:
            logger.info("Loading ML models for inference...")
            try:
                from prediction_service import HealthRiskPredictor
                model_dir = os.path.join(_ml_dir, "hea_results")
                if os.path.isdir(model_dir):
                    self._trained_predictor = HealthRiskPredictor(model_dir=model_dir)
                    if self._trained_predictor.load_models():
                        logger.info(f"✅ Trained ensemble model loaded from {model_dir}")
                    else:
                        self._trained_predictor = None
                        logger.warning("Trained model load returned False — using rule-based fallback")
                else:
                    logger.warning(f"No hea_results at {model_dir} — using rule-based fallback")
            except Exception as e:
                logger.warning(f"Could not load trained model: {e}. Using rule-based fallback.")
                self._trained_predictor = None
            self._loaded = True

    # ── Main Entry Point ──────────────────────────────────
    def assess_risk(
        self,
        symptom_text: Optional[str],
        emoji_inputs: list,
        checkbox_selections: list,
        daily_metrics: Optional[dict],
        historical_metrics: Optional[list] = None,
    ) -> dict:
        start_time = time.time()
        self._ensure_models_loaded()

        nlp_signals = self._analyze_text_signals(symptom_text, emoji_inputs, checkbox_selections)
        ts_signals = self._analyze_timeseries_signals(daily_metrics, historical_metrics)
        hrs_features = self._map_to_hrs_features(
            symptom_text, checkbox_selections, daily_metrics, nlp_signals, ts_signals
        )
        result = self._fuse_signals(nlp_signals, ts_signals, hrs_features)

        inference_time_ms = (time.time() - start_time) * 1000
        result["inference_time_ms"] = round(inference_time_ms, 2)
        logger.info(
            f"Risk assessment completed: level={result['risk_level']}, "
            f"confidence={result['confidence_score']:.2f}, "
            f"time={inference_time_ms:.1f}ms"
        )
        return result

    # ── HRS Feature Mapping ───────────────────────────────
    def _map_to_hrs_features(
        self,
        symptom_text: Optional[str],
        checkbox_selections: list,
        daily_metrics: Optional[dict],
        nlp_signals: dict,
        ts_signals: dict,
    ) -> dict:
        """
        Map user daily-input data to HRS survey features that the trained
        model expects. This is the critical bridge between the frontend UI
        and the trained LightGBM/XGBoost ensemble.
        """
        features = {}
        text_lower = (symptom_text or "").lower()

        # ── Self-Rated Health (r13shlt) — Top SHAP feature ──
        self_rated = 3  # default: "Good"
        if "self-rated health:" in text_lower:
            if "poor" in text_lower:
                self_rated = 5
            elif "fair" in text_lower:
                self_rated = 4
            elif "good" in text_lower and "very good" not in text_lower:
                self_rated = 3
            elif "very good" in text_lower:
                self_rated = 2
            elif "excellent" in text_lower:
                self_rated = 1
        else:
            total_score = nlp_signals["score"] * 0.6 + ts_signals["score"] * 0.4
            if total_score >= 0.7:
                self_rated = 5
            elif total_score >= 0.5:
                self_rated = 4
            elif total_score >= 0.3:
                self_rated = 3
            elif total_score >= 0.1:
                self_rated = 2
            else:
                self_rated = 1
        features["r13shlt"] = self_rated

        # ── Depression (r13cesd) ──
        cesd = 0
        dep_keywords = [
            "felt_depressed", "everything_effort", "sleep_restless",
            "not_happy", "felt_lonely", "people_unfriendly", "felt_sad"
        ]
        if "depression indicators:" in text_lower:
            for kw in dep_keywords:
                if kw in text_lower:
                    cesd += 1
        elif any(kw in text_lower for kw in ["depressed", "sad", "hopeless", "lonely", "anxious"]):
            cesd = 3
        features["r13cesd"] = cesd

        # ── Pain Frequency (r13painf) ──
        pain_freq = 0
        if "pain frequency: most" in text_lower or "pain frequency: every" in text_lower:
            pain_freq = 2
        elif "pain frequency: some" in text_lower:
            pain_freq = 1
        pain_checks = ["chest_pain", "back_pain", "joint_pain", "headache"]
        pain_count = len([c for c in checkbox_selections if c in pain_checks])
        if pain_count >= 2:
            pain_freq = max(pain_freq, 2)
        elif pain_count >= 1:
            pain_freq = max(pain_freq, 1)
        features["r13painf"] = pain_freq

        # ── Walking Difficulty (r13walkra) ──
        walk_diff = 0
        if "activity limitations:" in text_lower:
            if "walking" in text_lower or "climbing" in text_lower:
                walk_diff = 1
        features["r13walkra"] = walk_diff

        # ── ADL Limitations (r13adla) ──
        adl = 0
        adl_keys = ["dressing", "bathing", "getting_in_out_bed"]
        if "activity limitations:" in text_lower:
            for k in adl_keys:
                if k.replace("_", " ") in text_lower or k in text_lower:
                    adl += 1
        features["r13adla"] = adl

        # ── Sleep Problems (r13sleepr) ──
        sleep_prob = 0
        if daily_metrics:
            sleep_hrs = daily_metrics.get("sleep_hours", 7)
            if sleep_hrs < 5:
                sleep_prob = 2
            elif sleep_hrs < 6:
                sleep_prob = 1
        if "sleep_restless" in text_lower or "can't sleep" in text_lower:
            sleep_prob = max(sleep_prob, 1)
        features["r13sleepr"] = sleep_prob

        # ── BMI proxy (r13bmi) ──
        features["r13bmi"] = 26.5

        # ── Doctor visits proxy (r13doctim) ──
        doc_visits = 2
        if self_rated >= 4 or cesd >= 3:
            doc_visits = 6
        features["r13doctim"] = doc_visits

        # ── Age proxy (r13agey_b) ──
        features["r13agey_b"] = 55

        logger.info(f"HRS features mapped: {features}")
        return features

    # ── NLP Signal Detection ──────────────────────────────
    def _analyze_text_signals(
        self,
        symptom_text: Optional[str],
        emoji_inputs: list,
        checkbox_selections: list,
    ) -> dict:
        signals = []
        score = 0.0

        if symptom_text:
            text_lower = symptom_text.lower()

            # Self-rated health signal
            if "self-rated health:" in text_lower:
                for level in ["poor", "fair"]:
                    if level in text_lower:
                        signals.append({
                            "signal": f"Self-rated health: {'below average' if level == 'fair' else level}",
                            "weight": 0.65 if level == "poor" else 0.45,
                            "category": "nlp",
                        })
                        score = max(score, 0.65 if level == "poor" else 0.45)

            for keyword in self.HIGH_CONCERN_KEYWORDS:
                if keyword in text_lower:
                    signals.append({
                        "signal": f"Critical symptom detected: '{keyword}'",
                        "weight": 0.8,
                        "category": "nlp",
                    })
                    score = max(score, 0.8)

            for keyword in self.MODERATE_CONCERN_KEYWORDS:
                if keyword in text_lower:
                    signals.append({
                        "signal": f"Notable symptom: '{keyword}'",
                        "weight": 0.5,
                        "category": "nlp",
                    })
                    score = max(score, 0.5)

            if "depression indicators:" in text_lower:
                dep_count = text_lower.count(",") + 1 if "depression indicators:" in text_lower else 0
                if dep_count >= 3:
                    signals.append({
                        "signal": f"Multiple depression indicators reported ({dep_count} items)",
                        "weight": 0.6,
                        "category": "nlp",
                    })
                    score = max(score, 0.6)

        # Checkbox analysis
        high_concern_checks = ["chest_pain", "shortness_of_breath", "heart_palpitations"]
        moderate_concern_checks = ["headache", "fatigue", "insomnia", "anxiety", "dizziness",
                                   "nausea", "back_pain", "joint_pain", "muscle_weakness", "vision_changes"]

        for check in checkbox_selections:
            if check in high_concern_checks:
                signals.append({
                    "signal": f"Critical symptom: {check.replace('_', ' ')}",
                    "weight": 0.7,
                    "category": "nlp",
                })
                score = max(score, 0.7)
            elif check in moderate_concern_checks:
                signals.append({
                    "signal": f"Notable symptom: {check.replace('_', ' ')}",
                    "weight": 0.4,
                    "category": "nlp",
                })
                score = max(score, 0.4)

        if not signals:
            signals.append({
                "signal": "No concerning patterns detected in reported symptoms",
                "weight": 0.0,
                "category": "nlp",
            })

        return {"signals": signals, "score": round(score, 3)}

    # ── Time-Series Signal Detection ──────────────────────
    def _analyze_timeseries_signals(
        self,
        daily_metrics: Optional[dict],
        historical_metrics: Optional[list] = None,
    ) -> dict:
        signals = []
        score = 0.0

        if not daily_metrics:
            return {"signals": [{"signal": "Daily metrics within normal range", "weight": 0.0, "category": "timeseries"}], "score": 0.0}

        sleep = daily_metrics.get("sleep_hours")
        if sleep is not None:
            if sleep < 4:
                signals.append({"signal": f"Critically low sleep: {sleep}h", "weight": 0.7, "category": "timeseries"})
                score = max(score, 0.7)
            elif sleep < 6:
                signals.append({"signal": f"Below-average sleep: {sleep}h", "weight": 0.4, "category": "timeseries"})
                score = max(score, 0.4)

        mood = daily_metrics.get("mood_score")
        if mood is not None:
            if mood <= 2:
                signals.append({"signal": f"Very low mood: {mood}/10", "weight": 0.6, "category": "timeseries"})
                score = max(score, 0.6)
            elif mood <= 4:
                signals.append({"signal": f"Low mood: {mood}/10", "weight": 0.35, "category": "timeseries"})
                score = max(score, 0.35)

        energy = daily_metrics.get("energy_level")
        if energy is not None and energy <= 3:
            signals.append({"signal": f"Low energy: {energy}/10", "weight": 0.4, "category": "timeseries"})
            score = max(score, 0.4)

        stress = daily_metrics.get("stress_level")
        if stress is not None and stress >= 8:
            signals.append({"signal": f"High stress: {stress}/10", "weight": 0.5, "category": "timeseries"})
            score = max(score, 0.5)

        if not signals:
            signals.append({"signal": "Daily metrics within normal range", "weight": 0.0, "category": "timeseries"})

        return {"signals": signals, "score": round(score, 3)}

    # ── Fusion ────────────────────────────────────────────
    def _fuse_signals(self, nlp_signals: dict, ts_signals: dict, hrs_features: dict) -> dict:
        # Rule-based score
        rule_score = round(nlp_signals["score"] * 0.55 + ts_signals["score"] * 0.45, 3)
        rule_score = min(rule_score, 1.0)

        # Try trained model
        trained_result = None
        if self._trained_predictor is not None:
            try:
                trained_result = self._trained_predictor.predict(hrs_features)
                logger.info(
                    f"Trained model result: {trained_result['risk_level']} "
                    f"(prob={trained_result['risk_probability']:.3f})"
                )
            except Exception as e:
                logger.warning(f"Trained model prediction failed: {e}")

        # Determine final score
        if trained_result:
            combined_score = round(
                trained_result["risk_probability"] * 0.60 + rule_score * 0.40,
                3
            )
            model_version = "v2.0-ensemble (LightGBM+XGBoost)"
        else:
            combined_score = rule_score
            model_version = "v0.1.0-rule-based"

        # Determine risk level
        risk_level = "LOW"
        for level, (low, high) in self.RISK_THRESHOLDS.items():
            if low <= combined_score < high:
                risk_level = level
                break
        if combined_score >= 1.0:
            risk_level = "HIGH"

        # Generate explanation
        explanation = self._generate_explanation(
            risk_level, nlp_signals, ts_signals, combined_score, trained_result
        )

        # Combine all signal details
        all_signals = nlp_signals["signals"] + ts_signals["signals"]
        all_signals.sort(key=lambda s: s["weight"], reverse=True)

        return {
            "risk_level": risk_level,
            "confidence_score": combined_score,
            "explanation_text": explanation,
            "signal_details": {
                "nlp_score": nlp_signals["score"],
                "timeseries_score": ts_signals["score"],
                "combined_score": combined_score,
                "signals": all_signals[:10],
            },
            "model_version": model_version,
        }

    def _generate_explanation(self, risk_level, nlp_signals, ts_signals,
                              combined_score, trained_result=None):
        explanations = {
            "LOW": "Your recent inputs look good! No concerning patterns detected. Keep tracking to maintain your health awareness.",
            "WEAK": "We noticed a few subtle signals in your recent inputs. Nothing urgent, but worth keeping an eye on. Continue logging daily.",
            "MODERATE": "We've detected some patterns that suggest you might want to pay closer attention. Consider reviewing the signals below and consulting a healthcare professional.",
            "HIGH": "We've detected several concerning signals. We strongly recommend speaking with a healthcare professional soon. Remember, Hea is a wellness tool — not a medical diagnosis.",
        }

        base = explanations.get(risk_level, explanations["LOW"])

        top_nlp = [s for s in nlp_signals["signals"] if s["weight"] > 0.3]
        top_ts = [s for s in ts_signals["signals"] if s["weight"] > 0.3]

        details = []
        if top_nlp:
            details.append(f"Your symptom reports raised {len(top_nlp)} notable signal(s)")
        if top_ts:
            details.append(f"Your daily metrics showed {len(top_ts)} pattern change(s)")
        if trained_result:
            details.append(
                f"ML ensemble confidence: {trained_result['risk_probability']:.0%}"
            )

        if details:
            base += " " + ". ".join(details) + "."

        return base


# Singleton instance
inference_service = InferenceService()
