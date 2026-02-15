"""
Hea ML — Trained Model Prediction Service

Loads the trained LightGBM/XGBoost ensemble model and provides real-time
risk predictions with SHAP-based explanations.

Usage:
    from ml.prediction_service import prediction_service
    result = prediction_service.predict(health_features)
"""

import os
import logging
import numpy as np
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class HealthRiskPredictor:
    """
    Production prediction service using the trained ensemble model.
    
    Loads pre-trained LightGBM model, XGBoost model, meta-learner,
    imputer, and feature list from the results directory.
    """

    # Feature-to-human-readable mapping for SHAP explanations
    FEATURE_LABELS = {
        "r13shlt": "Self-rated health (latest)",
        "r13cesd": "Depression score (latest)",
        "r13bmi": "BMI (latest)",
        "r13adltot_m": "Daily activity limitations",
        "r13iadlza": "Instrumental ADL limitations",
        "r13paina": "Pain frequency",
        "r13sleepr": "Sleep problems",
        "r13agey_e": "Age",
        "ragender": "Gender",
        "slope_shlt": "Health trajectory (slope)",
        "slope_cesd": "Depression trajectory (slope)",
        "slope_bmi": "BMI trajectory (slope)",
        "volatility_shlt": "Health variability",
        "volatility_cesd": "Depression variability",
        "frailty_index": "Frailty index",
        "concern_count": "Number of health concerns",
        "total_missing_pct": "Data completeness",
        "health_x_depression": "Health × Depression interaction",
        "zscore_shlt": "Health vs. age-peers",
        "zscore_cesd": "Depression vs. age-peers",
    }

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir or os.environ.get("HEA_MODEL_DIR", "./hea_results"))
        self.lgb_model = None
        self.xgb_model = None
        self.meta_model = None
        self.imputer = None
        self.feature_cols = None
        self.threshold = None
        self.shap_explainer = None
        self._loaded = False

    def load_models(self) -> bool:
        """Load all trained model artifacts."""
        if not HAS_JOBLIB:
            logger.error("joblib not installed — cannot load models")
            return False

        try:
            self.lgb_model = joblib.load(self.model_dir / "lightgbm_model.pkl")
            self.xgb_model = joblib.load(self.model_dir / "xgboost_model.pkl")
            self.meta_model = joblib.load(self.model_dir / "meta_learner.pkl")
            self.imputer = joblib.load(self.model_dir / "imputer.pkl")
            self.feature_cols = joblib.load(self.model_dir / "feature_cols.pkl")
            self.threshold = joblib.load(self.model_dir / "threshold.pkl")

            # Initialize SHAP explainer
            if HAS_SHAP:
                try:
                    self.shap_explainer = shap.TreeExplainer(self.lgb_model)
                except Exception as e:
                    logger.warning(f"SHAP explainer init failed: {e}")

            self._loaded = True
            logger.info(f"Models loaded from {self.model_dir} ({len(self.feature_cols)} features)")
            return True

        except FileNotFoundError as e:
            logger.error(f"Model files not found in {self.model_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict(self, features: dict) -> dict:
        """
        Predict health risk for a single person.

        Args:
            features: Dict of {feature_name: value} for the person.
                      Missing features will be imputed with training medians.

        Returns:
            dict with: risk_level, risk_probability, confidence_score,
                       top_risk_factors, explanation
        """
        if not self._loaded:
            self.load_models()
            if not self._loaded:
                return {"error": "Models not loaded", "risk_level": "UNKNOWN"}

        # Build feature vector in correct order
        feature_vector = np.array([
            features.get(col, np.nan) for col in self.feature_cols
        ]).reshape(1, -1)

        # Impute missing values
        feature_vector = self.imputer.transform(feature_vector)

        # Get predictions from both models
        lgb_proba = self.lgb_model.predict_proba(feature_vector)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(feature_vector)[:, 1]

        # Meta-learner fusion
        meta_input = np.column_stack([lgb_proba, xgb_proba])
        final_proba = self.meta_model.predict_proba(meta_input)[:, 1][0]

        # Apply threshold
        is_at_risk = final_proba >= self.threshold

        # Determine risk level
        if final_proba >= 0.7:
            risk_level = "HIGH"
        elif final_proba >= 0.4:
            risk_level = "MODERATE"
        elif final_proba >= self.threshold:
            risk_level = "WEAK"
        else:
            risk_level = "LOW"

        # Generate SHAP explanation
        top_factors = self._explain(feature_vector)

        return {
            "risk_level": risk_level,
            "risk_probability": round(float(final_proba), 4),
            "is_at_risk": bool(is_at_risk),
            "confidence_score": round(float(min(abs(final_proba - 0.5) * 2, 1.0)), 4),
            "threshold": float(self.threshold),
            "top_risk_factors": top_factors,
            "model_version": "hea-ensemble-v2",
        }

    def _explain(self, feature_vector: np.ndarray, top_n: int = 8) -> list:
        """Generate top risk factor explanations using SHAP."""
        if self.shap_explainer is None:
            # Fallback: use feature importance
            importances = self.lgb_model.feature_importances_
            top_idx = np.argsort(importances)[-top_n:][::-1]
            return [
                {
                    "feature": self.feature_cols[i],
                    "label": self.FEATURE_LABELS.get(self.feature_cols[i], self.feature_cols[i]),
                    "importance": float(importances[i]),
                    "value": float(feature_vector[0, i]),
                }
                for i in top_idx
            ]

        try:
            shap_values = self.shap_explainer.shap_values(feature_vector)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (at-risk)

            abs_shap = np.abs(shap_values[0])
            top_idx = np.argsort(abs_shap)[-top_n:][::-1]

            factors = []
            for i in top_idx:
                feature_name = self.feature_cols[i]
                shap_val = float(shap_values[0][i])
                factors.append({
                    "feature": feature_name,
                    "label": self.FEATURE_LABELS.get(feature_name, feature_name),
                    "shap_value": round(shap_val, 4),
                    "direction": "increases risk" if shap_val > 0 else "decreases risk",
                    "value": round(float(feature_vector[0, i]), 2),
                    "importance": round(float(abs_shap[i]), 4),
                })
            return factors

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return []

    def batch_predict(self, feature_list: list[dict]) -> list[dict]:
        """Predict risk for multiple people."""
        return [self.predict(features) for features in feature_list]


# Singleton
prediction_service = HealthRiskPredictor()
