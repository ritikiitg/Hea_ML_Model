"""
Hea ML Explainability Module (SHAP)
====================================
Generates SHAP-based explanations for model predictions.
Maps SHAP values to Hea's existing ExplainabilityEngine format.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def generate_explanations(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    results_dir: str = "ml/results",
    model_name: str = "lightgbm",
    n_samples: int = 200,
) -> dict:
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model (LightGBM or XGBoost)
        X_test: Test feature matrix
        y_test: Test labels
        feature_cols: Feature names
        results_dir: Where to save outputs
        model_name: Name of the model for labeling
        n_samples: Number of background samples for SHAP
    
    Returns:
        Dict with SHAP values, feature importance, and individual explanations
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    if not HAS_SHAP:
        logger.warning("SHAP not available, generating basic feature importance only")
        return _fallback_importance(model, feature_cols, results_path)
    
    logger.info(f"Computing SHAP values for {model_name}...")
    
    # ─── Create SHAP Explainer ───────────────────────────────────────────
    try:
        # TreeExplainer is much faster for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # For binary classification, shap_values might be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}, using KernelExplainer")
        background = shap.sample(X_test, min(n_samples, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:min(500, len(X_test))])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # ─── Global Feature Importance ───────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    importance_df.to_csv(results_path / "shap_feature_importance.csv", index=False)
    logger.info(f"Top SHAP features:\n{importance_df.head(15).to_string()}")
    
    # ─── SHAP Summary Plot ───────────────────────────────────────────────
    if HAS_MPL:
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                            show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(results_path / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary plot to {results_path / 'shap_summary.png'}")
        except Exception as e:
            logger.warning(f"Failed to create SHAP summary plot: {e}")
        
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                            plot_type="bar", show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(results_path / "shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to create SHAP bar plot: {e}")
    
    # ─── Individual Explanations (top 10 high-risk) ──────────────────────
    individual_explanations = _generate_individual_explanations(
        shap_values, feature_cols, X_test, y_test, n_examples=10
    )
    
    with open(results_path / "individual_explanations.json", "w") as f:
        json.dump(individual_explanations, f, indent=2, default=str)
    
    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "individual_explanations": individual_explanations,
    }


def _generate_individual_explanations(
    shap_values: np.ndarray,
    feature_cols: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_examples: int = 10,
) -> list[dict]:
    """
    Create per-person explanations in Hea's ExplainabilityEngine format.
    """
    explanations = []
    
    # Find indices of highest risk predictions
    risk_scores = shap_values.sum(axis=1) if len(shap_values.shape) > 1 else shap_values
    top_risk_indices = np.argsort(risk_scores)[-n_examples:][::-1]
    
    for idx in top_risk_indices:
        person_shap = shap_values[idx] if len(shap_values.shape) > 1 else shap_values
        
        # Top contributing features for this person
        feature_contributions = sorted(
            zip(feature_cols, person_shap),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]
        
        signals = []
        for feat_name, shap_val in feature_contributions:
            direction = "increases" if shap_val > 0 else "decreases"
            signals.append({
                "feature": feat_name,
                "shap_value": round(float(shap_val), 4),
                "direction": direction,
                "signal": _feature_to_signal(feat_name, shap_val),
                "weight": round(min(abs(float(shap_val)) * 2, 1.0), 2),
            })
        
        explanations.append({
            "index": int(idx),
            "actual_outcome": int(y_test[idx]) if idx < len(y_test) else None,
            "risk_score": round(float(risk_scores[idx]), 4),
            "top_signals": signals,
        })
    
    return explanations


def _feature_to_signal(feature_name: str, shap_value: float) -> str:

    direction = "higher" if shap_value > 0 else "lower"
    
    signal_map = {
        "self_rated_health": f"Self-rated health {direction} than typical — contributes to risk assessment",
        "depression_score": f"Depression indicators {direction} — mental health signal detected",
        "depression_velocity": f"Depression score is {'rising' if shap_value > 0 else 'falling'} over time",
        "depression_sustained_rise": "Sustained increase in depression scores across multiple check-ins",
        "bmi": f"BMI is {direction} than average — weight-related signal",
        "adl_total": f"Daily activity limitations {'increasing' if shap_value > 0 else 'stable'}",
        "frailty_index": f"Physical frailty score is {direction}",
        "mental_burden_index": f"Mental health burden is {direction}",
        "age_x_depression": "Age combined with depression scores creates a compound signal",
        "health_declining": "Health has been on a declining trajectory",
        "health_sustained_decline": "Sustained decline in self-rated health over multiple waves",
        "pain_any": f"Pain reports are {direction} than typical",
        "sleep_problems": f"Sleep disruption {'detected' if shap_value > 0 else 'not a factor'}",
        "lifestyle_risk_score": f"Lifestyle risk factors are {direction}",
        "multi_domain_concern_count": f"Concerns detected across {'multiple' if shap_value > 0 else 'few'} health domains",
    }
    
    # Check for partial matches
    for key, description in signal_map.items():
        if key in feature_name:
            return description
    
    return f"{feature_name.replace('_', ' ').title()} — {direction} risk contribution"


def _fallback_importance(model, feature_cols: list[str], results_path: Path) -> dict:
    """Fallback when SHAP is not available — use model's native importance."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(results_path / "feature_importance.csv", index=False)
        return {"feature_importance": importance}
    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Explain model module loaded.")
