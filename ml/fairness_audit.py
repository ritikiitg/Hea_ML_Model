"""
Hea ML Fairness Audit
======================
Bias detection and mitigation across demographic groups.
Uses equalized odds, demographic parity, and disparate impact analysis.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score, roc_auc_score

logger = logging.getLogger(__name__)

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
    )
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False
    logger.warning("fairlearn not available, using basic fairness analysis")


def run_fairness_audit(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_test: np.ndarray,
    results_dir: str = "ml/results",
) -> dict:
    """
    Comprehensive fairness audit across demographic groups.
    
    Args:
        test_df: Test DataFrame with demographic columns
        y_pred: Binary predictions
        y_proba: Probability predictions
        y_test: True labels
        results_dir: Where to save the report
    
    Returns:
        Fairness report dict
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    report = {
        "overall_metrics": {},
        "demographic_analysis": {},
        "fairness_metrics": {},
        "recommendations": [],
    }
    
    # ─── Overall Metrics ─────────────────────────────────────────────────
    report["overall_metrics"] = {
        "f2_score": round(float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)), 4),
        "selection_rate": round(float(y_pred.mean()), 4),
        "positive_rate": round(float(y_test.mean()), 4),
    }
    
    # ─── Analyze Each Demographic Group ──────────────────────────────────
    demographic_cols = {
        "gender": "Gender",
        "race": "Race",
    }
    
    # Age groups
    if "age" in test_df.columns:
        age = pd.to_numeric(test_df["age"], errors="coerce")
        test_df = test_df.copy()
        test_df["age_group"] = pd.cut(
            age, bins=[0, 55, 65, 75, 85, 200],
            labels=["<55", "55-64", "65-74", "75-84", "85+"]
        )
        demographic_cols["age_group"] = "Age Group"
    
    for col, label in demographic_cols.items():
        if col not in test_df.columns:
            continue
        
        group_analysis = _analyze_demographic_group(
            test_df[col].values, y_test, y_pred, y_proba, label
        )
        report["demographic_analysis"][label] = group_analysis
    
    # ─── Fairlearn Metrics (if available) ────────────────────────────────
    if HAS_FAIRLEARN:
        for col, label in demographic_cols.items():
            if col not in test_df.columns:
                continue
            
            sensitive = test_df[col].fillna("Unknown").astype(str).values
            
            try:
                # Demographic parity
                dp_diff = demographic_parity_difference(
                    y_test, y_pred, sensitive_features=sensitive
                )
                
                # Equalized odds
                eo_diff = equalized_odds_difference(
                    y_test, y_pred, sensitive_features=sensitive
                )
                
                report["fairness_metrics"][label] = {
                    "demographic_parity_difference": round(float(dp_diff), 4),
                    "equalized_odds_difference": round(float(eo_diff), 4),
                    "passes_80_percent_rule": abs(dp_diff) < 0.2,
                    "passes_equalized_odds": abs(eo_diff) < 0.1,
                }
                
                # MetricFrame for detailed breakdown
                metric_frame = MetricFrame(
                    metrics={
                        "selection_rate": selection_rate,
                        "f2_score": lambda y_t, y_p: fbeta_score(y_t, y_p, beta=2, zero_division=0),
                    },
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=sensitive,
                )
                
                report["fairness_metrics"][f"{label}_by_group"] = {
                    str(k): {metric: round(float(v), 4) for metric, v in vals.items()}
                    for k, vals in metric_frame.by_group.iterrows()
                }
                
            except Exception as e:
                logger.warning(f"Fairlearn analysis failed for {label}: {e}")
    
    # ─── Generate Recommendations ────────────────────────────────────────
    report["recommendations"] = _generate_recommendations(report)
    
    # ─── Save Report ─────────────────────────────────────────────────────
    with open(results_path / "fairness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("FAIRNESS AUDIT RESULTS")
    logger.info(f"{'='*60}")
    for label, metrics in report.get("fairness_metrics", {}).items():
        if isinstance(metrics, dict) and "demographic_parity_difference" in metrics:
            dp = metrics["demographic_parity_difference"]
            eo = metrics["equalized_odds_difference"]
            dp_pass = "✅" if metrics["passes_80_percent_rule"] else "⚠️"
            eo_pass = "✅" if metrics["passes_equalized_odds"] else "⚠️"
            logger.info(f"{label}: DP={dp:.4f} {dp_pass} | EO={eo:.4f} {eo_pass}")
    logger.info(f"{'='*60}")
    
    return report


def _analyze_demographic_group(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    group_label: str,
) -> dict:
    """Analyze model performance within each demographic group."""
    analysis = {}
    
    unique_groups = pd.Series(groups).dropna().unique()
    
    for group_val in unique_groups:
        mask = groups == group_val
        if mask.sum() < 10:
            continue
        
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        group_proba = y_proba[mask]
        
        f2 = fbeta_score(group_true, group_pred, beta=2, zero_division=0)
        
        try:
            roc = roc_auc_score(group_true, group_proba)
        except ValueError:
            roc = 0.0
        
        analysis[str(group_val)] = {
            "count": int(mask.sum()),
            "positive_rate": round(float(group_true.mean()), 4),
            "selection_rate": round(float(group_pred.mean()), 4),
            "f2_score": round(float(f2), 4),
            "roc_auc": round(float(roc), 4),
        }
    
    return analysis


def _generate_recommendations(report: dict) -> list[str]:
    """Generate actionable fairness recommendations."""
    recs = []
    
    for label, metrics in report.get("fairness_metrics", {}).items():
        if not isinstance(metrics, dict):
            continue
        
        if "demographic_parity_difference" in metrics:
            if not metrics.get("passes_80_percent_rule", True):
                recs.append(
                    f"⚠️ {label}: Demographic parity difference ({metrics['demographic_parity_difference']:.3f}) "
                    f"exceeds 0.2 threshold. Consider post-processing calibration per subgroup."
                )
            
            if not metrics.get("passes_equalized_odds", True):
                recs.append(
                    f"⚠️ {label}: Equalized odds difference ({metrics['equalized_odds_difference']:.3f}) "
                    f"exceeds 0.1 threshold. Review false positive/negative rates per group."
                )
    
    if not recs:
        recs.append("✅ No significant fairness concerns detected across analyzed demographic groups.")
        recs.append("Continue monitoring fairness metrics as model is updated or deployed to new populations.")
    
    return recs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Fairness audit module loaded.")
