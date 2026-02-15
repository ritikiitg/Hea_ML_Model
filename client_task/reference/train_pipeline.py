"""
Hea ML Training Pipeline
=========================
Trains LightGBM + XGBoost ensemble with F2-score optimization.
Handles class imbalance, threshold tuning, and model serialization.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    fbeta_score, precision_recall_curve, auc,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ─── Optional imports with graceful fallback ─────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not available, SMOTE disabled")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    results_dir: str = "ml/results",
    use_smote: bool = True,
) -> dict:
    """
    Train a stacking ensemble (LightGBM + XGBoost → LogisticRegression meta).
    
    Args:
        train_df: Training data with 'target' column
        val_df: Validation data
        test_df: Test data
        feature_cols: List of feature column names
        results_dir: Where to save models and metrics
        use_smote: Whether to apply SMOTE for class balancing
    
    Returns:
        Dict with metrics, models, and predictions
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # ─── Prepare Arrays ─────────────────────────────────────────────────
    X_train, y_train = _prepare_arrays(train_df, feature_cols)
    X_val, y_val = _prepare_arrays(val_df, feature_cols)
    X_test, y_test = _prepare_arrays(test_df, feature_cols)
    
    logger.info(f"X_train shape: {X_train.shape}, y_train dist: {np.bincount(y_train.astype(int))}")
    logger.info(f"X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
    
    # ─── Impute Missing Values ───────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # ─── Handle Class Imbalance ──────────────────────────────────────────
    if use_smote and HAS_IMBLEARN:
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 5 and pos_count < neg_count:
            logger.info(f"Applying SMOTE (positive: {int(pos_count)}, negative: {int(neg_count)})")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, int(pos_count) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE: {np.bincount(y_train_balanced.astype(int))}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, using original data")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Calculate class weight for scale_pos_weight
    pos_ratio = y_train.sum() / len(y_train)
    scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-6)
    
    models = {}
    val_preds = {}
    test_preds = {}
    
    # ─── Model 1: LightGBM ──────────────────────────────────────────────
    if HAS_LGBM:
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        models["lightgbm"] = lgb_model
        val_preds["lightgbm"] = lgb_model.predict_proba(X_val)[:, 1]
        test_preds["lightgbm"] = lgb_model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": lgb_model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(results_path / "lgbm_feature_importance.csv", index=False)
        logger.info(f"LightGBM top features:\n{importance.head(10).to_string()}")
    
    # ─── Model 2: XGBoost ────────────────────────────────────────────────
    if HAS_XGB:
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models["xgboost"] = xgb_model
        val_preds["xgboost"] = xgb_model.predict_proba(X_val)[:, 1]
        test_preds["xgboost"] = xgb_model.predict_proba(X_test)[:, 1]
    
    # ─── Stacking Meta-Learner ───────────────────────────────────────────
    if len(models) >= 2:
        logger.info("Training stacking meta-learner...")
        # Stack validation predictions as features for meta-learner
        meta_val_features = np.column_stack([val_preds[name] for name in models.keys()])
        meta_test_features = np.column_stack([test_preds[name] for name in models.keys()])
        
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(meta_val_features, y_val)
        
        final_test_proba = meta_model.predict_proba(meta_test_features)[:, 1]
        models["meta_learner"] = meta_model
    elif len(models) == 1:
        model_name = list(models.keys())[0]
        final_test_proba = test_preds[model_name]
    else:
        raise RuntimeError("No ML libraries available! Install lightgbm or xgboost.")
    
    # ─── Threshold Tuning for F2-Score ───────────────────────────────────
    logger.info("Optimizing threshold for F2-score...")
    best_threshold, best_f2 = _optimize_f2_threshold(y_val, 
        meta_model.predict_proba(np.column_stack([val_preds[n] for n in models if n != "meta_learner"]))[:, 1]
        if len(models) >= 2 else val_preds[list(models.keys())[0]]
    )
    
    # ─── Final Predictions with Optimized Threshold ──────────────────────
    y_pred = (final_test_proba >= best_threshold).astype(int)
    
    # ─── Compute All Metrics ─────────────────────────────────────────────
    metrics = _compute_all_metrics(y_test, final_test_proba, y_pred, best_threshold)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS (threshold={best_threshold:.3f})")
    logger.info(f"{'='*60}")
    logger.info(f"F2-Score:  {metrics['f2_score']:.4f}")
    logger.info(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    logger.info(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    logger.info(f"{'='*60}")
    
    # ─── Save Everything ─────────────────────────────────────────────────
    # Save metrics
    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save models
    if HAS_JOBLIB:
        for name, model in models.items():
            joblib.dump(model, results_path / f"{name}_model.pkl")
        joblib.dump(imputer, results_path / "imputer.pkl")
        joblib.dump(feature_cols, results_path / "feature_cols.pkl")
        joblib.dump(best_threshold, results_path / "threshold.pkl")
    
    # Save predictions for analysis
    test_results = test_df[["HHIDPN", "wave", "target"]].copy()
    test_results["predicted_proba"] = final_test_proba
    test_results["predicted_class"] = y_pred
    test_results.to_csv(results_path / "test_predictions.csv", index=False)
    
    return {
        "metrics": metrics,
        "models": models,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "test_proba": final_test_proba,
        "test_pred": y_pred,
        "y_test": y_test,
    }


def _prepare_arrays(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    """Extract feature matrix and target vector from DataFrame."""
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    
    if missing_cols:
        logger.debug(f"Missing {len(missing_cols)} columns in split, will be NaN")
    
    X = df[available_cols].copy()
    
    # Add missing columns as NaN
    for col in missing_cols:
        X[col] = np.nan
    
    # Ensure correct column order
    X = X[feature_cols]
    
    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    
    y = df["target"].values.astype(float)
    
    return X.values, y


def _optimize_f2_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Find the classification threshold that maximizes F2-score.
    F2-score weights recall 4x more than precision.
    """
    best_f2 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
    
    logger.info(f"Best F2 threshold: {best_threshold:.3f} (F2={best_f2:.4f})")
    return best_threshold, best_f2


def _compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                         y_pred: np.ndarray, threshold: float) -> dict:
    """Compute all competition metrics."""
    # F2-Score
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    # ROC-AUC
    try:
        roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc = 0.0
    
    # Additional metrics for analysis
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "f2_score": round(float(f2), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": cm.tolist(),
        "precision": round(float(report.get("1", {}).get("precision", 0)), 4),
        "recall": round(float(report.get("1", {}).get("recall", 0)), 4),
        "f1_score": round(float(report.get("1", {}).get("f1-score", 0)), 4),
        "support_positive": int(y_true.sum()),
        "support_negative": int(len(y_true) - y_true.sum()),
        "num_features": len(y_pred),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training pipeline loaded. Use train_ensemble() to train.")
