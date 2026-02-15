"""
Hea ML — Full Pipeline Runner
==============================
Single-command entry point to run the entire training and evaluation pipeline.

Usage:
    python -m ml.run_full_pipeline --data_path <path_to_dta_file>
    python -m ml.run_full_pipeline --data_path ../../data/randhrs1992_2022v1.dta
    python -m ml.run_full_pipeline --data_path ../../data/randhrs1992_2022v1.dta --sample_frac 0.1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hea.pipeline")


def main():
    parser = argparse.ArgumentParser(description="Hea ML Full Pipeline")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to RAND HRS .dta file")
    parser.add_argument("--results_dir", type=str, default="ml/results",
                       help="Directory to save results")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                       help="Fraction of data to use (for quick testing, e.g. 0.1)")
    parser.add_argument("--no_smote", action="store_true",
                       help="Disable SMOTE oversampling")
    parser.add_argument("--target", type=str, default="chronic_condition_count",
                       help="Target condition variable name")
    parser.add_argument("--horizon", type=int, default=2,
                       help="Prediction horizon in waves")
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("  HEA — EARLY HEALTH RISK PREDICTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Data path:    {args.data_path}")
    logger.info(f"  Sample frac:  {args.sample_frac}")
    logger.info(f"  Target:       {args.target}")
    logger.info(f"  Horizon:      {args.horizon} waves")
    logger.info(f"  Results dir:  {args.results_dir}")
    logger.info("=" * 70)
    
    # ─── Step 1: Load Data ───────────────────────────────────────────────
    logger.info("\n📦 STEP 1/6: Loading RAND HRS Data...")
    step_start = time.time()
    
    from ml.data_loader import load_rand_hrs, build_longitudinal_dataset, prepare_splits
    
    raw_df = load_rand_hrs(args.data_path, sample_frac=args.sample_frac)
    logger.info(f"  Loaded {len(raw_df)} individuals in {time.time() - step_start:.1f}s")
    
    # ─── Step 2: Build Longitudinal Dataset ──────────────────────────────
    logger.info("\n🔧 STEP 2/6: Building Longitudinal Panel + Target...")
    step_start = time.time()
    
    panel = build_longitudinal_dataset(raw_df, target_condition=args.target,
                                        prediction_horizon=args.horizon)
    splits = prepare_splits(panel)
    
    logger.info(f"  Panel: {len(panel)} observations, target dist: "
               f"{panel['target'].value_counts().to_dict()}")
    logger.info(f"  Built in {time.time() - step_start:.1f}s")
    
    # ─── Step 3: Feature Engineering ─────────────────────────────────────
    logger.info("\n🧬 STEP 3/6: Feature Engineering...")
    step_start = time.time()
    
    from ml.feature_engineer import engineer_features
    
    # Engineer features for each split
    train_enhanced, feature_cols = engineer_features(splits["train"], splits["feature_cols"])
    val_enhanced, _ = engineer_features(splits["val"], splits["feature_cols"])
    test_enhanced, _ = engineer_features(splits["test"], splits["feature_cols"])
    
    logger.info(f"  {len(feature_cols)} features engineered in {time.time() - step_start:.1f}s")
    
    # ─── Step 4: Train Models ────────────────────────────────────────────
    logger.info("\n🏋️ STEP 4/6: Training Ensemble Models...")
    step_start = time.time()
    
    from ml.train_pipeline import train_ensemble
    
    results = train_ensemble(
        train_df=train_enhanced,
        val_df=val_enhanced,
        test_df=test_enhanced,
        feature_cols=feature_cols,
        results_dir=args.results_dir,
        use_smote=not args.no_smote,
    )
    
    logger.info(f"  Training completed in {time.time() - step_start:.1f}s")
    
    # ─── Step 5: Explainability ──────────────────────────────────────────
    logger.info("\n🔍 STEP 5/6: SHAP Explainability Analysis...")
    step_start = time.time()
    
    from ml.explain_model import generate_explanations
    
    # Use the LightGBM model for SHAP (most interpretable)
    primary_model = results["models"].get("lightgbm") or results["models"].get("xgboost")
    if primary_model:
        import numpy as np
        from ml.train_pipeline import _prepare_arrays
        from sklearn.impute import SimpleImputer
        
        X_test, y_test = _prepare_arrays(test_enhanced, feature_cols)
        imputer = results["imputer"]
        X_test = imputer.transform(X_test)
        
        explanation_results = generate_explanations(
            model=primary_model,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            results_dir=args.results_dir,
        )
        logger.info(f"  Explainability analysis completed in {time.time() - step_start:.1f}s")
    
    # ─── Step 6: Fairness Audit ──────────────────────────────────────────
    logger.info("\n⚖️ STEP 6/6: Fairness Audit...")
    step_start = time.time()
    
    from ml.fairness_audit import run_fairness_audit
    
    fairness_report = run_fairness_audit(
        test_df=test_enhanced,
        y_pred=results["test_pred"],
        y_proba=results["test_proba"],
        y_test=results["y_test"],
        results_dir=args.results_dir,
    )
    
    logger.info(f"  Fairness audit completed in {time.time() - step_start:.1f}s")
    
    # ─── Final Summary ───────────────────────────────────────────────────
    total_time = time.time() - start_time
    metrics = results["metrics"]
    
    logger.info("\n" + "=" * 70)
    logger.info("  🏆 PIPELINE COMPLETE — FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"  F2-Score:       {metrics['f2_score']:.4f}")
    logger.info(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    logger.info(f"  Threshold:      {metrics['threshold']:.4f}")
    logger.info(f"  Precision:      {metrics['precision']:.4f}")
    logger.info(f"  Recall:         {metrics['recall']:.4f}")
    logger.info(f"  Total Time:     {total_time:.1f}s")
    logger.info("=" * 70)
    logger.info(f"  Results saved to: {Path(args.results_dir).resolve()}")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
