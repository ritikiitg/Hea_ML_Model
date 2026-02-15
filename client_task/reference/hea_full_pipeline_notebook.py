"""
# ============================================================================
# HEA — EARLY HEALTH RISK PREDICTION PIPELINE  (v2 — Wide Format)
# ============================================================================
# Key change from v1: WIDE format (one row per person, ALL raw columns)
# instead of long panel format. This gives the model access to every
# variable across every wave simultaneously, preserving natural trajectory.
# ============================================================================
"""

# ── CELL 0: Environment Setup (must be FIRST) ─────────────────────────────
import os
os.environ["SCIPY_ARRAY_API"] = "1"  # Required for sklearn on Nebius

# ── CELL 1: Configuration ──────────────────────────────────────────────────
RESULTS_DIR = "./hea_results"
SAMPLE_FRAC = 1.0

# ── CELL 2: Find Data ────────────────────────────────────────────────────
import os, sys, glob, subprocess

print("📂 STEP 0: Finding data file...")
DATA_PATH = None

search_dirs = [
    "/workspace/bionemo2", "/workspace", ".",
    "/root/s3-new/extracted", "/root/s3-new", "/s3-new", "/s3",
    "/home", "./data", "/s3/data",
]
skip_patterns = ["Trash", ".cache", ".local/share/Trash", "__pycache__"]

for search_dir in search_dirs:
    if not os.path.isdir(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        if any(skip in root for skip in skip_patterns):
            continue
        for f in files:
            if f.endswith(".csv.gz") or f.endswith(".dta") or f.endswith(".csv"):
                candidate = os.path.join(root, f)
                fsize = os.path.getsize(candidate) / 1024 / 1024
                print(f"  Found: {candidate} ({fsize:.1f} MB)")
                if DATA_PATH is None or fsize > os.path.getsize(DATA_PATH) / 1024 / 1024:
                    DATA_PATH = candidate
    if DATA_PATH:
        break

if DATA_PATH is None:
    raise FileNotFoundError("No data file found! Upload hea_hrs_extract.csv.gz to /workspace/bionemo2/")

print(f"\n  📁 Using: {DATA_PATH} ({os.path.getsize(DATA_PATH)/1024/1024:.1f} MB)")

# ── CELL 3: Install Dependencies ─────────────────────────────────────────
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("\n📦 Installing dependencies...")
for pkg in ["lightgbm", "xgboost", "shap", "fairlearn", "imbalanced-learn",
            "matplotlib", "seaborn", "pandas", "numpy", "scikit-learn", "joblib"]:
    try:
        install(pkg)
    except Exception as e:
        print(f"  ⚠️ {pkg}: {e}")
print("✅ Dependencies ready!")

# ── CELL 4: Imports ───────────────────────────────────────────────────────
import json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score, precision_recall_curve, auc,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from fairlearn.metrics import (
        MetricFrame, demographic_parity_difference,
        equalized_odds_difference, selection_rate
    )
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

results_path = Path(RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)
print(f"✅ All imports ready. SHAP={HAS_SHAP}, Fairlearn={HAS_FAIRLEARN}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA (WIDE FORMAT — one row per person)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📦 STEP 1/6: LOADING DATA")
print("="*70)

step_start = time.time()

if DATA_PATH.endswith(".csv.gz") or DATA_PATH.endswith(".csv"):
    df = pd.read_csv(DATA_PATH)
elif DATA_PATH.endswith(".dta"):
    df = pd.read_stata(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

df.columns = [c.lower() for c in df.columns]

# Convert all columns to numeric
for col in df.columns:
    if col == "hhidpn":
        continue
    if df[col].dtype == object or str(df[col].dtype) == "category":
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
        else:
            df[col] = df[col].astype(str).str.extract(r'(\-?\d+\.?\d*)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"  Loaded: {df.shape[0]} people × {df.shape[1]} columns")
print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD TARGET + FEATURES (WIDE FORMAT)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🎯 STEP 2/6: BUILDING PREDICTION TARGET")
print("="*70)

step_start = time.time()

# Target: ANY new chronic condition onset between recent waves
# We compare waves 13-14 to waves 11-12 (condition count or individual diseases)
# Features: ALL data from waves 1-12

# --- Build compound target ---
target = pd.Series(0, index=df.index, dtype=float)
target_defined = pd.Series(False, index=df.index)

# Method A: Chronic condition count increases
for recent_w, baseline_w in [(14, 13), (15, 14), (13, 12), (16, 15)]:
    recent_col = f"r{recent_w}conde"
    baseline_col = f"r{baseline_w}conde"
    if recent_col in df.columns and baseline_col in df.columns:
        mask = df[recent_col].notna() & df[baseline_col].notna()
        target_defined |= mask
        target.loc[mask & (df[recent_col] > df[baseline_col])] = 1

# Method B: Individual disease onset (0→1 transition)
disease_suffixes = ["diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]
for disease in disease_suffixes:
    for recent_w, baseline_w in [(14, 13), (15, 14), (13, 12)]:
        recent_col = f"r{recent_w}{disease}"
        baseline_col = f"r{baseline_w}{disease}"
        if recent_col in df.columns and baseline_col in df.columns:
            mask = df[recent_col].notna() & df[baseline_col].notna()
            target_defined |= mask
            onset = mask & (df[baseline_col] == 0) & (df[recent_col] == 1)
            target.loc[onset] = 1

df["target"] = target
df = df[target_defined].copy()

print(f"  People with valid target: {len(df)}")
print(f"  Target: {df['target'].value_counts().to_dict()}")
print(f"  Positive rate: {df['target'].mean():.3f}")

# --- Define feature columns ---
# CRITICAL: Exclude ALL columns from waves 14-16 (target waves) to prevent leakage!
# The full .dta file has 19,880 columns. ANY wave 14+ column (medications, visits, etc.)
# can reveal the target. We must only use data from waves 1-13 (past).
import re

def is_future_wave_col(col_name):
    """Check if column belongs to wave 14, 15, or 16 (target/future waves)."""
    # Match patterns like r14xxx, r15xxx, r16xxx, s14xxx, h14xxx etc.
    match = re.match(r'^[a-z](\d+)', col_name)
    if match:
        wave_num = int(match.group(1))
        return wave_num >= 14
    return False

exclude_set = {"hhidpn", "target"}
leakage_count = 0
for col in df.columns:
    if col in exclude_set:
        continue
    if is_future_wave_col(col):
        exclude_set.add(col)
        leakage_count += 1

# Also exclude target variables from ALL waves (they directly encode the target)
for w in range(1, 14):
    for s in ["conde", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre"]:
        exclude_set.add(f"r{w}{s}")

feature_cols = [c for c in df.columns if c not in exclude_set]
print(f"  Excluded {leakage_count} future-wave columns (leakage prevention)")
print(f"  Excluded {len(exclude_set) - leakage_count - 2} target disease columns")

# Drop features with >70% NaN
nan_rates = df[feature_cols].isna().mean()
feature_cols = nan_rates[nan_rates < 0.7].index.tolist()

print(f"  Feature columns: {len(feature_cols)}")
print(f"  Avg NaN rate: {df[feature_cols].isna().mean().mean():.1%}")
print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🧬 STEP 3/6: FEATURE ENGINEERING")
print("="*70)

step_start = time.time()
new_features = []

# --- 3a: Cross-wave deltas (trajectory signal) ---
# For key health variables, compute change between consecutive waves
delta_suffixes = ["shlt", "cesd", "bmi", "conde", "adltot_m", "iadlza", "paina"]
for suffix in delta_suffixes:
    for w in range(2, 14):  # Deltas up to wave 13
        curr = f"r{w}{suffix}"
        prev = f"r{w-1}{suffix}"
        delta_name = f"delta_w{w}_{suffix}"
        if curr in df.columns and prev in df.columns:
            df[delta_name] = df[curr] - df[prev]
            new_features.append(delta_name)

# --- 3b: Recent trajectory (last 3 known waves) ---
for suffix in ["shlt", "cesd", "bmi"]:
    cols_ordered = [f"r{w}{suffix}" for w in range(1, 14) if f"r{w}{suffix}" in df.columns]
    if len(cols_ordered) >= 3:
        # Use last 3 available values
        last3 = df[cols_ordered[-3:]].copy()
        # Rate of change (slope)
        df[f"slope_{suffix}"] = last3.iloc[:, -1] - last3.iloc[:, 0]
        new_features.append(f"slope_{suffix}")
        # Volatility (std over recent waves)
        df[f"volatility_{suffix}"] = last3.std(axis=1)
        new_features.append(f"volatility_{suffix}")
        # Acceleration (second derivative)
        if last3.shape[1] >= 3:
            d1 = last3.iloc[:, 1] - last3.iloc[:, 0]
            d2 = last3.iloc[:, 2] - last3.iloc[:, 1]
            df[f"accel_{suffix}"] = d2 - d1
            new_features.append(f"accel_{suffix}")

# --- 3c: Multi-wave history aggregates ---
for suffix in ["shlt", "cesd", "bmi", "conde", "paina"]:
    all_cols = [f"r{w}{suffix}" for w in range(1, 14) if f"r{w}{suffix}" in df.columns]
    if len(all_cols) >= 2:
        df[f"mean_all_{suffix}"] = df[all_cols].mean(axis=1)
        df[f"max_all_{suffix}"] = df[all_cols].max(axis=1)
        df[f"min_all_{suffix}"] = df[all_cols].min(axis=1)
        df[f"range_all_{suffix}"] = df[all_cols].max(axis=1) - df[all_cols].min(axis=1)
        df[f"std_all_{suffix}"] = df[all_cols].std(axis=1)
        df[f"n_valid_{suffix}"] = df[all_cols].notna().sum(axis=1)
        new_features.extend([
            f"mean_all_{suffix}", f"max_all_{suffix}", f"min_all_{suffix}",
            f"range_all_{suffix}", f"std_all_{suffix}", f"n_valid_{suffix}"
        ])

# --- 3d: Cross-domain interactions ---
interaction_pairs = [
    ("r13shlt", "r13cesd", "health_x_depression"),
    ("r13bmi", "r13cesd", "bmi_x_depression"),
    ("r13paina", "r13adltot_m", "pain_x_adl"),
    ("r13cesd", "r13sleepr", "depression_x_sleep"),
    ("r13shlt", "r13adltot_m", "health_x_adl"),
]
for c1, c2, name in interaction_pairs:
    if c1 in df.columns and c2 in df.columns:
        v1 = pd.to_numeric(df[c1], errors="coerce").fillna(0)
        v2 = pd.to_numeric(df[c2], errors="coerce").fillna(0)
        df[name] = v1 * v2
        new_features.append(name)

# --- 3e: Composite risk indices ---
# Frailty-like index from latest wave
frailty_cols = [c for c in ["r13adltot_m", "r13iadlza", "r13walkra", "r13climsa", "r13chaira"]
                if c in df.columns]
if frailty_cols:
    frailty_df = df[frailty_cols].apply(pd.to_numeric, errors="coerce")
    for col in frailty_cols:
        cmin, cmax = frailty_df[col].min(), frailty_df[col].max()
        if cmax > cmin:
            frailty_df[col] = (frailty_df[col] - cmin) / (cmax - cmin)
    df["frailty_index"] = frailty_df.mean(axis=1)
    new_features.append("frailty_index")

# Multi-domain concern count
concern_flags = []
for col, thresh in [("r13shlt", 4), ("r13cesd", 4), ("r13paina", 1), ("r13adltot_m", 2)]:
    if col in df.columns:
        concern_flags.append((pd.to_numeric(df[col], errors="coerce").fillna(0) >= thresh).astype(float))
if concern_flags:
    df["concern_count"] = pd.concat(concern_flags, axis=1).sum(axis=1)
    new_features.append("concern_count")

# --- 3f: Age-relative z-scores ---
if "r13agey_e" in df.columns:
    age = pd.to_numeric(df["r13agey_e"], errors="coerce")
    age_bins = pd.cut(age, bins=[0, 55, 60, 65, 70, 75, 80, 85, 200], labels=False)
    for suffix, raw_col in [("shlt", "r13shlt"), ("cesd", "r13cesd"), ("bmi", "r13bmi")]:
        if raw_col in df.columns:
            vals = pd.to_numeric(df[raw_col], errors="coerce")
            gmeans = vals.groupby(age_bins).transform("mean")
            gstds = vals.groupby(age_bins).transform("std").replace(0, 1)
            df[f"zscore_{suffix}"] = (vals - gmeans) / gstds
            new_features.append(f"zscore_{suffix}")

# --- 3g: Missing data pattern features ---
# The PATTERN of missing data is informative (people who stop responding may be sicker)
for w in range(8, 14):
    cols_at_wave = [c for c in df.columns if c.startswith(f"r{w}")]
    if cols_at_wave:
        df[f"pct_missing_w{w}"] = df[cols_at_wave].isna().mean(axis=1)
        new_features.append(f"pct_missing_w{w}")

df["total_missing_pct"] = df[feature_cols].isna().mean(axis=1)
new_features.append("total_missing_pct")

# Update feature columns
feature_cols = feature_cols + [f for f in new_features if f in df.columns]
feature_cols = list(dict.fromkeys(feature_cols))
# Remove any that are still high-NaN
nan_rates = df[feature_cols].isna().mean()
feature_cols = nan_rates[nan_rates < 0.7].index.tolist()

print(f"  Engineered features: {len(new_features)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🏋️ STEP 4/6: TRAINING ENSEMBLE MODELS")
print("="*70)

step_start = time.time()

# --- Split (stratified random — wide format, each person appears once) ---
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
y = df["target"].values.astype(float)

# Save demographic columns for fairness audit BEFORE splitting
demo_data = df[[c for c in ["ragender", "raracem", "rahispan", "r13agey_e", "hhidpn"] if c in df.columns]].copy()

X_train_full, X_test, y_train_full, y_test, demo_train_full, demo_test = train_test_split(
    X, y, demo_data, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"  Train pos: {int(y_train.sum())}/{len(y_train)} ({y_train.mean():.3f})")
print(f"  Test pos:  {int(y_test.sum())}/{len(y_test)} ({y_test.mean():.3f})")

# Impute
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Class weight (moderate — don't over-compensate)
pos_rate = y_train.mean()
spw = min((1 - pos_rate) / max(pos_rate, 1e-6), 10.0)  # Cap at 10x
print(f"  Class weight: {spw:.2f}")

# ─── LightGBM ───────────────────────────────────────────────────────────
print("\n  Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000, max_depth=6, learning_rate=0.02,
    num_leaves=63, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=2.0,
    scale_pos_weight=spw,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
lgb_val_proba = lgb_model.predict_proba(X_val)[:, 1]
lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]

# Feature importance
lgb_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
lgb_importance.to_csv(results_path / "lgbm_feature_importance.csv", index=False)
print("  LightGBM top features:")
print(lgb_importance.head(10).to_string(index=False))

# ─── XGBoost ─────────────────────────────────────────────────────────────
print("\n  Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=6, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=2.0,
    scale_pos_weight=spw,
    random_state=42, n_jobs=-1, eval_metric="aucpr", verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# ─── Stacking Meta-Learner ──────────────────────────────────────────────
print("\n  Training meta-learner...")
meta_val_X = np.column_stack([lgb_val_proba, xgb_val_proba])
meta_test_X = np.column_stack([lgb_test_proba, xgb_test_proba])

meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(meta_val_X, y_val)
final_val_proba = meta_model.predict_proba(meta_val_X)[:, 1]
final_test_proba = meta_model.predict_proba(meta_test_X)[:, 1]

# ─── Threshold Optimization for F2-Score ─────────────────────────────────
print("\n  Optimizing F2 threshold...")
best_f2, best_threshold = 0, 0.5
for threshold in np.arange(0.05, 0.9, 0.005):
    y_pred_t = (final_val_proba >= threshold).astype(int)
    if y_pred_t.sum() == 0:
        continue
    f2 = fbeta_score(y_val, y_pred_t, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = threshold

print(f"  Best threshold: {best_threshold:.4f} (Val F2={best_f2:.4f})")

# ─── Final Predictions ──────────────────────────────────────────────────
y_pred = (final_test_proba >= best_threshold).astype(int)

f2_final = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, final_test_proba)
pr_auc_score = auc(recall_vals, precision_vals)
try:
    roc_auc_val = roc_auc_score(y_test, final_test_proba)
except ValueError:
    roc_auc_val = 0.0

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

metrics = {
    "f2_score": round(float(f2_final), 4),
    "pr_auc": round(float(pr_auc_score), 4),
    "roc_auc": round(float(roc_auc_val), 4),
    "threshold": round(float(best_threshold), 4),
    "confusion_matrix": cm.tolist(),
    "precision": round(float(report.get("1.0", report.get("1", {})).get("precision", 0)), 4),
    "recall": round(float(report.get("1.0", report.get("1", {})).get("recall", 0)), 4),
}

with open(results_path / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n  {'='*50}")
print(f"  📊 PRIMARY METRICS:")
print(f"  {'='*50}")
print(f"  F2-Score:  {metrics['f2_score']:.4f}")
print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
print(f"  Threshold: {metrics['threshold']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  {'='*50}")
print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: EXPLAINABILITY (SHAP)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🔍 STEP 5/6: SHAP EXPLAINABILITY")
print("="*70)

step_start = time.time()

if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(lgb_model)
        # Use a subsample for speed
        shap_sample_size = min(1000, X_test.shape[0])
        X_shap = X_test[:shap_sample_size]
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(results_path / "shap_feature_importance.csv", index=False)
        
        print("  Top 15 SHAP features:")
        print(shap_importance.head(15).to_string(index=False))
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(results_path / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                         plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(results_path / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  ✅ SHAP plots saved")
    except Exception as e:
        print(f"  ⚠️ SHAP failed: {e}")
else:
    print("  ⚠️ SHAP not available")

print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("⚖️ STEP 6/6: FAIRNESS AUDIT")
print("="*70)

step_start = time.time()
fairness_report = {"overall": metrics, "demographic_analysis": {}, "fairness_metrics": {}}

test_df = demo_test.reset_index(drop=True)
demo_cols = {}

if "ragender" in test_df.columns:
    demo_cols["ragender"] = "Gender"
if "raracem" in test_df.columns:
    demo_cols["raracem"] = "Race"
if "r13agey_e" in test_df.columns:
    age_num = pd.to_numeric(test_df["r13agey_e"], errors="coerce")
    test_df["age_group"] = pd.cut(age_num, bins=[0, 55, 65, 75, 85, 200],
                                   labels=["<55", "55-64", "65-74", "75-84", "85+"])
    demo_cols["age_group"] = "Age Group"

for col, label in demo_cols.items():
    if col not in test_df.columns:
        continue
    groups = test_df[col].values
    unique_groups = pd.Series(groups).dropna().unique()
    group_analysis = {}
    for gval in unique_groups:
        mask = groups == gval
        if mask.sum() < 10:
            continue
        g_true = y_test[mask]
        g_pred = y_pred[mask]
        g_proba = final_test_proba[mask]
        g_f2 = fbeta_score(g_true, g_pred, beta=2, zero_division=0)
        try:
            g_roc = roc_auc_score(g_true, g_proba)
        except:
            g_roc = 0.0
        group_analysis[str(gval)] = {
            "count": int(mask.sum()), "positive_rate": round(float(g_true.mean()), 4),
            "f2_score": round(float(g_f2), 4), "roc_auc": round(float(g_roc), 4),
        }
    fairness_report["demographic_analysis"][label] = group_analysis
    print(f"\n  {label} Analysis:")
    for k, v in group_analysis.items():
        print(f"    {k}: n={v['count']}, pos={v['positive_rate']:.3f}, F2={v['f2_score']:.3f}, ROC={v['roc_auc']:.3f}")

if HAS_FAIRLEARN:
    for col, label in demo_cols.items():
        if col not in test_df.columns:
            continue
        sensitive = test_df[col].astype(str).replace("nan", "Unknown").values
        try:
            dp = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
            eo = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)
            fairness_report["fairness_metrics"][label] = {
                "demographic_parity_diff": round(float(dp), 4),
                "equalized_odds_diff": round(float(eo), 4),
                "passes_80pct_rule": bool(abs(dp) < 0.2),
                "passes_eq_odds": bool(abs(eo) < 0.1),
            }
            print(f"  {label}: DP={dp:.4f}, EO={eo:.4f}")
        except Exception as e:
            print(f"  ⚠️ {label}: {e}")

with open(results_path / "fairness_report.json", "w") as f:
    json.dump(fairness_report, f, indent=2)
print(f"  Time: {time.time() - step_start:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE & VISUALIZE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("💾 SAVING MODELS & RESULTS")
print("="*70)

# Plots
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (threshold={metrics['threshold']:.3f})")
plt.tight_layout()
plt.savefig(results_path / "confusion_matrix.png", dpi=150)
plt.close()

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, final_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout()
plt.savefig(results_path / "roc_curve.png", dpi=150)
plt.close()

# PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f"PR-AUC = {metrics['pr_auc']:.4f}", linewidth=2)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
plt.legend(); plt.tight_layout()
plt.savefig(results_path / "pr_curve.png", dpi=150)
plt.close()

# Save models
joblib.dump(lgb_model, results_path / "lightgbm_model.pkl")
joblib.dump(xgb_model, results_path / "xgboost_model.pkl")
joblib.dump(meta_model, results_path / "meta_learner.pkl")
joblib.dump(imputer, results_path / "imputer.pkl")
joblib.dump(feature_cols, results_path / "feature_cols.pkl")
joblib.dump(best_threshold, results_path / "threshold.pkl")

# Save test predictions
test_results_df = demo_test[["hhidpn"]].copy().reset_index(drop=True)
test_results_df["target"] = y_test
test_results_df["predicted_proba"] = final_test_proba
test_results_df["predicted_class"] = y_pred
test_results_df.to_csv(results_path / "test_predictions.csv", index=False)

# List saved files
print(f"\n  ✅ All saved to {results_path}/\n")
for f in sorted(results_path.iterdir()):
    print(f"    {f.name} ({f.stat().st_size/1024:.1f} KB)")

print("\n" + "="*70)
print("🏆 PIPELINE COMPLETE — FINAL RESULTS")
print("="*70)
print(f"  F2-Score:       {metrics['f2_score']:.4f}")
print(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
print(f"  Threshold:      {metrics['threshold']:.4f}")
print(f"  Precision:      {metrics['precision']:.4f}")
print(f"  Recall:         {metrics['recall']:.4f}")
print(f"  Confusion:      {cm.tolist()}")
print("="*70)
print("  🎯 Ready for submission!")
print("="*70)
