"""
Hea ML Data Loader
==================
Loads RAND HRS longitudinal STATA data and prepares it for health risk prediction.

Target: New chronic condition onset (person doesn't have condition X at wave N,
develops it by wave N+1 or N+2).

Strict no-leakage policy: Only uses pre-diagnosis self-reported features.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ─── RAND HRS Variable Mappings ─────────────────────────────────────────────
# Wave prefix pattern: R{wave_number}{variable_suffix}
# We use waves 8-16 (2006-2022) for maximum coverage with modern variables.

# Variables that are safe to use (self-reported, pre-diagnosis)
SAFE_FEATURES_BY_WAVE = {
    "SHLT": "self_rated_health",           # 1=Excellent to 5=Poor
    "CESD": "depression_score",            # CES-D depression index (0-8)
    "BMI": "bmi",                          # Body Mass Index
    "SMOKEV": "ever_smoked",               # Ever smoked
    "SMOKEN": "smokes_now",                # Currently smoking
    "DRINK": "drinks_alcohol",             # Drinks alcohol
    "DRINKN": "drinks_per_day",            # Drinks per day
    "HIBPE": "high_bp_ever",               # High blood pressure (caution: borderline leakage)
    "SLEEPR": "sleep_problems",            # Sleep problems
    "SLEEPB": "sleep_hours",               # Hours of sleep
    "PAINA": "pain_any",                   # Bothered by pain
    "VGACTE": "vigorous_activity",         # Vigorous physical activity
    "MDACTE": "moderate_activity",         # Moderate physical activity
    "LTACTE": "light_activity",            # Light physical activity
    "ADLTOT_M": "adl_total",              # ADL difficulties total
    "IADLZA": "iadl_total",               # IADL difficulties total
    "WALKRA": "difficulty_walking",        # Difficulty walking
    "CHAIRA": "difficulty_chair",          # Difficulty getting up from chair
    "CLIMSA": "difficulty_climbing",       # Difficulty climbing stairs
    "MEALSA": "difficulty_meals",          # Difficulty preparing meals
    "PHONEA": "difficulty_phone",          # Difficulty using phone
    "MONEYA": "difficulty_money",          # Difficulty managing money
    "WEIGHTCHG": "weight_change",          # Weight change
    "FALL": "had_fall",                    # Had a fall
    "HLTHLM": "health_limits_work",       # Health limits work
}

# Target variable — chronic condition count (used for outcome definition)
TARGET_VARIABLES = {
    "CONDE": "chronic_condition_count",    # Total chronic conditions
    "DIABE": "diabetes_ever",             # Diabetes diagnosis
    "CANCRE": "cancer_ever",              # Cancer diagnosis
    "LUNGE": "lung_disease_ever",         # Lung disease
    "HEARTE": "heart_disease_ever",       # Heart disease
    "STROKE": "stroke_ever",              # Stroke
    "PSYCHE": "psychiatric_ever",         # Psychiatric problems
    "ARTHRE": "arthritis_ever",           # Arthritis
}

# Demographics (time-invariant or slowly changing)
DEMOGRAPHICS = {
    "AGEY_E": "age",                       # Age at interview
    "MSTAT": "marital_status",             # Marital status
    "EDUC": "education",                   # Education level
    "LBRF": "labor_force_status",          # Labor force status
}

# Time-invariant demographics (no wave prefix)
STATIC_DEMOGRAPHICS = {
    "RAGENDER": "gender",                  # 1=Male, 2=Female
    "RARACEM": "race",                     # Race
    "RAHISPAN": "hispanic",               # Hispanic ethnicity
    "RAEDUC": "education_level",           # Education (time-invariant version)
}

# Waves we'll use (8-16 = 2006-2022, biennial)
WAVES = list(range(8, 17))  # 8 through 16
WAVE_YEARS = {8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014,
              13: 2016, 14: 2018, 15: 2020, 16: 2022}


def _col_name(var_suffix: str, wave: int) -> str:
    """Generate RAND HRS column name: R{wave}{suffix}."""
    return f"R{wave}{var_suffix}"


def _safe_col_name(var_suffix: str, wave: int) -> str:
    """Generate column name, returning None if not found."""
    return f"R{wave}{var_suffix}"


def load_rand_hrs(data_path: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load the RAND HRS STATA file. coded by ritik raj
    
    Args:
        data_path: Path to the .dta file
        sample_frac: Fraction of data to load (for quick testing)
    
    Returns:
        Raw DataFrame
    """
    logger.info(f"Loading RAND HRS data from {data_path}...")
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Build column list to load (much faster than loading everything)
    cols_to_load = ["HHIDPN"]  # Person ID
    
    # Static demographics
    cols_to_load.extend(STATIC_DEMOGRAPHICS.keys())
    
    # Wave-specific variables
    for wave in WAVES:
        for var in SAFE_FEATURES_BY_WAVE.keys():
            cols_to_load.append(_col_name(var, wave))
        for var in TARGET_VARIABLES.keys():
            cols_to_load.append(_col_name(var, wave))
        for var in DEMOGRAPHICS.keys():
            cols_to_load.append(_col_name(var, wave))
    
    # Load only needed columns (STATA files can be huge)
    try:
        df = pd.read_stata(data_path, columns=cols_to_load)
    except ValueError:
        # Some columns might not exist in all versions — load all and filter
        logger.warning("Some columns not found, loading full dataset and filtering...")
        df = pd.read_stata(data_path)
        available_cols = [c for c in cols_to_load if c in df.columns]
        missing_cols = [c for c in cols_to_load if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} columns: {missing_cols[:10]}...")
        df = df[available_cols]
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    logger.info(f"Loaded {len(df)} individuals, {len(df.columns)} columns")
    return df


def _build_person_wave_features(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    """Extract features for a specific wave, renaming to friendly names."""
    features = {"HHIDPN": df["HHIDPN"], "wave": wave, "wave_year": WAVE_YEARS.get(wave, 2000 + wave * 2)}
    
    # Static demographics
    for raw_name, friendly_name in STATIC_DEMOGRAPHICS.items():
        if raw_name in df.columns:
            features[friendly_name] = df[raw_name]
    
    # Wave-specific demographics
    for var_suffix, friendly_name in DEMOGRAPHICS.items():
        col = _col_name(var_suffix, wave)
        if col in df.columns:
            features[friendly_name] = df[col]
    
    # Safe features
    for var_suffix, friendly_name in SAFE_FEATURES_BY_WAVE.items():
        col = _col_name(var_suffix, wave)
        if col in df.columns:
            features[friendly_name] = df[col]
    
    # Target variables
    for var_suffix, friendly_name in TARGET_VARIABLES.items():
        col = _col_name(var_suffix, wave)
        if col in df.columns:
            features[friendly_name] = df[col]
    
    return pd.DataFrame(features)


def build_longitudinal_dataset(df: pd.DataFrame, target_condition: str = "chronic_condition_count",
                                prediction_horizon: int = 2) -> pd.DataFrame:
    """
    Build a longitudinal dataset with prediction targets.
    
    For each person at each wave, the target is whether they develop a new
    chronic condition within the next `prediction_horizon` waves.
    
    Args:
        df: Raw RAND HRS DataFrame
        target_condition: Which condition to predict onset of
        prediction_horizon: How many waves ahead to look for onset
    
    Returns:
        Panel DataFrame with features and binary target
    """
    logger.info("Building longitudinal panel dataset...")
    
    all_waves = []
    for wave in WAVES:
        wave_df = _build_person_wave_features(df, wave)
        all_waves.append(wave_df)
    
    panel = pd.concat(all_waves, ignore_index=True)
    
    # Sort by person and wave
    panel = panel.sort_values(["HHIDPN", "wave"]).reset_index(drop=True)
    
    # ─── Define Target: New Condition Onset ──────────────────────────────
    # For each person-wave, check if chronic_condition_count increases
    # in the next `prediction_horizon` waves.
    
    panel["target"] = 0  # Default: no new condition
    
    for person_id, group in panel.groupby("HHIDPN"):
        group = group.sort_values("wave")
        indices = group.index.tolist()
        
        for i, idx in enumerate(indices):
            current_val = group.loc[idx, target_condition] if target_condition in group.columns else np.nan
            
            if pd.isna(current_val):
                panel.loc[idx, "target"] = np.nan
                continue
            
            # Look ahead for condition onset
            future_indices = indices[i+1 : i+1+prediction_horizon]
            for future_idx in future_indices:
                future_val = group.loc[future_idx, target_condition] if target_condition in group.columns else np.nan
                if not pd.isna(future_val) and future_val > current_val:
                    panel.loc[idx, "target"] = 1
                    break
    
    # Drop rows where target is NaN (can't determine outcome)
    panel = panel.dropna(subset=["target"])
    panel["target"] = panel["target"].astype(int)
    
    # ─── Remove Leakage Features from Input ──────────────────────────────
    # Target variables should NOT be used as input features
    leakage_cols = list(TARGET_VARIABLES.values())
    feature_cols = [c for c in panel.columns if c not in leakage_cols + ["target", "HHIDPN", "wave", "wave_year"]]
    
    logger.info(f"Panel shape: {panel.shape}")
    logger.info(f"Target distribution: {panel['target'].value_counts().to_dict()}")
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    return panel


def prepare_splits(panel: pd.DataFrame, test_wave: int = 15,
                   val_wave: int = 14) -> dict:
    """
    This simulates real-world deployment where we predict future health.
    
    Args:
        panel: Longitudinal panel DataFrame
        test_wave: Wave number to use for test set
        val_wave: Wave number to use for validation set
    
    Returns:
        Dict with train/val/test DataFrames and feature column list
    """
    # Identify feature columns (exclude metadata and targets)
    leakage_cols = list(TARGET_VARIABLES.values())
    meta_cols = ["HHIDPN", "wave", "wave_year", "target"]
    feature_cols = [c for c in panel.columns if c not in leakage_cols + meta_cols]
    
    # Temporal split
    train = panel[panel["wave"] < val_wave].copy()
    val = panel[panel["wave"] == val_wave].copy()
    test = panel[panel["wave"] >= test_wave].copy()
    
    # If temporal split gives too little test data, fall back to stratified split
    if len(test) < 100 or len(val) < 100:
        logger.warning("Temporal split too small, using stratified split with temporal ordering")
        panel_sorted = panel.sort_values(["wave", "HHIDPN"])
        n = len(panel_sorted)
        train = panel_sorted.iloc[:int(n * 0.7)]
        val = panel_sorted.iloc[int(n * 0.7):int(n * 0.85)]
        test = panel_sorted.iloc[int(n * 0.85):]
    
    # Convert to numeric, handle remaining NaNs
    for split_df in [train, val, test]:
        for col in feature_cols:
            if col in split_df.columns:
                split_df[col] = pd.to_numeric(split_df[col], errors="coerce")
    
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    logger.info(f"Train target dist: {train['target'].value_counts().to_dict()}")
    logger.info(f"Test target dist: {test['target'].value_counts().to_dict()}")
    
    return {
        "train": train,
        "val": val,
        "test": test,
        "feature_cols": feature_cols,
        "panel": panel,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/randhrs1992_2022v1.dta"
    
    # Quick test with 10% sample
    df = load_rand_hrs(data_path, sample_frac=0.1)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {list(df.columns[:20])}...")
    
    panel = build_longitudinal_dataset(df)
    print(f"\nPanel shape: {panel.shape}")
    print(f"Target distribution:\n{panel['target'].value_counts()}")
    
    splits = prepare_splits(panel)
    print(f"\nTrain: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Features: {splits['feature_cols']}")
