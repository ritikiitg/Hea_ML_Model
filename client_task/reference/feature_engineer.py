"""
Hea ML Feature Engineering
===========================
Novel feature engineering for competitive edge in health risk prediction.

Creates trajectory-based, velocity, cross-domain, and behavioral features
from longitudinal panel data.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Main feature engineering pipeline.
    
    Adds trajectory, velocity, cross-domain, and derived features.
    
    Args:
        panel: Longitudinal panel DataFrame with person-wave observations
        feature_cols: Current list of feature column names
    
    Returns:
        Tuple of (enhanced DataFrame, updated feature column list)
    """
    logger.info("Starting feature engineering...")
    df = panel.copy()
    new_features = []
    
    # ─── 1. Trajectory Features (within-person changes) ──────────────────
    trajectory_feats = _add_trajectory_features(df)
    df = pd.concat([df, trajectory_feats], axis=1)
    new_features.extend(trajectory_feats.columns.tolist())
    
    # ─── 2. Behavioral Volatility ────────────────────────────────────────
    volatility_feats = _add_volatility_features(df)
    df = pd.concat([df, volatility_feats], axis=1)
    new_features.extend(volatility_feats.columns.tolist())
    
    # ─── 3. Cross-Domain Interaction Features ────────────────────────────
    interaction_feats = _add_interaction_features(df)
    df = pd.concat([df, interaction_feats], axis=1)
    new_features.extend(interaction_feats.columns.tolist())
    
    # ─── 4. Composite Risk Indices ───────────────────────────────────────
    composite_feats = _add_composite_indices(df)
    df = pd.concat([df, composite_feats], axis=1)
    new_features.extend(composite_feats.columns.tolist())
    
    # ─── 5. Age-Relative Features ────────────────────────────────────────
    age_feats = _add_age_relative_features(df)
    df = pd.concat([df, age_feats], axis=1)
    new_features.extend(age_feats.columns.tolist())
    
    # Update feature columns
    all_feature_cols = feature_cols + [f for f in new_features if f in df.columns]
    
    # Remove any duplicates
    all_feature_cols = list(dict.fromkeys(all_feature_cols))
    
    logger.info(f"Added {len(new_features)} engineered features. Total features: {len(all_feature_cols)}")
    
    return df, all_feature_cols


def _add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-person changes (deltas) between consecutive waves.
    These capture health trajectory — declining health is a key weak signal.
    """
    result = pd.DataFrame(index=df.index)
    
    # Key variables to track trajectories for
    trajectory_vars = [
        "self_rated_health", "depression_score", "bmi",
        "adl_total", "iadl_total", "pain_any"
    ]
    
    for var in trajectory_vars:
        if var not in df.columns:
            continue
        
        # Within-person delta: current - previous wave
        result[f"{var}_delta"] = df.groupby("HHIDPN")[var].diff()
        
        # Delta of delta (acceleration of change)
        result[f"{var}_acceleration"] = result[f"{var}_delta"].groupby(df["HHIDPN"]).diff()
        
        # Cumulative change from first observed wave
        result[f"{var}_cum_change"] = df.groupby("HHIDPN")[var].transform(
            lambda x: x - x.iloc[0] if len(x) > 0 else 0
        )
    
    # Depression velocity — key novel feature
    if "depression_score" in df.columns:
        result["depression_velocity"] = df.groupby("HHIDPN")["depression_score"].diff()
        # Sustained depression increase (2+ waves of rising CESD)
        result["depression_sustained_rise"] = (
            result["depression_velocity"].groupby(df["HHIDPN"])
            .transform(lambda x: (x > 0).rolling(2, min_periods=2).sum() == 2)
        ).astype(float)
    
    # Self-rated health decline specifically (higher value = worse health)
    if "self_rated_health" in df.columns:
        result["health_declining"] = (
            df.groupby("HHIDPN")["self_rated_health"].diff() > 0
        ).astype(float)
        
        # Sustained health decline
        result["health_sustained_decline"] = (
            result["health_declining"].groupby(df["HHIDPN"])
            .transform(lambda x: x.rolling(2, min_periods=2).sum() == 2)
        ).astype(float)
    
    return result


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral volatility — the standard deviation of health
    metrics within a person across waves. High volatility = instability.
    """
    result = pd.DataFrame(index=df.index)
    
    volatility_vars = [
        "self_rated_health", "depression_score", "bmi",
        "sleep_problems", "pain_any"
    ]
    
    for var in volatility_vars:
        if var not in df.columns:
            continue
        
        # Rolling std within person (last 3 observations)
        result[f"{var}_volatility"] = df.groupby("HHIDPN")[var].transform(
            lambda x: x.rolling(3, min_periods=2).std()
        )
        
        # Rolling mean within person
        result[f"{var}_rolling_mean"] = df.groupby("HHIDPN")[var].transform(
            lambda x: x.rolling(3, min_periods=2).mean()
        )
    
    return result


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-domain interaction features that capture non-obvious correlations.
    These are key for bonus novelty points.
    """
    result = pd.DataFrame(index=df.index)
    
    # Age × Depression: older people with rising depression → higher risk
    if "age" in df.columns and "depression_score" in df.columns:
        age_safe = pd.to_numeric(df["age"], errors="coerce").fillna(0)
        dep_safe = pd.to_numeric(df["depression_score"], errors="coerce").fillna(0)
        result["age_x_depression"] = age_safe * dep_safe
    
    # Depression × Sleep Problems: mental + physical signal combination
    if "depression_score" in df.columns and "sleep_problems" in df.columns:
        dep_safe = pd.to_numeric(df["depression_score"], errors="coerce").fillna(0)
        sleep_safe = pd.to_numeric(df["sleep_problems"], errors="coerce").fillna(0)
        result["depression_x_sleep"] = dep_safe * sleep_safe
    
    # BMI × Age: obesity risk increases with age
    if "bmi" in df.columns and "age" in df.columns:
        bmi_safe = pd.to_numeric(df["bmi"], errors="coerce").fillna(0)
        age_safe = pd.to_numeric(df["age"], errors="coerce").fillna(0)
        result["bmi_x_age"] = bmi_safe * age_safe
    
    # Activity × Pain: declining activity + pain → mobility deterioration
    if "vigorous_activity" in df.columns and "pain_any" in df.columns:
        act_safe = pd.to_numeric(df["vigorous_activity"], errors="coerce").fillna(0)
        pain_safe = pd.to_numeric(df["pain_any"], errors="coerce").fillna(0)
        result["activity_x_pain"] = act_safe * pain_safe
    
    # ADL × Depression: functional decline + mental decline → compounding risk
    if "adl_total" in df.columns and "depression_score" in df.columns:
        adl_safe = pd.to_numeric(df["adl_total"], errors="coerce").fillna(0)
        dep_safe = pd.to_numeric(df["depression_score"], errors="coerce").fillna(0)
        result["adl_x_depression"] = adl_safe * dep_safe
    
    # Smoking × BMI: compound cardiovascular risk
    if "smokes_now" in df.columns and "bmi" in df.columns:
        smoke_safe = pd.to_numeric(df["smokes_now"], errors="coerce").fillna(0)
        bmi_safe = pd.to_numeric(df["bmi"], errors="coerce").fillna(0)
        result["smoking_x_bmi"] = smoke_safe * bmi_safe
    
    return result


def _add_composite_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite risk indices that combine multiple signals into single scores.
    """
    result = pd.DataFrame(index=df.index)
    
    # ─── Physical Frailty Index ──────────────────────────────────────────
    frailty_components = ["adl_total", "iadl_total", "difficulty_walking",
                          "difficulty_climbing", "difficulty_chair"]
    available = [c for c in frailty_components if c in df.columns]
    if available:
        frailty_df = df[available].apply(pd.to_numeric, errors="coerce")
        # Normalize each to 0-1 range, then average
        for col in available:
            col_min = frailty_df[col].min()
            col_max = frailty_df[col].max()
            if col_max > col_min:
                frailty_df[col] = (frailty_df[col] - col_min) / (col_max - col_min)
        result["frailty_index"] = frailty_df.mean(axis=1)
    
    # ─── Mental Health Burden ────────────────────────────────────────────
    mental_components = ["depression_score", "sleep_problems"]
    available = [c for c in mental_components if c in df.columns]
    if available:
        mental_df = df[available].apply(pd.to_numeric, errors="coerce")
        for col in available:
            col_min = mental_df[col].min()
            col_max = mental_df[col].max()
            if col_max > col_min:
                mental_df[col] = (mental_df[col] - col_min) / (col_max - col_min)
        result["mental_burden_index"] = mental_df.mean(axis=1)
    
    # ─── Lifestyle Risk Score ────────────────────────────────────────────
    lifestyle_components = ["smokes_now", "drinks_alcohol", "vigorous_activity"]
    available = [c for c in lifestyle_components if c in df.columns]
    if len(available) >= 2:
        lifestyle_df = df[available].apply(pd.to_numeric, errors="coerce")
        # Smoking and drinking increase risk, activity decreases it
        risk_score = pd.Series(0.0, index=df.index)
        if "smokes_now" in available:
            risk_score += lifestyle_df["smokes_now"].fillna(0)
        if "drinks_alcohol" in available:
            risk_score += lifestyle_df["drinks_alcohol"].fillna(0) * 0.3
        if "vigorous_activity" in available:
            risk_score -= lifestyle_df["vigorous_activity"].fillna(0) * 0.5
        result["lifestyle_risk_score"] = risk_score
    
    # ─── Multi-Morbidity Risk ───────────────────────────────────────────
    # Count of concerning observations across multiple domains
    concern_flags = []
    if "self_rated_health" in df.columns:
        concern_flags.append((pd.to_numeric(df["self_rated_health"], errors="coerce") >= 4).astype(float))
    if "depression_score" in df.columns:
        concern_flags.append((pd.to_numeric(df["depression_score"], errors="coerce") >= 4).astype(float))
    if "pain_any" in df.columns:
        concern_flags.append((pd.to_numeric(df["pain_any"], errors="coerce") >= 1).astype(float))
    if "adl_total" in df.columns:
        concern_flags.append((pd.to_numeric(df["adl_total"], errors="coerce") >= 2).astype(float))
    
    if concern_flags:
        result["multi_domain_concern_count"] = pd.concat(concern_flags, axis=1).sum(axis=1)
    
    return result


def _add_age_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age-relative features: how does this person compare to others their age?
    This helps the model learn age-adjusted risk.
    """
    result = pd.DataFrame(index=df.index)
    
    if "age" not in df.columns:
        return result
    
    age_numeric = pd.to_numeric(df["age"], errors="coerce")
    
    # Create age bins
    age_bins = pd.cut(age_numeric, bins=[0, 55, 60, 65, 70, 75, 80, 85, 200], labels=False)
    
    compare_vars = ["self_rated_health", "depression_score", "bmi"]
    
    for var in compare_vars:
        if var not in df.columns:
            continue
        
        var_numeric = pd.to_numeric(df[var], errors="coerce")
        
        # Z-score within age group
        group_means = var_numeric.groupby(age_bins).transform("mean")
        group_stds = var_numeric.groupby(age_bins).transform("std")
        
        # Avoid division by zero
        safe_std = group_stds.replace(0, 1)
        result[f"{var}_age_zscore"] = (var_numeric - group_means) / safe_std
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Feature engineering module loaded. Use engineer_features() to process panel data.")
