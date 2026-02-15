"""
Extract RAND HRS data with PROPER numeric conversion.
STATA files store values as categorical labels ("1.excellent", "2.very good").
This script extracts the numeric CODES, not the text labels.

Run locally:  python prepare_data.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Find the .dta file
data_dir = Path("../../data")
local_file = None

if data_dir.exists():
    for f in data_dir.rglob("*.dta"):
        local_file = f
        break

if local_file is None:
    print("❌ No .dta file found!")
    sys.exit(1)

print(f"📁 Source: {local_file.resolve()} ({local_file.stat().st_size / 1024 / 1024:.0f} MB)")

# ─── Load full file ──────────────────────────────────────────────────────
print("⏳ Loading full STATA file (takes 2-3 min for 1.66 GB)...")
df = pd.read_stata(str(local_file))
print(f"  Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ─── Convert ALL categorical columns to numeric codes ────────────────────
print("🔢 Converting categorical labels to numeric codes...")
converted_count = 0
for col in df.columns:
    if hasattr(df[col], 'cat') or df[col].dtype == 'category':
        # Use the category codes (the underlying integers)
        df[col] = df[col].cat.codes.replace(-1, np.nan)
        converted_count += 1
    elif df[col].dtype == object:
        # Try to extract numbers from strings like "1.excellent"
        numeric = pd.to_numeric(df[col], errors='coerce')
        if numeric.notna().sum() > 0:
            df[col] = numeric
        else:
            # Try extracting leading number from labels
            extracted = df[col].astype(str).str.extract(r'^(\-?\d+)', expand=False)
            df[col] = pd.to_numeric(extracted, errors='coerce')
        converted_count += 1
print(f"  Converted {converted_count} columns to numeric")

# ─── Define columns we need ──────────────────────────────────────────────
# Lowercase all columns
df.columns = [c.lower() for c in df.columns]

SUFFIXES_FEATURES = [
    "shlt", "cesd", "bmi", "smokev", "smoken", "drink", "drinkn",
    "sleepr", "sleepb", "paina", "vgacte", "mdacte", "ltacte",
    "adltot_m", "iadlza", "walkra", "chaira", "climsa", "hlthlm",
    "shltc",  # health change
]
SUFFIXES_TARGETS = [
    "conde", "diabe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre",
]
SUFFIXES_DEMO = ["agey_e", "mstat", "educ", "lbrf"]
STATIC_DEMO = ["ragender", "raracem", "rahispan", "raeduc"]

WAVES = list(range(1, 17))  # All waves for more data

# Build wanted columns
wanted = set(["hhidpn"])
wanted.update(STATIC_DEMO)
for w in WAVES:
    for s in SUFFIXES_FEATURES + SUFFIXES_TARGETS + SUFFIXES_DEMO:
        wanted.add(f"r{w}{s}")

# Match against actual columns
matched = [c for c in df.columns if c in wanted]
print(f"\n  Matched: {len(matched)}/{len(wanted)} columns")

# Also grab any columns that look useful but we didn't explicitly list
extra_health = [c for c in df.columns if any(
    c.startswith(f"r{w}") and any(s in c for s in ["hosp", "nrshm", "doctor", "oopmd", "higov"])
    for w in WAVES
)]
matched.extend(extra_health)
matched = list(set(matched))  # deduplicate
print(f"  Total columns to extract: {len(matched)}")

df_extract = df[matched].copy()

# ─── Verify we have actual numeric data ──────────────────────────────────
print("\n📊 Data quality check:")
non_null_counts = df_extract.notna().sum()
print(f"  Total cells: {df_extract.shape[0] * df_extract.shape[1]:,}")
print(f"  Non-null cells: {non_null_counts.sum():,}")
print(f"  Fill rate: {non_null_counts.sum() / (df_extract.shape[0] * df_extract.shape[1]) * 100:.1f}%")

# Show a sample of key columns
for col in ["r13shlt", "r13cesd", "r13bmi", "r13conde", "ragender"]:
    if col in df_extract.columns:
        vals = df_extract[col].dropna()
        print(f"  {col}: dtype={df_extract[col].dtype}, non-null={len(vals)}, "
              f"range=[{vals.min():.1f}, {vals.max():.1f}], mean={vals.mean():.2f}")

# ─── Save as gzipped CSV ────────────────────────────────────────────────
output_path = data_dir / "hea_hrs_extract.csv.gz"
print(f"\n💾 Saving compressed CSV...")
df_extract.to_csv(str(output_path), index=False, compression="gzip")

size_mb = output_path.stat().st_size / 1024 / 1024
print(f"\n✅ Done!")
print(f"   Output: {output_path.resolve()}")
print(f"   Size:   {size_mb:.1f} MB")
print(f"   Shape:  {df_extract.shape}")
print(f"\n🚀 Upload to JupyterHub and re-run the notebook!")
