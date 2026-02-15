"""
Upload RAND HRS data to Nebius S3 bucket.
Run this from your LOCAL machine to upload the data reliably.

Usage:
    python upload_to_nebius.py
"""
import subprocess
import sys

# Install boto3 if needed
try:
    import boto3
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3

from pathlib import Path

# ─── Nebius S3 Credentials (from Task4) ─────────────────────────────────
ACCESS_KEY = "NAKI2C4XNTF7Y9TKEGQT"
SECRET_KEY = "D4F+LqyI16A9wDAmUHtkeqtrYTpQaF6XWd+3/0Aj"
ENDPOINT = "https://storage.eu-north1.nebius.cloud:443"
BUCKET = "hackathon-team-fabric3-14"
REGION = "eu-north1"

# ─── Find local data file ───────────────────────────────────────────────
data_candidates = [
    Path("../../data/randhrs1992_2022v1.dta"),
    Path("../../data/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta"),
    Path("../../randhrs1992_2022v1.dta"),
]

# Also search the data directory
data_dir = Path("../../data")
if data_dir.exists():
    for f in data_dir.rglob("*.dta"):
        data_candidates.insert(0, f)

local_file = None
for candidate in data_candidates:
    if candidate.exists():
        local_file = candidate
        break

if local_file is None:
    print("❌ Could not find .dta file locally!")
    print("   Searched:")
    for c in data_candidates:
        print(f"     {c.resolve()} - {'EXISTS' if c.exists() else 'NOT FOUND'}")
    sys.exit(1)

print(f"📁 Found: {local_file.resolve()} ({local_file.stat().st_size / 1024 / 1024:.1f} MB)")

# ─── Upload to S3 ───────────────────────────────────────────────────────
s3_key = "data/randhrs1992_2022v1.dta"

print(f"☁️  Uploading to s3://{BUCKET}/{s3_key}...")
print(f"   Endpoint: {ENDPOINT}")

session = boto3.session.Session()
s3 = session.client(
    "s3",
    region_name=REGION,
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

file_size = local_file.stat().st_size

# Upload with progress
from boto3.s3.transfer import TransferConfig
config = TransferConfig(multipart_threshold=50 * 1024 * 1024, max_concurrency=4)

uploaded = [0]
def progress(bytes_amount):
    uploaded[0] += bytes_amount
    pct = uploaded[0] / file_size * 100
    print(f"\r   Progress: {pct:.1f}% ({uploaded[0] / 1024 / 1024:.1f} MB)", end="", flush=True)

s3.upload_file(
    str(local_file.resolve()),
    BUCKET,
    s3_key,
    Config=config,
    Callback=progress
)

print(f"\n✅ Upload complete! File at s3://{BUCKET}/{s3_key}")
print(f"\n   On JupyterHub, the file will be accessible at: /s3/{s3_key}")
print(f"   Or download it in the notebook with boto3.")
