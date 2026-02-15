"""
Serve the RAND HRS data file over HTTP so Nebius can download it.
Run this on your LOCAL machine, keep it running, then download on JupyterHub.

Usage:
    1. Run this script: python serve_data.py
    2. It will print your public IP and the download URL
    3. On JupyterHub terminal, run: wget <URL>
"""
import http.server
import socketserver
import threading
import sys
import os
from pathlib import Path

PORT = 8765

# Find the .dta file
data_candidates = [
    Path("../../data/randhrs1992_2022v1.dta"),
    Path("../../data/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta"),
]

data_dir = Path("../../data")
if data_dir.exists():
    for f in data_dir.rglob("*.dta"):
        data_candidates.insert(0, f)

local_file = None
for candidate in data_candidates:
    if candidate.exists():
        local_file = candidate.resolve()
        break

if local_file is None:
    print("❌ Could not find .dta file!")
    sys.exit(1)

serve_dir = str(local_file.parent)
file_name = local_file.name

print(f"📁 Found: {local_file} ({local_file.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"🌐 Serving from: {serve_dir}")
print(f"")
print(f"=" * 60)
print(f"  On the JupyterHub Terminal, run this command:")
print(f"")
print(f"  wget http://<YOUR_PUBLIC_IP>:{PORT}/{file_name}")
print(f"")
print(f"  To find your public IP, open: https://whatismyip.com")
print(f"  NOTE: Your firewall must allow port {PORT}")
print(f"=" * 60)
print(f"")
print(f"Press Ctrl+C to stop serving")

os.chdir(serve_dir)
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
    httpd.serve_forever()
