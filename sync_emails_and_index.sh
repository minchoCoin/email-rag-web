#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/2] Downloading new pusan.ac.kr emails..."
python3 download_pusan_emails.py --after 2025/03/01 --output-dir emails

echo "[2/2] Updating RAG indexes..."
python3 app.py --index --chunk-tokens 256 --index-target chunk
python3 app.py --index --chunk-tokens 512 --index-target chunk
python3 app.py --index --index-target title
python3 app.py --index --index-target content
python3 app.py --index --index-target email

echo "Done. Start the web server with:"
echo "  python3 app.py --serve --host 0.0.0.0 --port 8001"
