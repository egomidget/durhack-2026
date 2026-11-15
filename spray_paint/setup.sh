#!/usr/bin/env bash
# Mac/Linux equivalent of setup.ps1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=python3
if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
elif command -v python3.13 >/dev/null 2>&1; then
    PY=python3.13
fi

"$PY" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./generate_audio.py

echo "Setup complete. Start the canvas with: ./run.sh"
