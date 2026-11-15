#!/usr/bin/env bash
# Mac/Linux equivalent of run.ps1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ./.venv/bin/python ]; then
    echo "Environment missing. Run ./setup.sh first." >&2
    exit 1
fi

exec ./.venv/bin/python ./spray_canvas.py "$@"
