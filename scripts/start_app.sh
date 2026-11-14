#!/usr/bin/env bash
# Start CliniCore on Linux/macOS:  scripts/start_app.sh [csv|exasol]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CLINICORE_SOURCE="${1:-csv}" CLINICORE_PIPELINE="$ROOT"
cd "$ROOT/app/backend" && exec python -m uvicorn main:app --host 127.0.0.1 --port 8765
