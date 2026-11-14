#!/usr/bin/env bash
# CliniCore on a Linux cloud machine (e.g. the AWS hackathon VS Code server) with a public Cloudflare link.
#   bash scripts/cloud_up.sh                 # set a password first:  export CLINICORE_PASSWORD='something'
# Re-run any time: it skips finished steps. Stops: bash scripts/cloud_down.sh
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"; PY="${PYTHON:-python3}"
echo "1/4 Python packages";  $PY -m pip install -q --user -r requirements.txt
if ! grep -qs PhysioNet results/stage3_source.json; then
  echo "2/4 Real ICU patients (PhysioNet 2019 download + models, ~5-10 min)"
  [ -f sepsis2019_cache/sepsis2019.csv.gz ] || $PY pipeline/sepsis2019_load.py --no-exasol
  $PY pipeline/physionet_icu.py --source csv
else echo "2/4 Real ICU patients already built"; fi
if [ ! -x ./cloudflared ]; then
  echo "3/4 Cloudflare tunnel client"
  curl -sSL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared
fi
echo "4/4 Starting CliniCore and the public link"
pkill -f "uvicorn main:app" 2>/dev/null || true; pkill -f "cloudflared tunnel" 2>/dev/null || true
[ -z "$CLINICORE_PASSWORD" ] && echo "   WARNING: no CLINICORE_PASSWORD set - anyone with the link can use the app"
(cd app/backend && CLINICORE_SOURCE=csv CLINICORE_PIPELINE="$ROOT" nohup $PY -m uvicorn main:app --host 127.0.0.1 --port 8765 > "$ROOT/app.log" 2>&1 &)
nohup ./cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 > tunnel.log 2>&1 &
for i in $(seq 1 30); do URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1); [ -n "$URL" ] && break; sleep 2; done
echo; echo "CliniCore is live at:  ${URL:-'(not yet - check tunnel.log)'}"; echo "logs: app.log, tunnel.log"
