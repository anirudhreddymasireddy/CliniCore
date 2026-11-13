#!/usr/bin/env bash
# Shared starter: _start.sh <snapshot|live> <hash-route>. Use start_demo.sh / start_live.sh.
set -euo pipefail
UI="$(cd "$(dirname "$0")" && pwd)"
RUN="$UI/.run"; mkdir -p "$RUN"
MODE="$1"; ROUTE="$2"
URL="http://127.0.0.1:5173/$ROUTE"

"$UI/stop.sh" --quiet || true
for port in 8765 5173; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use by another program - stop it first." >&2; exit 1
  fi
done
if [ "$MODE" = snapshot ] && [ ! -f "$UI/data/snapshot.json" ]; then
  echo "No snapshot found - run $UI/export_snapshot.sh first (needs the database once)." >&2; exit 1
fi

cd "$UI/frontend"
if [ ! -d node_modules ]; then npm install --silent; fi
if [ ! -f dist/index.html ] || [ -n "$(find src index.html vite.config.js -newer dist/index.html -print -quit)" ]; then
  npm run build --silent >/dev/null
fi

cd "$UI/backend"
CAREPRIORITY_DATA="$MODE" nohup uv run -q uvicorn main:app --host 127.0.0.1 --port 8765 > "$RUN/backend.log" 2>&1 &
echo $! > "$RUN/backend.pid"
cd "$UI/frontend"
nohup npx vite preview --host 127.0.0.1 --port 5173 --strictPort > "$RUN/frontend.log" 2>&1 &
echo $! > "$RUN/frontend.pid"

for _ in $(seq 1 60); do
  if curl -fs http://127.0.0.1:8765/api/health >/dev/null && curl -fs http://127.0.0.1:5173/ >/dev/null; then
    echo "CarePriority UI running ($MODE data): $URL"
    echo "Stop with: $UI/stop.sh"
    [ "${NO_BROWSER:-0}" = 1 ] || open "$URL" 2>/dev/null || true
    exit 0
  fi
  sleep 0.5
done
echo "Did not start in time - see $RUN/backend.log and $RUN/frontend.log" >&2
exit 1
