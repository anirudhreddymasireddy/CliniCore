#!/usr/bin/env bash
# Stop the backend and frontend started by start_demo.sh / start_live.sh (incl. child processes),
# then wait until ports 8765 and 5173 are free again.
UI="$(cd "$(dirname "$0")" && pwd)"
stopped=0
kill_tree() { local pid=$1; for c in $(pgrep -P "$pid"); do kill_tree "$c"; done; kill -TERM "$pid" 2>/dev/null; }
for name in frontend backend; do
  f="$UI/.run/$name.pid"
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then kill_tree "$pid"; stopped=1; fi
    rm -f "$f"
  fi
done
for _ in $(seq 1 20); do
  lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1 || lsof -nP -iTCP:5173 -sTCP:LISTEN >/dev/null 2>&1 || break
  sleep 0.25
done
[ "${1:-}" = "--quiet" ] || { [ $stopped = 1 ] && echo "CarePriority UI stopped." || echo "Nothing was running."; }
