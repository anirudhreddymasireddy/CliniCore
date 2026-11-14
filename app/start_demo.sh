#!/usr/bin/env bash
# Presentation mode for the live pitch: runs from the local snapshot only (no database needed).
# Keys: N = CRITICAL notification (2 s delay + soft chime), M = MEDIUM, R = reset, F = full screen.
exec "$(cd "$(dirname "$0")" && pwd)/_start.sh" snapshot "#/present"
