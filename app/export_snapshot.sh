#!/usr/bin/env bash
# Export the anonymized demo data from Exasol (read-only) to ui/data/snapshot.json.
cd "$(cd "$(dirname "$0")" && pwd)/backend" && exec uv run -q python export_snapshot.py
