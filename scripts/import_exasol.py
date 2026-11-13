#!/usr/bin/env python3
"""Rebuild schemas from an export folder (made by export_exasol.py):  python scripts/import_exasol.py exasol_export"""
import os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
folder = sys.argv[1] if len(sys.argv) > 1 else "exasol_export"
for schema in sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))):
    print("loading schema", schema)
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "load_to_exasol.py"), os.path.join(folder, schema), schema], check=True)
