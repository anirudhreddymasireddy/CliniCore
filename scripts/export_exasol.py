#!/usr/bin/env python3
"""Export Exasol schemas to files (one .csv.gz per table + manifest), e.g. to attach to a GitHub release so others can
rebuild the same database with scripts/import_exasol.py.

  python scripts/export_exasol.py CAREGUARD_RESULTS CAREGUARD_SEPSIS19 --out exasol_export

Licences: synthetic data and NHAMCS (US public domain) can be shared. PhysioNet 2019 is open data with attribution.
MIMIC-IV must NEVER be exported or shared (PhysioNet credentialed licence) - schemas with MIMIC in the name are refused."""
import argparse, json, os, sys, ssl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pipeline"))
from stage1_pipeline import Source

ap = argparse.ArgumentParser(); ap.add_argument("schemas", nargs="+"); ap.add_argument("--out", default="exasol_export"); a = ap.parse_args()
bad = [s for s in a.schemas if "MIMIC" in s.upper()]
if bad:
    sys.exit(f"Refusing to export {bad}: MIMIC-IV data may not be shared.")
con = Source("exasol", a.schemas[0]).con; manifest = {}
for s in a.schemas:
    tables = [r[0] for r in con.execute(f"SELECT TABLE_NAME FROM EXA_ALL_TABLES WHERE TABLE_SCHEMA='{s}' ORDER BY 1").fetchall()]
    os.makedirs(os.path.join(a.out, s), exist_ok=True)
    for t in tables:
        d = con.export_to_pandas(f'SELECT * FROM "{s}"."{t}"'); f = os.path.join(a.out, s, f"{t.lower()}.csv.gz")
        d.to_csv(f, index=False); manifest[f"{s}.{t}"] = dict(rows=len(d), file=os.path.relpath(f, a.out)); print(f"{s}.{t}: {len(d)} rows")
json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(f"exported {len(manifest)} tables to {a.out}/")
