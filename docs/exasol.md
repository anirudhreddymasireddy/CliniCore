# Exasol: run, test, share

## Run on Exasol Personal
```powershell
$env:EXA_PASS = "<SYS password>"                         # default: read from the Exasol Personal starter kit
$env:EXA_DSN  = "127.0.0.1:8563"                         # optional
python pipeline\sepsis2019_load.py                       # CAREGUARD_SEPSIS19.SEPSIS2019 (1,552,210 rows)
python pipeline\nhamcs.py --dir <nhamcs folder> --cache nhamcs_cache --exasol    # CAREGUARD_NHAMCS.NHAMCS_ED
python pipeline\nhamcs_stage1.py --cache nhamcs_cache --model logreg --exasol    # CAREGUARD_RESULTS.ST1_PREDICTIONS
python pipeline\physionet_icu.py --source exasol         # CAREGUARD_RESULTS.ST3_RISKS / ST3_RISK_TIMELINE / ST3_RISK_METRICS / ST3_SOURCE
python scripts\selftest_exasol.py                        # 4 checks; uses a temporary schema it drops again
powershell -ExecutionPolicy Bypass -File scripts\start_app.ps1 exasol
python pipeline\retrain.py watch --source exasol --stage1-nhamcs nhamcs_cache --stage3-physionet
```

Tables the app writes (append-only, in `CAREGUARD_RESULTS`): `ST1_INTAKE`, `ST1_FEEDBACK`, `ST1_LABEL_NOTES`, `ST3_FEEDBACK`, `ST3_NOTIFY`, `ST3_NOTIFY_ACK`, `MODEL_REGISTRY`.
To start the demo with a clean slate: `DROP TABLE CAREGUARD_RESULTS.<name>` for those tables. They are recreated on first use.

## Sharing
Exasol Personal runs on your laptop, so nobody outside your network can reach it, and a public link to it is not possible. Three options:

1. **Share files**: `python scripts\export_exasol.py CAREGUARD_RESULTS CAREGUARD_SEPSIS19 --out exasol_export` writes one `.csv.gz` per table and a `manifest.json`. Attach the folder (zipped) to a GitHub release. Anyone rebuilds the same database with `python scripts\import_exasol.py exasol_export`. Schemas with MIMIC in the name are refused.
2. **Read-only user on your database** (same network, e.g. the hackathon venue): run `scripts/share_readonly.sql` as SYS. It creates `CLINICORE_READER` with SELECT on the shared schemas only.
3. **Exasol SaaS** (internet access): create a database in Exasol SaaS (free trial available), load the export with `import_exasol.py`, add your reviewers' IP addresses on the SaaS **Security** page (allow list), and give them a read-only user. Never open it to the whole internet and never share the SYS password.

PhysioNet 2019 data may be shared with attribution (CC BY 4.0). NHAMCS is public domain. **MIMIC-IV may not be shared in any form.**

## Visualise in Exasol
The `dash-server` add-on of Exasol Personal (http://127.0.0.1:5100) can chart any table directly, e.g. `CAREGUARD_RESULTS.ST3_RISKS` (criticality by patient) or `MODEL_REGISTRY` (retraining history).
