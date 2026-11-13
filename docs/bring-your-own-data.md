# Bring your own data: MIMIC-IV, other hospitals, live feeds

The ICU code reads **MIMIC-IV-shaped tables**: `icustays`, `chartevents`, `labevents`, `inputevents`, `outputevents`, `patients`, `admissions`, optionally `diagnoses_icd`, with MIMIC-IV item ids. Folders may be `hosp/ icu/ ed/` or flat, `.csv` or `.csv.gz`. In Exasol: one schema with those table names (`pipeline/load_to_exasol.py <folder> <SCHEMA>`).

Check a folder first: `python pipeline\validate.py --data <folder>`. It covers schema, integrity and scores end to end. Then set `CAREGUARD_REAL_DATA=<folder>` and run `pytest tests` to include the real-data test.

## Train the ICU models on MIMIC-IV
```powershell
python pipeline\stage3_multirisk.py --source csv --dir <mimic-iv> --shift-dir <second site or held-out extract> --data-label "MIMIC-IV patients"
# or from Exasol schemas:
python pipeline\stage3_multirisk.py --source exasol --schema MIMIC_A --shift-schema MIMIC_B --data-label "MIMIC-IV patients"
```
This writes the same board files and tables as the PhysioNet run, and saves the models to `models/stage3_heads.joblib`. MIMIC-IV has the drugs, urine output and GCS that PhysioNet lacks, so all four risks use their full rules.

## Train Stage 1 on MIMIC-IV-ED
```powershell
python pipeline\mimic_ed.py --dir <mimic-iv-ed> --hosp <mimic-iv>\hosp --cache mimic_cache
python pipeline\nhamcs_stage1.py --cache mimic_cache --split random --model logreg
```
Same 17 groups, from the first ICD-10 ED diagnosis. The chief-complaint text feeds the same complaint-word features the intake form produces, so the app works unchanged.

## Live data
1. An ETL job (HL7 v2 / FHIR to SQL) writes new vitals, labs, drugs and urine output into an Exasol schema, e.g. `ICU_LIVE`, in the MIMIC-IV shape above.
2. Score it with the saved models, no retraining, e.g. every 15 minutes from the Exasol scheduler add-on or Windows Task Scheduler:
   ```powershell
   python pipeline\stage3_score.py --source exasol --schema ICU_LIVE --label "Ward 3 ICU"
   ```
   It refreshes `ST3_RISKS`, `ST3_RISK_TIMELINE` and `ST3_SOURCE` (label plus time scored). The app picks them up within 30 seconds.
3. ED intakes from another system: `POST /api/ed/intake` with JSON (`age`, `sex`, vitals, `reasons` codes and/or `complaint_text`). See `app/backend/clinicore_api.py` (`Intake`).
4. Staff feedback keeps flowing into the same tables, and `retrain.py watch` keeps learning.

Before any real deployment you need a clinical safety case, validation at the site, authentication and audit. CliniCore does not provide these.
