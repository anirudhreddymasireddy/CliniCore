# Data, licences and how to get it

| Dataset | Used for | Licence / terms | In this repo? |
|---|---|---|---|
| **Synthetic MIMIC-IV-shaped** (`data/samples`, made by `pipeline/synth_mimic.py`) | tests, demo ICU board, shift profiles | ours (MIT) | yes |
| **NHAMCS ED 2018–2022** (CDC/NCHS) | Stage 1 training and test | US public domain (cite NCHS) | derived files only: `nhamcs_cache/`, `stage1_predictions.csv`, trained model |
| **PhysioNet/CinC Challenge 2019** | Stage 3 sepsis benchmark, real-patient ICU board | CC BY 4.0, citation required (below) | no (downloaded by `sepsis2019_load.py`) |
| **MIMIC-IV / MIMIC-IV-ED** | optional: train on your own credentialed copy | PhysioNet Credentialed Health Data License: **never commit, upload, share or publish** | never |

## Getting the data
- **NHAMCS**: download the ED public-use files for 2018–2022 in Stata format (`ed20XX-stata.zip`) from the CDC NHAMCS datasets page and unzip them into one folder. Then run `python pipeline/nhamcs.py --dir <folder> --cache nhamcs_cache`.
- **PhysioNet 2019**: `python pipeline/sepsis2019_load.py` downloads the Kaggle mirror (`farjanayesmin/the-physionet-challenge-2019-dataset`, 40,336 `.psv` files). You can also pass `--path` to the official `training_setA`/`training_setB` folders from physionet.org.
- **MIMIC-IV**: request credentialed access on physionet.org, then see [bring-your-own-data.md](bring-your-own-data.md).

## How labels are made
- **Stage 1**: the first ICD-10-CM diagnosis of the visit is mapped to one of 17 groups (`nhamcs.dx_group`). Administrative Z-codes and blank diagnoses are dropped. Adults only (18+).
- **Stage 3 sepsis**: the challenge's own Sepsis-3 label (`SepsisLabel`, which starts 6 h before onset). Hours after onset are removed. **Other risks**: whether the rule tier reaches High within the horizon.

## What the PhysioNet data lacks
Consciousness level (GCS), vasopressors, urine output, ventilation flag and PaO2 are not recorded. On real PhysioNet patients:
- kidney injury uses creatinine only;
- shock uses blood pressure, heart rate and lactate only;
- respiratory failure uses SpO2/FiO2 and breathing rate.

The ICU board shows this list next to the data label.

## Citations
- Reyna MA, Josef C, Jeter R, et al. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* PhysioNet (version 1.0.0), 2019. https://doi.org/10.13026/v64v-d857. Also: Reyna MA et al., *Critical Care Medicine* 48(2):210-217, 2020. https://doi.org/10.1097/CCM.0000000000004145
- National Center for Health Statistics. *National Hospital Ambulatory Medical Care Survey, Emergency Department public-use files 2018–2022.*
- If you use MIMIC-IV: Johnson A, Bulgarelli L, Pollard T, et al. *MIMIC-IV.* PhysioNet. (cite the version you use)
