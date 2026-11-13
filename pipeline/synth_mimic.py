#!/usr/bin/env python3
"""
synth_mimic.py - synthetic data in the MIMIC-IV schema (ED + hosp + icu modules) for CareGuard.

What it is for
  * Stage 1 (ED intake): triage, vitalsign, edstays, diagnosis (label), medrecon, pyxis, patients history.
  * Stage 3 (ICU monitoring): icustays, chartevents (real itemids), labevents, inputevents (pressors, antibiotics),
    outputevents (urine), prescriptions, microbiologyevents, transfers, admissions.
  * A side-table folder synth_truth/ with ground truth + edge-case tags. The truth tables are NEVER read by the
    scoring/feature code; they exist so you can measure whether the pipeline finds what was planted.

What it is NOT
  * Not real patients and not a source of headline accuracy numbers: a model trained on this data learns this
    generator. Use it to test pipelines, edge cases, shift detection and the doctor-in-the-loop retraining.

Schema parity: every table is reindexed to the column list of the real MIMIC-IV v3.1 demo
(data/schema/mimic_demo_schema.json), so code written against the real tables runs unchanged.
IDs are in the 90,000,000+ range and years in 2150-2195 so synthetic rows cannot be mistaken for real ones.

Profiles (distribution shift, for the "model fails -> doctor fixes it" demo)
  default        baseline hospital A
  hospital_b     different vitals offsets, charting every 2h, different wording, other lab calibration,
                 more atypical sepsis (no fever, no tachycardia on beta-blocker, hypotension first)
  hospital_c     no arterial lines (NIBP only), lactate rarely ordered, GCS every 12h: the model's strongest
                 features (lactate, MAP) vanish -> Stage 3 must degrade and the doctor loop must repair it
  sparse         vitals every 4h with 25% missing blocks, fewer labs (rural/low-resource ward)
  surge          respiratory surge incl. a NOVEL ED category never seen in default (label shift / unseen class)

Usage
  python synth_mimic.py --out out/synth_a --n-ed 3000 --n-icu-direct 300 --seed 1 --profile default
"""
from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "..", "data", "schema", "mimic_demo_schema.json")))
FMT = "%Y-%m-%d %H:%M:%S"
H = np.timedelta64(1, "h")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def ramp(t, t0, width, peak):
    return peak * sig(4.0 * (t - t0) / max(width, 1e-3))


# ----------------------------------------------------------------------------- profiles
@dataclass
class Profile:
    name: str = "default"
    vital_interval_h: float = 1.0
    vital_miss_p: float = 0.04
    gap_blocks_p: float = 0.0          # prob a stay has a 6-12h charting blackout
    hr_off: float = 0.0
    sbp_off: float = 0.0
    spo2_off: float = 0.0
    temp_off: float = 0.0
    cr_off: float = 0.0
    lab_interval_scale: float = 1.0
    gcs_interval_h: float = 4.0
    atypical_sepsis_w: float = 0.02
    lactate_never_p: float = 0.06      # share of ICU stays that never get a lactate
    art_line_p: float = 1.0            # 0 = no arterial lines at all (NIBP only)
    group_w: dict = field(default_factory=dict)      # ED group weight overrides
    wording: str = "A"
    novel_group: bool = False
    abx_names: tuple = ("Vancomycin", "Piperacillin-Tazobactam", "Cefepime", "CeftriaXONE", "Meropenem", "Metronidazole")


PROFILES = {
    "default": Profile(),
    "hospital_b": Profile(name="hospital_b", vital_interval_h=2.0, vital_miss_p=0.08, hr_off=8, sbp_off=-10,
                          spo2_off=-1.5, temp_off=-0.3, cr_off=0.2, gcs_interval_h=8.0, atypical_sepsis_w=0.12,
                          wording="B", group_w={"dyspnea_cardioresp": 0.16, "pneumonia": 0.10, "sepsis_infection": 0.14},
                          abx_names=("Cefepime", "Ampicillin-Sulbactam", "Levofloxacin", "Azithromycin", "Meropenem", "Metronidazole")),
    "hospital_c": Profile(name="hospital_c", vital_interval_h=2.0, vital_miss_p=0.10, hr_off=6, sbp_off=-6, temp_off=-0.2, gcs_interval_h=12.0,
                          lactate_never_p=0.85, art_line_p=0.0, atypical_sepsis_w=0.10, wording="B", lab_interval_scale=1.3),
    "sparse": Profile(name="sparse", vital_interval_h=4.0, vital_miss_p=0.15, gap_blocks_p=0.35, lab_interval_scale=2.0,
                      gcs_interval_h=12.0, wording="B"),
    "surge": Profile(name="surge", wording="B", novel_group=True,
                     group_w={"pneumonia": 0.14, "dyspnea_cardioresp": 0.16, "novel_resp_cluster": 0.12, "sepsis_infection": 0.12}),
}

# ----------------------------------------------------------------------------- stage 1: ED groups
# vit: (mean, sd) for temp F, hr, rr, spo2, sbp, dbp ; esi = probabilities for acuity 1..5
GROUPS = {
    "chest_pain": dict(w=.12, age=(60, 15), cc=["Chest pain", "Chest pain, Dyspnea", "Chest tightness", "Substernal chest pain"],
                       ccB=["CP", "chest pressure", "chest discomfort"],
                       dx10=[("I214", "Non-ST elevation (NSTEMI) myocardial infarction"), ("R079", "Chest pain, unspecified"), ("I2510", "Atherosclerotic heart disease of native coronary artery without angina pectoris")],
                       dx9=[("41071", "Subendocardial infarction, initial episode of care"), ("78650", "Chest pain, unspecified")],
                       vit=dict(temp=(98.2, .6), hr=(84, 16), rr=(18, 2.5), spo2=(97.5, 1.8), sbp=(145, 24), dbp=(82, 14)),
                       esi=[.04, .55, .38, .03, 0], admit=.55, icu=.06, pyx=["Aspirin", "Nitroglycerin", "Morphine Sulfate"]),
    "dyspnea_cardioresp": dict(w=.10, age=(68, 14), cc=["Dyspnea", "Shortness of breath", "Cough, Dyspnea", "Wheezing", "Leg swelling, Dyspnea"],
                               ccB=["SOB", "short of breath", "resp distress"],
                               dx10=[("I5033", "Acute on chronic diastolic (congestive) heart failure"), ("J441", "Chronic obstructive pulmonary disease with (acute) exacerbation")],
                               dx9=[("42823", "Acute on chronic systolic heart failure"), ("49121", "Obstructive chronic bronchitis with (acute) exacerbation")],
                               vit=dict(temp=(98.0, .7), hr=(98, 18), rr=(24, 5), spo2=(92, 4), sbp=(140, 28), dbp=(80, 16)),
                               esi=[.06, .52, .38, .04, 0], admit=.75, icu=.12, pyx=["Furosemide", "Albuterol Sulfate", "Ipratropium"]),
    "pneumonia": dict(w=.06, age=(66, 17), cc=["Cough, Fever", "Fever, Dyspnea", "Cough", "Weakness, Cough"],
                      ccB=["productive cough", "fever and cough"],
                      dx10=[("J189", "Pneumonia, unspecified organism"), ("J159", "Unspecified bacterial pneumonia")],
                      dx9=[("486", "Pneumonia, organism unspecified")],
                      vit=dict(temp=(100.4, 1.4), hr=(104, 18), rr=(24, 5), spo2=(92, 4), sbp=(128, 24), dbp=(72, 14)),
                      esi=[.04, .46, .46, .04, 0], admit=.8, icu=.14, pyx=["CeftriaXONE", "Azithromycin", "Acetaminophen"]),
    "sepsis_infection": dict(w=.10, age=(67, 17), cc=["Fever", "Altered mental status", "Weakness, Fever", "Dysuria", "Abd pain, Fever", "Hypotension"],
                             ccB=["febrile", "confused and febrile", "feels unwell"],
                             dx10=[("A419", "Sepsis, unspecified organism"), ("N390", "Urinary tract infection, site not specified"), ("L039", "Cellulitis, unspecified")],
                             dx9=[("03819", "Other staphylococcal septicemia"), ("5990", "Urinary tract infection, site not specified")],
                             vit=dict(temp=(100.8, 1.8), hr=(110, 20), rr=(22, 5), spo2=(95, 3.5), sbp=(112, 28), dbp=(62, 16)),
                             esi=[.12, .48, .36, .04, 0], admit=.85, icu=.2, pyx=["Vancomycin", "Piperacillin-Tazobactam", "Acetaminophen"]),
    "abd_gi": dict(w=.14, age=(52, 20), cc=["Abd pain", "Nausea/vomiting", "BRBPR", "Melena", "Abd pain, Nausea"],
                   ccB=["stomach pain", "belly pain", "vomiting blood"],
                   dx10=[("K3580", "Unspecified acute appendicitis"), ("K922", "Gastrointestinal hemorrhage, unspecified"), ("R1084", "Generalized abdominal pain")],
                   dx9=[("5409", "Acute appendicitis without mention of peritonitis"), ("5789", "Hemorrhage of gastrointestinal tract, unspecified")],
                   vit=dict(temp=(98.5, .8), hr=(92, 17), rr=(18, 3), spo2=(98, 1.6), sbp=(130, 24), dbp=(76, 14)),
                   esi=[.03, .3, .55, .12, 0], admit=.5, icu=.05, pyx=["Ondansetron", "Morphine Sulfate", "Famotidine"]),
    "stroke_neuro": dict(w=.07, age=(70, 15), cc=["Altered mental status", "Facial droop", "Slurred speech", "Seizure", "Headache", "Dizzy"],
                         ccB=["AMS", "stroke symptoms", "confused"],
                         dx10=[("I639", "Cerebral infarction, unspecified"), ("R4182", "Altered mental status, unspecified"), ("G40909", "Epilepsy, unspecified, not intractable, without status epilepticus")],
                         dx9=[("43491", "Cerebral artery occlusion, unspecified with cerebral infarction"), ("78097", "Altered mental status")],
                         vit=dict(temp=(98.2, .7), hr=(84, 16), rr=(17, 3), spo2=(97, 2.2), sbp=(158, 30), dbp=(88, 17)),
                         esi=[.1, .5, .36, .04, 0], admit=.7, icu=.1, pyx=["Lorazepam", "Labetalol"]),
    "trauma": dict(w=.12, age=(45, 22), cc=["Fall", "MVC", "Head injury", "Laceration", "Hip pain", "Wrist pain"],
                   ccB=["fell down", "car accident", "hit head"],
                   dx10=[("S0990XA", "Unspecified injury of head, initial encounter"), ("S72001A", "Fracture of unspecified part of neck of right femur, initial encounter"), ("S61409A", "Unspecified open wound of hand")],
                   dx9=[("8500", "Concussion with no loss of consciousness"), ("82009", "Other transcervical fracture of femur, closed")],
                   vit=dict(temp=(98.2, .6), hr=(88, 17), rr=(18, 2.8), spo2=(98, 1.5), sbp=(138, 25), dbp=(80, 15)),
                   esi=[.04, .22, .5, .22, .02], admit=.35, icu=.05, pyx=["Morphine Sulfate", "Acetaminophen", "Tetanus-Diphth-Acell Pertussis"]),
    "renal_metabolic": dict(w=.06, age=(58, 17), cc=["Weakness", "Hyperglycemia", "Vomiting", "Abnormal lab", "Lethargy"],
                            ccB=["high sugar", "feels weak", "dehydrated"],
                            dx10=[("N179", "Acute kidney failure, unspecified"), ("E1010", "Type 1 diabetes mellitus with ketoacidosis without coma"), ("E871", "Hypo-osmolality and hyponatremia")],
                            dx9=[("5849", "Acute kidney failure, unspecified"), ("25011", "Diabetes with ketoacidosis, type I")],
                            vit=dict(temp=(98.2, .8), hr=(100, 19), rr=(21, 5), spo2=(97, 2.4), sbp=(124, 26), dbp=(72, 15)),
                            esi=[.06, .42, .46, .06, 0], admit=.8, icu=.12, pyx=["Insulin", "Sodium Chloride 0.9% Bolus", "Ondansetron"]),
    "psych_substance": dict(w=.06, age=(38, 14), cc=["Suicidal ideation", "Overdose", "Alcohol intoxication", "Anxiety", "Agitation"],
                            ccB=["intoxicated", "ETOH", "wants to hurt self"],
                            dx10=[("F10129", "Alcohol abuse with intoxication, unspecified"), ("T391X1A", "Poisoning by 4-Aminophenol derivatives, accidental"), ("R45851", "Suicidal ideations")],
                            dx9=[("30500", "Alcohol abuse, unspecified"), ("9650", "Poisoning by opium")],
                            vit=dict(temp=(98.1, .6), hr=(94, 18), rr=(18, 3), spo2=(97.5, 2), sbp=(130, 20), dbp=(80, 13)),
                            esi=[.05, .26, .55, .14, 0], admit=.3, icu=.06, pyx=["Lorazepam", "Haloperidol", "Naloxone"]),
    "minor_msk": dict(w=.10, age=(42, 17), cc=["Back pain", "Dental pain", "Rash", "Sore throat", "Ankle pain", "Headache"],
                      ccB=["back ache", "tooth pain", "cut finger"],
                      dx10=[("M545", "Low back pain"), ("R519", "Headache, unspecified"), ("J029", "Acute pharyngitis, unspecified")],
                      dx9=[("7242", "Lumbago"), ("4619", "Acute sinusitis, unspecified")],
                      vit=dict(temp=(98.2, .5), hr=(80, 13), rr=(17, 2), spo2=(98.5, 1.2), sbp=(134, 20), dbp=(80, 12)),
                      esi=[0, .02, .28, .58, .12], admit=.05, icu=.0, pyx=["Ibuprofen", "Acetaminophen"]),
    "pe_af_other": dict(w=.07, age=(65, 16), cc=["Palpitations", "Dyspnea, Chest pain", "Syncope", "Leg pain", "Tachycardia"],
                        ccB=["racing heart", "passed out", "swollen leg"],
                        dx10=[("I2699", "Other pulmonary embolism without acute cor pulmonale"), ("I4891", "Unspecified atrial fibrillation"), ("R55", "Syncope and collapse")],
                        dx9=[("41519", "Other pulmonary embolism and infarction"), ("42731", "Atrial fibrillation")],
                        vit=dict(temp=(98.1, .6), hr=(108, 24), rr=(20, 4), spo2=(95, 3.5), sbp=(132, 26), dbp=(78, 15)),
                        esi=[.06, .5, .4, .04, 0], admit=.7, icu=.1, pyx=["Heparin", "Metoprolol Tartrate", "Diltiazem"]),
    # unseen class used only by the "surge" profile (label shift): the stage-1 model must route it to a doctor
    "novel_resp_cluster": dict(w=0.0, age=(52, 18), cc=["Fever, Dyspnea", "Cough, Anosmia", "Dyspnea, Myalgias", "Hypoxia"],
                               ccB=["can't taste, SOB", "flu-like, low O2"],
                               dx10=[("U071", "COVID-like illness, novel cluster (synthetic)")],
                               dx9=[("79999", "Other ill-defined conditions (synthetic novel cluster)")],
                               vit=dict(temp=(101.4, 1.2), hr=(106, 16), rr=(26, 5), spo2=(89, 4.5), sbp=(126, 20), dbp=(72, 12)),
                               esi=[.1, .5, .36, .04, 0], admit=.85, icu=.22, pyx=["Dexamethasone", "Acetaminophen"]),
}

ICU_FROM_GROUP = {
    "sepsis_infection": {"sepsis_floor_to_icu": .5, "septic_shock": .3, "infection_no_sepsis": .2},
    "pneumonia": {"resp_failure": .6, "sepsis_floor_to_icu": .3, "septic_shock": .1},
    "dyspnea_cardioresp": {"resp_failure": .6, "cardiogenic_cvicu": .4},
    "chest_pain": {"cardiogenic_cvicu": 1.0},
    "abd_gi": {"hemorrhagic": .5, "sirs_no_infection": .3, "sepsis_floor_to_icu": .2},
    "stroke_neuro": {"neuro_icu": 1.0},
    "trauma": {"hemorrhagic": .4, "neuro_icu": .3, "sirs_no_infection": .3},
    "renal_metabolic": {"dka_metabolic": .5, "aki_nephrotoxic": .5},
    "psych_substance": {"neuro_icu": .6, "resp_failure": .4},
    "minor_msk": {"stable_postop": 1.0},
    "pe_af_other": {"cardiogenic_cvicu": .5, "resp_failure": .5},
    "novel_resp_cluster": {"resp_failure": .7, "sepsis_floor_to_icu": .3},
}
# archetypes that start inside the ICU (direct admissions, no ED row)
DIRECT_ARCH = {"stable_postop": .24, "cardiogenic_cvicu": .12, "sepsis_in_icu": .16, "atypical_sepsis": .04, "liver_failure": .05,
               "neuro_icu": .08, "sirs_no_infection": .06, "infection_no_sepsis": .06, "aki_nephrotoxic": .05, "resp_failure": .05,
               "early_death_short": .04, "very_long_stay": .05, "septic_shock": .05, "hemorrhagic": .04, "dka_metabolic": .03,
               "sepsis_floor_to_icu": .04}
CAREUNIT = {"stable_postop": "Surgical Intensive Care Unit (SICU)", "cardiogenic_cvicu": "Cardiac Vascular Intensive Care Unit (CVICU)",
            "neuro_icu": "Neuro Surgical Intensive Care Unit (Neuro SICU)", "hemorrhagic": "Trauma SICU (TSICU)"}
GENERIC_UNITS = ["Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)", "Surgical Intensive Care Unit (SICU)",
                 "Coronary Care Unit (CCU)"]
COMORB = {"hypertension": (("I10", "Essential (primary) hypertension"), ("4019", "Unspecified essential hypertension"), ["lisinopril", "amlodipine", "metoprolol"], .38),
          "diabetes": (("E119", "Type 2 diabetes mellitus without complications"), ("25000", "Diabetes mellitus without mention of complication"), ["metformin", "insulin glargine"], .22),
          "chf": (("I509", "Heart failure, unspecified"), ("4280", "Congestive heart failure, unspecified"), ["furosemide", "carvedilol"], .1),
          "copd": (("J449", "Chronic obstructive pulmonary disease, unspecified"), ("4912", "Obstructive chronic bronchitis"), ["albuterol", "tiotropium"], .1),
          "ckd": (("N189", "Chronic kidney disease, unspecified"), ("5859", "Chronic kidney disease, unspecified"), ["calcitriol"], .09),
          "afib": (("I4891", "Unspecified atrial fibrillation"), ("42731", "Atrial fibrillation"), ["apixaban", "metoprolol"], .08)}

LAB_DEF = {  # itemid: (name, uom, lo, hi, category, fluid)
    50912: ("Creatinine", "mg/dL", 0.5, 1.2, "Chemistry", "Blood"), 50885: ("Bilirubin, Total", "mg/dL", 0.0, 1.5, "Chemistry", "Blood"),
    51265: ("Platelet Count", "K/uL", 150, 400, "Hematology", "Blood"), 51301: ("White Blood Cells", "K/uL", 4, 11, "Hematology", "Blood"),
    50813: ("Lactate", "mmol/L", 0.5, 2.0, "Blood Gas", "Blood"), 50821: ("pO2", "mm Hg", 80, 180, "Blood Gas", "Blood"),
    50818: ("pCO2", "mm Hg", 35, 45, "Blood Gas", "Blood"), 50820: ("pH", "units", 7.35, 7.45, "Blood Gas", "Blood"),
    50882: ("Bicarbonate", "mEq/L", 22, 32, "Chemistry", "Blood"), 51222: ("Hemoglobin", "g/dL", 12, 17, "Hematology", "Blood"),
    51237: ("INR(PT)", " ", 0.9, 1.1, "Hematology", "Blood"), 50931: ("Glucose", "mg/dL", 70, 100, "Chemistry", "Blood"),
    50983: ("Sodium", "mEq/L", 135, 145, "Chemistry", "Blood"), 50971: ("Potassium", "mEq/L", 3.5, 5.3, "Chemistry", "Blood"),
    51006: ("Urea Nitrogen", "mg/dL", 6, 20, "Chemistry", "Blood"), 50861: ("Alanine Aminotransferase (ALT)", "IU/L", 0, 40, "Chemistry", "Blood"),
    50878: ("Asparate Aminotransferase (AST)", "IU/L", 0, 40, "Chemistry", "Blood"),
}
ITEMS = {  # itemid: (label, abbrev, linksto, category, unit, ptype)
    220045: ("Heart Rate", "HR", "chartevents", "Routine Vital Signs", "bpm", "Numeric"),
    220210: ("Respiratory Rate", "RR", "chartevents", "Respiratory", "insp/min", "Numeric"),
    220277: ("O2 saturation pulseoxymetry", "SpO2", "chartevents", "Respiratory", "%", "Numeric"),
    220179: ("Non Invasive Blood Pressure systolic", "NBPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    220180: ("Non Invasive Blood Pressure diastolic", "NBPd", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    220181: ("Non Invasive Blood Pressure mean", "NBPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    220050: ("Arterial Blood Pressure systolic", "ABPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    220051: ("Arterial Blood Pressure diastolic", "ABPd", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    220052: ("Arterial Blood Pressure mean", "ABPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric"),
    223761: ("Temperature Fahrenheit", "Temperature F", "chartevents", "Routine Vital Signs", "°F", "Numeric"),
    223762: ("Temperature Celsius", "Temperature C", "chartevents", "Routine Vital Signs", "°C", "Numeric"),
    220739: ("GCS - Eye Opening", "Eye Opening", "chartevents", "Neurological", np.nan, "Text"),
    223900: ("GCS - Verbal Response", "Verbal Response", "chartevents", "Neurological", np.nan, "Text"),
    223901: ("GCS - Motor Response", "Motor Response", "chartevents", "Neurological", np.nan, "Text"),
    223835: ("Inspired O2 Fraction", "FiO2", "chartevents", "Respiratory", np.nan, "Numeric"),
    226732: ("O2 Delivery Device(s)", "O2 Delivery Device(s)", "chartevents", "Respiratory", np.nan, "Text"),
    226559: ("Foley", "Foley", "outputevents", "Output", "mL", "Numeric"),
    221906: ("Norepinephrine", "Norepinephrine", "inputevents", "Medications", "mg", "Solution"),
    221289: ("Epinephrine", "Epinephrine", "inputevents", "Medications", "mg", "Solution"),
    221662: ("Dopamine", "Dopamine", "inputevents", "Medications", "mg", "Solution"),
    221653: ("Dobutamine", "Dobutamine", "inputevents", "Medications", "mg", "Solution"),
    222315: ("Vasopressin", "Vasopressin", "inputevents", "Medications", "units", "Solution"),
    221749: ("Phenylephrine", "Phenylephrine", "inputevents", "Medications", "mg", "Solution"),
    225158: ("NaCl 0.9%", "NaCl 0.9%", "inputevents", "Fluids - Intake", "mL", "Solution"),
    225798: ("Vancomycin", "Vancomycin", "inputevents", "Antibiotics", "mg", "Solution"),
    225893: ("Piperacillin/Tazobactam (Zosyn)", "Zosyn", "inputevents", "Antibiotics", "mg", "Solution"),
    225851: ("Cefepime", "Cefepime", "inputevents", "Antibiotics", "mg", "Solution"),
    225883: ("Meropenem", "Meropenem", "inputevents", "Antibiotics", "mg", "Solution"),
    225884: ("Metronidazole", "Metronidazole", "inputevents", "Antibiotics", "mg", "Solution"),
}
ABX_ITEM = {"Vancomycin": 225798, "Piperacillin-Tazobactam": 225893, "Cefepime": 225851, "Meropenem": 225883, "Metronidazole": 225884}
GCS_EYE = {4: "4 Spontaneously", 3: "3 To Speech", 2: "2 To Pain", 1: "1 None"}
GCS_VERB = {5: "5 Oriented", 4: "4 Confused", 3: "3 Inappropriate Words", 2: "2 Incomprehensible sounds", 1: "1 No Response-ETT"}
GCS_MOT = {6: "6 Obeys Commands", 5: "5 Localizes Pain", 4: "4 Flex-withdraws", 3: "3 Abnormal flexion", 2: "2 Abnormal extension", 1: "1 No response"}
ORGS = ["BLOOD CULTURE", "URINE", "SPUTUM"]
BUGS = ["ESCHERICHIA COLI", "STAPH AUREUS COAG +", "KLEBSIELLA PNEUMONIAE", "PSEUDOMONAS AERUGINOSA", "ENTEROCOCCUS SP.", "STREPTOCOCCUS PNEUMONIAE"]

# ----------------------------------------------------------------------------- ICU latent trajectories
ZERO = ("infl", "shock", "resp", "renal", "liver", "coag", "neuro", "bleed", "metab")


def make_latent(arch: str, L: int, rng, prof: Profile):
    """Return (latent dict of arrays over hours 0..L-1, meta dict). meta has infection_h (hours rel. ICU in-time,
    may be negative = infection before ICU), truth_onset_h, death flag, tags."""
    t = np.arange(L, dtype=float)
    z = {k: np.zeros(L) for k in ZERO}
    meta = dict(infection_h=None, truth_onset_h=None, death=False, tags=[], pressors=False, vent_postop_h=0)
    u = lambda a, b: float(rng.uniform(a, b))
    if arch == "stable_postop":
        z["infl"] += ramp(t, 0, 4, .12) * (1 - sig((t - 10) / 3)); meta["vent_postop_h"] = u(2, 10)
    elif arch == "cardiogenic_cvicu":
        s = u(.2, .65); z["shock"] = s * (1 - sig((t - u(10, 40)) / 6)); z["resp"] = .2 * z["shock"]
        z["renal"] = ramp(t, u(10, 30), 18, u(0, .4)) * (s > .45); meta["vent_postop_h"] = u(3, 14)
    elif arch in ("sepsis_in_icu", "septic_shock", "atypical_sepsis"):
        t_inf = u(8, min(max(L - 10, 12), 90)); org = t_inf + u(-1, 10)
        peak = {"sepsis_in_icu": u(.35, .8), "septic_shock": u(.85, 1.2), "atypical_sepsis": u(.4, .9)}[arch]
        z["infl"] = ramp(t, t_inf - 2, 6, 1.0 if arch != "atypical_sepsis" else .25)
        z["shock"] = ramp(t, org, 6, peak * (1.0 if arch != "septic_shock" else 1.1))
        z["resp"] = ramp(t, org + u(0, 12), 10, peak * u(.2, .8)); z["renal"] = ramp(t, org + u(2, 18), 14, peak * u(.2, .9))
        z["coag"] = ramp(t, org + u(0, 20), 12, peak * u(.1, .7)); z["liver"] = ramp(t, org + u(6, 30), 16, peak * u(0, .5))
        z["neuro"] = ramp(t, org + u(0, 12), 10, peak * u(0, .5))
        meta.update(infection_h=t_inf, truth_onset_h=org + 1.5)  # cross-over of SOFA>=2 is a little after the ramp midpoint
        if arch == "atypical_sepsis":
            meta["tags"].append("atypical_sepsis_afebrile_no_tachycardia")
        meta["death"] = bool(rng.random() < (.45 if arch == "septic_shock" else .15))
    elif arch == "sepsis_floor_to_icu":   # sepsis already present on arrival
        z["infl"] = np.full(L, 1.0) * (1 - ramp(t, u(60, 140), 40, .7)); z["shock"] = ramp(t, -4, 8, u(.4, 1.0)) * (1 - ramp(t, u(24, 90), 20, .8))
        z["resp"] = ramp(t, -2, 8, u(0, .6)); z["renal"] = ramp(t, 0, 10, u(.1, .8)); z["coag"] = ramp(t, 0, 10, u(.1, .6))
        meta.update(infection_h=-u(2, 20), truth_onset_h=0.0, tags=["sepsis_present_on_admission"]); meta["death"] = bool(rng.random() < .2)
    elif arch == "resp_failure":
        t0 = u(-2, 30); z["resp"] = ramp(t, t0, 12, u(.5, 1.1)) * (1 - ramp(t, t0 + u(40, 120), 30, .9)); z["infl"] = .4 * z["resp"]
        if rng.random() < .4:        # pneumonia with respiratory failure = Sepsis-3 sepsis (infection + SOFA>=2)
            meta["infection_h"] = t0 - u(0, 6); meta["truth_onset_h"] = max(0.0, t0 + 2.0)
    elif arch == "aki_nephrotoxic":
        z["renal"] = ramp(t, u(6, 40), 20, u(.5, 1.1)) * (1 - ramp(t, u(60, 140), 30, .7)); z["metab"] = .35 * z["renal"]
    elif arch == "hemorrhagic":
        t0 = u(-1, 18); z["bleed"] = ramp(t, t0, 5, u(.5, 1.1)) * (1 - ramp(t, t0 + u(10, 24), 8, .95)); z["shock"] = .8 * z["bleed"]
        z["coag"] = .4 * z["bleed"]; z["renal"] = ramp(t, t0 + 10, 16, .3) * (z["bleed"].max() > .8)
    elif arch == "dka_metabolic":
        z["metab"] = np.clip(u(.6, 1.0) * (1 - ramp(t, u(10, 24), 8, 1.0)), 0, 1); z["renal"] = .3 * z["metab"]
    elif arch == "liver_failure":
        z["liver"] = ramp(t, u(0, 30), 30, u(.5, 1.1)); z["coag"] = .8 * z["liver"]; z["neuro"] = ramp(t, u(10, 40), 20, .6 * z["liver"].max())
        z["renal"] = ramp(t, u(20, 60), 24, .5 * z["liver"].max())
    elif arch == "neuro_icu":
        z["neuro"] = np.full(L, u(.4, 1.0)) * (1 - ramp(t, u(40, 160), 40, .6))
    elif arch == "sirs_no_infection":      # hard negative: pancreatitis / polytrauma
        z["infl"] = np.full(L, u(.6, 1.0)) * (1 - ramp(t, u(30, 70), 20, .8)); z["shock"] = ramp(t, u(2, 20), 10, u(0, .25)); meta["tags"].append("sirs_without_infection")
    elif arch == "infection_no_sepsis":    # hard negative: infection treated, no organ dysfunction
        z["infl"] = ramp(t, 2, 6, u(.3, .7)) * (1 - ramp(t, u(40, 80), 20, .8)); meta.update(infection_h=u(-6, 6), tags=["infection_without_organ_dysfunction"])
    elif arch == "early_death_short":
        z["shock"] = ramp(t, 0, 3, 1.2); z["resp"] = ramp(t, 0, 3, 1.0); z["neuro"] = ramp(t, 0, 5, .8); meta["death"] = True
        meta["tags"].append("short_stay_death_lt12h")
    elif arch == "very_long_stay":
        for _ in range(int(rng.integers(2, 4))):
            t0 = u(20, max(L - 30, 40)); w = u(10, 30); a = u(.3, .9)
            z["resp"] += ramp(t, t0, 8, a) * (1 - ramp(t, t0 + w, 8, 1)); z["infl"] += .5 * ramp(t, t0 - 3, 6, a) * (1 - ramp(t, t0 + w, 8, 1))
            z["shock"] += .6 * ramp(t, t0, 8, a) * (1 - ramp(t, t0 + w, 8, 1))
        for k in z:
            z[k] = np.clip(z[k], 0, 1.3)
        z["renal"] = ramp(t, L * .5, 50, .4); meta.update(tags=["very_long_stay_gt20d"])
        meta["infection_h"] = u(L * .3, L * .5)    # infection planted, sepsis truth deliberately not asserted (multi-episode)
    else:
        raise ValueError(arch)
    if meta["death"]:      # deterioration in the last hours
        for k, a in (("shock", 1.1), ("resp", 1.0), ("neuro", .7)):
            z[k] = np.maximum(z[k], ramp(t, L - 8, 6, a))
    for k in z:
        z[k] = np.clip(z[k], 0, 1.4)
    return z, meta


# ----------------------------------------------------------------------------- the generator
class Gen:
    def __init__(self, seed=0, profile="default", n_ed=3000, n_icu_direct=300, max_icu_los_h=None):
        self.rng = np.random.default_rng(seed)
        self.prof = PROFILES[profile] if isinstance(profile, str) else profile
        self.n_ed, self.n_icu_direct = n_ed, n_icu_direct
        self.next_subject, self.next_hadm, self.next_stay = 90_000_000, 90_000_000, 90_000_000
        self.next_lab, self.next_spec, self.next_micro, self.next_order = 9_000_000, 90_000_000, 9_000_000, 90_000_000
        self.T = {k: [] for k in ["patients", "admissions", "transfers", "edstays", "triage", "vitalsign", "diagnosis", "medrecon", "pyxis",
                                  "diagnoses_icd", "prescriptions", "microbiologyevents", "labevents", "icustays", "chartevents",
                                  "inputevents", "outputevents"]}
        self.truth_ed, self.truth_icu, self.truth_lat = [], [], []
        self.subjects = {}      # subject_id -> dict(gender, age, anchor_year, comorb list)

    # --- helpers
    def _id(self, name):
        v = getattr(self, name); setattr(self, name, v + 1); return v

    def rand_time(self):
        y = int(self.rng.integers(2150, 2195)); doy = int(self.rng.integers(0, 365)); sec = int(self.rng.integers(0, 86400))
        return np.datetime64(f"{y}-01-01T00:00:00") + np.timedelta64(doy, "D") + np.timedelta64(sec, "s")

    def new_subject(self, age_mu, age_sd, comorb_boost=1.0):
        sid = self._id("next_subject")
        age = int(np.clip(self.rng.normal(age_mu, age_sd), 18, 91))
        if self.rng.random() < .01:
            age = 18          # age-boundary edge case
        if self.rng.random() < .02:
            age = 91          # MIMIC caps shifted ages at 91
        ay = int(self.rng.integers(2150, 2190))
        cm = [k for k, v in COMORB.items() if self.rng.random() < min(.95, v[3] * comorb_boost * (age / 60))]
        self.subjects[sid] = dict(gender=self.rng.choice(["F", "M"]), age=age, ay=ay, comorb=cm, dod=None)
        return sid

    # --- Stage 1: ED stay
    def ed_stay(self, group, subject=None, arrive=None):
        g = GROUPS[group]; rng = self.rng; P = self.prof
        sid = subject or self.new_subject(*g["age"])
        S = self.subjects[sid]
        stay = self._id("next_stay"); t0 = arrive if arrive is not None else self.rand_time()
        los_h = float(np.clip(rng.lognormal(1.6, .5), .8, 30)); t1 = t0 + np.timedelta64(int(los_h * 3600), "s")
        tags = []
        esi = int(rng.choice([1, 2, 3, 4, 5], p=np.array(g["esi"]) / sum(g["esi"])))
        # triage vitals
        v = {k: rng.normal(m, s) for k, (m, s) in g["vit"].items()}
        v["hr"] += P.hr_off; v["sbp"] += P.sbp_off; v["spo2"] += P.spo2_off; v["temp"] += P.temp_off * 1.8
        if "pneumonia" == group or group in ("sepsis_infection", "novel_resp_cluster"):
            if group == "sepsis_infection" and rng.random() < (P.atypical_sepsis_w + .04):
                v["temp"] = 97.0; v["hr"] = rng.normal(76, 8); tags.append("atypical_sepsis_presentation")
        tri = dict(temperature=round(v["temp"], 1), heartrate=round(v["hr"]), resprate=round(v["rr"]), o2sat=round(min(v["spo2"], 100)),
                   sbp=round(v["sbp"]), dbp=round(v["dbp"]), pain=str(int(rng.choice(np.arange(0, 11)))), acuity=float(esi))
        if rng.random() < .03:
            tri["temperature"] = round((tri["temperature"] - 32) * 5 / 9, 1); tags.append("temp_in_celsius_in_F_column")
        for k, p, tag in (("temperature", .06, "missing_temp"), ("heartrate", .02, "missing_hr"), ("resprate", .04, "missing_rr"),
                          ("o2sat", .04, "missing_spo2"), ("sbp", .03, "missing_bp"), ("acuity", .03, "missing_acuity")):
            if rng.random() < p:
                tri[k] = np.nan; tags.append(tag)
                if k == "sbp":
                    tri["dbp"] = np.nan
        if rng.random() < .01:
            tri["dbp"] = 879; tags.append("bp_typo_outlier")
        if rng.random() < .008:
            tri["heartrate"] = 0; tags.append("hr_zero_artifact")
        if rng.random() < .1:
            tri["pain"] = str(rng.choice(["UA", "unable", "13", "Critical"]))
        if rng.random() < .01:
            tri["o2sat"] = 10; tags.append("spo2_implausible_low")
        cc = rng.choice(g["ccB"] if (P.wording == "B" and rng.random() < .6) else g["cc"])
        if rng.random() < .03:
            cc = ""; tags.append("missing_chief_complaint")
        tri["chiefcomplaint"] = cc
        admit = rng.random() < g["admit"]
        dispo = "ADMITTED" if admit else str(rng.choice(["HOME", "HOME", "HOME", "TRANSFER", "LEFT WITHOUT BEING SEEN", "LEFT AGAINST MEDICAL ADVICE", "ELOPED"], p=[.6, .1, .1, .05, .06, .05, .04]))
        hadm = self._id("next_hadm") if admit else np.nan
        arrival = str(rng.choice(["AMBULANCE", "WALK IN", "UNKNOWN", "OTHER"], p=([.62, .33, .03, .02] if esi <= 2 else [.25, .71, .03, .01])))
        self.T["edstays"].append(dict(subject_id=sid, hadm_id=hadm, stay_id=stay, intime=t0, outtime=t1, gender=S["gender"],
                                      race=str(rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - PUERTO RICAN", "ASIAN", "OTHER", "UNKNOWN"], p=[.55, .2, .08, .06, .07, .04])),
                                      arrival_transport=arrival, disposition=dispo))
        self.T["triage"].append(dict(subject_id=sid, stay_id=stay, **tri))
        # ED vitalsign rows
        n_v = int(rng.integers(1, 6)); tv = np.sort(rng.uniform(0, los_h, n_v))
        for i, h in enumerate(tv):
            dec = 1 - .15 * (i + 1) * (g["admit"] < .5)
            self.T["vitalsign"].append(dict(subject_id=sid, stay_id=stay, charttime=t0 + np.timedelta64(int(h * 3600), "s"),
                                            temperature=round(v["temp"] + rng.normal(0, .3), 1) if rng.random() > .4 else np.nan,
                                            heartrate=round(v["hr"] * dec + rng.normal(0, 5)), resprate=round(v["rr"] + rng.normal(0, 1.5)),
                                            o2sat=round(min(100, v["spo2"] + rng.normal(0, 1))), sbp=round(v["sbp"] + rng.normal(0, 8)), dbp=round(v["dbp"] + rng.normal(0, 5)),
                                            rhythm=("Sinus Tachycardia" if v["hr"] > 100 and rng.random() < .3 else np.nan), pain=tri["pain"]))
        # diagnosis (the stage-1 label lives here) + comorbidities
        n9 = rng.random() < .45
        pool = g["dx9"] if n9 else g["dx10"]; code, title = pool[int(rng.integers(len(pool)))]
        seq = 1; self.T["diagnosis"].append(dict(subject_id=sid, stay_id=stay, seq_num=seq, icd_code=code, icd_version=9 if n9 else 10, icd_title=title))
        for cm in S["comorb"][:3]:
            seq += 1; c10, c9 = COMORB[cm][0], COMORB[cm][1]; c = c9 if n9 else c10
            self.T["diagnosis"].append(dict(subject_id=sid, stay_id=stay, seq_num=seq, icd_code=c[0], icd_version=9 if n9 else 10, icd_title=c[1]))
        # medrecon (home meds from comorbidities) + pyxis (ED meds)
        for cm in S["comorb"]:
            for drug in COMORB[cm][2][:2]:
                self.T["medrecon"].append(dict(subject_id=sid, stay_id=stay, charttime=t0 + np.timedelta64(int(rng.integers(300, 3600)), "s"), name=drug,
                                               gsn=int(rng.integers(1000, 99999)), ndc=int(rng.integers(10 ** 6, 10 ** 9)), etc_rn=1, etccode=float(rng.integers(10 ** 5, 10 ** 6)), etcdescription=f"{cm} agent"))
        for i, d in enumerate(g["pyx"][:int(rng.integers(0, len(g["pyx"]) + 1))]):
            self.T["pyxis"].append(dict(subject_id=sid, stay_id=stay, charttime=t0 + np.timedelta64(int(rng.integers(600, 7200)), "s"), med_rn=i + 1, name=d, gsn_rn=1, gsn=int(rng.integers(1000, 9999))))
        self.truth_ed.append(dict(ed_stay_id=stay, subject_id=sid, hadm_id=hadm, dx_group=group, is_novel_class=int(group == "novel_resp_cluster"),
                                  esi_true=esi, tags="|".join(tags), profile=P.name))
        return dict(subject=sid, stay=stay, hadm=hadm, t0=t0, t1=t1, admit=admit, group=group, esi=esi)

    # --- Stage 3: ICU stay (+ hosp context)
    def icu_stay(self, arch, subject, hadm, icu_in, from_ed=None, planted_tags=None):
        rng = self.rng; P = self.prof; S = self.subjects[subject]
        stay = self._id("next_stay")
        base_L = {"early_death_short": rng.uniform(4, 11), "very_long_stay": rng.uniform(480, 900), "stable_postop": rng.uniform(12, 60),
                  "dka_metabolic": rng.uniform(20, 60)}.get(arch, float(np.clip(rng.lognormal(4.1, .55), 20, 400)))
        L = int(base_L) + 1
        z, meta = make_latent(arch, L, rng, P)
        tags = list(meta["tags"]) + (planted_tags or [])
        if arch == "atypical_sepsis" or (arch == "sepsis_in_icu" and rng.random() < P.atypical_sepsis_w):
            tags.append("beta_blocker_masks_tachycardia")
        icu_out = icu_in + np.timedelta64(int(base_L * 3600), "s")
        t = np.arange(L, dtype=float); tt = icu_in + (t * 3600).astype("timedelta64[s]")
        wt = float(np.clip(rng.normal(80, 20), 42, 143))
        unit = CAREUNIT.get(arch, GENERIC_UNITS[int(rng.integers(len(GENERIC_UNITS)))])
        # ---------- physiology
        infl, sh, rs, rn, lv, cg, nu, bl, mt = (z[k] for k in ZERO)
        hr0 = rng.normal(76, 9) + P.hr_off; atyp = "atypical" in arch or "beta_blocker_masks_tachycardia" in tags
        hr = hr0 + (6 if atyp else 25) * infl + (8 if atyp else 35) * sh + 18 * rs + 32 * bl + 8 * mt + rng.normal(0, 4, L)
        rr = rng.normal(16, 1.8) + 9 * rs + 5 * infl + 4 * sh + 9 * mt + rng.normal(0, 1.4, L)
        vent = (rs > .65) | (nu > .75) | (t < meta["vent_postop_h"]) | (arch == "early_death_short")
        o2dev = np.where(vent, "Endotracheal tube", np.where(rs > .5, "High flow nasal cannula", np.where(rs > .3, "Face tent", np.where(rs > .12, "Nasal cannula", "None"))))
        if rng.random() < .1:
            o2dev = np.where((rs > .5) & ~vent, "Bipap mask ", o2dev)
        fio2 = np.where(o2dev == "None", 21, np.where(o2dev == "Nasal cannula", 28 + 20 * rs, np.where(o2dev == "Endotracheal tube", 35 + 60 * rs, 40 + 55 * rs)))
        fio2 = np.clip(np.round(fio2 / 5) * 5, 21, 100)
        spo2 = 97.5 - 14 * rs * (~vent) - 7 * rs * vent + rng.normal(0, 1, L) + P.spo2_off
        spo2 = np.where(vent, np.maximum(spo2, 90 + rng.normal(0, 1.5, L)), spo2)
        temp = 36.8 + P.temp_off + (1.9 * infl if not atyp else -0.9 * infl - 0.4 * sh) - (.7 * sh if sh.max() > .9 else 0) + rng.normal(0, .25, L)
        map_un = rng.normal(84, 6) - 30 * sh - 24 * bl - 5 * infl + rng.normal(0, 3, L)
        press = (sh > .45) | ((bl > .85) & (sh > .4))
        meta["pressors"] = bool(press.any())
        mp = np.where(press, np.maximum(map_un, 63 + rng.normal(0, 3, L)), map_un)
        sbp = mp * 1.45 + 12 + P.sbp_off * .5 + rng.normal(0, 6, L); dbp = (3 * mp - sbp) / 2
        # GCS (verbal 1 when intubated, as charted in MIMIC) -- sedation ignored on purpose (documented limitation)
        gcs_total = np.clip(np.round(15 - 9 * nu - 3 * sh * (sh > .8) + rng.normal(0, .4, L)), 3, 15)
        eye = np.clip(np.round(gcs_total / 15 * 4 + rng.normal(0, .3, L)), 1, 4)
        motor = np.clip(np.round(gcs_total / 15 * 6 + rng.normal(0, .3, L)), 1, 6)
        verb = np.clip(gcs_total - eye - motor, 1, 5); verb = np.where(vent, 1, verb)
        # ---------- chart events
        rows = []
        def emit(itemid, hours, vals, uom=None, textvals=None, warn=0):
            hours = np.asarray(hours, dtype=float); times = tm(hours); vals = np.asarray(vals, dtype=float) if textvals is None else vals
            n = len(times)
            if n == 0:
                return
            d = dict(subject_id=subject, hadm_id=hadm, stay_id=stay, caregiver_id=rng.integers(1000, 99999, n), charttime=times,
                     storetime=times + (rng.integers(-15, 25, n) * 60).astype("timedelta64[s]"), itemid=itemid,
                     value=textvals if textvals is not None else vals, valuenum=vals if textvals is None or np.issubdtype(np.asarray(vals).dtype, np.number) else np.nan,
                     valueuom=uom, warning=warn)
            rows.append(pd.DataFrame(d))
        step = P.vital_interval_h
        grid = np.arange(0, L, step) + rng.uniform(-.2, .2) * (step > 1)
        grid = grid[(grid >= 0) & (grid < L)]
        gap = None
        if rng.random() < P.gap_blocks_p:
            a = rng.uniform(4, max(L - 14, 5)); gap = (a, a + rng.uniform(6, 12)); tags.append("charting_blackout_6_12h")
        def hours_ok(ts, p_miss=None):
            keep = rng.random(len(ts)) > (P.vital_miss_p if p_miss is None else p_miss)
            if gap:
                keep &= ~((ts >= gap[0]) & (ts <= gap[1]))
            return ts[keep]
        interp = lambda arr, ts: np.interp(ts, t, arr)
        def tm(ts):
            return icu_in + (ts * 3600 + rng.uniform(-300, 300, len(ts))).astype("timedelta64[s]")
        has_art = bool(P.art_line_p > 0 and ((sh.max() > .4) or (rs.max() > .6) or rng.random() < .25))
        n_dev_missing = rng.random() < .03
        if n_dev_missing:
            tags.append("no_chartevents_at_all")
        else:
            ts = hours_ok(grid); T_ = tm(ts)
            emit(220045, ts, np.round(interp(hr, ts)).clip(0, 400), "bpm")
            ts2 = hours_ok(grid); T2 = tm(ts2)
            emit(220210, ts2, np.round(interp(rr, ts2)).clip(0, 90), "insp/min")
            ts3 = hours_ok(grid); T3 = tm(ts3)
            emit(220277, ts3, np.round(np.clip(interp(spo2, ts3), 40, 100)), "%")
            ts4 = hours_ok(grid)
            if has_art:
                a_m, a_s, a_d = interp(mp, ts4), interp(sbp, ts4), interp(dbp, ts4)
                for it, arr in ((220052, a_m), (220050, a_s), (220051, a_d)):
                    emit(it, ts4, np.round(arr), "mmHg")
            tsn = hours_ok(np.arange(0, L, 1.0 if not has_art else 2.0))
            for it, arr in ((220181, interp(mp, tsn) + rng.normal(0, 4, len(tsn))), (220179, interp(sbp, tsn) + rng.normal(0, 6, len(tsn))), (220180, interp(dbp, tsn) + rng.normal(0, 4, len(tsn)))):
                emit(it, tsn, np.round(arr), "mmHg")
            # temperature: some stays chart F, some C (real data has both); occasional C items reading in F range
            tts = hours_ok(np.arange(0, L, max(4.0, step)))
            use_c = rng.random() < .22
            tv = interp(temp, tts) + rng.normal(0, .1, len(tts))
            emit(223762 if use_c else 223761, tts, np.round(tv if use_c else tv * 9 / 5 + 32, 1), "°C" if use_c else "°F")
            if use_c and rng.random() < .3:
                tags.append("mixed_temp_units")
                f_ts = tts[:2]; emit(223761, f_ts, np.round(interp(temp, f_ts) * 9 / 5 + 32, 1), "°F")
            # GCS
            tg = hours_ok(np.arange(0, L, P.gcs_interval_h), p_miss=.1)
            for it, arr, mp_ in ((220739, eye, GCS_EYE), (223900, verb, GCS_VERB), (223901, motor, GCS_MOT)):
                vals = np.round(interp(arr, tg)).astype(int); emit(it, tg, vals.astype(float), None, textvals=np.array([mp_.get(v_, str(v_)) for v_ in vals], dtype=object))
            # FiO2 + O2 device (on change or q4h)
            tf = hours_ok(np.arange(0, L, 4.0), p_miss=.05)
            idx = np.clip(tf.astype(int), 0, L - 1)
            fio_vals = fio2[idx]
            emit(223835, tf, fio_vals.astype(float), None)
            emit(226732, tf, np.full(len(tf), np.nan), None, textvals=o2dev[idx].astype(object))
            # ---- outliers planted on top of otherwise normal charting
            ptag = lambda name: tags.append(name)
            rh = lambda: [float(rng.integers(1, max(2, L)))]
            if rng.random() < .2 and len(ts):
                emit(220045, [ts[int(rng.integers(len(ts)))]], [0.0], "bpm", warn=1); ptag("hr_zero_artifact")
            if has_art and rng.random() < .15:
                emit(220052, rh(), [801.0], "mmHg", warn=1); ptag("abp_mean_801_artifact")
            if rng.random() < .08:
                emit(220052, rh(), [-23.0], "mmHg", warn=1); ptag("negative_map")
            if use_c and rng.random() < .25:
                emit(223762, rh(), [99.0], "°C", warn=1); ptag("temp_c_99_sentinel")
            if rng.random() < .12:
                emit(220277, rh(), [29.0], "%", warn=1); ptag("spo2_29_spike")
            if rng.random() < .05:
                emit(220210, rh(), [88.0], "insp/min", warn=1); ptag("rr_88_spike")
            if rng.random() < .1:
                dup = pd.concat(rows).sample(n=min(30, sum(len(r) for r in rows)), random_state=int(rng.integers(1 << 30)))
                rows.append(dup); ptag("duplicate_chartevents_rows")
            if rng.random() < .1:
                emit(220045, [-float(rng.integers(1, 5))], [float(rng.integers(150, 200))], "bpm"); ptag("pre_icu_charttime")
        if rows:
            ce = pd.concat(rows, ignore_index=True)
            if "pre_icu_charttime" in tags or rng.random() < .02:
                pass
            self.T["chartevents"].append(ce)
        # ---------- inputevents: pressors, abx, fluids ; outputevents: urine
        ine = []
        def drip(itemid, rate_series, uom="mcg/kg/min", cat="01-Drips"):
            on = np.where(rate_series > 0)[0]
            for h in on:
                s = icu_in + np.timedelta64(int(h * 3600), "s")
                ine.append(dict(subject_id=subject, hadm_id=hadm, stay_id=stay, caregiver_id=int(rng.integers(1000, 99999)), starttime=s, endtime=s + np.timedelta64(3600 - 1, "s"),
                                storetime=s + np.timedelta64(int(rng.integers(0, 900)), "s"), itemid=itemid, amount=round(float(rate_series[h]) * wt * 60 / 1000, 4), amountuom="mg",
                                rate=round(float(rate_series[h]), 4), rateuom=uom, orderid=int(self._id("next_order")), linkorderid=int(self.next_order), ordercategoryname=cat,
                                secondaryordercategoryname=np.nan, ordercomponenttypedescription="Main order parameter", ordercategorydescription="Continuous Med", patientweight=round(wt, 1),
                                totalamount=np.nan, totalamountuom=np.nan, isopenbag=0, continueinnextdept=0, statusdescription="FinishedRunning", originalamount=0, originalrate=round(float(rate_series[h]), 4)))
        if press.any():
            if arch == "cardiogenic_cvicu":
                drip(221653, np.where(press, rng.uniform(2.5, 8, L), 0))
                if sh.max() > .55:
                    drip(221289, np.where(sh > .55, rng.uniform(.02, .08, L), 0))
            else:
                drip(221906, np.where(sh > .45, .03 + .35 * np.clip(sh, 0, 1.2) + rng.normal(0, .01, L), 0).clip(0))
                if sh.max() > .85:
                    drip(222315, np.where(sh > .8, 2.4, 0), uom="units/hour")
                if sh.max() > 1.0 and rng.random() < .5:
                    drip(221289, np.where(sh > 1.0, rng.uniform(.03, .15, L), 0))
        if rng.random() < .03 and sh.max() < .4:
            drip(221749, np.where(t < 3, 0.5, 0), uom="mcg/kg/min"); tags.append("phenylephrine_only_post_induction")
        # infection evidence
        inf_h = meta["infection_h"]
        if inf_h is not None:
            t_cul = icu_in + np.timedelta64(int(inf_h * 3600), "s")
            delay = float(rng.normal(1.5, 2.5)); t_abx = t_cul + np.timedelta64(int(delay * 3600), "s")
            if rng.random() < .15:
                tags.append("antibiotic_before_culture")
                t_abx = t_cul - np.timedelta64(int(abs(delay) * 3600) + 1800, "s")
            for k, name in enumerate(rng.choice(list(P.abx_names), size=int(rng.integers(1, 3)), replace=False)):
                ts_ = t_abx + np.timedelta64(k * 600, "s")
                self.T["prescriptions"].append(dict(subject_id=subject, hadm_id=hadm, pharmacy_id=int(self._id("next_order")), starttime=ts_, stoptime=ts_ + np.timedelta64(int(rng.integers(3, 8)) * 86400, "s"),
                                                   drug_type="MAIN", drug=name, route="IV", dose_val_rx=str(rng.choice(["1000", "2000", "500", "4500"])), dose_unit_rx="mg"))
                if name in ABX_ITEM and ts_ >= icu_in:
                    ine.append(dict(subject_id=subject, hadm_id=hadm, stay_id=stay, caregiver_id=int(rng.integers(1000, 99999)), starttime=ts_, endtime=ts_ + np.timedelta64(3600, "s"),
                                    storetime=ts_ + np.timedelta64(300, "s"), itemid=ABX_ITEM[name], amount=1.0, amountuom="dose", rate=np.nan, rateuom=np.nan, orderid=int(self._id("next_order")),
                                    linkorderid=int(self.next_order), ordercategoryname="08-Antibiotics (IV)", secondaryordercategoryname=np.nan, ordercomponenttypedescription="Main order parameter",
                                    ordercategorydescription="Drug Push", patientweight=round(wt, 1), totalamount=500, totalamountuom="ml", isopenbag=0, continueinnextdept=0, statusdescription="FinishedRunning", originalamount=1, originalrate=1))
            spec = str(rng.choice(ORGS, p=[.55, .25, .2])); grow = rng.random() < .55
            self.T["microbiologyevents"].append(dict(microevent_id=int(self._id("next_micro")), subject_id=subject, hadm_id=(hadm if rng.random() > .35 else np.nan),
                                                    micro_specimen_id=int(self._id("next_spec")), chartdate=t_cul.astype("datetime64[D]"), charttime=t_cul, spec_itemid=70012, spec_type_desc=spec, test_seq=1,
                                                    storedate=(t_cul + np.timedelta64(3, "D")).astype("datetime64[D]"), storetime=t_cul + np.timedelta64(3, "D"), test_itemid=90201,
                                                    test_name="Blood Culture, Routine" if spec == "BLOOD CULTURE" else f"{spec.title()} Culture", org_itemid=80235.0 if grow else np.nan,
                                                    org_name=str(rng.choice(BUGS)) if grow else np.nan, isolate_num=1.0 if grow else np.nan, comments=np.nan if grow else "NO GROWTH."))
            if rng.random() < .1:   # negative-cultures-but-treated edge case
                tags.append("culture_negative_but_treated")
        if sh.max() > .5 or rs.max() > .6:
            for h in np.unique(np.clip(rng.integers(0, max(2, L // 2), 2), 0, L - 1)):
                s = icu_in + np.timedelta64(int(h * 3600), "s")
                ine.append(dict(subject_id=subject, hadm_id=hadm, stay_id=stay, caregiver_id=int(rng.integers(1000, 99999)), starttime=s, endtime=s + np.timedelta64(3600, "s"), storetime=s, itemid=225158,
                                amount=500.0, amountuom="ml", rate=500.0, rateuom="mL/hour", orderid=int(self._id("next_order")), linkorderid=int(self.next_order), ordercategoryname="02-Fluids (Crystalloids)",
                                secondaryordercategoryname=np.nan, ordercomponenttypedescription="Main order parameter", ordercategorydescription="Bolus", patientweight=round(wt, 1), totalamount=500,
                                totalamountuom="ml", isopenbag=0, continueinnextdept=0, statusdescription="FinishedRunning", originalamount=500, originalrate=500))
        if ine:
            self.T["inputevents"].append(pd.DataFrame(ine))
        # urine output hourly
        uo = np.clip(70 * (1 - np.clip(rn, 0, 1.0) * 1.05) * rng.uniform(.6, 1.3, L) + rng.normal(0, 8, L), 0, 400)
        uo = np.where(rn > .85, rng.uniform(0, 15, L), uo)
        uts = hours_ok(np.arange(0, L, 1.0), p_miss=.05)
        if len(uts) and not n_dev_missing:
            self.T["outputevents"].append(pd.DataFrame(dict(subject_id=subject, hadm_id=hadm, stay_id=stay, caregiver_id=rng.integers(1000, 99999, len(uts)),
                                                          charttime=tm(uts), storetime=tm(uts) + np.timedelta64(600, "s"), itemid=226559, value=np.round(uo[uts.astype(int).clip(0, L - 1)]), valueuom="mL")))
        # ---------- labs (hadm-level)
        cr0 = max(.4, rng.normal(.9 + (.5 if "ckd" in S["comorb"] else 0), .25)) + P.cr_off
        def labvals(h):
            h = int(min(h, L - 1))
            f = lambda k: float(z[k][h])
            return {50912: cr0 * (1 + 4.8 * f("renal")) + rng.normal(0, .05), 50885: max(.2, .7 + 11 * f("liver") + rng.normal(0, .15)),
                    51265: max(4, 235 - 200 * f("coag") - 35 * f("infl") + rng.normal(0, 14)), 51301: max(.5, 8.5 + 11 * f("infl") - (14 * f("infl") if (atyp and f("infl") > .2 and rng.random() < .3) else 0) + rng.normal(0, 1.5)),
                    50813: max(.4, 1.0 + 7 * f("shock") + 2.5 * f("bleed") + rng.normal(0, .25)),
                    50821: max(30, (450 - 380 * f("resp")) * (fio2[h] / 100) * rng.uniform(.9, 1.1)), 50818: 40 + 18 * f("resp") - 10 * f("metab") + rng.normal(0, 3),
                    50820: 7.40 - .26 * f("shock") - .18 * f("metab") - .05 * f("resp") + rng.normal(0, .015), 50882: 24 - 15 * f("metab") - 7 * f("shock") + rng.normal(0, 1.2),
                    51222: 12.8 - 6.5 * f("bleed") - 1.5 * f("infl") + rng.normal(0, .5), 51237: 1.1 + 2.6 * f("liver") + 1.2 * f("coag") + rng.normal(0, .06),
                    50931: 115 + 380 * f("metab") + rng.normal(0, 18), 50983: 139 - 6 * f("metab") + rng.normal(0, 2), 50971: 4.1 + 1.4 * f("renal") + rng.normal(0, .3),
                    51006: 16 + 55 * f("renal") + rng.normal(0, 3), 50861: 28 + 600 * f("liver") + rng.normal(0, 8), 50878: 30 + 700 * f("liver") + rng.normal(0, 10)}
        unstable = float(max(sh.max(), rs.max(), bl.max(), mt.max()))
        lab_hours = [-3.0]
        h_ = 0.5
        while h_ < L:
            lab_hours.append(h_)
            iv = (6 if unstable > .5 else 12 if unstable > .25 else 22) * P.lab_interval_scale * rng.uniform(.85, 1.2)
            h_ += iv
        full_group = {50912, 51265, 51301, 51222, 50983, 50971, 51006, 50882, 50931}
        abg_group = {50813, 50821, 50818, 50820}
        lft_group = {50885, 51237, 50861, 50878}
        no_lactate = rng.random() < P.lactate_never_p
        if no_lactate:
            tags.append("lactate_never_measured")
        for k, lh in enumerate(lab_hours):
            tl = icu_in + np.timedelta64(int(lh * 3600), "s"); hh = max(0, lh); vals = labvals(hh)
            items = set(full_group)
            if (unstable > .3 or k == 0 and (rs.max() > .3 or sh.max() > .3)) or rng.random() < .12:
                items |= abg_group
            if (k == 0 or int(lh) % 24 < 12) and (z["liver"].max() > .1 or arch in ("sepsis_in_icu", "septic_shock", "sepsis_floor_to_icu") or rng.random() < .3):
                items |= lft_group
            if no_lactate:
                items.discard(50813)
            spec_id = int(self._id("next_spec"))
            for itemid in items:
                nm, uom, lo, hi, cat, fl = LAB_DEF[itemid]; v = float(vals[itemid])
                v = round(v, 1 if itemid not in (50820, 51237) else 2)
                val, vnum = v, v
                if rng.random() < .004:
                    val, vnum = (">1000" if itemid in (50861, 50878) else "ERROR"), np.nan; tags.append("lab_string_value_no_valuenum")
                if rng.random() < .003:
                    val, vnum = "___", np.nan
                flag = "abnormal" if (isinstance(vnum, float) and not np.isnan(vnum) and (vnum < lo or vnum > hi)) else np.nan
                self.T["labevents"].append(dict(labevent_id=int(self._id("next_lab")), subject_id=subject, hadm_id=hadm, specimen_id=spec_id, itemid=itemid, charttime=tl,
                                                storetime=tl + np.timedelta64(int(rng.integers(20, 120)), "m"), value=val, valuenum=vnum, valueuom=uom, ref_range_lower=lo, ref_range_upper=hi, flag=flag,
                                                priority="STAT" if unstable > .5 else "ROUTINE", comments=("HEMOLYZED SPECIMEN" if rng.random() < .004 else np.nan)))
        # ---------- icu / transfers / admissions rows
        died = meta["death"]; icu_out_t = icu_out
        self.T["icustays"].append(dict(subject_id=subject, hadm_id=hadm, stay_id=stay, first_careunit=unit, last_careunit=unit, intime=icu_in, outtime=icu_out_t, los=round(base_L / 24, 4)))
        self.T["transfers"].append(dict(subject_id=subject, hadm_id=hadm, transfer_id=int(self._id("next_order")), eventtype="transfer", careunit=unit, intime=icu_in, outtime=icu_out_t))
        # latent dump for debugging
        lat = pd.DataFrame({k: z[k] for k in ZERO}); lat.insert(0, "hour", tt); lat.insert(0, "stay_id", stay)
        self.truth_lat.append(lat.round(3))
        onset_h = meta["truth_onset_h"]
        self.truth_icu.append(dict(stay_id=stay, subject_id=subject, hadm_id=hadm, archetype=arch, from_ed_stay_id=from_ed if from_ed is not None else np.nan,
                                   is_sepsis_truth=int(onset_h is not None),
                                   sepsis_truth_onset=(icu_in + np.timedelta64(int(onset_h * 3600), "s")) if onset_h is not None else pd.NaT,
                                   infection_time=(icu_in + np.timedelta64(int(inf_h * 3600), "s")) if inf_h is not None else pd.NaT, died=int(died), los_h=round(base_L, 1),
                                   tags="|".join(sorted(set(tags))), profile=P.name))
        return dict(stay=stay, icu_in=icu_in, icu_out=icu_out_t, died=died, L=L)

    # --- hospital admission wrapper + hosp-level tables
    def finish_admission(self, subject, hadm, adm_t, icu_res, group_or_arch, ed=None, elective=False):
        rng = self.rng
        extra_days = float(np.clip(rng.lognormal(1.0, .7), .5, 25))
        base_end = (icu_res["icu_out"] if icu_res else adm_t + np.timedelta64(int(rng.integers(12, 120)), "h"))
        disch = base_end + np.timedelta64(int(extra_days * 86400 * (0 if (icu_res and icu_res["died"]) else 1)), "s")
        died = bool(icu_res and icu_res["died"]) or (rng.random() < .02)
        dtime = disch if died else pd.NaT
        self.T["admissions"].append(dict(subject_id=subject, hadm_id=hadm, admittime=adm_t, dischtime=disch, deathtime=dtime, admission_type=("ELECTIVE" if elective else "EW EMER."),
                                         admission_location=("PHYSICIAN REFERRAL" if elective else "EMERGENCY ROOM"), discharge_location=("DIED" if died else str(rng.choice(["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB"], p=[.5, .2, .2, .1]))),
                                         insurance=str(rng.choice(["Medicare", "Other", "Medicaid"])), language="ENGLISH", marital_status=str(rng.choice(["MARRIED", "SINGLE", "WIDOWED", "DIVORCED"])),
                                         race="WHITE", edregtime=(ed["t0"] if ed else pd.NaT), edouttime=(ed["t1"] if ed else pd.NaT), hospital_expire_flag=int(died)))
        S = self.subjects[subject]
        if died:
            S["dod"] = disch
        self.T["transfers"].append(dict(subject_id=subject, hadm_id=hadm, transfer_id=int(self._id("next_order")), eventtype="admit", careunit="Medicine", intime=adm_t, outtime=adm_t + np.timedelta64(3600, "s")))
        self.T["transfers"].append(dict(subject_id=subject, hadm_id=hadm, transfer_id=int(self._id("next_order")), eventtype="discharge", careunit=np.nan, intime=disch, outtime=pd.NaT))
        # admission labs for non-ICU admits (so stage-1/2 style history has labs)
        if not icu_res:
            spec = int(self._id("next_spec"))
            for itemid in (50912, 51265, 51301, 51222, 50983):
                nm, uom, lo, hi, *_ = LAB_DEF[itemid]
                v = {50912: rng.normal(1.0, .3), 51265: rng.normal(240, 60), 51301: rng.normal(9.5, 3), 51222: rng.normal(12.5, 1.8), 50983: rng.normal(139, 3)}[itemid]
                self.T["labevents"].append(dict(labevent_id=int(self._id("next_lab")), subject_id=subject, hadm_id=hadm, specimen_id=spec, itemid=itemid, charttime=adm_t, storetime=adm_t + np.timedelta64(90, "m"),
                                                value=round(v, 1), valuenum=round(v, 1), valueuom=uom, ref_range_lower=lo, ref_range_upper=hi, flag=np.nan, priority="ROUTINE", comments=np.nan))
        # discharge diagnoses (history for next stays)
        seq = 1
        if group_or_arch in GROUPS:
            g = GROUPS[group_or_arch]; c, ttl = g["dx10"][0]; self.T["diagnoses_icd"].append(dict(subject_id=subject, hadm_id=hadm, seq_num=seq, icd_code=c, icd_version=10))
        for cm in S["comorb"][:4]:
            seq += 1; self.T["diagnoses_icd"].append(dict(subject_id=subject, hadm_id=hadm, seq_num=seq, icd_code=COMORB[cm][0][0], icd_version=10))

    # --- orchestration
    def run(self):
        rng = self.rng; P = self.prof
        gnames = [k for k in GROUPS if k != "novel_resp_cluster" or P.novel_group]
        w = np.array([P.group_w.get(k, GROUPS[k]["w"]) for k in gnames]); w = w / w.sum()
        reuse_pool = []
        for _ in range(self.n_ed):
            grp = str(rng.choice(gnames, p=w))
            subj = None
            if reuse_pool and rng.random() < .1:
                subj = int(rng.choice(reuse_pool))                     # repeat visitor: real "history" for stage 1
            e = self.ed_stay(grp, subject=subj)
            reuse_pool.append(e["subject"])
            if not e["admit"]:
                continue
            icu_go = rng.random() < GROUPS[grp]["icu"]
            icu_res = None
            if icu_go:
                m = ICU_FROM_GROUP[grp]; arch = str(rng.choice(list(m), p=np.array(list(m.values())) / sum(m.values())))
                if arch == "sepsis_floor_to_icu" and rng.random() < P.atypical_sepsis_w:
                    arch = "atypical_sepsis"
                icu_in = e["t1"] + np.timedelta64(int(rng.integers(1, 12) * 3600), "s")
                icu_res = self.icu_stay(arch, e["subject"], e["hadm"], icu_in, from_ed=e["stay"])
            self.finish_admission(e["subject"], e["hadm"], e["t1"] - np.timedelta64(1800, "s"), icu_res, grp, ed=e)
        arch_names = list(DIRECT_ARCH); aw = np.array(list(DIRECT_ARCH.values())); aw = aw / aw.sum()
        for _ in range(self.n_icu_direct):
            arch = str(rng.choice(arch_names, p=aw))
            if arch == "sepsis_in_icu" and rng.random() < P.atypical_sepsis_w:
                arch = "atypical_sepsis"
            subj = self.new_subject(62, 15, comorb_boost=1.3); hadm = self._id("next_hadm"); t0 = self.rand_time()
            icu_res = self.icu_stay(arch, subj, hadm, t0 + np.timedelta64(int(rng.integers(0, 3)) * 3600, "s"))
            self.finish_admission(subj, hadm, t0, icu_res, arch, elective=(arch == "stable_postop"))
        return self.finalize()

    def finalize(self):
        out = {}
        for name, parts in self.T.items():
            if not parts:
                df = pd.DataFrame(columns=SCHEMA[name]["columns"])
            elif isinstance(parts[0], dict):
                df = pd.DataFrame(parts)
            else:
                df = pd.concat(parts, ignore_index=True)
            out[name] = df
        pts = pd.DataFrame([dict(subject_id=s, gender=v["gender"], anchor_age=v["age"], anchor_year=v["ay"],
                                 anchor_year_group=f"{(v['ay'] // 3) * 3 - 1} - {(v['ay'] // 3) * 3 + 1}", dod=(pd.Timestamp(v["dod"]).strftime("%Y-%m-%d") if v["dod"] is not None else np.nan))
                            for s, v in self.subjects.items()])
        out["patients"] = pts
        # dictionaries (minimal, only items used) - written from built-in constants, not copied from MIMIC
        out["d_items"] = pd.DataFrame([dict(itemid=i, label=v[0], abbreviation=v[1], linksto=v[2], category=v[3], unitname=v[4], param_type=v[5], lownormalvalue=np.nan, highnormalvalue=np.nan) for i, v in ITEMS.items()])
        out["d_labitems"] = pd.DataFrame([dict(itemid=i, label=v[0], fluid=v[5], category=v[4]) for i, v in LAB_DEF.items()])
        codes = {}
        for gname, g in GROUPS.items():
            for c, t in g["dx10"]:
                codes[(c, 10)] = t
            for c, t in g["dx9"]:
                codes[(c, 9)] = t
        for cm, v in COMORB.items():
            codes[(v[0][0], 10)] = v[0][1]; codes[(v[1][0], 9)] = v[1][1]
        out["d_icd_diagnoses"] = pd.DataFrame([dict(icd_code=c, icd_version=ver, long_title=t) for (c, ver), t in codes.items()])
        # drop ED stays duplicated charttime types, enforce schema
        for name in list(out):
            cols = SCHEMA[name]["columns"]
            df = out[name]
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[cols]
            for c in df.columns:
                if c.endswith("time") or c in ("chartdate", "storedate"):
                    df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(FMT)
            out[name] = df
        truth = dict(truth_icu=pd.DataFrame(self.truth_icu), truth_ed=pd.DataFrame(self.truth_ed),
                     truth_latent=pd.concat(self.truth_lat, ignore_index=True) if self.truth_lat else pd.DataFrame())
        for k in ("sepsis_truth_onset", "infection_time"):
            if k in truth["truth_icu"]:
                truth["truth_icu"][k] = pd.to_datetime(truth["truth_icu"][k]).dt.strftime(FMT)
        if len(truth["truth_latent"]):
            truth["truth_latent"]["hour"] = pd.to_datetime(truth["truth_latent"]["hour"]).dt.strftime(FMT)
        return out, truth


def write_all(tables, truth, out_dir):
    for name, df in tables.items():
        mod = SCHEMA[name]["module"]; os.makedirs(os.path.join(out_dir, mod), exist_ok=True)
        df.to_csv(os.path.join(out_dir, mod, f"{name}.csv.gz"), index=False, compression="gzip")
    td = os.path.join(out_dir, "synth_truth"); os.makedirs(td, exist_ok=True)
    for name, df in truth.items():
        df.to_csv(os.path.join(td, f"{name}.csv.gz"), index=False, compression="gzip")


def generate(out_dir, n_ed=3000, n_icu_direct=300, seed=0, profile="default"):
    g = Gen(seed=seed, profile=profile, n_ed=n_ed, n_icu_direct=n_icu_direct)
    tables, truth = g.run()
    write_all(tables, truth, out_dir)
    return tables, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n-ed", type=int, default=3000)
    ap.add_argument("--n-icu-direct", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="default", choices=list(PROFILES))
    a = ap.parse_args()
    tb, tr = generate(a.out, a.n_ed, a.n_icu_direct, a.seed, a.profile)
    print({k: len(v) for k, v in tb.items()}); print({k: len(v) for k, v in tr.items()})
