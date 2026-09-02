"""
Filter ESC-50 audio files by category using the metadata CSV, and copy
them into the correct stationary/impulsive bucket folders.

Run this from the project root:
    python filter_esc50.py

Requires: pandas
"""

import shutil
from pathlib import Path

import pandas as pd

ESC50_ROOT = Path("data/raw/noise/esc50_temp")
META_CSV = ESC50_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = ESC50_ROOT / "audio"

STATIONARY_OUT = Path("data/raw/noise/stationary")
IMPULSIVE_OUT = Path("data/raw/noise/impulsive")

# ESC-50 category names to pull. Check meta/esc50.csv yourself for the
# exact full list of 50 categories if you want to add more.
STATIONARY_CATEGORIES = {"helicopter", "siren", "engine", "airplane", "train", "chainsaw"}
IMPULSIVE_CATEGORIES = {"fireworks"}  # weak stand-in only, see note below

STATIONARY_OUT.mkdir(parents=True, exist_ok=True)
IMPULSIVE_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META_CSV)
print("Available categories in ESC-50:", sorted(df["category"].unique()))

for _, row in df.iterrows():
    src = AUDIO_DIR / row["filename"]
    if not src.exists():
        continue
    if row["category"] in STATIONARY_CATEGORIES:
        shutil.copy(src, STATIONARY_OUT / row["filename"])
    elif row["category"] in IMPULSIVE_CATEGORIES:
        shutil.copy(src, IMPULSIVE_OUT / row["filename"])

n_stat = len(list(STATIONARY_OUT.glob("*.wav")))
n_imp = len(list(IMPULSIVE_OUT.glob("*.wav")))
print(f"Copied {n_stat} stationary files, {n_imp} impulsive files.")
print("NOTE: 'fireworks' is a weak stand-in for gunshot/artillery.")
print("Pull real gun_shot clips from UrbanSound8K and add them to impulsive/ separately.")