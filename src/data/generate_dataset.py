"""
Generate the full noisy/clean training dataset.

Owner: Track 1 (Data)

Loops over every clean speech file x every noise file (both buckets) x
every target SNR, writes mixed audio to data/processed/{train,val,test}/,
and writes data/manifest.csv directly (so build_manifest.py is not needed
as a separate step — this script does both).

Run from the project root:
    python src/data/generate_dataset.py
"""

import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ---- CONFIG ----------------------------------------------------------

CLEAN_SPEECH_DIR = Path("data/raw/clean_speech")  # LibriSpeech root (nested by speaker/chapter)
NOISE_DIRS = {
    "stationary": Path("data/raw/noise/stationary"),
    "impulsive": Path("data/raw/noise/impulsive"),
}
OUT_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/manifest.csv")

SNR_LEVELS_DB = [-5, 0, 5, 10]
TARGET_SR = 16000

# split fractions — by SPEAKER for clean, by FILE for noise, so no file
# or speaker appears in more than one split (avoids leakage)
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# cap total pairs generated — LibriSpeech x noise x SNR can explode fast.
# Set to None to generate everything (only do this if you've checked disk
# space and have time to spare).
MAX_PAIRS_PER_SPLIT = 800

# ---- CORE MIXING LOGIC (same as mix_snr.py) --------------------------


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    return clean + scale * noise


def match_length(clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
    len_clean, len_noise = len(clean), len(noise)
    if len_noise >= len_clean:
        return noise[:len_clean]
    n_repeats = int(np.ceil(len_clean / len_noise))
    return np.tile(noise, n_repeats)[:len_clean]


def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ---- DISCOVER SOURCE FILES --------------------------------------------


def find_clean_files():
    """Find all .flac/.wav clean speech files, grouped by speaker ID.

    Assumes LibriSpeech-style nesting: clean_speech/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac
    Adjust the glob if your extracted structure differs.
    """
    files = list(CLEAN_SPEECH_DIR.rglob("*.flac")) + list(CLEAN_SPEECH_DIR.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No clean speech files found under {CLEAN_SPEECH_DIR}. "
            "Check the path and that LibriSpeech was extracted correctly."
        )

    by_speaker = {}
    for f in files:
        # LibriSpeech path: .../<speaker_id>/<chapter_id>/<file>.flac
        # speaker_id is the parent-of-parent directory name
        speaker_id = f.parent.parent.name
        by_speaker.setdefault(speaker_id, []).append(f)
    return by_speaker


def find_noise_files():
    """Return dict: bucket -> list of noise file paths."""
    out = {}
    for bucket, d in NOISE_DIRS.items():
        files = list(d.glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No noise files found under {d}")
        out[bucket] = files
    return out


# ---- SPLIT BY SPEAKER / NOISE FILE (avoid leakage) --------------------


def split_list(items, val_frac, test_frac, seed):
    items = sorted(items)  # sort first for reproducibility across runs
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, int(n * val_frac)) if n > 2 else 0
    n_test = max(1, int(n * test_frac)) if n > 2 else 0
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return {"train": train, "val": val, "test": test}


# ---- MAIN GENERATION LOOP ----------------------------------------------


def generate():
    clean_by_speaker = find_clean_files()
    noise_by_bucket = find_noise_files()

    speaker_ids = list(clean_by_speaker.keys())
    speaker_splits = split_list(speaker_ids, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

    # noise files also split independently, per bucket, so a noise clip
    # used in train never also appears in val/test
    noise_splits = {
        bucket: split_list(files, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED + 1)
        for bucket, files in noise_by_bucket.items()
    }

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for split in ["train", "val", "test"]:
        split_speakers = speaker_splits[split]
        split_clean_files = []
        for sid in split_speakers:
            split_clean_files.extend(clean_by_speaker[sid])

        split_noise_files = {
            bucket: noise_splits[bucket][split] for bucket in noise_by_bucket
        }

        pairs = []
        for clean_path in split_clean_files:
            for bucket, noise_files in split_noise_files.items():
                for noise_path in noise_files:
                    for snr_db in SNR_LEVELS_DB:
                        pairs.append((clean_path, noise_path, bucket, snr_db))

        rng = random.Random(RANDOM_SEED + 2)
        rng.shuffle(pairs)
        if MAX_PAIRS_PER_SPLIT is not None:
            pairs = pairs[:MAX_PAIRS_PER_SPLIT]

        print(f"[{split}] generating {len(pairs)} pairs "
              f"({len(split_speakers)} speakers, "
              f"{sum(len(v) for v in split_noise_files.values())} noise files)")

        for clean_path, noise_path, bucket, snr_db in tqdm(pairs, desc=split):
            try:
                clean = load_mono_resampled(clean_path, TARGET_SR)
                noise = load_mono_resampled(noise_path, TARGET_SR)
            except Exception as e:
                print(f"Skipping pair due to load error: {e}")
                continue

            noise = match_length(clean, noise)
            noisy = mix_at_snr(clean, noise, snr_db)

            clean_stem = clean_path.stem
            noise_stem = noise_path.stem
            noise_type = noise_stem  # filename doubles as noise_type label; refine if needed
            snr_tag = f"{snr_db}db".replace("-", "neg")

            noisy_name = f"{clean_stem}__{noise_type}__{snr_tag}__noisy.wav"
            clean_name = f"{clean_stem}__{noise_type}__{snr_tag}__clean.wav"

            noisy_path = OUT_DIR / split / noisy_name
            clean_out_path = OUT_DIR / split / clean_name

            sf.write(str(noisy_path), noisy, TARGET_SR)
            sf.write(str(clean_out_path), clean, TARGET_SR)

            manifest_rows.append({
                "noisy_path": str(noisy_path),
                "clean_path": str(clean_out_path),
                "noise_type": noise_type,
                "bucket": bucket,
                "snr_db": snr_db,
                "split": split,
            })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {MANIFEST_PATH}")
    print(df.groupby(["split", "bucket"]).size())


if __name__ == "__main__":
    generate()