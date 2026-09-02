"""
Mix clean speech with noise at a target SNR.

Owner: Track 1 (Data)

TODO:
- Loop this over your clean speech files x noise files x SNR range
- Save each output triplet (noisy, clean, noise_type) to data/processed/
- Log every entry to data/manifest.csv via build_manifest.py

Run manually to sanity-check on one file pair before wiring the full loop.
"""

import librosa
import numpy as np
import soundfile as sf


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean speech and noise at a target SNR (dB).

    Assumes clean and noise are already the same length (crop/tile noise
    to match clean's length before calling this).
    """
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    noisy = clean + scale * noise
    return noisy


def match_length(clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Crop or tile noise so it matches clean's length."""
    len_clean = len(clean)
    len_noise = len(noise)

    if len_noise >= len_clean:
        # noise is longer (or equal) — crop it
        return noise[:len_clean]
    else:
        # noise is shorter — tile it until it covers clean's length, then crop
        n_repeats = int(np.ceil(len_clean / len_noise))
        tiled = np.tile(noise, n_repeats)
        return tiled[:len_clean]


if __name__ == "__main__":
    # Point these at one real file from each bucket you've already sourced.
    CLEAN_PATH = "data/raw/clean_speech/LibriSpeech/dev-clean/422/122949/422-122949-0005.flac"
    NOISE_PATH = "data/raw/noise/stationary/1-11687-A-47.wav"

    clean, sr_clean = sf.read(CLEAN_PATH)
    noise, sr_noise = sf.read(NOISE_PATH)

    # resample noise to match clean's sample rate if they differ
    if sr_noise != sr_clean:
        print(f"Resampling noise from {sr_noise}Hz to {sr_clean}Hz")
        noise = librosa.resample(noise, orig_sr=sr_noise, target_sr=sr_clean)
        sr_noise = sr_clean

    assert sr_clean == sr_noise, (
        f"Sample rate mismatch: clean={sr_clean}, noise={sr_noise}. "
        "Resample one of them before mixing — mismatched rates will "
        "produce garbage output."
    )

    # if either file is stereo, collapse to mono (mean across channels)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    noise = match_length(clean, noise)

    for snr_db in [-5, 0, 5, 10]:
        noisy = mix_at_snr(clean, noise, snr_db)
        out_path = f"test_mix_{snr_db}db.wav"
        sf.write(out_path, noisy, sr_clean)
        print(f"Wrote {out_path}")

    print("Now go listen to each test_mix_*.wav file.")
    print("-5dB should sound noise-dominated. 10dB should sound mostly clean speech.")