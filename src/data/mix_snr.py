"""
Mix clean speech with noise at a target SNR.

Owner: Track 1 (Data)

TODO:
- Loop this over your clean speech files x noise files x SNR range
- Save each output triplet (noisy, clean, noise_type) to data/processed/
- Log every entry to data/manifest.csv via build_manifest.py

Run manually to sanity-check on one file pair before wiring the full loop.
"""

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
    """Crop or tile noise so it matches clean's length. TODO: implement."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: quick manual test — load one clean file + one noise file,
    # mix at a couple of SNR levels, sf.write() the results, listen to them.
    pass
