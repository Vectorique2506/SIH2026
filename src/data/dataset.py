"""
PyTorch Dataset reading data/manifest.csv.

Owner: Track 1 (Data), consumed by Track 2 in train.py

This is the contract between the two tracks — the __getitem__ output shape
must match what train.py / your model expects (batch of waveforms or
spectrograms, check your chosen model's input format).
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path
from torch.utils.data import Dataset

FIXED_LENGTH = 16000  # 1 second at 16kHz — crop/pad every clip to this

# project root = two levels up from this file (src/data/dataset.py -> src/data -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ANCDataset(Dataset):
    def __init__(self, manifest_csv: str, split: str = None, bucket: str = None):
        """
        Args:
            manifest_csv: path to data/manifest.csv
            split: optional filter, "train"/"val"/"test" or None for all
            bucket: optional filter, "stationary" or "impulsive" or None for all
        """
        self.df = pd.read_csv(manifest_csv)
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if bucket is not None:
            self.df = self.df[self.df["bucket"] == bucket].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(PROJECT_ROOT / p)

    def _load_fixed_length(self, path: str) -> np.ndarray:
        audio, _ = sf.read(self._resolve_path(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) >= FIXED_LENGTH:
            audio = audio[:FIXED_LENGTH]
        else:
            pad = FIXED_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad))
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        noisy = self._load_fixed_length(row["noisy_path"])
        clean = self._load_fixed_length(row["clean_path"])
        # shape: (1, FIXED_LENGTH) — matches (batch, 1, time) once batched
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)
        return noisy_t, clean_t