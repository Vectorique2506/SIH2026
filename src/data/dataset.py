"""
PyTorch Dataset reading data/manifest.csv.

Owner: Track 1 (Data), consumed by Track 2 in train.py

This is the contract between the two tracks — the __getitem__ output shape
must match what train.py / your model expects (batch of waveforms or
spectrograms, check your chosen model's input format).
"""

import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset


class ANCDataset(Dataset):
    def __init__(self, manifest_csv: str, bucket: str = None):
        """
        Args:
            manifest_csv: path to data/manifest.csv
            bucket: optional filter, "stationary" or "impulsive" or None for all
        """
        self.df = pd.read_csv(manifest_csv)
        if bucket is not None:
            self.df = self.df[self.df["bucket"] == bucket].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # TODO: load noisy_path and clean_path with soundfile, return as tensors
        # noisy, _ = sf.read(row["noisy_path"])
        # clean, _ = sf.read(row["clean_path"])
        raise NotImplementedError
