"""
Walk data/processed/ and generate data/manifest.csv.

Owner: Track 1 (Data)

Schema (see README):
    noisy_path,clean_path,noise_type,bucket,snr_db

bucket must be one of: "stationary", "impulsive"

TODO:
- Decide your on-disk naming convention for processed files so this script
  can parse noise_type / bucket / snr_db back out of filenames or a sidecar.
- Write out data/manifest.csv with pandas.
"""

import pandas as pd


def build_manifest(processed_dir: str, out_csv: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: call build_manifest() with your actual paths
    pass
