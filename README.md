# SIH26052 — AI/ML-Enabled Adaptive Noise Cancellation

DRDO | Hardware category | Smart Vehicles theme

## Problem
Suppress stationary, non-stationary, and impulsive defence noises (gunshots,
artillery, rotor, engine, sirens, wind) from speech while maintaining
intelligibility and real-time performance on embedded hardware.

## Targets
- SNR > 15 dB
- STOI > 0.85
- PESQ > 2.5
- Real-time end-to-end latency (mic capture + inference + playback)

## Current scope (update this daily — be honest about cuts)
- Noise classes covered: TBD
- Model: DCCRN / Conv-TasNet (pick one)
- Hardware target: Raspberry Pi 5 (8GB)
- Dataset: synthetic SNR-mixed pairs only (no real recordings yet — flag this in the pitch)

## Current metrics (fill in as you get them, split by bucket)
| Bucket      | SNR (dB) | STOI | PESQ | Latency (ms) |
|-------------|----------|------|------|--------------|
| Stationary  |          |      |      |              |
| Impulsive   |          |      |      |              |

## Manifest schema (agree before Track 1 / Track 2 merge)
`data/manifest.csv` columns:
`noisy_path,clean_path,noise_type,bucket,snr_db`

- `bucket` is one of: `stationary`, `impulsive`
- `noise_type` is the specific class, e.g. `rotor`, `gunshot`, `engine`

## Team ownership
- Track 1 (data): owns `src/data/`
- Track 2 (model): owns `src/models/`, `train.py`
- Merge point: swap dummy tensors in `train.py` for real `ANCDataset` reading `manifest.csv`
