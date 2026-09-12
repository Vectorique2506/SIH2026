"""
Evaluate trained model against SNR / STOI / PESQ targets.

Reports metrics SPLIT BY BUCKET (stationary vs impulsive).
Targets: SNR > 15 dB, STOI > 0.85, PESQ > 2.5
"""


import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pystoi import stoi
from pesq import pesq

from data.dataset import ANCDataset
from models.conv_tasnet import ConvTasNet


def snr_improvement(clean: np.ndarray, noisy: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Calculate SNR improvement in dB."""
    noise_orig = noisy - clean
    noise_est = estimate - clean

    snr_orig = 10 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise_orig ** 2) + eps))
    snr_est = 10 * np.log10((np.sum(clean ** 2) + eps) / (np.sum(noise_est ** 2) + eps))
    return snr_est - snr_orig


def evaluate_bucket(model, dataset, device, bucket_name: str, sample_rate: int = 16000):
    """
    Run model on all examples of a specific bucket and return average metrics.
    """
    model.eval()
    snr_list, stoi_list, pesq_list = [], [], []

    print(f"\nEvaluating bucket: {bucket_name} ({len(dataset)} samples)")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=bucket_name):
            noisy, clean = dataset[i]          # shape: (1, T)
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Model expects [batch, T]
            estimate = model(noisy.squeeze(0).unsqueeze(0))  # → [1, 1, T]
            estimate = estimate.squeeze().cpu().numpy()
            clean_np = clean.squeeze().cpu().numpy()
            noisy_np = noisy.squeeze().cpu().numpy()

            # Make sure lengths match
            min_len = min(len(estimate), len(clean_np), len(noisy_np))
            estimate = estimate[:min_len]
            clean_np = clean_np[:min_len]
            noisy_np = noisy_np[:min_len]

            # --- Metrics ---
            try:
                snr_imp = snr_improvement(clean_np, noisy_np, estimate)
                snr_list.append(snr_imp)

                stoi_score = stoi(clean_np, estimate, sample_rate, extended=False)
                stoi_list.append(stoi_score)

                # PESQ needs 16kHz and can fail on some clips → protect it
                pesq_score = pesq(sample_rate, clean_np, estimate, 'wb')
                pesq_list.append(pesq_score)
            except Exception as e:
                # Skip problematic clips instead of crashing
                continue

    results = {
        "snr": np.mean(snr_list) if snr_list else 0.0,
        "stoi": np.mean(stoi_list) if stoi_list else 0.0,
        "pesq": np.mean(pesq_list) if pesq_list else 0.0,
        "count": len(snr_list)
    }
    return results


def main():
    # ---------- Config ----------
    CHECKPOINT = "../checkpoints/prototype.pt"   # change if you want another epoch
    MANIFEST = "../data/manifest.csv"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    SAMPLE_RATE = 16000

    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT}")

    # ---------- Load Model ----------
    model = ConvTasNet(
        N=64, L=20, B=64, H=128, P=3, X=6, R=2, C=1,
        norm_type="gLN", causal=False, mask_nonlinear='relu'
    ).to(DEVICE)

    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # ---------- Evaluate both buckets ----------
    results = {}

    for bucket in ["stationary", "impulsive"]:
        dataset = ANCDataset(MANIFEST, split="test", bucket=bucket)
        if len(dataset) == 0:
            print(f"Warning: No samples found for bucket '{bucket}'")
            continue
        results[bucket] = evaluate_bucket(model, dataset, DEVICE, bucket, SAMPLE_RATE)

    # ---------- Print Final Table ----------
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS (Test Set)")
    print("="*60)
    print(f"{'Bucket':<12} | {'SNR (dB)':>10} | {'STOI':>8} | {'PESQ':>8} | {'Samples'}")
    print("-"*60)

    for bucket, res in results.items():
        print(f"{bucket:<12} | {res['snr']:10.2f} | {res['stoi']:8.3f} | {res['pesq']:8.3f} | {res['count']}")

    print("="*60)
    print("Targets → SNR > 15 dB | STOI > 0.85 | PESQ > 2.5")
    print("="*60)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    