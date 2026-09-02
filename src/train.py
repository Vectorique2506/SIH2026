"""
Training entrypoint.

Owner: Track 2 (Model)

Run from inside src/:
    python train.py
"""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import si_snr_loss
from models.conv_tasnet import ConvTasNet
from data.dataset import ANCDataset

os.makedirs("../checkpoints", exist_ok=True)


def train(manifest_csv: str = "../data/manifest.csv", epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ANCDataset(manifest_csv, split="train")
    print(f"Training on {len(dataset)} pairs")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ConvTasNet(
        N=64, L=20, B=64, H=128, P=3, X=6, R=2, C=1,
        norm_type="gLN", causal=False, mask_nonlinear='relu'
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for noisy, clean in tqdm(loader, desc=f"epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            # model expects [batch, T] (2D), dataset gives [batch, 1, T] (3D)
            estimate = model(noisy.squeeze(1))  # -> [batch, C=1, T]

            loss = si_snr_loss(estimate, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"epoch {epoch+1}/{epochs}  loss {avg_loss:.4f}")
        torch.save(model.state_dict(), f"../checkpoints/epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "../checkpoints/prototype.pt")
    print("Saved final checkpoint to checkpoints/prototype.pt")


if __name__ == "__main__":
    train()
    
    
    
    
    
    