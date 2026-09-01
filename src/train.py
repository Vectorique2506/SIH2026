"""
Training entrypoint.

Owner: Track 2 (Model) — merges with Track 1's dataset.py once manifest.csv exists.

Stage 1 (today, before real data exists): validate the pipeline plumbing
with dummy tensors — confirms forward/backward pass works before real data
is mixed in as a second variable.

Stage 2 (merge point): swap dummy_input for the real ANCDataset + DataLoader.
"""

import torch
from torch.utils.data import DataLoader

from losses import si_snr_loss
# from models.dccrn import DCCRN
# from models.conv_tasnet import ConvTasNet
# from data.dataset import ANCDataset


def validate_pipeline():
    """Stage 1: confirm the model runs end-to-end on dummy data."""
    dummy_input = torch.randn(4, 1, 16000)  # batch=4, mono, 1 sec @ 16kHz
    # model = DCCRN()  # or ConvTasNet()
    # output = model(dummy_input)
    # loss = si_snr_loss(output, dummy_input)  # dummy target for now
    # loss.backward()
    # print("Pipeline runs. Output shape:", output.shape)
    raise NotImplementedError("Uncomment above once a model is wired in")


def train(manifest_csv: str, epochs: int = 10, batch_size: int = 4, lr: float = 1e-3):
    """Stage 2: real training loop, called once manifest.csv exists."""
    # dataset = ANCDataset(manifest_csv)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # model = DCCRN()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # for epoch in range(epochs):
    #     for noisy, clean in loader:
    #         optimizer.zero_grad()
    #         estimate = model(noisy)
    #         loss = si_snr_loss(estimate, clean)
    #         loss.backward()
    #         optimizer.step()
    #     print(f"epoch {epoch} loss {loss.item():.4f}")
    #     torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
    raise NotImplementedError


if __name__ == "__main__":
    validate_pipeline()
    # train("data/manifest.csv")
