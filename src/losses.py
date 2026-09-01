"""
Loss functions: SI-SNR + perceptual loss.

Owner: Track 2 (Model)
"""

import torch


def si_snr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR loss (negative SI-SNR, since we minimize).

    estimate, target: (batch, time) waveforms
    """
    # TODO: implement SI-SNR
    # 1. zero-mean both signals
    # 2. project estimate onto target direction -> s_target
    # 3. e_noise = estimate - s_target
    # 4. SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    # return negative mean SI-SNR as the loss
    raise NotImplementedError


def perceptual_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Placeholder for a perceptual loss term (e.g. STFT-magnitude based).

    Keep this simple initially — an STFT L1 loss is a reasonable stand-in
    for a full differentiable PESQ/STOI approximation under time pressure.
    """
    raise NotImplementedError
