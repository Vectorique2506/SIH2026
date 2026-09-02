"""
Loss functions: SI-SNR + perceptual loss.

Owner: Track 2 (Model)
"""

import torch


def si_snr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR loss (negative SI-SNR, since we minimize).

    estimate, target: (batch, 1, time) or (batch, time) waveforms
    """
    # flatten channel dim if present: (batch, 1, time) -> (batch, time)
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    # 1. zero-mean both signals (per-example, along time)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # 2. project estimate onto target direction -> s_target
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    s_target = dot * target / target_energy

    # 3. e_noise = estimate - s_target
    e_noise = estimate - s_target

    # 4. SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    s_target_energy = torch.sum(s_target ** 2, dim=-1) + eps
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps
    si_snr = 10 * torch.log10(s_target_energy / e_noise_energy)

    # return negative mean SI-SNR as the loss (minimizing loss = maximizing SI-SNR)
    return -si_snr.mean()


def perceptual_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Placeholder for a perceptual loss term (e.g. STFT-magnitude based).

    Keep this simple initially — an STFT L1 loss is a reasonable stand-in
    for a full differentiable PESQ/STOI approximation under time pressure.
    """
    raise NotImplementedError