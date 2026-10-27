"""TILDE-Q loss helpers. Taken from repo: https://github.com/HyunWookL/TILDE-Q"""

from __future__ import annotations

import numpy as np
import torch


def amp_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Amplitude consistency loss."""

    _, _, horizon = outputs.shape
    fft_size = 1 << (2 * horizon - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim=-1)

    out_norm = torch.norm(outputs, dim=-1, keepdim=True)
    tgt_norm = torch.norm(targets, dim=-1, keepdim=True)

    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(horizon - 1) :], auto_corr[..., :horizon]], dim=-1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim=-1).real
    cross_corr = torch.cat(
        [cross_corr[..., -(horizon - 1) :], cross_corr[..., :horizon]],
        dim=-1,
    )
    nac_out = cross_corr / (tgt_norm * out_norm)

    return torch.mean(torch.abs(nac_tgt - nac_out))


def ashift_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Amplitude-shift loss."""

    _, _, horizon = outputs.shape
    shifted = torch.softmax(outputs - targets, dim=-1)
    return horizon * torch.mean(torch.abs(1 / horizon - shifted))


def phase_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Frequency-domain phase loss."""

    _, _, horizon = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim=-1)
    tgt_fourier = torch.fft.fft(targets, dim=-1)
    tgt_power = tgt_fourier.real.square() + tgt_fourier.imag.square()

    mask = (tgt_power > horizon).float()
    topk_indices = tgt_power.topk(k=int(horizon**0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.0)
    mask[..., 0] = 1.0
    mask = torch.where(mask > 0, 1.0, 0.0).bool()

    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)

    mask_f = mask.float()
    mask_f /= torch.mean(mask_f)
    abs_error = torch.abs(out_fourier - tgt_fourier) * mask_f
    abs_error = torch.where(torch.isnan(abs_error), torch.zeros_like(abs_error), abs_error)

    return (torch.mean(zero_error) + torch.mean(abs_error)) / (horizon**0.5)


def tildeq_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 0.01,
) -> torch.Tensor:
    """Compute the TILDE-Q loss from forecast and target tensors."""

    outputs = outputs.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)

    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        raise ValueError("Invalid values detected in outputs.")

    loss = (
        alpha * ashift_loss(outputs, targets)
        + (1 - alpha) * phase_loss(outputs, targets)
        + gamma * amp_loss(outputs, targets)
    )

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("Invalid TILDE-Q loss value produced.")
    return loss


def compute_tilde_q(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute TILDE-Q for one univariate forecast."""

    true_t = torch.as_tensor(np.asarray(y_true), dtype=torch.float32).view(1, -1, 1)
    pred_t = torch.as_tensor(np.asarray(y_pred), dtype=torch.float32).view(1, -1, 1)
    return float(tildeq_loss(pred_t, true_t).detach().cpu().item())
