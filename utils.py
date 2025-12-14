import torch
from torch import Tensor
from globals import device


def circular_weigthed_mean(x: Tensor) -> Tensor:
    n = x.shape[-1]
    w = torch.exp(1j * 2 * torch.pi * torch.arange(-n // 2 + 1, n // 2 + 1) / n).to(device=device)
    avg = (x * w / x.sum(dim=-1, keepdim=True)).sum(dim=-1)

    peak_idx = n * avg.angle() / (2 * torch.pi)
    peak_idx = peak_idx + n // 2
    mag = avg.abs() ** 2
    return mag, peak_idx