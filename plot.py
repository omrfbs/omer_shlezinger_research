from torch import Tensor
import torch

import matplotlib.pyplot as plt


def save_rd_map(rd_map: Tensor, file_path: str = "figs/rd_map.png") -> None:
    fig, ax = plt.subplots()
    rd_map_db = 20 * torch.log10(rd_map.abs())
    ax.pcolormesh(rd_map_db.detach().cpu())
    fig.savefig(file_path)


def save_spectrum(rd_map: Tensor, file_path: str = "figs/spectrum.png") -> None:
    fig, ax = plt.subplots()
    rd_map_abs = rd_map.abs()
    max_global_index = rd_map_abs.argmax()
    max_doppler_index = max_global_index // rd_map.shape[-1]
    max_range_index = max_global_index - max_doppler_index * rd_map.shape[-1]
    rd_map_db = 20 * torch.log10(rd_map_abs)
    ax.plot(rd_map_db[:, max_range_index].detach().cpu())
    fig.savefig(file_path)
