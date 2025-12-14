# %%
from globals import device
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from datasets import IqDiscDataset
from radar_simulator.signal_processing.signal_processing import (
    pulse_compression,
    produce_rd_map,
)
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from typing import Tuple


# %%
class OptimizeSNR(nn.Module):

    def __init__(self, seq_len: int, kernel: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.weights_mag = nn.Parameter(torch.ones(seq_len, 1))
        self.weights_angle = nn.Parameter(
            0.001 * torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
        )

        # nn.init.normal_(self.weights_mag)
        nn.init.uniform_(self.weights_angle)
        self.mf_coeffs = (
            kernel.repeat(self.seq_len, 1).unsqueeze(1).conj().to(device=device)
        )
        self.padding = int(kernel.shape[-1] / 2)

    def real_to_complex(self, x: Tensor) -> Tensor:
        tensor_shape = list(x.shape)
        tensor_shape[-1] = tensor_shape[-1] // 2
        tensor_shape.append(2)
        return torch.view_as_complex(x.reshape(tensor_shape))

    def complex_to_real(self, x: Tensor) -> Tensor:
        return torch.view_as_real(x).flatten(start_dim=-2)

    def normalize(self, x: Tensor) -> Tensor:
        x /= torch.std(x, dim=(1, 2), keepdim=True)

        return x

    def forward(self, x: Tensor):
        x = x.to(torch.complex64)
        x = F.conv1d(x, self.mf_coeffs, padding=self.padding, groups=self.seq_len)
        phase = torch.atan(self.weights_angle)
        # pulse_axis = torch.arange(x.shape[0], device=device)
        # X = torch.stack([pulse_axis ** i for i in range(2)], dim=1).to(torch.float32)
        # coeffs = torch.linalg.lstsq(X.to(device=device), phase).solution
        # phase = phase - coeffs[0] - coeffs[1] * pulse_axis.unsqueeze(-1)
        # x = torch.softmax(self.weights_mag.abs(), dim=0) * x * torch.exp(-1j * phase)
        norm_mag = x.shape[0] * torch.softmax(self.weights_mag, dim=0)
        x = norm_mag * x * torch.exp(-1j * phase)
        return x


def circular_weigthed_mean(x: Tensor) -> Tensor:
    n = x.shape[0]
    w = torch.exp(1j * 2 * torch.pi * torch.arange(-n // 2 + 1, n // 2 + 1) / n).to(
        device=device
    )
    avg = (x * w / x.sum()).sum()
    # diff = (avg.angle() - w.angle()).abs()
    # vals, idxs = diff.sort()
    # alpha = vals[0] / (vals[0] + vals[1])
    # peak_idx = alpha * idxs[1] + (1 - alpha) * idxs[0]
    peak_idx = n * avg.angle() / (2 * torch.pi)
    peak_idx = peak_idx + n // 2
    mag = avg.abs() ** 2
    return mag, peak_idx


def fit_linear_ls(phase: Tensor) -> Tensor:
    pulse_axis = torch.arange(phase.shape[0], device=device)
    X = torch.stack([pulse_axis**i for i in range(2)], dim=1).to(torch.float32)
    return torch.linalg.lstsq(X.to(device=device), phase).solution


def snr(rd_map: Tensor) -> Tuple[Tensor, float]:
    nfft = rd_map.shape[0]
    rd_map_energy = rd_map.abs() ** 2
    max_global_index = rd_map_energy.argmax()
    max_doppler_index = max_global_index // rd_map_energy.shape[1]
    max_range_index = max_global_index - max_doppler_index * rd_map_energy.shape[1]
    mag, peak_idx = circular_weigthed_mean(rd_map_energy[:, max_range_index])
    idx = torch.round(peak_idx).to(torch.int) % nfft
    # idx2 = (idx1 + 1) % nfft
    peak = rd_map_energy[idx, max_range_index]
    # peak2 = rd_map_energy[idx2, max_range_index]
    # alpha = peak1 / (peak1 + peak2)
    # peak = peak1 * alpha + peak2 * (1-alpha)
    global_noise = (rd_map_energy.sum() - peak) / (rd_map_energy.numel() - 1)
    doppler_noise = rd_map_energy[:, max_range_index].sum() - peak
    avg_noise = doppler_noise
    return -10 * torch.log10(peak) + 10 * torch.log10(avg_noise), peak_idx
    # return -mag, peak_idx


# %%
dataset = IqDiscDataset(
    n_samples=1000,
    r_min=2000,
    r_max=20000,
    v_min=0,
    v_max=90,
    rotation_max=150,
    radius=0.2,
    target_type="disc",
)
# %%
n_iter = 1000
index = 2

model = OptimizeSNR(seq_len=dataset.radar.n_pulses, kernel=dataset.kernel).to(
    device=device
)
optimizer = optim.Adam(lr=1e-2, params=model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
iq, (range_index, doppler_index) = dataset[index]

model.train()  # Set model to training mode
pbar = tqdm(range(n_iter), desc="iter")
losses = torch.zeros(n_iter)
err = torch.zeros(n_iter)
best_loss = Tensor([torch.inf]).to(device=device)
best_rd_map = None

nfft = 21
for i in pbar:
    optimizer.zero_grad()
    iq_fixed = model(iq)
    # iq_fixed = iq_fixed / iq_fixed.abs().max(dim=1)[0].unsqueeze(-1)
    rd_map_fixed = torch.fft.fftshift(torch.fft.fft(iq_fixed, n=nfft, dim=-2), dim=-2)
    max_idx = rd_map_fixed.abs().argmax()
    doppler_index_pred = max_idx // rd_map_fixed.shape[1]
    err[i] = doppler_index_pred.cpu() - doppler_index
    loss_snr, doppler_index_opt = snr(rd_map_fixed)
    p = model.weights_angle.atan().squeeze()
    a_spectrum = torch.fft.fftshift(torch.fft.fft(torch.exp(1j * p))).abs() ** 2
    noise = (a_spectrum.sum() - a_spectrum[10]) / (a_spectrum.shape[0] - 1)
    linearity = 10 * torch.log10(a_spectrum[10]) - 10 * torch.log10(noise)
    loss = loss_snr + 0.0 * linearity

    if loss < best_loss:
        best_loss = loss
        best_model = deepcopy(model)
        best_rd_map = rd_map_fixed
        best_doppler_index = doppler_index_opt.cpu().item()
    if i < n_iter:
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        scheduler.step()
    losses[i] = loss
    # print(f'SNR={-loss:.2f}')
# %% PLOT
max_idx = best_rd_map.abs().argmax()
max_doppler_index = max_idx // best_rd_map.shape[1]
max_range_index = max_idx - max_doppler_index * best_rd_map.shape[1]

after_mf = pulse_compression(
    iq.cpu().to(torch.complex128), dataset.radar.pulse_bb.to(torch.complex128)
)
rd_map = produce_rd_map(after_mf, n=nfft)
rd_map_abs = rd_map.abs() ** 2
rd_map_db = 10 * torch.log10(rd_map_abs)
max_classic = rd_map_abs.max()
noise_classic = (rd_map_abs.sum() - max_classic) / (rd_map_abs.numel() - 1)
snr_classic = 10 * torch.log10(max_classic / noise_classic)
print(snr_classic)
fig, ax = plt.subplots()
im = ax.pcolormesh(rd_map_db)
ax.set_title("Classic RD-MAP")
fig.colorbar(im, ax=ax)

fig, ax = plt.subplots()
rd_map_fixed_db = 20 * torch.log10(best_rd_map.abs().cpu().detach().squeeze())
im = ax.pcolormesh(rd_map_fixed_db)
ax.set_title("Pred RD-MAP")
fig.colorbar(im, ax=ax)

fig, ax = plt.subplots()
ax.plot(
    rd_map_db[:, max_range_index] - rd_map_db[:, max_range_index].max(), label="Classic"
)
ax.plot(
    rd_map_fixed_db[:, max_range_index] - rd_map_fixed_db[:, max_range_index].max(),
    label="Pred",
)
ax.axvline(
    x=doppler_index * nfft / dataset.radar.n_pulses, color="r", label="True Doppler"
)
ax.set_title("Spectrums")
ax.legend()
ax.set_ylabel("dB")
ax.set_xlabel("Doppler Index")
# %%
import pandas as pd

n_iter = 1000

columns = [
    "Range",
    "Doppler",
    "Doppler Opt Peak",
    "Doppler Classic Peak",
    "Doppler Opt Avg",
    "Doppler Classic Avg",
    "SNR Opt",
    "SNR Classic",
]
df = pd.DataFrame(columns=columns, index=range(dataset.__len__()), data=None)

for index, (iq, (range_index, doppler_index)) in tqdm(enumerate(dataset)):
    model = OptimizeSNR(seq_len=dataset.radar.n_pulses, kernel=dataset.kernel).to(
        device=device
    )
    optimizer = optim.Adam(lr=1e-2, params=model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

    model.train()  # Set model to training mode
    losses = torch.zeros(n_iter)
    err = torch.zeros(n_iter)
    best_loss = Tensor([torch.inf]).to(device=device)
    best_rd_map = None

    nfft = 21
    for i in range(n_iter):
        optimizer.zero_grad()
        iq_fixed = model(iq)
        # iq_fixed = iq_fixed / iq_fixed.abs().max(dim=1)[0].unsqueeze(-1)
        rd_map_fixed = torch.fft.fftshift(
            torch.fft.fft(iq_fixed, n=nfft, dim=-2), dim=-2
        )
        max_idx = rd_map_fixed.abs().argmax()
        doppler_index_pred = max_idx // rd_map_fixed.shape[1]
        err[i] = doppler_index_pred.cpu() - doppler_index

        loss_snr, doppler_idx = snr(rd_map_fixed)
        p = model.weights_angle.atan().squeeze()
        a_spectrum = torch.fft.fftshift(torch.fft.fft(torch.exp(1j * p))).abs() ** 2
        noise = (a_spectrum.sum() - a_spectrum[10]) / (a_spectrum.shape[0] - 1)
        linearity = 10 * torch.log10(a_spectrum[10]) - 10 * torch.log10(noise)
        loss = loss_snr + 0.0 * linearity

        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(model)
            best_rd_map = rd_map_fixed

        if i < n_iter:
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            scheduler.step()
        losses[i] = loss

    df.loc[index, "Range"] = range_index.item()
    df.loc[index, "Doppler"] = doppler_index.item()

    # PRED
    best_rd_map = best_rd_map.abs() ** 2
    max_idx = best_rd_map.argmax()
    max_doppler_index = max_idx // best_rd_map.shape[1]
    max_range_index = max_idx - max_doppler_index * best_rd_map.shape[1]
    max_opt = best_rd_map.max()
    noise_opt = (best_rd_map[:, max_range_index].sum() - max_opt) / (
        best_rd_map.shape[0] - 1
    )
    snr_opt = 10 * torch.log10(max_opt / noise_opt)

    mag_opt, doppler_index_opt = circular_weigthed_mean(
        best_rd_map[:, max_range_index].to(device=device)
    )
    _, doppler_opt_avg = circular_weigthed_mean(best_rd_map[:, max_range_index])
    df.loc[index, "Doppler Opt Peak"] = max_doppler_index.item()
    df.loc[index, "Doppler Opt Avg"] = doppler_opt_avg.item()
    df.loc[index, "SNR Opt"] = snr_opt.cpu().item()

    after_mf = pulse_compression(
        iq.cpu().to(torch.complex128), dataset.radar.pulse_bb.to(torch.complex128)
    )
    rd_map = produce_rd_map(after_mf, n=nfft)
    rd_map_classic = rd_map.abs() ** 2
    max_idx = rd_map_classic.argmax()
    max_doppler_index = max_idx // rd_map_classic.shape[1]
    max_range_index = max_idx - max_doppler_index * rd_map_classic.shape[1]
    max_classic = rd_map_classic.max()
    noise_classic = (rd_map_classic[:, max_range_index].sum() - max_classic) / (
        rd_map_classic.shape[0] - 1
    )
    snr_classic = 10 * torch.log10(max_classic / noise_classic)

    mag_classic, doppler_index_classic = circular_weigthed_mean(
        rd_map_classic[:, max_range_index].to(device=device)
    )

    df.loc[index, "Doppler Classic Peak"] = max_doppler_index.item()
    df.loc[index, "Doppler Classic Avg"] = doppler_index_classic.item()
    df.loc[index, "SNR Classic"] = snr_classic.item()
    # print(f'SNR={-loss:.2f}')
# %% Statistics

df_diff_classic = df["Doppler Classic Avg"] - df["Doppler"]
df_diff_classic[df_diff_classic < -10] += 20
df_diff_classic[df_diff_classic > 10] -= 20

df_diff_opt = df["Doppler Opt Avg"] - df["Doppler"]
df_diff_opt[df_diff_opt < -10] += 20
df_diff_opt[df_diff_opt > 10] -= 20

plt.hist(
    df_diff_classic,
    bins=50,
    alpha=0.75,
    label=f"Classic - STD = {df_diff_classic.std(): .2f}",
)
plt.hist(
    df_diff_opt, bins=50, alpha=0.75, label=f"Opt - STD = {df_diff_opt.std(): .2f}"
)
plt.legend()
plt.title("Doppler Errors Distribution")
plt.xlabel("Doppler Index")
# %%
p = model.weights_angle.squeeze().atan().cpu().detach()
s = torch.fft.fftshift(torch.fft.fft(torch.exp(1j * p)))
plt.plot(s.abs() ** 2)
# %%
