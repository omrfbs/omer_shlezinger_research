# %%
import pickle
from globals import device
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from datasets import IqDiscDataset
from radar_simulator.signal_processing.signal_processing import (
    pulse_compression,
    produce_rd_map,
)
from utils import circular_weigthed_mean
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from typing import Tuple
from dataclasses import dataclass
from basic_nn import BasicNeuralNetwork, NeuralNetworkParams

# %% Dataset
# train_dataset = IqDiscDataset(
#     n_samples=10000,
#     r_min=0,
#     r_max=29000,
#     v_min=0,
#     v_max=90,
#     rotation_max=150,
#     radius=0.2,
#     target_type="disc",
#     # use_custom_rcs=True,
# )
# train_dataset.produce_iq()

with open("dataset_files/dataset_10000.pk", "rb") as fp:
    train_dataset = pickle.load(fp)


# %% Loss Function
def snr(rd_map: Tensor) -> Tensor:
    nfft = rd_map.shape[1]
    rd_map_energy = rd_map.abs() ** 2
    flattened = rd_map_energy.view(rd_map.shape[0], -1)  # shape (4, 5*6)
    max_val, max_global_index = flattened.max(dim=1)
    max_doppler_index = max_global_index // rd_map_energy.shape[2]
    max_range_index = max_global_index - max_doppler_index * rd_map_energy.shape[2]
    mag, peak_idx = circular_weigthed_mean(
        rd_map_energy[torch.arange(rd_map.shape[0], device=device), :, max_range_index]
    )
    idx = torch.round(peak_idx).to(torch.int) % nfft
    peak = rd_map_energy[
        torch.arange(rd_map.shape[0], device=device), idx, max_range_index
    ]
    doppler_noise = (
        rd_map_energy[
            torch.arange(rd_map.shape[0], device=device), :, max_range_index
        ].sum(dim=-1)
        - peak
    ) / (nfft - 1)
    return 10 * torch.log10(peak) - 10 * torch.log10(doppler_noise)


# %%
@dataclass
class DataInSteps:
    data: Tensor  # original IQ after Matched Filter
    w_mag: Tensor
    w_phase: Tensor

    def iq(self, step: int = -1) -> Tensor:
        return self.iq * w_mag[:, step] * torch.exp(-1j * w_phase[:, step])

    def rd_map(self, step: int = -1) -> Tensor:
        _iq = self.iq(step)
        return torch.fft.fftshift(torch.fft.fft(_iq, dim=0), dim=0)

    def loss(step: int = -1) -> Tensor:
        _rd_map = rd_map_in_step(step=step)
        return snr(_rd_map)


# %% Optimizer
class DeepUnfolding(nn.Module):

    def __init__(self, n_steps: int, seq_len: int, kernel: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.log_step_size_mag = nn.Parameter(-2 * torch.ones(seq_len, n_steps))
        self.log_step_size_phase = nn.Parameter(-2 * torch.ones(seq_len, n_steps))
        self.mf_coeffs = (
            kernel.repeat(self.seq_len, 1).unsqueeze(1).conj().to(device=device)
        )
        self.padding = int(kernel.shape[-1] / 2)
        self.iq_fixed: Tensor = None
        self.w_mag: Tensor = None
        self.w_phase: Tensor = None

    def produce_fixed_rd_map(self, x: Tensor, w_mag: Tensor, w_phase: Tensor) -> Tensor:
        phase = torch.atan(w_phase)
        # Force sum of magnitudes to be seq_len
        norm_mag = self.seq_len * torch.softmax(w_mag, dim=-1)
        # Fix IQ
        iq_fixed = norm_mag.unsqueeze(-1) * x * torch.exp(-1j * phase.unsqueeze(-1))
        # Produce RD-MAP from IQ (FFT along doppler dimention)
        rd_map_fixed = torch.fft.fftshift(torch.fft.fft(iq_fixed, dim=-2), dim=-2)

        self.iq_fixed = iq_fixed
        self.w_mag = w_mag.clone().detach()
        self.w_phase = w_phase.clone().detach()

        return rd_map_fixed

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        # Init Weights
        w_mag = torch.ones(batch_size, self.seq_len, device=device)
        w_phase = torch.rand(batch_size, self.seq_len, device=device)

        # Matched Filter
        x = x.to(torch.complex64)
        x = F.conv1d(x, self.mf_coeffs, padding=self.padding, groups=self.seq_len)

        for i in range(self.n_steps):
            with torch.enable_grad():
                w_phase_temp = w_phase.clone().detach().requires_grad_(True)
                w_mag_temp = w_mag.clone().detach().requires_grad_(True)

                rd_map_fixed = self.produce_fixed_rd_map(x, w_mag_temp, w_phase_temp)
                # Calculate loss
                loss = snr(rd_map_fixed)
                # Calculate Gradients w.r.t w_mag and w_phase
                grad_w_mag, grad_w_phase = torch.autograd.grad(
                    loss,
                    (w_mag_temp, w_phase_temp),
                    create_graph=True,
                    grad_outputs=torch.ones_like(loss),
                )
                # Apply the gradient decent step (maximize)
                lr_mag = torch.exp(self.log_step_size_mag[:, i] * math.log(10))
                lr_phase = torch.exp(self.log_step_size_phase[:, i] * math.log(10))
                w_mag = w_mag + lr_mag * grad_w_mag
                w_phase = w_phase + lr_phase * grad_w_phase

        rd_map_fixed = self.produce_fixed_rd_map(x, w_mag, w_phase)
        loss = snr(rd_map_fixed)
        return loss.mean()


# %% Train NN
nn_params = NeuralNetworkParams(
    n_epochs=200,
    batch_size=256,
    train_size=0.9,
    shuffle=True,
    lambda1=0,
    direction=-1,  # maximize
)

nn_model = DeepUnfolding(
    n_steps=10,
    seq_len=train_dataset.radar.n_pulses,
    kernel=train_dataset.kernel,
).to(device=device)

optimizer = optim.Adam(lr=1e-2, params=nn_model.parameters())


def criterion(x: Tensor, *args, **kwargs) -> Tensor:
    return x


model = BasicNeuralNetwork(
    nn_model=nn_model,
    criterion=criterion,
    optimizer=optimizer,
    dataset=train_dataset,
    params=nn_params,
    # scheduler=scheduler,
)
# %%
model.fit()
# %%
test_dataset = IqDiscDataset(
    n_samples=100,
    r_min=2000,
    r_max=20000,
    v_min=0,
    v_max=90,
    rotation_max=150,
    radius=0.2,
    target_type="disc",
    # use_custom_rcs=True,
)

idx = 0
iq, (range_index, doppler_index) = test_dataset[idx]
loss = model.nn_model(iq.unsqueeze(0))
iq_fixed = model.nn_model.iq_fixed.squeeze()
best_rd_map = torch.fft.fftshift(torch.fft.fft(iq_fixed, dim=-2), dim=-2)
best_rd_map = (best_rd_map.abs() ** 2).squeeze()
plt.pcolormesh(10 * torch.log10(best_rd_map.cpu().detach()))
# %%
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
best_rd_map = best_rd_map.detach().cpu()

after_mf = pulse_compression(
    iq.cpu().to(torch.complex128), test_dataset.radar.pulse_bb.to(torch.complex128)
)
rd_map = produce_rd_map(after_mf, n=21)
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

plt.plot(
    10
    * torch.log10(
        best_rd_map[:, max_range_index] / best_rd_map[:, max_range_index].max()
    )
)
plt.plot(
    10
    * torch.log10(
        rd_map_classic[:, max_range_index] / rd_map_classic[:, max_range_index].max()
    )
)
plt.axvline(x=doppler_index.to(device="cpu"), color="r", label="True Doppler")
# %%
