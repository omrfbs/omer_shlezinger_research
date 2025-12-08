# %%
from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from datasets import IqDiscDataset
from basic_nn import BasicNeuralNetwork, NeuralNetworkParams
from signal_processing.signal_processing import cell_under_test_to_neigbhors

from globals import device



#%%
train_dataset = IqDiscDataset(
    n_samples=100000,
    r_min=0,
    r_max=29000,
    v_min=0,
    v_max=90,
    rotation_max=50,
    radius=0.2,
    target_type='point',
    use_custom_rcs=True,
)
train_dataset.produce_iq()

# with open(r'/nas_users/a50710/repos/radar_simulator/datasets/100k_all_ranges.pk', 'rb') as fp:
#     train_dataset = pickle.load(fp)

# valid_indexes = ((train_dataset.range_index > 19) & (train_dataset.range_index < 380)).any(dim=1)
# train_dataset.valid_indexes = torch.arange(train_dataset.n_samples, device=device)[valid_indexes]
# train_dataset.n_samples = train_dataset.valid_indexes.shape[0]
# train_dataset.valid_indexes = torch.arange(train_dataset.n_samples)


class ComplexSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, num_heads: int = 1):
        """
        embed_dim: the original complex embedding size
        num_heads: number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        positional_encoding = [torch.cos(torch.pi/2 * (1 - i % 2) - torch.arange(seq_len)/10000**(2*i/embed_dim)).unsqueeze(-1) for i in range(embed_dim)]
        self.positional_encoding = 0.5 * torch.concat(positional_encoding, dim=-1).unsqueeze(0).to(device=device)
        self.attn_full = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=0.5)
        self.norm = nn.LayerNorm(self.embed_dim)  # normalize both channels
        # self.fc = nn.Linear(embed_dim, embed_dim)
    def forward(self, x: Tensor):
        x += self.positional_encoding
        attn_output_full, _ = self.attn_full(x, x, x, average_attn_weights=False)
        # attn_output_full = self.norm(attn_output_full + x)
        # attn_output_full = self.fc(attn_output_full)
        return attn_output_full


class SpectrumCompression(nn.Module):

    def __init__(self, seq_len: int,
                embed_dim: int,
                kernel: Tensor,
                num_heads: int = 1,
                latent_dim: int= None,
                apply_fft: bool = True,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.apply_fft = apply_fft
        self.seq_len = seq_len
        latent_dim = embed_dim * 2 if latent_dim is None else latent_dim
        self.attn_angle = ComplexSelfAttention(latent_dim,  seq_len, num_heads)
        self.attn_mag = ComplexSelfAttention(latent_dim,  seq_len, num_heads)
        self.encoder = nn.Linear(embed_dim * 2, latent_dim)
        self.decoder = nn.Linear(latent_dim, embed_dim * 2)
        self.fc_mag = nn.Linear(latent_dim, 1)
        self.fc_angle = nn.Linear(latent_dim, 1)
        self.mf_coeffs = kernel.repeat(self.seq_len, 1).unsqueeze(1).conj().to(device=device)
        self.padding = int(kernel.shape[-1] / 2)

    def real_to_complex(self, x: Tensor)->Tensor:
        tensor_shape = list(x.shape)
        tensor_shape[-1] = tensor_shape[-1] // 2
        tensor_shape.append(2)
        return torch.view_as_complex(x.reshape(tensor_shape))

    def complex_to_real(self, x: Tensor)->Tensor:
        return torch.view_as_real(x).flatten(start_dim=-2)

    def normalize(self, x: Tensor)->Tensor:
        x /= torch.std(x, dim=(1, 2), keepdim=True)
        return x

    def forward(self, x: Tensor):
        x_complex = x.to(torch.complex64)
        x_complex = F.conv1d(x_complex, self.mf_coeffs, padding=self.padding, groups=self.seq_len)
        x_complex = self.normalize(x_complex)
        x_real = self.complex_to_real(x_complex)
        x_mag = self.attn_mag(x_real)
        x_mag = self.fc_mag(x_mag)
        x_angle = self.attn_angle(x_real)
        x_angle = self.fc_angle(x_angle)
        x_out = x_complex * torch.exp(-1j * x_angle)
        # x = self.real_to_complex(x)
        if self.apply_fft:
            x_out = torch.fft.fftshift(torch.fft.fft(x_out, dim=-2), dim=-2)
        return x_out

class SimpleNet(nn.Module):

    def __init__(self, seq_len: int, kernel: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.weights_mag = nn.Parameter(torch.ones(1, seq_len, 1))
        self.weights_angle = nn.Parameter(torch.ones(1, seq_len, 1))
        self.mf_coeffs = kernel.repeat(self.seq_len, 1).unsqueeze(1).conj().to(device=device)
        self.padding = int(kernel.shape[-1] / 2)

    def real_to_complex(self, x: Tensor)->Tensor:
        tensor_shape = list(x.shape)
        tensor_shape[-1] = tensor_shape[-1] // 2
        tensor_shape.append(2)
        return torch.view_as_complex(x.reshape(tensor_shape))

    def complex_to_real(self, x: Tensor)->Tensor:
        return torch.view_as_real(x).flatten(start_dim=-2)

    def normalize(self, x: Tensor)->Tensor:
        x /= torch.std(x, dim=(1, 2), keepdim=True)
        return x

    def forward(self, x: Tensor):
        x = x.to(torch.complex64)
        x = F.conv1d(x, self.mf_coeffs, padding=self.padding, groups=self.seq_len)
        x = self.normalize(x)
        x = self.weights_mag * x * torch.exp(-1j * self.weights_angle) #+ 1j * x.imag * self.weights_imag
        x = torch.fft.fftshift(torch.fft.fft(x, dim=-2), dim=-2)
        return x
# %%
nn_params = NeuralNetworkParams(
    n_epochs=100,
    batch_size=128,
    train_size=0.9,
    shuffle=True,
    lambda1=0,
    direction=-1,  # maximize
)

nn_model = SpectrumCompression(
    seq_len=train_dataset.radar.n_pulses,
    embed_dim=train_dataset[0][0].shape[-1],
    num_heads=1,
    kernel=train_dataset.kernel,
    latent_dim=None,
    apply_fft=True,
).to(device=device)

# nn_model = SimpleNet(seq_len=train_dataset.radar.n_pulses, kernel=train_dataset.kernel).to(device=device)
optimizer = optim.Adam(lr=1e-4, params=nn_model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def rd_map_max_idx(rd_map: Tensor, indexes: List[Tensor]) -> Tensor:
    # n_r = 5
    start_index = 10
    rd_map = rd_map[:, :, start_index:]
    rd_map = rd_map[:, :, :388]
    doppler_indexes = indexes[1].squeeze().round().to(torch.int32)
    range_indexes = indexes[0].squeeze() - start_index
    peak = rd_map[torch.arange(rd_map.shape[0], device=device), doppler_indexes, range_indexes]

    noise_range = (rd_map[torch.arange(rd_map.shape[0], device=device), doppler_indexes, :]).sum(dim=-1) - peak
    noise_range /= rd_map.shape[2] - 1
    noise_range_db = 10 * torch.log10(noise_range + 1e-8)

    noise_doppler = (rd_map[torch.arange(rd_map.shape[0], device=device), :, range_indexes]).sum(dim=-1) - peak
    noise_doppler /= rd_map.shape[1] - 1
    noise_doppler_db = 10 * torch.log10(noise_doppler + 1e-8)
    peak_db = 10*torch.log10(peak + 1e-8)
    snr_doppler = peak_db - noise_doppler_db
    snr_range = peak_db - noise_range_db
    spiky = 10 * torch.log10(rd_map[torch.arange(rd_map.shape[0], device=device), :, range_indexes]+ 1e-8)
    spiky = spiky.diff(dim=1).abs().mean(dim=1)
    return snr_doppler.mean() #+ snr_range.mean()) #- spiky.mean()

def snr(rd_map: Tensor)->Tensor:
    peak = rd_map.max()
    noise = (rd_map.sum() - peak)/(rd_map.numel()-1)
    return 10*torch.log10(peak + 1e-8) - 10*torch.log10(noise + 1e-8)

def snr_local(rd_maps: Tensor, indexes: List[Tensor]) -> Tensor:
    start_index = 10
    rd_maps = rd_maps[:, :, start_index:]
    doppler_indexes = indexes[1].squeeze().round().to(torch.int32)
    range_indexes = indexes[0].squeeze() - start_index
    n_r = 5
    n_d = 2

    local_snr = Tensor([0]).to(device=device)
    global_snr = Tensor([0]).to(device=device)
    for i, rd_map in enumerate(rd_maps):
        i_r = range_indexes[i]
        i_d = doppler_indexes[i]
        peak = rd_map[i_d, i_r]
        peak_db = 10 * torch.log10(peak + 1e-8)
        i_d_start = max(i_d-n_d, 0)
        i_d_end = min(i_d+n_d + 1, rd_map.shape[0])
        i_r_start = max(i_r-n_r, 0)
        i_r_end = min(i_r+n_r + 1, rd_map.shape[1])
        noise_local = (rd_map[i_d_start:i_d_end, i_r_start:i_r_end]).sum() - peak
        noise_local /= (i_r_end - i_r_start) * (i_d_end - i_d_start)
        local_snr += peak_db - 10 * torch.log10(noise_local + 1e-8)
    local_snr /= rd_maps.shape[0]

    # peak = rd_maps[torch.arange(rd_maps.shape[0], device=device), doppler_indexes, range_indexes]
    # noise_doppler = (rd_maps[torch.arange(rd_maps.shape[0], device=device), :, range_indexes]).sum(dim=-1) - peak
    # noise_doppler /= rd_maps.shape[1] - 1
    # peak_db = 10*torch.log10(peak + 1e-8)
    # snr_doppler = peak_db - 10 * torch.log10(noise_doppler + 1e-8)

    return local_snr

def snr_local_and_global(rd_map: Tensor, indexes: List[Tensor])->Tensor:
    patch_size = 10
    n_ignore = 4
    local_snrs = cell_under_test_to_neigbhors(rd_map, patch_size, n_ignore, is_1d=True).squeeze()
    doppler_indexes = indexes[1].squeeze().round().to(torch.int32)
    range_indexes = indexes[0].squeeze()
    target_snr = local_snrs[torch.arange(local_snrs.shape[0], device=device), doppler_indexes, range_indexes]
    global_snr = local_snrs.sum(dim=(1, 2)) - target_snr
    global_snr = global_snr / (local_snrs.shape[1]*local_snrs.shape[2] - 1)

    return (10 * torch.log10(target_snr + 1e-8) - 10 * torch.log10(global_snr + 1e-8)).mean()

snr_local_compiled = torch.compile(snr_local)

def velocity_estimation(rd_map: Tensor, indexes: List[Tensor]) -> Tensor:
    doppler_indexes = indexes[1].squeeze()
    range_indexes = indexes[0].squeeze()
    spectrum = rd_map[torch.arange(rd_map.shape[0], device=device), :, range_indexes] ** 2
    index_estimation = torch.mean(torch.arange(spectrum.shape[1], device=device) * spectrum / spectrum.sum(dim=1).unsqueeze(-1), dim=1)
    return -torch.abs(index_estimation - doppler_indexes).mean()


def patch_snr(rd_map: Tensor, indexes: List[Tensor])->Tensor:
    doppler_index = indexes[1].squeeze()
    peak = rd_map[torch.arange(rd_map.shape[0], device=device), doppler_index.round().to(int), range_index]
    noise = (rd_map.sum(dim=(1, 2)) - peak)/(rd_map.shape[1]*rd_map.shape[2])
    snr = 10 * torch.log10(peak + 1e-8) - 10 * torch.log10(noise + 1e-8)
    return snr.mean()

def min_range_max_harmony(iq: Tensor, indexes: List[Tensor])->Tensor:
    start_index = 10
    iq = iq[:, :, start_index:]
    iq = iq[:, :, :388]

    doppler_indexes = indexes[1].squeeze() - 10
    range_indexes = indexes[0].squeeze() - start_index

    slow_in_target_range = iq[torch.arange(iq.shape[0], device=device), :, range_indexes]
    slow_norm = slow_in_target_range * torch.exp(-1j * slow_in_target_range[:, [0]].angle()) / slow_in_target_range.norm(dim=1, keepdim=True)
    ref = (1/math.sqrt(slow_norm.shape[1]))*torch.exp(1j * 2 * torch.pi * torch.outer(doppler_indexes, torch.arange(slow_norm.shape[1], device=device) / slow_norm.shape[1]))
    mse = ((slow_norm - ref).abs()).mean()
    return mse

def criterion(rd_map: Tensor, indexes: List[Tensor], alpha: float = 0.1) -> Tensor:
    rd_map = rd_map.abs() ** 2
    loss = rd_map_max_idx(rd_map)
    # loss2 = rd_map_max_idx(rd_map, indexes)
    # loss = min_range_max_harmony(rd_map, indexes)
    return loss#0.5*(loss1 + loss2)

# %%
# train_dataset.doppler_index = 20 - train_dataset.doppler_index
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
import matplotlib.pyplot as plt

from signal_processing.signal_processing import cell_under_test_to_neigbhors, produce_rd_map, pulse_compression

test_dataset = IqDiscDataset(
    n_samples=100,
    r_min=2000,
    r_max=20000,
    v_min=0,
    v_max=90,
    rotation_max=50,
    radius=0.2,
    target_type='point',
    use_custom_rcs=True,
)
# test_dataset.custom_rcs = None
test_dataset.valid_indexes =torch.arange(test_dataset.n_samples)
idx = 0

iq, indxs = test_dataset[idx]
iq = iq.to(device=device)
rotation = test_dataset.rotation[idx][test_dataset.rotation[idx] > 0] / (2 * torch.pi)
res = model.nn_model(iq.unsqueeze(0)).detach().squeeze().cpu()
# res = torch.fft.fftshift(torch.fft.fft(res, dim=0), dim=0)
res_abs = res.abs()
res_iq = torch.fft.ifft(torch.fft.ifftshift(res, dim=0), dim=0)
# res_iq = iq
res_db = 20 * torch.log10(res_abs)
plt.pcolormesh(res_db)
plt.colorbar()
doppler_index = indxs[1].cpu()
doppler_index_round = doppler_index.round().to(torch.int32)
range_index = indxs[0]
# range_index = [Tensor([19]).to(torch.int)]
peak = res_abs[doppler_index_round, range_index]** 2
# peak = res_abs.max() ** 2
noise = torch.sum(res_abs[:, range_index[0]] ** 2) - peak
noise /= res_abs.shape[0] - 1
snr_res = 10 * torch.log10(peak / noise).item()
plt.title(f"r_i={range_index[0].item()}, r_d={doppler_index.item():.2f}, rotation={rotation.item():.2f}Hz, SNR={snr_res:.0f}")

flat_index = torch.argmax(res_abs)

# Convert back to row and col
row = flat_index // res_abs.size(1)
col = flat_index % res_abs.size(1)

print(f"col={col.item()}, row={row.item()}")
fig, ax = plt.subplots()
ax.plot(res_db[:, range_index])
# %%
after_mf = pulse_compression(iq.cpu().to(torch.complex128), test_dataset.radar.pulse_bb.to(torch.complex128))
rd_map = produce_rd_map(after_mf)
rd_map_abs = rd_map.abs()
rd_map_db = 20 * torch.log10(rd_map_abs)
plt.pcolormesh(rd_map_db)
peak = rd_map_abs.max() ** 2
noise = torch.var(rd_map[:, 300:])
snr_rd_map = 10 * torch.log10(peak / noise)
plt.title(f"SNR = {snr_rd_map:.0f}")
plt.colorbar()
# Convert back to row and col
flat_index = torch.argmax(rd_map_abs)
row = flat_index // rd_map_abs.size(1)
col = flat_index % res_abs.size(1)

print(f"row={row.item()}, col={col.item()}")
fig, ax = plt.subplots()
ax.plot(rd_map_db[:, range_index]-rd_map_db[:, range_index].max(), label='classic')
ax.plot(res_db[:, range_index]-res_db[:, range_index].max(), label='NN')
plt.legend()
# %%
after_mf /= torch.max(after_mf.abs(), dim=1, keepdim=True).values
rd_map = produce_rd_map(after_mf)
rd_map_abs = rd_map.abs()
rd_map_db = 20 * torch.log10(rd_map_abs)
plt.pcolormesh(rd_map_db)
peak = rd_map_abs.max() ** 2
noise = torch.var(rd_map[:, 300:])
snr_rd_map = 10 * torch.log10(peak / noise)
plt.title(f"SNR = {snr_rd_map:.0f}")
plt.colorbar()
# Convert back to row and col
flat_index = torch.argmax(rd_map_abs)
row = flat_index // rd_map_abs.size(1)
col = flat_index % res_abs.size(1)

print(f"row={row.item()}, col={col.item()}")
fig, ax = plt.subplots()
ax.plot(rd_map_db[:, range_index])
# %%
