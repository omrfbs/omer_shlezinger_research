import os
import pickle

import torch
from tqdm import tqdm
from radar_simulator.simulations.antenna import RxArray
from radar_simulator.simulations.radar import RadarSim
from radar_simulator.simulations.targets import Disc, PointTarget
from globals import device
from torch.utils.data import Dataset
from torch import Tensor


class IqDiscDataset(Dataset):

    def __init__(
        self,
        target_type: str,
        n_samples: int,
        r_min: float,
        r_max: float,
        v_max: float,
        v_min: float,
        rotation_max: float,
        radius: float,
        use_custom_rcs: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.valid_indexes = torch.arange(n_samples)
        _radial_velocities = torch.rand(n_samples, 1) * (v_max - v_min) + v_min
        _distances = torch.rand(n_samples, 1) * (r_max - r_min) + r_min
        _dir_velocities = torch.rand(size=(n_samples, 3)) - 0.5
        _dir_velocities /= torch.linalg.norm(_dir_velocities, dim=1).unsqueeze(-1)
        _dir_locations = torch.rand(size=(n_samples, 3))
        _dir_locations /= torch.linalg.norm(_dir_locations, dim=1).unsqueeze(-1)
        self.locations = _dir_locations * _distances
        self.rotation = torch.zeros(size=(n_samples, 3))
        # rounds per second
        self.rotation[torch.arange(n_samples), torch.randint(size=(n_samples,), high=3)] = (
            torch.rand(size=(n_samples,)) * rotation_max * 2 * torch.pi
        )
        self.radius = radius

        self.radar = RadarSim(
            fc=4e9,
            pri=2 * 1e-4,
            tp=1e-5,
            b=2e6,
            pt=1e6,
            n_pulses=21,
            antenna=RxArray(theta_tilt=0, height=0, n_rx_x=1, n_rx_y=1),
        )

        cos_alpha = (_dir_locations * _dir_velocities).sum(dim=1).unsqueeze(dim=-1)
        signed_radial_velocities = _radial_velocities * torch.sign(cos_alpha)
        self.velocities = _dir_velocities * _radial_velocities / cos_alpha.abs()
        self.doppler_index = 10 - 10 * signed_radial_velocities / self.radar.v_max
        self.range_index = ((_distances + self.radar.r_pulse / 2) / self.radar.dr_bb).round().to(torch.int32)
        self.iq = None
        self.target_type = target_type

        ampliudes = torch.arange(1, 22) * 21/231
        phases = Tensor([0, 0, 0, 0, 0.1, 0.4, -0.9, 2.41, -1.4, 1.31, 0.9, 0.5, 0.1, 0.05, 0.01, 0, 0, 0, 0, 0, 0])
        self.custom_rcs = torch.polar(ampliudes, phases) if use_custom_rcs else None
    @property
    def kernel(self) -> Tensor:
        return self.radar.pulse_bb

    @property
    def wavelength(self) -> float:
        return self.radar.wavelength

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int):
        i = self.valid_indexes[i]
        kwargs = dict(
                velocity=self.velocities[i, :],
                rotation=self.rotation[i, :],
                center=self.locations[i, :],
                wavelength=self.wavelength,
            )
        if self.iq is None:
            if self.target_type == 'disc':
                target_class = Disc
                kwargs['radius'] = self.radius
            elif self.target_type == 'point':
                target_class = PointTarget
            target = target_class(**kwargs)

            self.radar.targets = [target]
            self.radar.start_simulation(seed=i, custom_rcs=self.custom_rcs)
            iq = self.radar.fast_slow_channel_matrix.squeeze().T
        else:
            iq = self.iq[i, :, :]

        return iq, (self.range_index[i], self.doppler_index[i])

    def save(self, path: str) -> None:
        for i in tqdm(range(self.n_samples)):
            with open(os.path.join(path, f"{i}.pk"), "wb") as fp:
                pickle.dump(self[i], fp)

    def produce_iq(self) -> None:
        iq = [None] * self.n_samples
        for i in tqdm(range(self.n_samples), desc="Generating IQ"):
            iq[i] = self[i][0].unsqueeze(dim=0).to(torch.complex64)

        self.iq = torch.concat(iq, dim=0).to(device=device)
        self.doppler_index = self.doppler_index.to(device=device)
        self.range_index = self.range_index.to(device=device)
