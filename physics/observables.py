"""
Physical observables for the Fermi-Hubbard model.

Computes measurable quantities from the NQS wave function:
- Spin structure factor S(q)
- Pairing correlations (d-wave superconductivity)
- Double occupancy <D>
- Momentum distribution n(k)
- Spin-spin correlation <S_i^z S_j^z>
"""

import torch
import numpy as np
from typing import Optional

from .hubbard import SquareLattice


class ObservableCalculator:
    """
    Compute physical observables from MCMC samples.

    All observables are computed as Monte Carlo averages over samples
    drawn from |psi(sigma)|^2.
    """

    def __init__(self, lattice: SquareLattice):
        self.lattice = lattice
        self.N = lattice.N
        self.Lx = lattice.Lx
        self.Ly = lattice.Ly

        # Precompute site coordinates
        self.coords = np.array([lattice.site_coords(i) for i in range(self.N)])

    def spin_z(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute S_i^z = (n_i_up - n_i_down) / 2 for each site.

        Args:
            configs: (B, N) tensor with values {0,1,2,3}
        Returns:
            sz: (B, N) tensor
        """
        up = ((configs == 1) | (configs == 3)).float()
        down = ((configs == 2) | (configs == 3)).float()
        return 0.5 * (up - down)

    def density(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute n_i = n_i_up + n_i_down for each site.

        Args:
            configs: (B, N) tensor
        Returns:
            n: (B, N) tensor
        """
        up = ((configs == 1) | (configs == 3)).float()
        down = ((configs == 2) | (configs == 3)).float()
        return up + down

    def double_occupancy(self, configs: torch.Tensor) -> float:
        """
        Compute average double occupancy: <D> = (1/N) sum_i <n_i_up n_i_down>

        Args:
            configs: (B, N) tensor
        Returns:
            scalar
        """
        d = (configs == 3).float()
        return d.mean().item()

    def spin_spin_correlation(self, configs: torch.Tensor) -> np.ndarray:
        """
        Compute spin-spin correlation matrix: C[i,j] = <S_i^z S_j^z>

        Args:
            configs: (B, N) tensor
        Returns:
            (N, N) numpy array
        """
        sz = self.spin_z(configs)  # (B, N)
        # <S_i^z S_j^z> = mean over samples of sz_i * sz_j
        corr = torch.einsum('bi,bj->ij', sz, sz) / sz.shape[0]
        return corr.cpu().numpy()

    def spin_structure_factor(self, configs: torch.Tensor) -> dict:
        """
        Compute spin structure factor S(q) = (1/N) sum_{i,j} <S_i.S_j> exp(iq.(r_i-r_j))

        For a square lattice, returns S(q) at high-symmetry points:
        - Gamma (0,0): ferromagnetic signal
        - M (pi,pi): antiferromagnetic signal
        - X (pi,0): stripe order signal

        Args:
            configs: (B, N) tensor
        Returns:
            dict with q-points and S(q) values
        """
        corr = self.spin_spin_correlation(configs)

        results = {}

        # High-symmetry q-points
        q_points = {
            'Gamma': (0, 0),
            'X': (np.pi, 0),
            'M': (np.pi, np.pi),
            'Y': (0, np.pi),
        }

        for name, (qx, qy) in q_points.items():
            sq = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    xi, yi = self.coords[i]
                    xj, yj = self.coords[j]
                    phase = qx * (xi - xj) + qy * (yi - yj)
                    sq += corr[i, j] * np.cos(phase)
            results[name] = sq / self.N

        # Full S(q) on the reciprocal lattice
        sq_full = np.zeros((self.Ly, self.Lx))
        for iqy in range(self.Ly):
            for iqx in range(self.Lx):
                qx = 2 * np.pi * iqx / self.Lx
                qy = 2 * np.pi * iqy / self.Ly
                sq = 0.0
                for i in range(self.N):
                    for j in range(self.N):
                        xi, yi = self.coords[i]
                        xj, yj = self.coords[j]
                        phase = qx * (xi - xj) + qy * (yi - yj)
                        sq += corr[i, j] * np.cos(phase)
                sq_full[iqy, iqx] = sq / self.N
        results['full'] = sq_full

        return results

    def density_density_correlation(self, configs: torch.Tensor) -> np.ndarray:
        """
        Compute density-density correlation: C_nn[i,j] = <n_i n_j> - <n_i><n_j>

        Args:
            configs: (B, N) tensor
        Returns:
            (N, N) numpy array (connected correlation)
        """
        n = self.density(configs)  # (B, N)
        nn_corr = torch.einsum('bi,bj->ij', n, n) / n.shape[0]
        n_mean = n.mean(dim=0)  # (N,)
        connected = nn_corr - torch.outer(n_mean, n_mean)
        return connected.cpu().numpy()

    def compute_all(self, configs: torch.Tensor) -> dict:
        """
        Compute all observables at once.

        Args:
            configs: (B, N) tensor

        Returns:
            dict with all observable values
        """
        results = {
            'double_occupancy': self.double_occupancy(configs),
            'spin_structure_factor': self.spin_structure_factor(configs),
            'spin_correlation': self.spin_spin_correlation(configs),
            'density_correlation': self.density_density_correlation(configs),
            'mean_density': self.density(configs).mean().item(),
            'mean_magnetization': self.spin_z(configs).mean().item(),
        }

        # Add spin structure factor at key points
        sf = results['spin_structure_factor']
        results['S_pi_pi'] = sf['M']  # AF signal
        results['S_0_0'] = sf['Gamma']  # FM signal

        return results
