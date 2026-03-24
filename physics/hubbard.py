"""
Fermi-Hubbard model Hamiltonian on a 2D square lattice.

H = -t sum_{<ij>,s} (c+_is c_js + h.c.) + U sum_i n_i_up n_i_down

This module provides:
- Lattice geometry and neighbor lists
- Configuration representation (second quantization occupation basis)
- Local energy computation with Jordan-Wigner signs
- Sparse Hamiltonian construction for exact diagonalization
"""

import numpy as np
import torch
from typing import Optional
from itertools import combinations


class SquareLattice:
    """2D square lattice with periodic or open boundary conditions."""

    def __init__(self, Lx: int, Ly: int, pbc: bool = True):
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly  # number of sites
        self.pbc = pbc

        # Build neighbor list
        self.neighbors = self._build_neighbors()
        # Snake-scan ordering for serialization
        self.snake_order = self._build_snake_order()
        self.inverse_snake = np.argsort(self.snake_order)

    def site_index(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to linear site index (row-major)."""
        return y * self.Lx + x

    def site_coords(self, idx: int) -> tuple[int, int]:
        """Convert linear site index to (x, y) coordinates."""
        return idx % self.Lx, idx // self.Lx

    def _build_neighbors(self) -> list[list[int]]:
        """Build nearest-neighbor list for each site."""
        neighbors = [[] for _ in range(self.N)]
        for y in range(self.Ly):
            for x in range(self.Lx):
                i = self.site_index(x, y)
                # Right neighbor
                if x + 1 < self.Lx:
                    j = self.site_index(x + 1, y)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                elif self.pbc and self.Lx > 1:
                    j = self.site_index(0, y)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                # Up neighbor
                if y + 1 < self.Ly:
                    j = self.site_index(x, y + 1)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
                elif self.pbc and self.Ly > 1:
                    j = self.site_index(x, 0)
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors

    def _build_snake_order(self) -> np.ndarray:
        """Build snake (boustrophedon) scan ordering."""
        order = []
        for y in range(self.Ly):
            if y % 2 == 0:
                order.extend(range(y * self.Lx, (y + 1) * self.Lx))
            else:
                order.extend(range((y + 1) * self.Lx - 1, y * self.Lx - 1, -1))
        return np.array(order)

    def get_neighbor_pairs(self) -> list[tuple[int, int]]:
        """Get list of unique nearest-neighbor pairs (i, j) with i < j."""
        pairs = set()
        for i, nbrs in enumerate(self.neighbors):
            for j in nbrs:
                pairs.add((min(i, j), max(i, j)))
        return sorted(pairs)

    def translation_group(self) -> list[np.ndarray]:
        """Get all lattice translation permutations (for PBC only)."""
        if not self.pbc:
            return [np.arange(self.N)]

        translations = []
        for dy in range(self.Ly):
            for dx in range(self.Lx):
                perm = np.zeros(self.N, dtype=int)
                for y in range(self.Ly):
                    for x in range(self.Lx):
                        i = self.site_index(x, y)
                        nx = (x + dx) % self.Lx
                        ny = (y + dy) % self.Ly
                        j = self.site_index(nx, ny)
                        perm[i] = j
                translations.append(perm)
        return translations


class FermiHubbardHamiltonian:
    """
    Fermi-Hubbard Hamiltonian on a square lattice.

    Configuration representation:
    Each site has state in {0, 1, 2, 3}:
      0 = empty
      1 = spin-up only
      2 = spin-down only
      3 = doubly occupied (up + down)

    Internally we also work with a split representation:
      up_config[i]   = 1 if site i has spin-up electron, 0 otherwise
      down_config[i] = 1 if site i has spin-down electron, 0 otherwise
    """

    def __init__(self, lattice: SquareLattice, t: float = 1.0, U: float = 4.0):
        self.lattice = lattice
        self.t = t
        self.U = U
        self.N = lattice.N
        self.neighbor_pairs = lattice.get_neighbor_pairs()

    def config_to_updown(self, config: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert {0,1,2,3} config to separate up/down occupation arrays."""
        up = np.zeros_like(config)
        down = np.zeros_like(config)
        up[config == 1] = 1
        up[config == 3] = 1
        down[config == 2] = 1
        down[config == 3] = 1
        return up, down

    def updown_to_config(self, up: np.ndarray, down: np.ndarray) -> np.ndarray:
        """Convert separate up/down arrays to {0,1,2,3} config."""
        return up + 2 * down

    def config_to_updown_torch(self, config: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert {0,1,2,3} config to separate up/down occupation tensors."""
        up = ((config == 1) | (config == 3)).long()
        down = ((config == 2) | (config == 3)).long()
        return up, down

    def updown_to_config_torch(self, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """Convert separate up/down tensors to {0,1,2,3} config."""
        return up + 2 * down

    def count_double_occupancy(self, config: np.ndarray) -> int:
        """Count number of doubly-occupied sites."""
        return int(np.sum(config == 3))

    def diagonal_energy(self, config: np.ndarray) -> float:
        """Compute diagonal (interaction) part of local energy: U * n_double."""
        return self.U * self.count_double_occupancy(config)

    def jordan_wigner_sign(self, occ: np.ndarray, site_from: int, site_to: int) -> int:
        """
        Compute Jordan-Wigner sign for hopping an electron from site_from to site_to.

        The sign is (-1)^(number of occupied orbitals strictly between
        min(site_from, site_to) and max(site_from, site_to) in the linear ordering).

        Args:
            occ: occupation array for one spin species, shape (N,)
            site_from: site the electron hops from (must be occupied)
            site_to: site the electron hops to (must be empty)

        Returns:
            +1 or -1
        """
        lo = min(site_from, site_to)
        hi = max(site_from, site_to)
        # Count occupied sites strictly between lo and hi
        n_between = int(np.sum(occ[lo + 1:hi]))
        return 1 - 2 * (n_between % 2)  # (-1)^n_between

    def get_connected_configs(self, config: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Get all configurations connected to 'config' by the Hamiltonian,
        along with the matrix element <config|H|config'>.

        Returns:
            List of (config', matrix_element) tuples.
            Does NOT include the diagonal term (same config).
        """
        up, down = self.config_to_updown(config)
        connected = []

        # Hopping terms: -t * (c+_is c_js + h.c.) for each neighbor pair and spin
        for i, j in self.neighbor_pairs:
            # Spin-up hopping: i -> j
            if up[i] == 1 and up[j] == 0:
                new_up = up.copy()
                new_up[i] = 0
                new_up[j] = 1
                sign = self.jordan_wigner_sign(up, i, j)
                new_config = self.updown_to_config(new_up, down)
                connected.append((new_config, -self.t * sign))

            # Spin-up hopping: j -> i
            if up[j] == 1 and up[i] == 0:
                new_up = up.copy()
                new_up[j] = 0
                new_up[i] = 1
                sign = self.jordan_wigner_sign(up, j, i)
                new_config = self.updown_to_config(new_up, down)
                connected.append((new_config, -self.t * sign))

            # Spin-down hopping: i -> j
            if down[i] == 1 and down[j] == 0:
                new_down = down.copy()
                new_down[i] = 0
                new_down[j] = 1
                sign = self.jordan_wigner_sign(down, i, j)
                new_config = self.updown_to_config(up, new_down)
                connected.append((new_config, -self.t * sign))

            # Spin-down hopping: j -> i
            if down[j] == 1 and down[i] == 0:
                new_down = down.copy()
                new_down[j] = 0
                new_down[i] = 1
                sign = self.jordan_wigner_sign(down, j, i)
                new_config = self.updown_to_config(up, new_down)
                connected.append((new_config, -self.t * sign))

        return connected

    def local_energy_single(self, config: np.ndarray, log_psi_fn) -> complex:
        """
        Compute local energy for a single configuration.

        E_loc(sigma) = <sigma|H|psi> / <sigma|psi>
                     = U * n_double + sum_{sigma'} H_{sigma,sigma'} * psi(sigma')/psi(sigma)

        Args:
            config: shape (N,), values in {0,1,2,3}
            log_psi_fn: callable that takes config array -> complex log(psi)

        Returns:
            Complex local energy
        """
        # Diagonal part
        e_loc = self.diagonal_energy(config) + 0j

        # Off-diagonal part
        log_psi_sigma = log_psi_fn(config)
        connected = self.get_connected_configs(config)
        for config_prime, h_element in connected:
            log_psi_sigma_prime = log_psi_fn(config_prime)
            # psi(sigma')/psi(sigma) = exp(log_psi(sigma') - log_psi(sigma))
            ratio = np.exp(log_psi_sigma_prime - log_psi_sigma)
            e_loc += h_element * ratio

        return e_loc

    def get_connected_configs_batched(
        self, configs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully vectorized batched version: get all connected configurations.

        Enumerates all possible single-electron hops across all neighbor pairs,
        both spin species, and both directions. Invalid hops (occupied target or
        empty source) are masked out.

        Args:
            configs: shape (B, N), values in {0,1,2,3}

        Returns:
            connected_configs: shape (B, max_connected, N)
            matrix_elements: shape (B, max_connected) - Hamiltonian matrix elements
            n_connected: shape (B,) - number of valid connected configs per sample
            diagonal_energy: shape (B,) - diagonal energy for each config
        """
        B, N = configs.shape
        device = configs.device

        up, down = self.config_to_updown_torch(configs)  # (B, N) each

        # Diagonal energy: U * number of doubly-occupied sites
        diag_e = self.U * (configs == 3).sum(dim=1).float()

        # Build neighbor pair tensors
        pairs = self.neighbor_pairs  # list of (i, j)
        n_pairs = len(pairs)
        pair_i = torch.tensor([p[0] for p in pairs], device=device)
        pair_j = torch.tensor([p[1] for p in pairs], device=device)

        # We enumerate 4 hops per pair: up i->j, up j->i, down i->j, down j->i
        # Total max_conn = n_pairs * 4
        max_conn = n_pairs * 4

        # Gather occupations at pair sites: (B, n_pairs)
        up_i = up[:, pair_i]    # (B, n_pairs)
        up_j = up[:, pair_j]    # (B, n_pairs)
        dn_i = down[:, pair_i]  # (B, n_pairs)
        dn_j = down[:, pair_j]  # (B, n_pairs)

        # Valid hop masks: source occupied AND target empty (for that spin)
        # Shape: (B, n_pairs) each
        valid_up_ij = (up_i == 1) & (up_j == 0)   # up hop i->j
        valid_up_ji = (up_j == 1) & (up_i == 0)   # up hop j->i
        valid_dn_ij = (dn_i == 1) & (dn_j == 0)   # down hop i->j
        valid_dn_ji = (dn_j == 1) & (dn_i == 0)   # down hop j->i

        # Stack all validity masks: (B, max_conn)
        valid = torch.cat([valid_up_ij, valid_up_ji, valid_dn_ij, valid_dn_ji], dim=1)

        # Build connected configs for ALL possible hops (valid or not)
        # For up hops: modify up occupations, recombine with down
        # For down hops: modify down occupations, recombine with up

        # Source and target site indices for each of the 4 hop types
        src_sites = torch.cat([pair_i, pair_j, pair_i, pair_j])  # (max_conn,)
        dst_sites = torch.cat([pair_j, pair_i, pair_j, pair_i])  # (max_conn,)
        is_up_hop = torch.cat([
            torch.ones(n_pairs * 2, dtype=torch.bool, device=device),
            torch.zeros(n_pairs * 2, dtype=torch.bool, device=device),
        ])  # (max_conn,)

        # Build connected configs: start from original, apply hops
        # (B, max_conn, N)
        conn_configs = configs.unsqueeze(1).expand(B, max_conn, N).clone()

        # Batch indices
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_conn)
        hop_idx = torch.arange(max_conn, device=device).unsqueeze(0).expand(B, max_conn)

        # Apply hops only where valid
        # For up spin: src -= 1 (1->0 or 3->2), dst += 1 (0->1 or 2->3)
        # For down spin: src -= 2 (2->0 or 3->1), dst += 2 (0->2 or 1->3)
        up_valid = valid & is_up_hop.unsqueeze(0)    # (B, max_conn)
        dn_valid = valid & (~is_up_hop).unsqueeze(0) # (B, max_conn)

        # Apply up hops
        if up_valid.any():
            uv_b, uv_h = up_valid.nonzero(as_tuple=True)
            uv_src = src_sites[uv_h]
            uv_dst = dst_sites[uv_h]
            conn_configs[uv_b, uv_h, uv_src] -= 1  # remove up from source
            conn_configs[uv_b, uv_h, uv_dst] += 1  # add up to dest

        # Apply down hops
        if dn_valid.any():
            dv_b, dv_h = dn_valid.nonzero(as_tuple=True)
            dv_src = src_sites[dv_h]
            dv_dst = dst_sites[dv_h]
            conn_configs[dv_b, dv_h, dv_src] -= 2  # remove down from source
            conn_configs[dv_b, dv_h, dv_dst] += 2  # add down to dest

        # Compute Jordan-Wigner signs vectorized
        # sign = (-1)^(number of occupied orbitals strictly between src and dst)
        # For up hops, count up occupations between; for down hops, count down.
        lo = torch.min(src_sites, dst_sites)  # (max_conn,)
        hi = torch.max(src_sites, dst_sites)  # (max_conn,)

        # Build range mask: for each hop, which sites are strictly between lo and hi
        site_range = torch.arange(N, device=device)  # (N,)
        # (max_conn, N): True if site is strictly between lo[h] and hi[h]
        between_mask = (site_range.unsqueeze(0) > lo.unsqueeze(1)) & \
                       (site_range.unsqueeze(0) < hi.unsqueeze(1))  # (max_conn, N)

        # Count occupied sites between lo and hi for the relevant spin
        # up hops: count up electrons between
        # down hops: count down electrons between
        # (B, max_conn)
        up_between = (up.unsqueeze(1) * between_mask.unsqueeze(0)).sum(dim=2)    # (B, max_conn)
        dn_between = (down.unsqueeze(1) * between_mask.unsqueeze(0)).sum(dim=2)  # (B, max_conn)

        # Select appropriate count based on spin
        n_between = torch.where(is_up_hop.unsqueeze(0), up_between, dn_between)  # (B, max_conn)
        signs = 1 - 2 * (n_between % 2)  # (B, max_conn), values +1 or -1

        # Matrix elements: -t * sign (only for valid hops)
        mat_elements = (-self.t * signs.float()) * valid.float()  # (B, max_conn)

        # Count valid connections per sample
        n_conn = valid.sum(dim=1)  # (B,)

        return conn_configs, mat_elements, n_conn, diag_e

    def compute_local_energy_batch(
        self,
        model: torch.nn.Module,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local energies for a batch of configurations using a neural network.

        Args:
            model: NQS model that takes configs (B, N) -> log_psi (B,) complex
            configs: (B, N) tensor, values in {0,1,2,3}

        Returns:
            e_loc: (B,) complex tensor of local energies
        """
        B, N = configs.shape
        device = configs.device

        # Get connected configurations
        conn_configs, mat_elements, n_conn, diag_e = self.get_connected_configs_batched(configs)
        max_conn = conn_configs.shape[1]

        # Validity mask: mat_elements != 0 means valid hop
        conn_mask = mat_elements != 0  # (B, max_conn)

        # Compute log psi for original configs
        with torch.no_grad():
            log_psi_orig = model(configs)  # (B,) complex

        # Flatten valid connections
        flat_configs = conn_configs.reshape(B * max_conn, N)
        flat_mask = conn_mask.reshape(B * max_conn)

        # Only evaluate non-zero connections
        valid_indices = flat_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            valid_configs = flat_configs[valid_indices]
            with torch.no_grad():
                log_psi_conn = model(valid_configs)  # (n_valid,) complex

            # Scatter back
            all_log_psi = torch.zeros(B * max_conn, dtype=log_psi_conn.dtype, device=device)
            all_log_psi[valid_indices] = log_psi_conn
            all_log_psi = all_log_psi.reshape(B, max_conn)
        else:
            all_log_psi = torch.zeros(B, max_conn, dtype=torch.complex64, device=device)

        # Compute psi ratios: exp(log_psi(sigma') - log_psi(sigma))
        log_ratios = all_log_psi - log_psi_orig.unsqueeze(1)
        psi_ratios = torch.exp(log_ratios)  # (B, max_conn)

        # Zero out invalid connections
        psi_ratios = psi_ratios * conn_mask.float()

        # E_loc = U * n_double + sum_connected h_element * psi_ratio
        mat_elements_complex = mat_elements.to(psi_ratios.dtype)
        e_loc = diag_e.to(psi_ratios.dtype) + (mat_elements_complex * psi_ratios).sum(dim=1)

        return e_loc


def generate_random_configs(
    n_samples: int,
    n_sites: int,
    n_up: int,
    n_down: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate random valid fermion configurations at fixed particle numbers.

    Args:
        n_samples: number of configurations to generate
        n_sites: number of lattice sites
        n_up: number of spin-up electrons
        n_down: number of spin-down electrons
        rng: numpy random generator (optional)

    Returns:
        configs: (n_samples, n_sites) array with values in {0,1,2,3}
    """
    if rng is None:
        rng = np.random.default_rng()

    configs = np.zeros((n_samples, n_sites), dtype=np.int64)
    for i in range(n_samples):
        # Place spin-up electrons
        up_sites = rng.choice(n_sites, n_up, replace=False)
        up = np.zeros(n_sites, dtype=np.int64)
        up[up_sites] = 1

        # Place spin-down electrons
        down_sites = rng.choice(n_sites, n_down, replace=False)
        down = np.zeros(n_sites, dtype=np.int64)
        down[down_sites] = 1

        configs[i] = up + 2 * down

    return configs
