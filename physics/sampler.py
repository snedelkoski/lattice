"""
Metropolis-Hastings MCMC sampler for fermionic lattice configurations.

Samples from |psi(sigma)|^2 using single-electron hop proposals that
preserve particle number conservation.
"""

import torch
import numpy as np
from typing import Optional

from .hubbard import SquareLattice


class MetropolisSampler:
    """
    Metropolis-Hastings sampler with fermion-hop proposals.

    Runs multiple parallel MCMC chains on GPU. Each proposal picks a random
    occupied site and a random neighboring empty site, and proposes moving
    the electron there. This preserves N_up and N_down.
    """

    def __init__(
        self,
        lattice: SquareLattice,
        n_chains: int = 256,
        n_sweeps: int = 10,
        n_thermalize: int = 200,
        device: str = "cuda",
    ):
        self.lattice = lattice
        self.N = lattice.N
        self.n_chains = n_chains
        self.n_sweeps = n_sweeps
        self.n_thermalize = n_thermalize
        self.device = device

        # Precompute neighbor list as tensor
        max_neighbors = max(len(nbrs) for nbrs in lattice.neighbors)
        self.neighbor_table = torch.full(
            (self.N, max_neighbors), -1, dtype=torch.long, device=device
        )
        self.n_neighbors = torch.zeros(self.N, dtype=torch.long, device=device)
        for i, nbrs in enumerate(lattice.neighbors):
            self.n_neighbors[i] = len(nbrs)
            for j, nbr in enumerate(nbrs):
                self.neighbor_table[i, j] = nbr

        # Chain states
        self.configs = None  # (n_chains, N)
        self.log_psi = None  # (n_chains,) complex
        self.n_accepted = 0
        self.n_proposed = 0
        self.last_sweep_acceptance_rate = 0.0
        self.last_sample_acceptance_rate = 0.0

    def initialize_chains(
        self,
        n_up: int,
        n_down: int,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize chains with random valid configurations."""
        if rng is None:
            rng = np.random.default_rng()

        configs = np.zeros((self.n_chains, self.N), dtype=np.int64)
        for i in range(self.n_chains):
            up_sites = rng.choice(self.N, n_up, replace=False)
            down_sites = rng.choice(self.N, n_down, replace=False)
            up = np.zeros(self.N, dtype=np.int64)
            down = np.zeros(self.N, dtype=np.int64)
            up[up_sites] = 1
            down[down_sites] = 1
            configs[i] = up + 2 * down

        self.configs = torch.from_numpy(configs).to(self.device)
        self.log_psi = None
        self.n_accepted = 0
        self.n_proposed = 0
        self.last_sweep_acceptance_rate = 0.0
        self.last_sample_acceptance_rate = 0.0

    @torch.no_grad()
    def _propose_hop(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propose single-electron hops for all chains.

        For each chain:
        1. Randomly pick spin species (up or down)
        2. Find an occupied site of that species
        3. Pick a random neighbor of that site
        4. If neighbor is empty (for that species), propose the hop

        Returns:
            proposed_configs: (n_chains, N)
            valid_mask: (n_chains,) bool - which proposals are valid
            hop_info: tuple of tensors for bookkeeping
        """
        B = self.n_chains
        configs = self.configs

        # Decompose into up/down
        up = ((configs == 1) | (configs == 3)).long()
        down = ((configs == 2) | (configs == 3)).long()

        # Pick spin species randomly
        spin_choice = torch.randint(0, 2, (B,), device=self.device)  # 0=up, 1=down

        # Get occupation for chosen spin
        occ = torch.where(spin_choice.unsqueeze(1) == 0, up, down)  # (B, N)

        # Find occupied sites
        # For each chain, pick a random occupied site
        occupied_counts = occ.sum(dim=1)  # (B,)

        # Generate random index into occupied sites
        rand_idx = torch.rand(B, device=self.device)
        rand_idx = (rand_idx * occupied_counts.float()).long()
        rand_idx = rand_idx.clamp(max=occupied_counts - 1)

        # Convert to actual site indices using cumsum trick
        cumsum = occ.cumsum(dim=1)  # (B, N)
        # site_from[b] = first site where cumsum > rand_idx[b]
        target = rand_idx + 1  # 1-indexed
        site_mask = cumsum >= target.unsqueeze(1)
        # First True position in each row
        site_from = site_mask.long().argmax(dim=1)  # (B,)

        # Pick a random neighbor of site_from
        max_nbrs = self.neighbor_table.shape[1]
        n_nbrs = self.n_neighbors[site_from]  # (B,)
        rand_nbr_idx = torch.rand(B, device=self.device)
        rand_nbr_idx = (rand_nbr_idx * n_nbrs.float()).long()
        rand_nbr_idx = rand_nbr_idx.clamp(max=n_nbrs - 1)
        site_to = self.neighbor_table[site_from, rand_nbr_idx]  # (B,)

        # Check if target site is empty (for the chosen spin)
        occ_at_target = occ[torch.arange(B, device=self.device), site_to]
        valid = occ_at_target == 0  # Valid if target is empty

        # Build proposed configs (fully vectorized, no Python loop)
        proposed = configs.clone()

        if valid.any():
            valid_idx = valid.nonzero(as_tuple=True)[0]
            sf = site_from[valid_idx]
            st = site_to[valid_idx]
            sc = spin_choice[valid_idx]

            # Current values at source and target
            val_sf = proposed[valid_idx, sf]
            val_st = proposed[valid_idx, st]

            is_up = (sc == 0)
            is_down = ~is_up

            # UP spin hops: remove up from sf, add up to st
            # sf: 1->0, 3->2 (remove up bit)
            # st: 0->1, 2->3 (add up bit)
            if is_up.any():
                up_idx = valid_idx[is_up]
                up_sf = sf[is_up]
                up_st = st[is_up]
                proposed[up_idx, up_sf] = proposed[up_idx, up_sf] - 1  # remove up: 1->0 or 3->2
                proposed[up_idx, up_st] = proposed[up_idx, up_st] + 1  # add up: 0->1 or 2->3

            # DOWN spin hops: remove down from sf, add down to st
            # sf: 2->0, 3->1 (remove down bit)
            # st: 0->2, 1->3 (add down bit)
            if is_down.any():
                dn_idx = valid_idx[is_down]
                dn_sf = sf[is_down]
                dn_st = st[is_down]
                proposed[dn_idx, dn_sf] = proposed[dn_idx, dn_sf] - 2  # remove down: 2->0 or 3->1
                proposed[dn_idx, dn_st] = proposed[dn_idx, dn_st] + 2  # add down: 0->2 or 1->3

        return proposed, valid, (site_from, site_to, spin_choice)

    @torch.no_grad()
    def sweep(self, model: torch.nn.Module) -> float:
        """
        Perform one MCMC sweep (N proposals per chain).

        Returns acceptance rate.
        """
        n_accepted = 0
        n_proposed = 0

        for _ in range(self.N):
            proposed, valid, _ = self._propose_hop()

            if not valid.any():
                n_proposed += self.n_chains
                continue

            # Evaluate log psi for proposed configs (only valid ones)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            valid_proposed = proposed[valid_idx]

            log_psi_proposed = model(valid_proposed)

            # Ensure we have log_psi for current configs
            if self.log_psi is None:
                self.log_psi = model(self.configs)

            log_psi_current = self.log_psi[valid_idx]

            # Acceptance probability: |psi(proposed)|^2 / |psi(current)|^2
            # = exp(2 * Re(log_psi_proposed - log_psi_current))
            log_ratio = 2.0 * (log_psi_proposed - log_psi_current).real
            log_uniform = torch.log(torch.rand(len(valid_idx), device=self.device))
            accept = log_uniform < log_ratio

            # Update accepted chains
            accepted_idx = valid_idx[accept]
            if len(accepted_idx) > 0:
                self.configs[accepted_idx] = proposed[accepted_idx]
                self.log_psi[accepted_idx] = log_psi_proposed[accept]

            n_accepted += accept.sum().item()
            n_proposed += self.n_chains

        self.n_accepted += n_accepted
        self.n_proposed += n_proposed
        self.last_sweep_acceptance_rate = n_accepted / max(n_proposed, 1)
        return self.last_sweep_acceptance_rate

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples from |psi(sigma)|^2.

        If chains are not initialized, this will raise an error.
        Performs n_sweeps sweeps between returned samples.

        Returns:
            configs: (n_chains, N) current chain states
        """
        if self.configs is None:
            raise RuntimeError("Chains not initialized. Call initialize_chains() first.")

        # Thermalilze if log_psi is None (first call)
        if self.log_psi is None:
            model.eval()
            self.log_psi = model(self.configs)
            for _ in range(self.n_thermalize):
                self.sweep(model)
        else:
            # Model parameters changed after optimizer updates; refresh cached
            # log psi on the current chain states before proposing new moves.
            self.log_psi = model(self.configs)

        # Decorrelation sweeps
        sample_accept_rates = []
        for _ in range(self.n_sweeps):
            sample_accept_rates.append(self.sweep(model))

        if sample_accept_rates:
            self.last_sample_acceptance_rate = sum(sample_accept_rates) / len(sample_accept_rates)
        else:
            self.last_sample_acceptance_rate = self.last_sweep_acceptance_rate

        return self.configs.clone()

    @property
    def acceptance_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    @property
    def recent_acceptance_rate(self) -> float:
        return self.last_sample_acceptance_rate
