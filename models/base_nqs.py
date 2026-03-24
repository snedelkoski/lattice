"""
Abstract base class for Neural Quantum State (NQS) ansatz.

Transformer-NQS inherits from this class.
The NQS takes a lattice configuration sigma in {0,1,2,3}^N and outputs
log psi(sigma) = log|psi| + i*phase as a complex number.

Two output head modes:
  - "scalar": Mean-pool over sites -> linear heads for log_amp and phase.
  - "backflow_det": Transformer outputs backflow orbital matrices, and the
    wavefunction is a sum of Slater determinants:
        psi(n) = sum_{k=1}^{K} det[Phi^k(n)]
    This naturally handles the fermionic sign structure.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseNQS(nn.Module, ABC):
    """
    Base class for Neural Quantum State models.

    Input: configs (B, N) with values in {0,1,2,3}
    Output: log_psi (B,) complex tensor = log|psi| + i*phase
    """

    def __init__(
        self,
        n_sites: int,
        d_model: int = 128,
        vocab_size: int = 4,
        max_sites: int = 144,
        force_real: bool = False,
        # Backflow determinant settings
        head_mode: str = "scalar",  # "scalar" or "backflow_det"
        n_determinants: int = 4,  # K: number of Slater determinants
        n_up: int = 0,  # for backflow: number of spin-up electrons
        n_down: int = 0,  # for backflow: number of spin-down electrons
    ):
        super().__init__()
        self.n_sites = n_sites
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.force_real = force_real
        self.head_mode = head_mode

        # Shared input encoding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_sites, d_model)

        if head_mode == "scalar":
            # Output heads for scalar mode
            self.log_amp_head = nn.Linear(d_model, 1)
            self.phase_head = nn.Linear(d_model, 1)

            # Initialize output heads to small values
            nn.init.zeros_(self.log_amp_head.bias)
            nn.init.normal_(self.log_amp_head.weight, std=0.01)
            nn.init.zeros_(self.phase_head.bias)
            nn.init.normal_(self.phase_head.weight, std=0.01)

        elif head_mode == "backflow_det":
            self.n_determinants = n_determinants
            self.n_up = n_up
            self.n_down = n_down
            self.n_electrons = n_up + n_down

            # The transformer output is (B, N, d_model) — one vector per site.
            # We need to produce orbital matrices M^k of shape (2*N, N_e)
            # where 2*N = spin-up channels + spin-down channels.
            #
            # Following Gu et al.: a linear layer maps each site's d_model
            # vector to K * N_e values for that site's up-row and K * N_e
            # for that site's down-row.
            #
            # So: (B, N, d_model) -> (B, N, 2 * K * N_e)
            # Reshape to (B, 2*N, K * N_e) then (B, K, 2*N, N_e)
            self.orbital_head = nn.Linear(
                d_model, 2 * n_determinants * self.n_electrons
            )
            # Initialize to small values for stable determinants near identity-ish
            nn.init.normal_(self.orbital_head.weight, std=0.01)
            nn.init.zeros_(self.orbital_head.bias)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def encode_input(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Encode input configurations to embeddings.

        Args:
            configs: (B, N) long tensor, values in {0,1,2,3}
        Returns:
            (B, N, d_model) float tensor
        """
        B, N = configs.shape
        device = configs.device

        tok_emb = self.token_embed(configs)  # (B, N, d_model)
        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_ids)  # (B, N, d_model)

        return tok_emb + pos_emb

    def decode_output(self, h: torch.Tensor, configs: torch.Tensor = None) -> torch.Tensor:
        """
        Decode backbone output to log psi.

        Args:
            h: (B, N, d_model) backbone output
            configs: (B, N) input configurations (needed for backflow_det mode)
        Returns:
            log_psi: (B,) complex tensor
        """
        if self.head_mode == "scalar":
            return self._decode_scalar(h)
        elif self.head_mode == "backflow_det":
            assert configs is not None, "backflow_det mode requires configs"
            return self._decode_backflow_det(h, configs)
        else:
            raise ValueError(f"Unknown head_mode: {self.head_mode}")

    def _decode_scalar(self, h: torch.Tensor) -> torch.Tensor:
        """Original scalar output head: mean-pool -> log_amp + phase."""
        # Mean pooling over sequence dimension
        h_pool = h.mean(dim=1)  # (B, d_model)

        log_amp = self.log_amp_head(h_pool).squeeze(-1)  # (B,)
        if self.force_real:
            phase = torch.zeros_like(log_amp)
        else:
            phase = self.phase_head(h_pool).squeeze(-1)  # (B,)

        # log psi = log|psi| + i * phase
        log_psi = torch.complex(log_amp.float(), phase.float())
        return log_psi

    def _decode_backflow_det(
        self, h: torch.Tensor, configs: torch.Tensor
    ) -> torch.Tensor:
        """
        Backflow determinant output head.

        psi(n) = sum_{k=1}^{K} det[Phi^k(n)]

        where Phi^k is formed by selecting rows of the orbital matrix M^k
        that correspond to occupied orbitals in config n.

        Args:
            h: (B, N, d_model) backbone output
            configs: (B, N) configurations, values in {0,1,2,3}
        Returns:
            log_psi: (B,) complex tensor
        """
        B, N, d = h.shape
        K = self.n_determinants
        N_e = self.n_electrons

        # Produce orbital coefficients: (B, N, 2 * K * N_e)
        orb_raw = self.orbital_head(h)  # (B, N, 2*K*N_e)

        # Reshape to separate spin-up and spin-down orbital rows:
        # (B, N, 2, K, N_e) -> (B, 2, N, K, N_e) -> (B, K, 2*N, N_e)
        orb = orb_raw.reshape(B, N, 2, K, N_e)
        orb = orb.permute(0, 3, 2, 1, 4)  # (B, K, 2, N, N_e)
        orb = orb.reshape(B, K, 2 * N, N_e)  # (B, K, 2N, N_e)

        # Build occupied-orbital indices from configs.
        # For each config, we need to find which "rows" of the 2N orbital
        # matrix are occupied.
        #
        # Row layout: rows 0..N-1 are spin-up orbitals for sites 0..N-1
        #             rows N..2N-1 are spin-down orbitals for sites 0..N-1
        #
        # Config encoding: 0=empty, 1=up, 2=down, 3=double
        # Spin-up at site i: config[i] == 1 or config[i] == 3
        # Spin-down at site i: config[i] == 2 or config[i] == 3
        has_up = (configs == 1) | (configs == 3)  # (B, N) bool
        has_down = (configs == 2) | (configs == 3)  # (B, N) bool

        # Occupied row indices: first all up positions, then all down positions
        # Since N_up and N_down are fixed, each config has exactly N_e occupied rows
        up_indices = has_up.float()  # (B, N)
        down_indices = has_down.float()  # (B, N)

        # Build the selection mask: (B, 2N) with 1 where occupied
        occ_mask = torch.cat([up_indices, down_indices], dim=1)  # (B, 2N)

        # Get sorted indices of occupied rows for each sample
        # Each sample has exactly N_e occupied positions
        # We use topk or nonzero; since count is fixed, argsort the mask
        # descending and take the first N_e
        occ_idx = occ_mask.argsort(dim=1, descending=True)[:, :N_e]  # (B, N_e)
        occ_idx = occ_idx.sort(dim=1).values  # sort for consistency

        # Select rows from orbital matrices: Phi^k_{ij} = M^k_{occ_idx[i], j}
        # orb: (B, K, 2N, N_e), occ_idx: (B, N_e)
        # Need to gather: for each (b, k), select rows occ_idx[b] from orb[b, k]
        occ_expanded = occ_idx.unsqueeze(1).unsqueeze(-1).expand(
            B, K, N_e, N_e
        )  # (B, K, N_e, N_e)
        phi = torch.gather(
            orb, 2, occ_expanded
        )  # (B, K, N_e, N_e) — the Slater matrices

        # Compute determinants: (B, K)
        dets = torch.linalg.det(phi)  # (B, K)

        # Wavefunction: psi = sum_k det[Phi^k]
        psi = dets.sum(dim=1)  # (B,)

        # Convert to log_psi = log|psi| + i*arg(psi)
        log_abs_psi = torch.log(psi.abs() + 1e-30)
        phase = torch.atan2(psi.imag, psi.real) if psi.is_complex() else torch.zeros_like(log_abs_psi)

        # psi from real determinants is real-valued
        # But can be negative, so we need to handle the sign
        if not psi.is_complex():
            # Real case: sign gives phase of 0 or pi
            phase = torch.where(psi >= 0, torch.zeros_like(psi), torch.full_like(psi, torch.pi))

        log_psi = torch.complex(log_abs_psi.float(), phase.float())
        return log_psi

    @abstractmethod
    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        The backbone network (Transformer).

        Args:
            x: (B, N, d_model) input embeddings
        Returns:
            h: (B, N, d_model) output embeddings
        """
        pass

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: configs -> log psi.

        Args:
            configs: (B, N) long tensor, values in {0,1,2,3}
        Returns:
            log_psi: (B,) complex tensor
        """
        x = self.encode_input(configs)
        h = self.backbone(x)
        return self.decode_output(h, configs=configs)

    def log_psi_and_grad(
        self, configs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log psi and its gradient w.r.t. parameters.
        Used for VMC gradient estimation.

        Returns:
            log_psi: (B,) complex tensor
            O_k: dict of parameter gradients (for SR)
        """
        log_psi = self.forward(configs)
        return log_psi, None  # Gradients computed via autograd in VMC loop

    def get_backbone_state_dict(self) -> dict:
        """Get only the backbone parameters (for JEPA pretraining transfer)."""
        state_dict = {}
        exclude_prefixes = (
            'log_amp_head', 'phase_head', 'orbital_head'
        )
        for name, param in self.named_parameters():
            if not any(name.startswith(p) for p in exclude_prefixes):
                state_dict[name] = param
        return state_dict

    def load_backbone_state_dict(self, state_dict: dict, strict: bool = False):
        """Load pretrained backbone weights (from JEPA)."""
        own_state = self.state_dict()
        loaded = 0
        for name, param in state_dict.items():
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
                else:
                    print(f"Shape mismatch for {name}: "
                          f"{own_state[name].shape} vs {param.shape}")
            else:
                print(f"Skipping {name} (not in model)")
        print(f"Loaded {loaded}/{len(state_dict)} pretrained parameters")

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_force_real(self, enabled: bool = True):
        """Force wavefunction phase to zero in forward pass."""
        self.force_real = enabled
