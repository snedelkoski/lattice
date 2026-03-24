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

            # Backflow orbital head.
            # Option A (factored=True, default): SEPARATE up/down determinants
            #   psi = sum_k det[Phi^k_up] * det[Phi^k_down]
            #   Phi^k_up is (n_up x n_up), Phi^k_down is (n_down x n_down)
            # Option B (factored=False): Combined determinant
            #   psi = sum_k det[Phi^k]
            #   Phi^k is (N_e x N_e) where N_e = n_up + n_down
            self.factored_det = True  # can be toggled

            if self.factored_det:
                # Each site produces K * n_up (for up) + K * n_down (for down)
                self.orbital_head = nn.Linear(
                    d_model, n_determinants * (n_up + n_down)
                )
            else:
                # Each site produces 2 * K * N_e (up rows + down rows)
                self.orbital_head = nn.Linear(
                    d_model, 2 * n_determinants * self.n_electrons
                )
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
        Backflow determinant output head with SEPARATE up/down determinants.

        psi(n) = sum_{k=1}^{K} det[Phi^k_up(n)] * det[Phi^k_down(n)]

        where Phi^k_up is (n_up x n_up) and Phi^k_down is (n_down x n_down).
        This factored form naturally handles the fermionic sign structure
        for each spin species independently.

        Args:
            h: (B, N, d_model) backbone output
            configs: (B, N) configurations, values in {0,1,2,3}
        Returns:
            log_psi: (B,) complex tensor
        """
        B, N, d = h.shape
        K = self.n_determinants
        n_up = self.n_up
        n_down = self.n_down

        # Produce orbital coefficients: (B, N, K * (n_up + n_down))
        orb_raw = self.orbital_head(h)  # (B, N, K*(n_up+n_down))

        # Split into up and down orbital blocks
        # (B, N, K*n_up), (B, N, K*n_down)
        orb_up_raw, orb_down_raw = orb_raw.split([K * n_up, K * n_down], dim=-1)

        # Reshape to (B, K, N, n_up) and (B, K, N, n_down)
        orb_up = orb_up_raw.reshape(B, N, K, n_up).permute(0, 2, 1, 3)  # (B, K, N, n_up)
        orb_down = orb_down_raw.reshape(B, N, K, n_down).permute(0, 2, 1, 3)  # (B, K, N, n_down)

        # Get occupied site indices for each spin
        # Config encoding: 0=empty, 1=up, 2=down, 3=double
        has_up = (configs == 1) | (configs == 3)  # (B, N) bool
        has_down = (configs == 2) | (configs == 3)  # (B, N) bool

        # Get sorted occupied site indices
        # Each sample has exactly n_up up-occupied sites and n_down down-occupied sites
        up_occ_idx = has_up.float().argsort(dim=1, descending=True)[:, :n_up]  # (B, n_up)
        up_occ_idx = up_occ_idx.sort(dim=1).values
        down_occ_idx = has_down.float().argsort(dim=1, descending=True)[:, :n_down]  # (B, n_down)
        down_occ_idx = down_occ_idx.sort(dim=1).values

        # Select occupied rows from orbital matrices
        # For up: orb_up is (B, K, N, n_up), select rows up_occ_idx -> (B, K, n_up, n_up)
        up_idx_exp = up_occ_idx.unsqueeze(1).unsqueeze(-1).expand(B, K, n_up, n_up)
        phi_up = torch.gather(orb_up, 2, up_idx_exp)  # (B, K, n_up, n_up)

        # For down: orb_down is (B, K, N, n_down), select rows down_occ_idx -> (B, K, n_down, n_down)
        down_idx_exp = down_occ_idx.unsqueeze(1).unsqueeze(-1).expand(B, K, n_down, n_down)
        phi_down = torch.gather(orb_down, 2, down_idx_exp)  # (B, K, n_down, n_down)

        # Compute determinants: (B, K) for each spin
        det_up = torch.linalg.det(phi_up)  # (B, K)
        det_down = torch.linalg.det(phi_down)  # (B, K)

        # Wavefunction: psi = sum_k det_up^k * det_down^k
        psi = (det_up * det_down).sum(dim=1)  # (B,)

        # Convert to log_psi = log|psi| + i*arg(psi)
        log_abs_psi = torch.log(psi.abs() + 1e-30)

        # psi from real determinants products is real-valued but can be negative
        # sign gives phase of 0 or pi
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
