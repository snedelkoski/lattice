"""
Lattice symmetry projection for Neural Quantum States.

Projects the NQS wave function onto the symmetric (k=0) sector of the
translation group, improving variational accuracy and ensuring the
ground state has the correct quantum numbers.

For a square lattice with PBC, the symmetrized wave function is:
    psi_sym(sigma) = (1/|G|) sum_{g in G} psi(g * sigma)

where G is the group of lattice translations and g*sigma applies
the permutation g to the configuration sigma.

In log space (for numerical stability):
    log psi_sym(sigma) = log_sum_exp_{g in G}[ log psi(g * sigma) ] - log|G|

This module wraps an NQS model and applies symmetry projection
externally, so the network itself remains unsymmetrized. This is
important for JEPA pretraining, where we don't want symmetry constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .hubbard import SquareLattice


def complex_logsumexp(log_z: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Numerically stable log-sum-exp for complex log values.

    Given log(z_i) = a_i + i*b_i, computes:
        log(sum_i z_i) = log(sum_i exp(a_i + i*b_i))

    We factor out max(a_i) for stability:
        = max_a + log(sum_i exp(a_i - max_a) * exp(i*b_i))

    Args:
        log_z: complex tensor of log values
        dim: dimension to sum over

    Returns:
        complex scalar (or reduced tensor): log(sum exp(log_z))
    """
    # Separate real and imaginary parts
    log_amp = log_z.real  # a_i = log|z_i|
    phase = log_z.imag    # b_i = arg(z_i)

    # Shift for numerical stability
    max_log_amp = log_amp.max(dim=dim, keepdim=True).values
    shifted = log_amp - max_log_amp  # a_i - max_a

    # sum_i exp(a_i - max_a) * exp(i*b_i)
    # = sum_i exp(shifted_i) * (cos(b_i) + i*sin(b_i))
    weights = torch.exp(shifted)
    real_part = (weights * torch.cos(phase)).sum(dim=dim)
    imag_part = (weights * torch.sin(phase)).sum(dim=dim)

    # log of the complex sum
    result_amp = torch.sqrt(real_part ** 2 + imag_part ** 2)
    result_phase = torch.atan2(imag_part, real_part)

    # Add back max_log_amp
    max_log_amp = max_log_amp.squeeze(dim)
    result = torch.complex(
        torch.log(result_amp + 1e-30) + max_log_amp,
        result_phase,
    )
    return result


class SymmetryProjector:
    """
    Projects an NQS wave function onto the k=0 translation-symmetric sector.

    Usage:
        projector = SymmetryProjector(lattice)
        log_psi_sym = projector.project(model, configs)

    The projection is applied outside the network, so the model itself
    doesn't need to know about symmetries. This allows the same model
    to be used with or without symmetry projection.
    """

    def __init__(
        self,
        lattice: SquareLattice,
        use_point_group: bool = False,
    ):
        """
        Args:
            lattice: SquareLattice with PBC
            use_point_group: if True, also include C4v point group
                           (rotations + reflections). Default: translations only.
        """
        self.lattice = lattice
        self.N = lattice.N

        # Get translation permutations
        self.translations = lattice.translation_group()
        self.n_translations = len(self.translations)

        # Optionally add point group symmetries
        if use_point_group and lattice.Lx == lattice.Ly:
            self.symmetries = self._build_full_space_group()
        else:
            self.symmetries = self.translations

        self.n_symmetries = len(self.symmetries)
        self.log_n_sym = np.log(self.n_symmetries)

        # Precompute permutation tensors
        self._perm_tensor = None  # lazily initialized on correct device

    def _build_full_space_group(self) -> list[np.ndarray]:
        """
        Build the full space group for a square lattice with PBC:
        translations x C4v point group.

        C4v = {identity, R90, R180, R270, Rx, Ry, Rd1, Rd2}
        where R = rotation, Rx/Ry = reflections, Rd = diagonal reflections.

        Returns unique permutations only.
        """
        L = self.lattice.Lx
        assert self.lattice.Lx == self.lattice.Ly, "Point group requires square lattice"

        # Point group generators for square lattice
        def rotate_90(x, y):
            """90-degree counterclockwise rotation."""
            return (L - 1 - y) % L, x % L

        def reflect_x(x, y):
            """Reflection across x-axis."""
            return x, (L - y) % L

        def reflect_y(x, y):
            """Reflection across y-axis."""
            return (L - x) % L, y

        def reflect_d1(x, y):
            """Reflection across main diagonal."""
            return y, x

        def reflect_d2(x, y):
            """Reflection across anti-diagonal."""
            return (L - 1 - y) % L, (L - 1 - x) % L

        # Build point group operations
        point_ops = []

        # Identity
        def identity(x, y):
            return x, y
        point_ops.append(identity)

        # Rotations
        def r180(x, y):
            return (L - x) % L, (L - y) % L
        def r270(x, y):
            return y % L, (L - 1 - x) % L

        point_ops.extend([rotate_90, r180, r270])

        # Reflections
        point_ops.extend([reflect_x, reflect_y, reflect_d1, reflect_d2])

        # Compose translations with point group
        seen = set()
        full_group = []

        for trans_perm in self.translations:
            for point_op in point_ops:
                # Build combined permutation: first apply point_op, then translation
                perm = np.zeros(self.N, dtype=int)
                for i in range(self.N):
                    x, y = self.lattice.site_coords(i)
                    # Apply point group operation
                    xp, yp = point_op(x, y)
                    # Apply translation (encoded in trans_perm)
                    j = self.lattice.site_index(xp, yp)
                    perm[i] = trans_perm[j]

                # Deduplicate
                key = tuple(perm)
                if key not in seen:
                    seen.add(key)
                    full_group.append(perm)

        return full_group

    def _get_perm_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Get permutation tensor on the correct device.

        Returns:
            (n_symmetries, N) long tensor of permutation indices
        """
        if self._perm_tensor is None or self._perm_tensor.device != device:
            perms = np.stack(self.symmetries, axis=0)  # (n_sym, N)
            self._perm_tensor = torch.from_numpy(perms).long().to(device)
        return self._perm_tensor

    def apply_symmetry(self, configs: torch.Tensor, perm_idx: int) -> torch.Tensor:
        """
        Apply a single symmetry operation to a batch of configurations.

        Args:
            configs: (B, N) tensor
            perm_idx: index into self.symmetries

        Returns:
            (B, N) tensor with permuted sites
        """
        perm = self._get_perm_tensor(configs.device)[perm_idx]  # (N,)
        return configs[:, perm]

    def apply_all_symmetries(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Apply all symmetry operations to a batch of configurations.

        Args:
            configs: (B, N) tensor

        Returns:
            (n_sym, B, N) tensor where out[g, b, :] = configs[b, perms[g, :]]
        """
        perms = self._get_perm_tensor(configs.device)  # (n_sym, N)
        # Expand configs: (1, B, N) and perms: (n_sym, 1, N)
        # Use gather: configs_expanded[g, b, perms[g, i]] -> need advanced indexing
        expanded = configs.unsqueeze(0).expand(self.n_symmetries, -1, -1)  # (n_sym, B, N)
        perm_expanded = perms.unsqueeze(1).expand(-1, configs.shape[0], -1)  # (n_sym, B, N)
        return torch.gather(expanded, 2, perm_expanded)

    @torch.no_grad()
    def project(
        self,
        model: nn.Module,
        configs: torch.Tensor,
        max_batch: int = 4096,
    ) -> torch.Tensor:
        """
        Compute symmetry-projected log psi.

        log psi_sym(sigma) = log_sum_exp_g[ log psi(g*sigma) ] - log|G|

        This evaluates the network |G| times (once per symmetry operation).
        For large |G| (e.g., 16*16=256 translations), this can be expensive.

        Args:
            model: NQS model
            configs: (B, N) configurations
            max_batch: maximum batch size for forward passes (to manage memory)

        Returns:
            log_psi_sym: (B,) complex tensor
        """
        B, N = configs.shape
        device = configs.device
        perms = self._get_perm_tensor(device)  # (n_sym, N)
        n_sym = perms.shape[0]

        # Collect log psi for all symmetry-transformed configs
        log_psi_all = torch.zeros(n_sym, B, dtype=torch.complex64, device=device)

        for g in range(n_sym):
            # Apply permutation g to all configs
            transformed = configs[:, perms[g]]  # (B, N)

            # Forward pass in chunks to manage memory
            if B <= max_batch:
                log_psi_all[g] = model(transformed)
            else:
                results = []
                for start in range(0, B, max_batch):
                    end = min(start + max_batch, B)
                    results.append(model(transformed[start:end]))
                log_psi_all[g] = torch.cat(results, dim=0)

        # log_sum_exp over symmetries (dim=0), then subtract log|G|
        log_psi_sym = complex_logsumexp(log_psi_all, dim=0) - self.log_n_sym

        return log_psi_sym

    def project_with_grad(
        self,
        model: nn.Module,
        configs: torch.Tensor,
        max_batch: int = 4096,
    ) -> torch.Tensor:
        """
        Compute symmetry-projected log psi WITH gradient tracking.

        Same as project() but allows gradients to flow through for VMC training.

        Args:
            model: NQS model
            configs: (B, N) configurations
            max_batch: max batch size per forward pass

        Returns:
            log_psi_sym: (B,) complex tensor (with grad)
        """
        B, N = configs.shape
        device = configs.device
        perms = self._get_perm_tensor(device)
        n_sym = perms.shape[0]

        # Collect log psi for all transformed configs
        log_psi_list = []

        for g in range(n_sym):
            transformed = configs[:, perms[g]]  # (B, N)

            if B <= max_batch:
                log_psi_g = model(transformed)
            else:
                chunks = []
                for start in range(0, B, max_batch):
                    end = min(start + max_batch, B)
                    chunks.append(model(transformed[start:end]))
                log_psi_g = torch.cat(chunks, dim=0)

            log_psi_list.append(log_psi_g)

        # Stack: (n_sym, B)
        log_psi_all = torch.stack(log_psi_list, dim=0)

        # Differentiable complex_logsumexp
        log_psi_sym = _differentiable_complex_logsumexp(log_psi_all, dim=0) - self.log_n_sym

        return log_psi_sym


def _differentiable_complex_logsumexp(
    log_z: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """
    Differentiable complex log-sum-exp.

    This version supports autograd by avoiding in-place operations
    and using differentiable primitives.

    Args:
        log_z: (n_sym, B) complex tensor
        dim: dimension to reduce

    Returns:
        (B,) complex tensor
    """
    log_amp = log_z.real
    phase = log_z.imag

    # Shift for stability
    max_log_amp = log_amp.max(dim=dim, keepdim=True).values.detach()
    shifted = log_amp - max_log_amp

    # Weighted sum
    weights = torch.exp(shifted)
    real_sum = (weights * torch.cos(phase)).sum(dim=dim)
    imag_sum = (weights * torch.sin(phase)).sum(dim=dim)

    # Magnitude and phase of the sum
    sum_amp = torch.sqrt(real_sum ** 2 + imag_sum ** 2 + 1e-30)
    sum_phase = torch.atan2(imag_sum, real_sum)

    max_log_amp = max_log_amp.squeeze(dim)
    result = torch.complex(
        torch.log(sum_amp) + max_log_amp,
        sum_phase,
    )
    return result


class SymmetrizedNQS(nn.Module):
    """
    Wrapper that adds symmetry projection to any NQS model.

    This wraps a BaseNQS and applies translation symmetry projection.
    The wrapped model can be used interchangeably with the original
    in VMC training.

    Usage:
        model = TransformerNQS(...)
        sym_model = SymmetrizedNQS(model, lattice)
        log_psi = sym_model(configs)  # symmetry-projected
    """

    def __init__(
        self,
        model: nn.Module,
        lattice: SquareLattice,
        use_point_group: bool = False,
    ):
        super().__init__()
        self.model = model
        self.projector = SymmetryProjector(
            lattice, use_point_group=use_point_group
        )
        # Expose model attributes that VMC trainer needs
        self.n_sites = getattr(model, 'n_sites', lattice.N)

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with symmetry projection.

        Args:
            configs: (B, N) long tensor
        Returns:
            log_psi_sym: (B,) complex tensor
        """
        return self.projector.project_with_grad(self.model, configs)

    def parameters(self, recurse=True):
        """Delegate to inner model."""
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        """Delegate to inner model."""
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def train(self, mode=True):
        """Delegate to inner model."""
        self.model.train(mode)
        return self

    def eval(self):
        """Delegate to inner model."""
        self.model.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Delegate to inner model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Delegate to inner model."""
        return self.model.load_state_dict(*args, **kwargs)

    def count_parameters(self) -> int:
        """Delegate to inner model."""
        if hasattr(self.model, 'count_parameters'):
            return self.model.count_parameters()
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MarshallSignNQS(nn.Module):
    """
    Wrapper that applies the Marshall sign rule to an NQS model.

    For a bipartite lattice at half-filling, the exact ground state
    wavefunction satisfies:
        psi(sigma) = (-1)^{N_down_B} * |psi(sigma)|

    where N_down_B is the number of down-spin electrons on sublattice B.

    This wrapper forces the model to output a real wavefunction with the
    correct Marshall sign, eliminating the need to learn the phase:
        log psi(sigma) = log|psi(sigma)| + i * pi * N_down_B

    The model itself only needs to output log|psi| (real part).

    Encoding: config values are {0=empty, 1=up, 2=down, 3=double}.
    Down electrons exist at sites where config == 2 or config == 3.
    """

    def __init__(self, model: nn.Module, lattice: SquareLattice):
        super().__init__()
        self.model = model
        self.n_sites = getattr(model, 'n_sites', lattice.N)

        # Build sublattice mask: True for sublattice B sites
        # Sublattice B = sites where (x + y) is odd
        sublattice_b = np.zeros(lattice.N, dtype=bool)
        for i in range(lattice.N):
            x, y = lattice.site_coords(i)
            sublattice_b[i] = (x + y) % 2 == 1
        self._sublattice_b = sublattice_b  # numpy for registration
        self.register_buffer(
            'sublattice_b',
            torch.from_numpy(sublattice_b),
            persistent=False,
        )

        # Force the inner model to output real wavefunctions
        if hasattr(model, 'set_force_real'):
            model.set_force_real(True)

    def _marshall_sign(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute (-1)^{N_down_B} for each config in the batch.

        Args:
            configs: (B, N) long tensor, values in {0, 1, 2, 3}

        Returns:
            sign: (B,) tensor of +1 or -1
        """
        # Down electrons at sites where config == 2 (down only) or == 3 (double occ)
        has_down = (configs == 2) | (configs == 3)  # (B, N) bool
        # Count down electrons on sublattice B
        n_down_b = (has_down & self.sublattice_b.unsqueeze(0)).sum(dim=1)  # (B,)
        # (-1)^n = 1 - 2*(n % 2)
        sign = 1 - 2 * (n_down_b % 2)
        return sign.float()

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Marshall sign.

        Args:
            configs: (B, N) long tensor
        Returns:
            log_psi: (B,) complex tensor with Marshall sign encoded in phase
        """
        log_psi = self.model(configs)  # (B,) complex, but imaginary is 0
        log_amp = log_psi.real

        # Marshall sign: phase = pi * (N_down_B mod 2)
        has_down = (configs == 2) | (configs == 3)
        sublattice_b = self.sublattice_b.to(configs.device)
        n_down_b = (has_down & sublattice_b.unsqueeze(0)).sum(dim=1)
        phase = np.pi * (n_down_b % 2).float()

        return torch.complex(log_amp, phase)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def count_parameters(self) -> int:
        if hasattr(self.model, 'count_parameters'):
            return self.model.count_parameters()
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
