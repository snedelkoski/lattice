"""
Optimizers for VMC training.

Provides:
- MinSR (Minimum-step Stochastic Reconfiguration) — matrix-free natural gradient
  using the kernel trick with efficient batched Jacobian via torch.func.
- MARCH (Moment-Adaptive ReConfiguration Heuristic) — MinSR + momentum +
  per-parameter adaptive second-moment scaling. Like "Adam meets SR".

References:
  MinSR: Chen & Heyl, Nature Physics 20, 1476 (2024)
  MARCH: Gu et al., arXiv:2507.02644v2 (2025)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch.func import functional_call, vmap, jacrev


class MinSR:
    """
    Minimum-step Stochastic Reconfiguration (MinSR).

    Natural gradient for VMC using the kernel trick. Instead of building
    the P x P S-matrix (where P = n_params), uses the N_s x N_s kernel
    matrix T = O_bar @ O_bar^T where O_bar is the centered Jacobian.

    This is efficient when N_samples << N_params (always true for deep NQS).

    Algorithm:
        1. Compute Jacobian O = d(log psi) / d(theta) for all samples
           using torch.func.vmap + jacrev (fully batched, no per-sample loop)
        2. Center: O_bar = O - mean(O)
        3. Compute kernel: T = O_bar @ O_bar^T / N_s  (N_s x N_s matrix)
        4. Regularize: T_reg = T + epsilon * I
        5. Solve: T_reg @ alpha = e_centered
        6. Compute delta = O_bar^T @ alpha / N_s
        7. Update: theta -= lr * delta

    Reference: Chen & Heyl, Nature Physics 20, 1476 (2024)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.02,
        diag_shift: float = 0.01,
        max_norm: float = 1.0,
        chunk_size: int = 32,  # vmap chunk size to reduce memory
    ):
        self.model = model
        self.lr = lr
        self.diag_shift = diag_shift
        self.max_norm = max_norm
        self.chunk_size = chunk_size

        # Cache: extract param structure once
        self._param_names = []
        self._param_shapes = []
        self._n_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._param_names.append(name)
                self._param_shapes.append(p.shape)
                self._n_params += p.numel()

        # Build the functional call infrastructure
        self._setup_functional()

    def _setup_functional(self):
        """Set up torch.func infrastructure for batched Jacobian computation."""
        pass  # Done lazily in _compute_jacobian

    def _compute_jacobian(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Jacobian O_{i,k} = d(Re[log psi(sigma_i)]) / d(theta_k)
        for all samples i, using torch.func.vmap + jacrev.

        Args:
            configs: (N_s, N) sampled configurations

        Returns:
            O: (N_s, N_params) real Jacobian matrix
        """
        # Separate params into {name: tensor} dict for functional_call
        params = {name: p for name, p in self.model.named_parameters()
                  if p.requires_grad}

        # Define function from params -> log_psi for a single config
        def func(params_dict, single_config):
            """Compute Re[log psi] for a single configuration."""
            # single_config: (N,) -> need (1, N) for the model
            log_psi = functional_call(
                self.model, params_dict, (single_config.unsqueeze(0),)
            )
            # log_psi is (1,) complex -> take real part scalar
            return log_psi.real.squeeze(0)

        # jacrev over params, then vmap over configs (samples)
        # jacrev(func, argnums=0) gives gradient w.r.t. params_dict
        # vmap over the config dimension with chunking
        jac_func = vmap(jacrev(func, argnums=0), in_dims=(None, 0),
                        chunk_size=self.chunk_size)

        # Compute: jac is a dict {name: (N_s, *param_shape)}
        jac = jac_func(params, configs)

        # Flatten each param's jacobian and concatenate
        jac_flat = []
        for name in self._param_names:
            j = jac[name]  # (N_s, *param_shape)
            jac_flat.append(j.reshape(configs.shape[0], -1))

        # O: (N_s, N_params)
        O = torch.cat(jac_flat, dim=1)
        return O

    @torch.no_grad()
    def step(
        self,
        configs: torch.Tensor,
        e_loc: torch.Tensor,
    ) -> dict:
        """
        Perform one MinSR update step.

        Args:
            configs: (N_s, N) sampled configurations
            e_loc: (N_s,) local energies (complex)

        Returns:
            dict with 'param_update_norm' for logging
        """
        N_s = configs.shape[0]

        # 1. Compute Jacobian (need grad for jacrev, then detach)
        with torch.enable_grad():
            O = self._compute_jacobian(configs)  # (N_s, N_params)

        # 2. Center the Jacobian
        O_bar = O - O.mean(dim=0, keepdim=True)

        # 3. Centered local energies
        e_mean = e_loc.real.mean()
        e_centered = e_loc.real - e_mean

        # 4. Kernel: T = O_bar @ O_bar^T / N_s
        T = O_bar @ O_bar.t() / N_s

        # 5. Regularize: T_reg = T + epsilon * I
        T.diagonal().add_(self.diag_shift)

        # 6. Solve: T_reg @ alpha = e_centered
        try:
            alpha = torch.linalg.solve(T, e_centered)
        except torch.linalg.LinAlgError:
            # Fallback: use pseudoinverse
            alpha = T.pinverse() @ e_centered

        # 7. Parameter update: delta = O_bar^T @ alpha / N_s
        delta = O_bar.t() @ alpha / N_s

        # Clip update norm
        update_norm = delta.norm().item()
        if self.max_norm > 0 and update_norm > self.max_norm:
            delta = delta * (self.max_norm / update_norm)
            update_norm = self.max_norm

        # 8. Apply update to model parameters
        idx = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                n = p.numel()
                p.data -= self.lr * delta[idx:idx+n].reshape(p.shape)
                idx += n

        return {
            'param_update_norm': update_norm,
            'energy': e_mean.item(),
        }

    def set_lr(self, lr: float):
        """Update learning rate."""
        self.lr = lr

    def set_diag_shift(self, diag_shift: float):
        """Update diagonal regularization."""
        self.diag_shift = diag_shift


class MARCH:
    """
    Moment-Adaptive ReConfiguration Heuristic (MARCH).

    Enhances MinSR with:
    - Momentum (first moment): accelerates convergence like SGD+momentum
    - Adaptive per-parameter scaling (second moment): like Adam's v_t,
      suppresses volatile parameters and boosts stable ones

    The update rule minimizes:
        dtheta = argmin (1/lambda) ||O_bar @ dtheta - e_centered||^2
                       + ||diag(v)^{1/4} (dtheta - phi)||^2

    where phi is the momentum term and v tracks gradient volatility.

    Closed-form solution via Woodbury identity (O(B^2 * N_params)):
        U = O_bar @ diag(v)^{-1/4}
        zeta = e_centered - O_bar @ phi
        pi = U^T @ (lambda*I + U @ U^T)^{-1} @ zeta
        dtheta = diag(v)^{-1/4} @ pi + phi

    Reference: Gu et al., arXiv:2507.02644v2 (2025)
    """

    def __init__(
        self,
        model: nn.Module,
        norm_constraint: float = 0.1,
        damping: float = 0.001,
        mu: float = 0.95,  # momentum decay (beta1)
        beta: float = 0.995,  # second moment decay (beta2)
        max_norm: float = 0.0,  # 0 = use norm_constraint instead
        chunk_size: int = 32,
        norm_decay_start: int = 8000,  # step after which norm_constraint decays
    ):
        self.model = model
        self.norm_constraint = norm_constraint
        self.damping = damping
        self.mu = mu
        self.beta = beta
        self.max_norm = max_norm
        self.chunk_size = chunk_size
        self.norm_decay_start = norm_decay_start

        # Cache param structure
        self._param_names = []
        self._param_shapes = []
        self._n_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._param_names.append(name)
                self._param_shapes.append(p.shape)
                self._n_params += p.numel()

        # State: momentum and second moment
        self.phi = torch.zeros(self._n_params, device=next(model.parameters()).device)
        self.v = torch.ones(self._n_params, device=next(model.parameters()).device)
        self.prev_dtheta = torch.zeros(self._n_params, device=next(model.parameters()).device)
        self.step_count = 0

    def _compute_jacobian(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian O_{i,k} = d(Re[log psi(sigma_i)]) / d(theta_k).
        Same as MinSR but shared here.
        """
        params = {name: p for name, p in self.model.named_parameters()
                  if p.requires_grad}

        def func(params_dict, single_config):
            log_psi = functional_call(
                self.model, params_dict, (single_config.unsqueeze(0),)
            )
            return log_psi.real.squeeze(0)

        jac_func = vmap(jacrev(func, argnums=0), in_dims=(None, 0),
                        chunk_size=self.chunk_size)
        jac = jac_func(params, configs)

        jac_flat = []
        for name in self._param_names:
            j = jac[name]
            jac_flat.append(j.reshape(configs.shape[0], -1))

        return torch.cat(jac_flat, dim=1)

    def _get_norm_constraint(self) -> float:
        """Get the current norm constraint with optional decay."""
        if self.step_count <= self.norm_decay_start:
            return self.norm_constraint
        # 1/t decay after norm_decay_start
        excess = self.step_count - self.norm_decay_start
        return self.norm_constraint / (1.0 + excess / self.norm_decay_start)

    @torch.no_grad()
    def step(
        self,
        configs: torch.Tensor,
        e_loc: torch.Tensor,
    ) -> dict:
        """
        Perform one MARCH update step.

        Args:
            configs: (N_s, N) sampled configurations
            e_loc: (N_s,) local energies (complex)

        Returns:
            dict with diagnostics
        """
        N_s = configs.shape[0]
        device = configs.device
        sqrt_Ns = N_s ** 0.5

        # 1. Compute Jacobian
        with torch.enable_grad():
            O = self._compute_jacobian(configs)  # (N_s, N_params)

        # 2. Center and normalize the Jacobian: Ō = (O - mean(O)) / √N_s
        O_bar = (O - O.mean(dim=0, keepdim=True)) / sqrt_Ns

        # 3. Centered and normalized local energies: ε̃ = -(E_loc - <E_loc>) / √N_s
        e_mean = e_loc.real.mean()
        e_centered = -(e_loc.real - e_mean) / sqrt_Ns

        # 4. Compute adaptive scaling: v^{-1/4}
        v_inv_quarter = (self.v.clamp(min=1e-10) ** (-0.25))  # (N_params,)

        # 5. Build the Woodbury system
        # U = Ō @ diag(v^{-1/4})
        U = O_bar * v_inv_quarter.unsqueeze(0)  # (N_s, N_params)

        # ζ = ε̃ - Ō @ φ
        zeta = e_centered - O_bar @ self.phi  # (N_s,)

        # Kernel: K = U @ U^T + λI  (N_s, N_s)
        K = U @ U.t()  # (N_s, N_s)
        K.diagonal().add_(self.damping)

        # Solve: K @ π̂ = ζ
        try:
            pi_hat = torch.linalg.solve(K, zeta)
        except torch.linalg.LinAlgError:
            pi_hat = K.pinverse() @ zeta

        # 6. Compute update: dθ = D^{-1/4} (U^T π̂) + φ
        pi = U.t() @ pi_hat  # (N_params,)
        dtheta = v_inv_quarter * pi + self.phi  # (N_params,)

        # 7. Apply norm constraint
        nc = self._get_norm_constraint()
        update_norm = dtheta.norm().item()
        if update_norm > 0:
            # Scale update to have norm = norm_constraint
            dtheta = dtheta * (nc / max(update_norm, nc))

        # 8. Apply update to model parameters
        idx = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                n = p.numel()
                p.data += dtheta[idx:idx+n].reshape(p.shape)
                idx += n

        # 9. Update momentum and second moment
        self.phi = self.mu * dtheta
        self.v = self.beta * self.v + (dtheta - self.prev_dtheta) ** 2
        self.prev_dtheta = dtheta.clone()

        self.step_count += 1

        return {
            'param_update_norm': update_norm,
            'energy': e_mean.item(),
            'norm_constraint': nc,
        }

    def set_norm_constraint(self, nc: float):
        """Update norm constraint."""
        self.norm_constraint = nc

    def reset_state(self):
        """Reset momentum and second moment (useful when U changes)."""
        device = self.phi.device
        self.phi = torch.zeros(self._n_params, device=device)
        self.v = torch.ones(self._n_params, device=device)
        self.prev_dtheta = torch.zeros(self._n_params, device=device)
        # Don't reset step_count — it controls norm decay schedule

    # Compatibility with MinSR interface
    @property
    def lr(self):
        return self.norm_constraint

    def set_lr(self, lr: float):
        self.norm_constraint = lr

    def set_diag_shift(self, diag_shift: float):
        self.damping = diag_shift
