"""
SIGReg: Sketch Isotropic Gaussian Regularizer.

Anti-collapse regularizer for JEPA based on the Epps-Pulley statistical
test for Gaussianity using random projections. Enforces that the embedding
distribution matches a standard isotropic Gaussian N(0, I).

Reference: LeWorldModel (arXiv:2603.19312, Maes et al., 2026)
"""

import torch
import torch.nn as nn
import math


class SIGReg(nn.Module):
    """
    SIGReg anti-collapse loss.

    Tests whether embeddings follow a standard Gaussian distribution
    by comparing the empirical characteristic function (on random 1D
    projections) to the Gaussian characteristic function exp(-t^2/2).

    If embeddings collapse (all identical), the empirical CF deviates
    strongly from the Gaussian CF, producing a large loss.

    Only hyperparameter: lambda (weight relative to prediction loss).
    """

    def __init__(
        self,
        num_projections: int = 256,
        knots: int = 17,
        t_max: float = 3.0,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.knots = knots
        self.t_max = t_max

        # Pre-compute knot points and weights
        t = torch.linspace(0, t_max, knots)
        self.register_buffer('t', t)

        # Trapezoidal weights with Gaussian window
        dt = t_max / (knots - 1)
        weights = torch.ones(knots) * 2.0
        weights[0] = 1.0
        weights[-1] = 1.0
        weights = weights * dt * torch.exp(-t ** 2 / 2)
        self.register_buffer('weights', weights)

        # Target: Gaussian CF real part = exp(-t^2/2), imaginary = 0
        self.register_buffer('gaussian_cf', torch.exp(-t ** 2 / 2))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            embeddings: (B, D) tensor of embeddings

        Returns:
            Scalar loss (0 if embeddings match standard Gaussian)
        """
        B, D = embeddings.shape
        device = embeddings.device

        # Standardize embeddings (zero mean, unit variance per dimension)
        emb = embeddings - embeddings.mean(dim=0, keepdim=True)
        std = emb.std(dim=0, keepdim=True).clamp(min=1e-8)
        emb = emb / std

        # Random unit projections
        A = torch.randn(D, self.num_projections, device=device)
        A = A / A.norm(dim=0, keepdim=True)

        # Project embeddings onto random directions
        p = emb @ A  # (B, num_projections)

        # Compute empirical characteristic function at knot points
        # x_t[b, proj, knot] = p[b, proj] * t[knot]
        x_t = p.unsqueeze(-1) * self.t.unsqueeze(0).unsqueeze(0)  # (B, num_proj, knots)

        # Empirical CF: E[exp(i*t*X)] = E[cos(tX)] + i*E[sin(tX)]
        ecf_real = x_t.cos().mean(dim=0)  # (num_proj, knots)
        ecf_imag = x_t.sin().mean(dim=0)  # (num_proj, knots)

        # Error: squared difference from Gaussian CF
        err_real = (ecf_real - self.gaussian_cf.unsqueeze(0)) ** 2
        err_imag = ecf_imag ** 2  # should be 0 for symmetric distribution
        err = err_real + err_imag  # (num_proj, knots)

        # Weighted integration
        statistic = (err @ self.weights) * B  # (num_proj,)

        return statistic.mean()
