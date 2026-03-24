"""
Transformer-based Neural Quantum State ansatz.

Standard Transformer encoder (bidirectional/full attention).
Non-autoregressive: takes full configuration, outputs log psi.

Supports two architecture modes:
  - "standard": Pre-norm with LayerNorm, 2-layer FFN (d -> d_ff -> d)
  - "nqs": No LayerNorm, single-layer FFN with SiLU (following Gu et al. 2025)
    This simplified architecture works better for VMC optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base_nqs import BaseNQS


class TransformerBlock(nn.Module):
    """Transformer encoder block. Supports standard (pre-norm) or NQS (no-norm) mode."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_layernorm:
            # Standard 2-layer FFN: d -> d_ff -> d
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
        else:
            # NQS-style single-layer FFN: d -> d with SiLU
            # Following Gu et al.: FFN(X) = SiLU(X @ W_F)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            # Pre-norm architecture
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # No-norm architecture (post-add residual, no normalization)
            attn_out, _ = self.attn(x, x, x)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(x))
        return x


class TransformerStack(nn.Module):
    """Stack of Transformer encoder blocks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_layernorm)
            for _ in range(num_layers)
        ])
        # Final norm only if using LayerNorm
        self.final_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TransformerNQS(BaseNQS):
    """
    Transformer-based Neural Quantum State ansatz.

    Uses standard Transformer encoder (bidirectional attention).
    Non-autoregressive: takes full configuration, outputs log psi.

    Supports two head modes:
      - "scalar": Mean-pool -> linear heads for log_amp and phase
      - "backflow_det": Backflow orbital matrices -> sum of Slater determinants
    """

    def __init__(
        self,
        n_sites: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        vocab_size: int = 4,
        max_sites: int = 144,
        use_layernorm: bool = True,
        # Backflow determinant settings (forwarded to BaseNQS)
        head_mode: str = "scalar",
        n_determinants: int = 4,
        n_up: int = 0,
        n_down: int = 0,
    ):
        super().__init__(
            n_sites=n_sites,
            d_model=d_model,
            vocab_size=vocab_size,
            max_sites=max_sites,
            head_mode=head_mode,
            n_determinants=n_determinants,
            n_up=n_up,
            n_down=n_down,
        )
        self.backbone_net = TransformerStack(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_net(x)
