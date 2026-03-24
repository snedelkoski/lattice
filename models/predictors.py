"""
JEPA predictor network.

Small Transformer that takes context embeddings (from visible sites)
and learnable mask tokens (for masked sites) and predicts the target
embeddings at masked positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PredictorTransformer(nn.Module):
    """
    Predictor network for JEPA.

    Takes a sequence of embeddings where visible positions have encoder
    outputs and masked positions have learnable mask tokens. Runs a small
    Transformer to predict the target embeddings at masked positions.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_sites: int = 144,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding for the predictor
        self.pos_embed = nn.Embedding(max_sites, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            PredictorBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target embeddings at masked positions.

        Args:
            context_embeddings: (B, N, d_model) encoder output for ALL positions
            mask: (B, N) bool tensor, True = visible, False = masked

        Returns:
            predicted: (B, N, d_model) predictions at ALL positions
                       (only masked positions matter for loss)
        """
        B, N, D = context_embeddings.shape
        device = context_embeddings.device

        # Build input: visible positions get encoder output, masked get mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(context_embeddings)  # (B, N, D)
        mask_tokens = self.mask_token.expand(B, N, D)  # (B, N, D)

        x = torch.where(mask_expanded, context_embeddings, mask_tokens)

        # Add positional encoding
        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)

        # Run Transformer
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x


class PredictorBlock(nn.Module):
    """Transformer block for predictor (pre-norm, full attention)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
