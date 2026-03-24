"""
Lattice-JEPA: Joint Embedding Predictive Architecture for lattice configurations.

Self-supervised pretraining module that learns lattice representations by
predicting masked site embeddings from context. Uses SIGReg anti-collapse
regularizer (no target encoder, no EMA, end-to-end training).

Architecture:
  Encoder: Transformer backbone (shared with NQS)
  Projector: MLP -> embedding space (where SIGReg operates)
  Predictor: Small Transformer that fills in masked positions
  Loss: MSE(predicted, target) + lambda * SIGReg(embeddings)

Reference: LeWorldModel (arXiv:2603.19312) for the end-to-end JEPA approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigreg import SIGReg
from .predictors import PredictorTransformer
from .masking import generate_batch_masks


class Projector(nn.Module):
    """MLP projector with BatchNorm (maps backbone dim -> embedding dim)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_in) or (B*N, d_in)
        Returns:
            (same shape with d_in replaced by d_out)
        """
        shape = x.shape
        if len(shape) == 3:
            B, N, D = shape
            x = x.reshape(B * N, D)
            out = self.net(x)
            return out.reshape(B, N, -1)
        else:
            return self.net(x)


class LatticeJEPA(nn.Module):
    """
    Lattice-JEPA pretraining module.

    Wraps an NQS backbone (Transformer) with:
    - Projector: maps backbone output to embedding space
    - Predictor: predicts masked position embeddings
    - SIGReg: prevents representation collapse

    Training procedure:
    1. Encode full configuration -> embeddings for all sites
    2. Project to embedding space
    3. Feed visible embeddings + mask tokens to predictor
    4. Predict target embeddings at masked positions
    5. Loss = MSE(prediction, target) + lambda * SIGReg(all embeddings)
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 128,
        d_embed: int = 64,
        d_proj_hidden: int = 512,
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        predictor_d_ff: int = 256,
        max_sites: int = 144,
        lambda_sigreg: float = 0.09,
        sigreg_num_proj: int = 256,
        sigreg_knots: int = 17,
    ):
        super().__init__()

        self.backbone = backbone
        self.d_model = d_model
        self.d_embed = d_embed
        self.lambda_sigreg = lambda_sigreg

        # Projector: backbone dim -> embedding space
        self.projector = Projector(d_model, d_proj_hidden, d_embed)

        # Predictor: small Transformer
        self.predictor = PredictorTransformer(
            d_model=d_model,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
            d_ff=predictor_d_ff,
            max_sites=max_sites,
        )

        # Predictor projector: predictor output -> embedding space
        self.pred_projector = Projector(d_model, d_proj_hidden, d_embed)

        # SIGReg regularizer
        self.sigreg = SIGReg(
            num_projections=sigreg_num_proj,
            knots=sigreg_knots,
        )

    def forward(
        self,
        configs: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for JEPA pretraining.

        Args:
            configs: (B, N) lattice configurations, values in {0,1,2,3}
            masks: (B, N) bool tensor, True = visible, False = masked

        Returns:
            dict with:
                'loss': total loss
                'mse_loss': prediction MSE
                'sigreg_loss': SIGReg regularization
                'embeddings': projected embeddings (for visualization)
        """
        B, N = configs.shape

        # 1. Encode all sites (full configuration, no masking here)
        x = self.backbone.encode_input(configs)  # (B, N, d_model)
        h = self.backbone.backbone(x)  # (B, N, d_model)

        # 2. Project to embedding space
        z = self.projector(h)  # (B, N, d_embed)

        # 3. Predict masked positions
        pred_h = self.predictor(h, masks)  # (B, N, d_model)
        z_pred = self.pred_projector(pred_h)  # (B, N, d_embed)

        # 4. MSE loss on masked positions only
        masked = ~masks  # True where masked
        if masked.any():
            z_target = z[masked]  # (n_masked, d_embed)
            z_predicted = z_pred[masked]  # (n_masked, d_embed)
            mse_loss = F.mse_loss(z_predicted, z_target)
        else:
            mse_loss = torch.tensor(0.0, device=configs.device)

        # 5. SIGReg on all embeddings (mean-pooled per sample)
        z_pooled = z.mean(dim=1)  # (B, d_embed)
        sigreg_loss = self.sigreg(z_pooled)

        # Total loss
        total_loss = mse_loss + self.lambda_sigreg * sigreg_loss

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'sigreg_loss': sigreg_loss,
            'embeddings': z_pooled.detach(),
        }

    def get_backbone_state_dict(self) -> dict:
        """Extract backbone parameters for transfer to NQS."""
        return {k: v.clone() for k, v in self.backbone.state_dict().items()}
