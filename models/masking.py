"""
Spatial masking strategies for lattice configurations.

For JEPA pretraining, we mask contiguous blocks of lattice sites
to force the predictor to learn spatial correlations. This is analogous
to I-JEPA's block masking but adapted for 2D lattice geometry.
"""

import torch
import numpy as np
from typing import Optional


def generate_block_mask(
    Lx: int,
    Ly: int,
    block_size: int = 2,
    n_blocks: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a spatial block mask for a 2D lattice.

    Masks one or more contiguous block_size x block_size regions.

    Args:
        Lx, Ly: lattice dimensions
        block_size: size of masked block (block_size x block_size)
        n_blocks: number of blocks to mask
        device: torch device

    Returns:
        mask: (Lx * Ly,) bool tensor. True = visible, False = masked.
    """
    mask = torch.ones(Ly, Lx, dtype=torch.bool, device=device)

    for _ in range(n_blocks):
        # Random top-left corner (with wrapping for PBC)
        x0 = torch.randint(0, Lx, (1,)).item()
        y0 = torch.randint(0, Ly, (1,)).item()

        for dy in range(block_size):
            for dx in range(block_size):
                x = (x0 + dx) % Lx
                y = (y0 + dy) % Ly
                mask[y, x] = False

    return mask.flatten()


def generate_random_mask(
    n_sites: int,
    mask_ratio: float = 0.25,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a random (non-spatial) mask.

    Args:
        n_sites: total number of sites
        mask_ratio: fraction of sites to mask
        device: torch device

    Returns:
        mask: (n_sites,) bool tensor. True = visible, False = masked.
    """
    n_mask = max(1, int(n_sites * mask_ratio))
    mask = torch.ones(n_sites, dtype=torch.bool, device=device)
    mask_indices = torch.randperm(n_sites, device=device)[:n_mask]
    mask[mask_indices] = False
    return mask


def generate_row_mask(
    Lx: int,
    Ly: int,
    n_rows: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Mask entire rows of the lattice.

    Args:
        Lx, Ly: lattice dimensions
        n_rows: number of rows to mask
        device: torch device

    Returns:
        mask: (Lx * Ly,) bool tensor. True = visible, False = masked.
    """
    mask = torch.ones(Ly, Lx, dtype=torch.bool, device=device)
    rows = torch.randperm(Ly, device=device)[:n_rows]
    for r in rows:
        mask[r, :] = False
    return mask.flatten()


def generate_batch_masks(
    batch_size: int,
    Lx: int,
    Ly: int,
    mask_type: str = "block",
    block_size: int = 2,
    mask_ratio: float = 0.25,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a batch of masks.

    Args:
        batch_size: number of masks to generate
        Lx, Ly: lattice dimensions
        mask_type: "block", "random", or "row"
        block_size: for block masking
        mask_ratio: for random masking
        device: torch device

    Returns:
        masks: (batch_size, Lx*Ly) bool tensor
    """
    N = Lx * Ly
    masks = torch.ones(batch_size, N, dtype=torch.bool, device=device)

    for b in range(batch_size):
        if mask_type == "block":
            # Determine number of blocks to reach ~mask_ratio
            block_area = block_size * block_size
            n_blocks = max(1, int(N * mask_ratio / block_area))
            masks[b] = generate_block_mask(Lx, Ly, block_size, n_blocks, device)
        elif mask_type == "random":
            masks[b] = generate_random_mask(N, mask_ratio, device)
        elif mask_type == "row":
            n_rows = max(1, int(Ly * mask_ratio))
            masks[b] = generate_row_mask(Lx, Ly, n_rows, device)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    return masks
