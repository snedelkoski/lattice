"""
Stage 1: JEPA pretraining script.

Trains the Transformer backbone on a self-supervised JEPA task
using random lattice configurations with spatial block masking.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_nqs import TransformerNQS
from models.jepa import LatticeJEPA
from models.masking import generate_batch_masks
from physics.hubbard import generate_random_configs


def create_backbone(backbone_type: str, n_sites: int, cfg: dict) -> nn.Module:
    """Create Transformer backbone."""
    model_cfg = cfg['model']
    if backbone_type == 'transformer':
        return TransformerNQS(
            n_sites=n_sites,
            d_model=model_cfg['d_model'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_blocks'],
            d_ff=model_cfg['d_ff'],
            vocab_size=model_cfg['vocab_size'],
            max_sites=model_cfg['max_sites'],
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")


def pretrain(
    backbone_type: str = 'transformer',
    config_path: str = 'configs/default.yaml',
    save_dir: str = 'results/pretrain',
    device: str = 'cuda',
):
    """Run JEPA pretraining."""

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    physics_cfg = cfg['physics']
    jepa_cfg = cfg['jepa']
    model_cfg = cfg['model']

    Lx = physics_cfg['Lx']
    Ly = physics_cfg['Ly']
    N = Lx * Ly
    n_up = N // 2 if physics_cfg['n_up'] == -1 else physics_cfg['n_up']
    n_down = N // 2 if physics_cfg['n_down'] == -1 else physics_cfg['n_down']

    os.makedirs(save_dir, exist_ok=True)

    print(f"JEPA Pretraining: {backbone_type} backbone, {Lx}x{Ly} lattice")

    # Generate pretraining data
    print(f"Generating {jepa_cfg['n_pretrain_samples']} random configurations...")
    configs_np = generate_random_configs(
        n_samples=jepa_cfg['n_pretrain_samples'],
        n_sites=N,
        n_up=n_up,
        n_down=n_down,
    )
    configs_tensor = torch.from_numpy(configs_np).long()
    dataset = TensorDataset(configs_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=jepa_cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    backbone = create_backbone(backbone_type, N, cfg)
    jepa_model = LatticeJEPA(
        backbone=backbone,
        d_model=model_cfg['d_model'],
        d_embed=model_cfg['d_embed'],
        predictor_layers=jepa_cfg['predictor_layers'],
        predictor_heads=jepa_cfg['predictor_heads'],
        lambda_sigreg=jepa_cfg['lambda_sigreg'],
        sigreg_num_proj=jepa_cfg['sigreg_num_proj'],
        sigreg_knots=jepa_cfg['sigreg_knots'],
        max_sites=model_cfg['max_sites'],
    ).to(device)

    n_params = sum(p.numel() for p in jepa_model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        jepa_model.parameters(),
        lr=jepa_cfg['lr'],
        weight_decay=jepa_cfg['weight_decay'],
    )

    # LR scheduler: cosine with warmup
    total_steps = jepa_cfg['epochs'] * len(dataloader)
    warmup_steps = jepa_cfg['warmup_epochs'] * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Determine mask block size
    block_size = jepa_cfg['mask_block_size']
    if Lx >= 6 or Ly >= 6:
        block_size = 3  # larger blocks for larger lattices

    # Training loop
    history = {'loss': [], 'mse': [], 'sigreg': [], 'lr': []}
    best_loss = float('inf')

    for epoch in range(jepa_cfg['epochs']):
        jepa_model.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_sigreg = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{jepa_cfg['epochs']}")
        for (batch_configs,) in pbar:
            batch_configs = batch_configs.to(device)
            B = batch_configs.shape[0]

            # Generate masks
            masks = generate_batch_masks(
                batch_size=B,
                Lx=Lx,
                Ly=Ly,
                mask_type='block',
                block_size=block_size,
                mask_ratio=jepa_cfg['mask_ratio'],
                device=device,
            )

            # Forward pass
            if device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = jepa_model(batch_configs, masks)
            else:
                outputs = jepa_model(batch_configs, masks)

            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                jepa_model.parameters(), jepa_cfg['grad_clip']
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mse += outputs['mse_loss'].item()
            epoch_sigreg += outputs['sigreg_loss'].item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['mse_loss'].item():.4f}",
                'sig': f"{outputs['sigreg_loss'].item():.4f}",
            })

        # Epoch stats
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_sigreg = epoch_sigreg / n_batches
        current_lr = scheduler.get_last_lr()[0]

        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        history['sigreg'].append(avg_sigreg)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, mse={avg_mse:.4f}, "
              f"sigreg={avg_sigreg:.4f}, lr={current_lr:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, f'{backbone_type}_jepa_best.pt')
            torch.save({
                'backbone_state_dict': jepa_model.get_backbone_state_dict(),
                'full_state_dict': jepa_model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': cfg,
            }, save_path)
            print(f"  Saved best model (loss={avg_loss:.4f})")

    # Save final model and history
    final_path = os.path.join(save_dir, f'{backbone_type}_jepa_final.pt')
    torch.save({
        'backbone_state_dict': jepa_model.get_backbone_state_dict(),
        'full_state_dict': jepa_model.state_dict(),
        'epoch': jepa_cfg['epochs'],
        'loss': avg_loss,
        'config': cfg,
        'history': history,
    }, final_path)

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    print(f"Saved to: {save_dir}")

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JEPA Pretraining')
    parser.add_argument('--backbone', type=str, default='transformer',
                       choices=['transformer'])
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--save-dir', type=str, default='results/pretrain')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    pretrain(
        backbone_type=args.backbone,
        config_path=args.config,
        save_dir=args.save_dir,
        device=args.device,
    )
