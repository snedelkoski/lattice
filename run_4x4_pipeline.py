#!/usr/bin/env python
"""
Full pipeline: JEPA Pretraining -> VMC Finetuning -> Evaluation
on 4x4 Fermi-Hubbard at U/t=4, half-filling.

Runs:
  1. JEPA pretraining (self-supervised, random lattice configs)
  2. VMC finetuning: random init (baseline)
  3. VMC finetuning: JEPA full + differential LR (best strategy from 2x2)
  4. VMC finetuning: JEPA full (naive, for comparison)
  5. Compare: energy, convergence speed, variance

ED reference: E/N = -1.0959 (4x4, U=4, half-filling, PBC)

Memory budget: RTX 3050 3.7GB VRAM
  - d_model=128, 4 layers, 4 heads: ~357K params
  - MARCH with 512 chains: ~2.5GB peak (fits)
"""

import torch
import numpy as np
import yaml
import os
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_nqs import TransformerNQS
from models.jepa import LatticeJEPA
from models.masking import generate_batch_masks
from physics.hubbard import SquareLattice, FermiHubbardHamiltonian, generate_random_configs
from physics.vmc import VMCTrainer


# ── Constants ──────────────────────────────────────────────────
CONFIG_PATH = "configs/4x4_pipeline.yaml"
SAVE_DIR = "results/4x4_pipeline"
ED_REFERENCE = -1.0959  # E/N for 4x4 U=4 half-filling PBC


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Stage 1: JEPA Pretraining ─────────────────────────────────
def run_jepa_pretraining(cfg, device='cuda'):
    """Run JEPA pretraining and save backbone checkpoint."""
    print("\n" + "=" * 60)
    print("STAGE 1: JEPA PRETRAINING (4x4)")
    print("=" * 60)

    physics = cfg['physics']
    model_cfg = cfg['model']
    jepa_cfg = cfg['jepa']

    Lx, Ly = physics['Lx'], physics['Ly']
    N = Lx * Ly
    n_up = N // 2 if physics['n_up'] == -1 else physics['n_up']
    n_down = N // 2 if physics['n_down'] == -1 else physics['n_down']

    save_dir = os.path.join(SAVE_DIR, "pretrain")
    os.makedirs(save_dir, exist_ok=True)

    # Generate pretraining data
    print(f"Generating {jepa_cfg['n_pretrain_samples']} random configs "
          f"for {Lx}x{Ly} lattice...")
    configs_np = generate_random_configs(
        n_samples=jepa_cfg['n_pretrain_samples'],
        n_sites=N, n_up=n_up, n_down=n_down,
    )
    configs_tensor = torch.from_numpy(configs_np).long()
    dataset = torch.utils.data.TensorDataset(configs_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=jepa_cfg['batch_size'],
        shuffle=True, drop_last=True,
    )

    # Create backbone (same architecture as VMC model)
    backbone = TransformerNQS(
        n_sites=N,
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_blocks'],
        d_ff=model_cfg['d_ff'],
        vocab_size=model_cfg['vocab_size'],
        max_sites=model_cfg['max_sites'],
        use_layernorm=model_cfg.get('use_layernorm', True),
    )

    # Wrap in JEPA
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
    n_backbone_params = sum(p.numel() for p in backbone.parameters())
    print(f"JEPA total params: {n_params:,} (backbone: {n_backbone_params:,})")

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

    # Masking strategy
    mask_type = jepa_cfg.get('mask_type', 'block')
    block_size = jepa_cfg.get('mask_block_size', 2)
    if mask_type == 'block' and block_size >= min(Lx, Ly):
        print(f"  block_size={block_size} >= lattice dim, using random masking")
        mask_type = 'random'

    print(f"Masking: type={mask_type}, block_size={block_size}, "
          f"ratio={jepa_cfg['mask_ratio']}")
    print(f"Training: {jepa_cfg['epochs']} epochs, "
          f"{len(dataloader)} batches/epoch, {total_steps} total steps")

    # Training
    history = {'loss': [], 'mse': [], 'sigreg': []}
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(jepa_cfg['epochs']):
        jepa_model.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_sigreg = 0
        n_batches = 0

        for (batch_configs,) in dataloader:
            batch_configs = batch_configs.to(device)
            B = batch_configs.shape[0]

            masks = generate_batch_masks(
                batch_size=B, Lx=Lx, Ly=Ly,
                mask_type=mask_type,
                block_size=block_size,
                mask_ratio=jepa_cfg['mask_ratio'],
                device=device,
            )

            outputs = jepa_model(batch_configs, masks)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jepa_model.parameters(),
                                           jepa_cfg['grad_clip'])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mse += outputs['mse_loss'].item()
            epoch_sigreg += outputs['sigreg_loss'].item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_sigreg = epoch_sigreg / n_batches

        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        history['sigreg'].append(avg_sigreg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{jepa_cfg['epochs']}: "
                  f"loss={avg_loss:.4f} mse={avg_mse:.4f} sig={avg_sigreg:.4f} "
                  f"[{elapsed:.0f}s]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, "jepa_best.pt")
            torch.save({
                'backbone_state_dict': jepa_model.get_backbone_state_dict(),
                'full_state_dict': jepa_model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': cfg,
            }, checkpoint_path)

    elapsed = time.time() - t_start
    print(f"\nJEPA pretraining complete in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final MSE: {history['mse'][-1]:.4f}")
    print(f"  Final SIGReg: {history['sigreg'][-1]:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Save history
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f)

    return checkpoint_path, history


# ── Stage 2: VMC Finetuning ───────────────────────────────────
def run_vmc_finetuning(cfg, pretrained_path=None, label="random",
                       device='cuda', seed=42, transfer_mode="full",
                       backbone_lr_multiplier=1.0):
    """
    Run VMC finetuning with various JEPA transfer strategies.

    Args:
        transfer_mode: How to load JEPA weights:
            "full" — load entire backbone (token_embed + pos_embed + transformer layers)
            "embed_only" — load only token_embed + pos_embed
            "none" — random init (no pretrained weights)
        backbone_lr_multiplier: LR multiplier for backbone params relative to head.
            e.g., 10.0 means backbone gets 10x the base LR.
    """
    init_desc = f"{transfer_mode}" if pretrained_path else "random"
    if backbone_lr_multiplier != 1.0 and pretrained_path:
        init_desc += f" (backbone_lr={backbone_lr_multiplier}x)"
    print(f"\n{'=' * 60}")
    print(f"STAGE 2: VMC FINETUNING ({init_desc} init, seed={seed})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    physics = cfg['physics']
    model_cfg = cfg['model']
    vmc_cfg = cfg['vmc']

    Lx, Ly = physics['Lx'], physics['Ly']
    N = Lx * Ly
    t = physics['t']
    U = physics['U']
    n_up = N // 2 if physics['n_up'] == -1 else physics['n_up']
    n_down = N // 2 if physics['n_down'] == -1 else physics['n_down']

    run_dir = os.path.join(SAVE_DIR, f"vmc_{label}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Create model
    model = TransformerNQS(
        n_sites=N,
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_blocks'],
        d_ff=model_cfg['d_ff'],
        vocab_size=model_cfg['vocab_size'],
        max_sites=model_cfg['max_sites'],
        use_layernorm=model_cfg.get('use_layernorm', True),
        head_mode=model_cfg.get('head_mode', 'scalar'),
        n_determinants=model_cfg.get('n_determinants', 4),
        n_up=n_up,
        n_down=n_down,
    ).to(device)

    # Load pretrained backbone
    if pretrained_path and os.path.exists(pretrained_path) and transfer_mode != "none":
        print(f"Loading JEPA backbone from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device,
                                weights_only=False)
        backbone_state = checkpoint.get('backbone_state_dict', checkpoint)

        if transfer_mode == "embed_only":
            embed_state = {k: v for k, v in backbone_state.items()
                          if 'embed' in k}
            print(f"  Transfer mode: embed_only — "
                  f"loading {list(embed_state.keys())}")
            model.load_backbone_state_dict(embed_state, strict=False)
        else:
            print(f"  Transfer mode: full backbone")
            model.load_backbone_state_dict(backbone_state, strict=False)
    elif pretrained_path and transfer_mode != "none":
        print(f"WARNING: pretrained path not found: {pretrained_path}")

    n_params = model.count_parameters()
    print(f"Model params: {n_params:,}")
    print(f"Head: {model_cfg.get('head_mode', 'scalar')}, "
          f"K={model_cfg.get('n_determinants', 4)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create lattice and trainer
    lattice = SquareLattice(Lx, Ly, pbc=physics['pbc'])

    trainer = VMCTrainer(
        model=model,
        lattice=lattice,
        t=t, U=U,
        n_up=n_up, n_down=n_down,
        lr=vmc_cfg['lr'],
        weight_decay=vmc_cfg['weight_decay'],
        n_chains=vmc_cfg['n_chains'],
        n_sweeps=vmc_cfg['n_sweeps'],
        n_thermalize=vmc_cfg['n_thermalize'],
        grad_clip=vmc_cfg['grad_clip'],
        e_loc_clip=vmc_cfg['e_loc_clip'],
        device=device,
        use_amp=False,
        log_amp_reg=vmc_cfg.get('log_amp_reg', 0.0),
        log_amp_reg_decay_steps=vmc_cfg.get('log_amp_reg_decay_steps', 0),
        phase_reg=vmc_cfg.get('phase_reg', 0.0),
        lr_schedule_mode=vmc_cfg.get('lr_schedule_mode', 'global_cosine'),
        min_lr=vmc_cfg.get('min_lr', 1e-5),
        exact_sampling=vmc_cfg.get('exact_sampling', False),
        keep_best_state=vmc_cfg.get('keep_best_state', True),
        use_sr=vmc_cfg.get('use_sr', False),
        sr_lr=vmc_cfg.get('sr_lr', 0.02),
        sr_diag_shift=vmc_cfg.get('sr_diag_shift', 0.01),
        sr_n_chains=vmc_cfg.get('sr_n_chains', 128),
        sr_min_U=vmc_cfg.get('sr_min_U', 2.0),
        sr_optimizer=vmc_cfg.get('sr_optimizer', 'minsr'),
        sr_momentum=vmc_cfg.get('sr_momentum', 0.95),
        sr_beta=vmc_cfg.get('sr_beta', 0.995),
        sr_norm_decay_start=vmc_cfg.get('sr_norm_decay_start', 8000),
        sr_chunk_size=vmc_cfg.get('sr_chunk_size', 32),
        raw_model=model,
    )

    # Apply differential LR if needed
    # For MARCH, we need to modify the AdamW optimizer used for non-SR steps
    # AND ensure the backbone gets higher LR in the SR update
    if backbone_lr_multiplier != 1.0 and pretrained_path and transfer_mode == "full":
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if name.startswith('orbital_head'):
                head_params.append(p)
            else:
                backbone_params.append(p)
        base_lr = vmc_cfg['lr']
        trainer.optimizer = torch.optim.AdamW([
            {'params': backbone_params,
             'lr': base_lr * backbone_lr_multiplier},
            {'params': head_params, 'lr': base_lr},
        ], weight_decay=vmc_cfg['weight_decay'])
        print(f"  Differential LR: backbone={base_lr * backbone_lr_multiplier:.1e}, "
              f"head={base_lr:.1e}")

    # U-ramp config
    uramp_cfg = cfg.get('uramp', {})

    # Train
    n_steps = vmc_cfg['n_steps']
    log_interval = cfg['experiment']['log_interval']

    if uramp_cfg.get('enabled', False):
        schedule = uramp_cfg['schedule']
        total_uramp_steps = sum(s['steps'] for s in schedule)
        print(f"\nU-ramp enabled: {len(schedule)} stages, "
              f"{total_uramp_steps} total steps")
        for s in schedule:
            print(f"  U={s['U']:.1f} for {s['steps']} steps")

        # Override total steps from uramp
        n_steps = total_uramp_steps

    t_start = time.time()

    history = trainer.train(
        n_steps=n_steps,
        log_interval=log_interval,
        u_ramp_schedule=uramp_cfg.get('schedule') if uramp_cfg.get('enabled') else None,
    )

    elapsed = time.time() - t_start

    # Get results
    trainer.restore_best_state()
    best_e = trainer.get_best_energy()

    print(f"\nVMC complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best E/N = {best_e:.6f} (ED ref = {ED_REFERENCE:.6f})")
    rel_error = abs(best_e - ED_REFERENCE) / abs(ED_REFERENCE) * 100
    print(f"Relative error: {rel_error:.2f}%")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_energy_per_site': best_e,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'elapsed_seconds': elapsed,
    }, os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "history.json"), 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.items()}, f)

    return best_e, history, run_dir


# ── Stage 3: Evaluation & Comparison ─────────────────────────
def evaluate_and_compare(ed_ref, results, save_dir):
    """Print comparison table and save summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (4x4 Fermi-Hubbard, U/t=4)")
    print("=" * 60)
    print(f"\nED Reference E/N = {ed_ref:.6f}")
    print(f"{'':─<70}")
    print(f"{'Init':<20} {'Best E/N':>10} {'Error%':>8} {'Final Var':>10} "
          f"{'Conv@5%':>8} {'Conv@1%':>8}")
    print(f"{'':─<70}")

    summary = {}
    N = 16  # 4x4
    for label, (best_e, history) in results.items():
        rel_error = abs(best_e - ed_ref) / abs(ed_ref) * 100

        # Final variance (average of last 10%)
        variances = history['variance']
        n_avg = max(1, len(variances) // 10)
        final_var = np.mean(variances[-n_avg:])

        # Steps to converge within X% of ED
        energies = [e / N for e in history['energy']]
        steps_1pct = None
        steps_5pct = None
        for i, e in enumerate(energies):
            err = abs(e - ed_ref) / abs(ed_ref) * 100
            if steps_5pct is None and err < 5.0:
                steps_5pct = i + 1
            if steps_1pct is None and err < 1.0:
                steps_1pct = i + 1

        s1 = f"{steps_1pct}" if steps_1pct else "N/A"
        s5 = f"{steps_5pct}" if steps_5pct else "N/A"

        print(f"{label:<20} {best_e:>+10.6f} {rel_error:>7.2f}% "
              f"{final_var:>10.4f} {s5:>8} {s1:>8}")

        summary[label] = {
            'best_e_per_site': best_e,
            'relative_error_pct': rel_error,
            'final_variance': final_var,
            'steps_to_1pct': steps_1pct,
            'steps_to_5pct': steps_5pct,
        }

    print(f"{'':─<70}")

    # Find best and worst
    best_label = min(summary, key=lambda k: summary[k]['relative_error_pct'])
    worst_label = max(summary, key=lambda k: summary[k]['relative_error_pct'])
    if best_label != worst_label:
        best_err = summary[best_label]['relative_error_pct']
        worst_err = summary[worst_label]['relative_error_pct']
        print(f"\nBest:  {best_label} ({best_err:.2f}% error)")
        print(f"Worst: {worst_label} ({worst_err:.2f}% error)")

    # Save summary
    summary['ed_reference'] = ed_ref
    with open(os.path.join(save_dir, "comparison.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Main ──────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    os.makedirs(SAVE_DIR, exist_ok=True)

    cfg = load_config()
    physics = cfg['physics']
    model_cfg = cfg['model']
    vmc_cfg = cfg['vmc']
    print(f"\nTarget: {physics['Lx']}x{physics['Ly']} Fermi-Hubbard, "
          f"U/t={physics['U']}, half-filling, PBC")
    print(f"Model: d_model={model_cfg['d_model']}, layers={model_cfg['num_blocks']}, "
          f"heads={model_cfg['num_heads']}, K={model_cfg.get('n_determinants', 4)}")
    print(f"VMC: {vmc_cfg['n_chains']} chains, {vmc_cfg['n_sweeps']} sweeps, "
          f"MARCH, {vmc_cfg['n_steps']} steps")
    print(f"ED reference: E/N = {ED_REFERENCE}")

    # Step 1: JEPA pretraining
    pretrained_path, jepa_history = run_jepa_pretraining(cfg, device=device)

    # Step 2: Run VMC variants
    # Based on 2x2 findings: differential LR is the best strategy
    variants = [
        {
            'label': 'random',
            'pretrained_path': None,
            'transfer_mode': 'none',
            'backbone_lr_multiplier': 1.0,
            'desc': 'Random init (baseline)',
        },
        {
            'label': 'jepa_diff_lr',
            'pretrained_path': pretrained_path,
            'transfer_mode': 'full',
            'backbone_lr_multiplier': 10.0,
            'desc': 'JEPA full + differential LR (backbone=10x)',
        },
        {
            'label': 'jepa_full',
            'pretrained_path': pretrained_path,
            'transfer_mode': 'full',
            'backbone_lr_multiplier': 1.0,
            'desc': 'JEPA full backbone transfer (naive)',
        },
    ]

    results = {}
    for v in variants:
        print(f"\n>>> Running variant: {v['desc']}")
        best_e, hist, run_dir = run_vmc_finetuning(
            cfg,
            pretrained_path=v['pretrained_path'],
            label=v['label'],
            device=device,
            seed=42,
            transfer_mode=v['transfer_mode'],
            backbone_lr_multiplier=v['backbone_lr_multiplier'],
        )
        results[v['label']] = (best_e, hist)

        # Clear GPU memory between runs
        torch.cuda.empty_cache()

    # Step 3: Compare all variants
    summary = evaluate_and_compare(ED_REFERENCE, results, SAVE_DIR)

    print(f"\nAll results saved to: {SAVE_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
