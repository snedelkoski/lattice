"""
Stage 2: VMC fine-tuning script.

Trains a Transformer NQS model on the VMC objective to find
the ground state of the Fermi-Hubbard model. Optionally loads pretrained
JEPA weights for the backbone.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import json
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_nqs import TransformerNQS
from physics.hubbard import SquareLattice, FermiHubbardHamiltonian
from physics.symmetry import SymmetrizedNQS, MarshallSignNQS
from physics.vmc import VMCTrainer


ED_REFERENCES = {
    (4, 4, 4.0): -1.0959,
    (4, 4, 8.0): -1.0288,
}


def create_model(
    backbone_type: str, n_sites: int, cfg: dict,
    n_up: int = 0, n_down: int = 0,
) -> nn.Module:
    """Create Transformer NQS model."""
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
            use_layernorm=model_cfg.get('use_layernorm', True),
            head_mode=model_cfg.get('head_mode', 'scalar'),
            n_determinants=model_cfg.get('n_determinants', 4),
            n_up=n_up,
            n_down=n_down,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")


def finetune(
    backbone_type: str = 'transformer',
    pretrained_path: str = None,
    config_path: str = 'configs/default.yaml',
    save_dir: str = 'results/vmc',
    device: str = 'cuda',
    target_U: float = None,
    seed: int = 42,
):
    """Run VMC fine-tuning."""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    physics_cfg = cfg['physics']
    vmc_cfg = cfg['vmc']

    Lx = physics_cfg['Lx']
    Ly = physics_cfg['Ly']
    N = Lx * Ly
    t = physics_cfg['t']
    U = target_U if target_U is not None else physics_cfg['U']
    n_up = N // 2 if physics_cfg['n_up'] == -1 else physics_cfg['n_up']
    n_down = N // 2 if physics_cfg['n_down'] == -1 else physics_cfg['n_down']
    pbc = physics_cfg['pbc']

    init_type = 'jepa' if pretrained_path else 'random'
    run_name = f'{backbone_type}_{init_type}_U{U:.0f}_seed{seed}'
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"VMC Fine-tuning: {backbone_type} backbone, {init_type} init")
    print(f"Lattice: {Lx}x{Ly}, U/t={U/t:.1f}, PBC={pbc}")
    print(f"Particles: {n_up} up, {n_down} down (half-filling={n_up==N//2})")

    # Create lattice and model
    lattice = SquareLattice(Lx, Ly, pbc=pbc)
    model = create_model(
        backbone_type, N, cfg, n_up=n_up, n_down=n_down
    ).to(device)

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        backbone_state = checkpoint.get('backbone_state_dict', checkpoint)
        model.load_backbone_state_dict(backbone_state, strict=False)
    elif pretrained_path:
        print(f"WARNING: Pretrained path not found: {pretrained_path}")

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Keep reference to the raw (unwrapped) model for SR Jacobian computation
    raw_model = model

    head_mode = cfg['model'].get('head_mode', 'scalar')
    print(f"Head mode: {head_mode}")
    if head_mode == 'backflow_det':
        print(f"  K={cfg['model'].get('n_determinants', 4)} determinants, "
              f"N_e={n_up + n_down} electrons")

    # Wrap model with symmetry projection if enabled
    use_symmetry = vmc_cfg.get('use_symmetry', False)
    use_marshall = vmc_cfg.get('marshall_sign', False)
    force_real = vmc_cfg.get('force_real', False)

    # For backflow_det mode, Marshall sign and force_real are handled
    # naturally by the determinant — skip those wrappers
    if head_mode == 'scalar':
        # Apply Marshall sign rule first (modifies model's force_real setting)
        if use_marshall and pbc:
            print("Marshall sign rule: ENABLED")
            model = MarshallSignNQS(model, lattice)
        elif force_real:
            model.set_force_real(True)
            print("Force real wavefunction: ENABLED")

    # Then wrap with symmetry projection
    if use_symmetry and pbc:
        print("Symmetry projection: ENABLED (translations)")
        vmc_model = SymmetrizedNQS(model, lattice, use_point_group=False)
    else:
        vmc_model = model
        if use_symmetry and not pbc:
            print("Symmetry projection: DISABLED (requires PBC)")
        else:
            print("Symmetry projection: DISABLED")

    vmc_model = vmc_model.to(device)

    # Build U-ramp schedule
    u_ramp = None
    uramp_cfg = cfg.get('uramp', {})
    if uramp_cfg.get('enabled', False) and U > 2.0:
        u_ramp = []
        for stage in uramp_cfg['schedule']:
            if stage['U'] <= U:
                u_ramp.append(stage)
        # Add final U if not in schedule
        if not u_ramp or u_ramp[-1]['U'] < U:
            u_ramp.append({'U': U, 'steps': vmc_cfg['n_steps'] // 2})
        print(f"U-ramping: {[s['U'] for s in u_ramp]}")

    # Create VMC trainer
    trainer = VMCTrainer(
        model=vmc_model,
        lattice=lattice,
        t=t,
        U=U,
        n_up=n_up,
        n_down=n_down,
        lr=vmc_cfg['lr'],
        weight_decay=vmc_cfg['weight_decay'],
        n_chains=vmc_cfg['n_chains'],
        n_sweeps=vmc_cfg['n_sweeps'],
        n_thermalize=vmc_cfg['n_thermalize'],
        grad_clip=vmc_cfg['grad_clip'],
        e_loc_clip=vmc_cfg['e_loc_clip'],
        device=device,
        use_amp=(vmc_cfg.get('precision', 'fp32') == 'bf16'),
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
        raw_model=raw_model,
    )

    # Train
    if u_ramp:
        history = trainer.train(
            n_steps=sum(s['steps'] for s in u_ramp),
            log_interval=cfg['experiment']['log_interval'],
            u_ramp_schedule=u_ramp,
        )
    else:
        history = trainer.train(
            n_steps=vmc_cfg['n_steps'],
            log_interval=cfg['experiment']['log_interval'],
        )

    # Save results
    restored_target = trainer.restore_best_state_for_U(U)
    if restored_target:
        best_target_e = trainer.get_best_energy_for_U(U)
        print(f"\nBest E/N at target U={U/t:.1f}: {best_target_e:.6f}")
    else:
        best_target_e = float('inf')
        print(f"\nBest E/N at target U={U/t:.1f}: not available")

    trainer.restore_best_state()
    best_e = trainer.get_best_energy()
    print(f"Best E/N (global across all stages) = {best_e:.6f}")
    ed_ref = ED_REFERENCES.get((Lx, Ly, float(U)))
    if ed_ref is not None:
        print(f"ED reference ({Lx}x{Ly}, U/t={U/t:.1f}): E/N = {ed_ref:.4f}")

    # Save model
    model_path = os.path.join(run_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'backbone_type': backbone_type,
        'init_type': init_type,
        'U': U,
        'seed': seed,
        'best_energy_per_site': best_e,
        'best_target_u_energy_per_site': best_target_e,
    }, model_path)

    # Save history
    history_path = os.path.join(run_dir, 'history.json')
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)

    # Save summary
    summary = {
        'backbone': backbone_type,
        'init': init_type,
        'lattice': f'{Lx}x{Ly}',
        'U_over_t': U / t,
        'n_params': n_params,
        'best_energy_per_site': best_e,
        'best_target_u_energy_per_site': best_target_e,
        'ed_reference_per_site': ed_ref,
        'final_variance': history['variance'][-1] if history['variance'] else None,
        'final_acceptance_rate': history['acceptance_rate'][-1] if history['acceptance_rate'] else None,
        'seed': seed,
    }
    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {run_dir}")
    return history, best_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VMC Fine-tuning')
    parser.add_argument('--backbone', type=str, default='transformer',
                       choices=['transformer'])
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained JEPA checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--save-dir', type=str, default='results/vmc')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--U', type=float, default=None,
                       help='Target U/t (overrides config)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    finetune(
        backbone_type=args.backbone,
        pretrained_path=args.pretrained,
        config_path=args.config,
        save_dir=args.save_dir,
        device=args.device,
        target_U=args.U,
        seed=args.seed,
    )
