"""
Main experiment runner for Lattice-JEPA.

Runs the full experiment matrix:
- Transformer backbone
- 2 initializations (random, JEPA-pretrained)
- 2 coupling strengths (U/t=4, U/t=8)
- Multiple lattice sizes (4x4, 6x6, 8x8)
- Multiple seeds for statistical significance

Usage:
    python experiments/run_experiment.py --phase all
    python experiments/run_experiment.py --phase pretrain --backbone transformer
    python experiments/run_experiment.py --phase vmc --lattice 4x4 --U 4
"""

import torch
import numpy as np
import yaml
import os
import json
import argparse
import copy
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pretrain import pretrain, create_backbone
from training.finetune import finetune, create_model
from physics.hubbard import SquareLattice
from physics.observables import ObservableCalculator
from physics.symmetry import SymmetrizedNQS


# ---- Experiment configuration ----

BACKBONES = ['transformer']
INIT_TYPES = ['random', 'jepa']
U_VALUES = [4.0, 8.0]

LATTICE_SIZES = {
    'small': [(4, 4)],
    'medium': [(4, 4), (6, 6)],
    'full': [(4, 4), (6, 6), (8, 8)],
}

SEEDS = [42, 123, 456, 789, 1024]

# ED reference energies per site (for validation)
ED_REFERENCE = {
    (4, 4, 4.0): -1.0959,
    (4, 4, 8.0): -1.0288,
}

AFQMC_REFERENCE = {
    (8, 8, 4.0): -0.8603,
    (8, 8, 8.0): -0.5257,
}

# U-ramp schedules for different target U values
U_RAMP_SCHEDULES = {
    4.0: [
        {'U': 1.0, 'steps': 500},
        {'U': 2.0, 'steps': 500},
        {'U': 4.0, 'steps': 2000},
    ],
    8.0: [
        {'U': 1.0, 'steps': 500},
        {'U': 2.0, 'steps': 500},
        {'U': 4.0, 'steps': 1000},
        {'U': 6.0, 'steps': 1000},
        {'U': 8.0, 'steps': 2000},
    ],
}


def load_config(config_path: str) -> dict:
    """Load and return config dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def modify_config_for_lattice(cfg: dict, Lx: int, Ly: int) -> dict:
    """Create a modified config for a specific lattice size."""
    cfg = copy.deepcopy(cfg)
    cfg['physics']['Lx'] = Lx
    cfg['physics']['Ly'] = Ly

    # Adjust mask block size for larger lattices
    N = Lx * Ly
    if Lx >= 6 or Ly >= 6:
        cfg['jepa']['mask_block_size'] = 3
    else:
        cfg['jepa']['mask_block_size'] = 2

    # Adjust JEPA pretrain samples for larger lattices
    if N > 36:
        cfg['jepa']['n_pretrain_samples'] = 300000

    # Adjust batch sizes for memory constraints
    if N > 36:
        cfg['jepa']['batch_size'] = 128
        cfg['vmc']['n_chains'] = 128

    return cfg


def get_pretrained_path(save_root: str, backbone: str, Lx: int, Ly: int) -> str:
    """Get path to pretrained JEPA checkpoint."""
    return os.path.join(
        save_root, 'pretrain', f'{Lx}x{Ly}', f'{backbone}_jepa_best.pt'
    )


def run_pretrain_phase(
    cfg: dict,
    backbone: str,
    Lx: int,
    Ly: int,
    save_root: str,
    device: str,
):
    """Run JEPA pretraining for one backbone + lattice size."""
    lattice_cfg = modify_config_for_lattice(cfg, Lx, Ly)

    save_dir = os.path.join(save_root, 'pretrain', f'{Lx}x{Ly}')
    os.makedirs(save_dir, exist_ok=True)

    # Save the modified config
    config_path = os.path.join(save_dir, f'{backbone}_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(lattice_cfg, f)

    print(f"\n{'='*60}")
    print(f"JEPA PRETRAINING: {backbone}, {Lx}x{Ly} lattice")
    print(f"{'='*60}")

    t_start = time.time()
    history = pretrain(
        backbone_type=backbone,
        config_path=config_path,
        save_dir=save_dir,
        device=device,
    )
    elapsed = time.time() - t_start

    print(f"Pretraining completed in {elapsed:.1f}s")
    return history


def run_vmc_phase(
    cfg: dict,
    backbone: str,
    init_type: str,
    U: float,
    Lx: int,
    Ly: int,
    seed: int,
    save_root: str,
    device: str,
    use_symmetry: bool = False,
):
    """Run VMC fine-tuning for one configuration."""
    lattice_cfg = modify_config_for_lattice(cfg, Lx, Ly)

    # Set U-ramp schedule
    if U in U_RAMP_SCHEDULES:
        lattice_cfg['uramp']['enabled'] = True
        lattice_cfg['uramp']['schedule'] = U_RAMP_SCHEDULES[U]
    else:
        lattice_cfg['uramp']['enabled'] = False

    # Save modified config
    save_dir = os.path.join(save_root, 'vmc', f'{Lx}x{Ly}')
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, f'{backbone}_{init_type}_U{U:.0f}_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(lattice_cfg, f)

    # Get pretrained path if needed
    pretrained_path = None
    if init_type == 'jepa':
        pretrained_path = get_pretrained_path(save_root, backbone, Lx, Ly)
        if not os.path.exists(pretrained_path):
            print(f"WARNING: Pretrained model not found at {pretrained_path}")
            print(f"  Run pretraining first: --phase pretrain --backbone {backbone}")
            return None

    print(f"\n{'='*60}")
    print(f"VMC: {backbone} + {init_type}, {Lx}x{Ly}, U/t={U}, seed={seed}")
    if use_symmetry:
        print(f"  (with translation symmetry projection)")
    print(f"{'='*60}")

    t_start = time.time()
    history, best_e = finetune(
        backbone_type=backbone,
        pretrained_path=pretrained_path,
        config_path=config_path,
        save_dir=os.path.join(save_root, 'vmc', f'{Lx}x{Ly}'),
        device=device,
        target_U=U,
        seed=seed,
    )
    elapsed = time.time() - t_start

    # Compare to reference
    ref_key = (Lx, Ly, U)
    ref = ED_REFERENCE.get(ref_key) or AFQMC_REFERENCE.get(ref_key)
    if ref is not None:
        rel_err = abs(best_e - ref) / abs(ref) * 100
        print(f"  Reference E/N = {ref:.4f}, relative error = {rel_err:.2f}%")

    print(f"VMC completed in {elapsed:.1f}s, best E/N = {best_e:.6f}")

    return {
        'backbone': backbone,
        'init': init_type,
        'U': U,
        'lattice': f'{Lx}x{Ly}',
        'seed': seed,
        'best_energy_per_site': best_e,
        'time_seconds': elapsed,
        'reference': ref,
        'use_symmetry': use_symmetry,
    }


def run_observables(
    cfg: dict,
    backbone: str,
    init_type: str,
    U: float,
    Lx: int,
    Ly: int,
    seed: int,
    save_root: str,
    device: str,
):
    """Compute physical observables for a trained model."""
    from physics.sampler import MetropolisSampler

    lattice_cfg = modify_config_for_lattice(cfg, Lx, Ly)
    N = Lx * Ly
    n_up = N // 2
    n_down = N // 2

    # Load trained model
    run_name = f'{backbone}_{init_type}_U{U:.0f}_seed{seed}'
    model_path = os.path.join(save_root, 'vmc', f'{Lx}x{Ly}', run_name, 'model.pt')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    print(f"\nComputing observables: {run_name}, {Lx}x{Ly}")

    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(backbone, N, lattice_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Sample configurations
    lattice = SquareLattice(Lx, Ly, pbc=True)
    sampler = MetropolisSampler(
        lattice=lattice,
        n_chains=1024,
        n_sweeps=20,
        n_thermalize=500,
        device=device,
    )
    sampler.initialize_chains(n_up=n_up, n_down=n_down)

    with torch.no_grad():
        configs = sampler.sample(model)

    # Compute observables
    obs_calc = ObservableCalculator(lattice)
    observables = obs_calc.compute_all(configs)

    # Save
    obs_dir = os.path.join(save_root, 'observables', f'{Lx}x{Ly}', run_name)
    os.makedirs(obs_dir, exist_ok=True)

    # Convert numpy arrays for JSON serialization
    serializable = {}
    for k, v in observables.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v

    with open(os.path.join(obs_dir, 'observables.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"  Double occupancy: {observables['double_occupancy']:.4f}")
    print(f"  S(pi,pi): {observables['S_pi_pi']:.4f}")

    return observables


def collect_results(save_root: str) -> list[dict]:
    """Collect all VMC results into a summary table."""
    results = []
    vmc_dir = os.path.join(save_root, 'vmc')

    if not os.path.exists(vmc_dir):
        return results

    for lattice_dir in sorted(os.listdir(vmc_dir)):
        lattice_path = os.path.join(vmc_dir, lattice_dir)
        if not os.path.isdir(lattice_path):
            continue

        for run_dir in sorted(os.listdir(lattice_path)):
            summary_path = os.path.join(lattice_path, run_dir, 'summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                summary['run_dir'] = run_dir
                results.append(summary)

    return results


def print_results_table(results: list[dict]):
    """Print a formatted table of all results."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f"{'RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Backbone':<12} {'Init':<8} {'Lattice':<8} {'U/t':<6} "
          f"{'E/N':<12} {'Ref':<12} {'Err%':<8} {'Var':<10}")
    print(f"{'-'*80}")

    for r in sorted(results, key=lambda x: (x['lattice'], x['U_over_t'], x['backbone'], x['init'])):
        backbone = r['backbone']
        init = r['init']
        lattice = r['lattice']
        u_over_t = r['U_over_t']
        e_n = r['best_energy_per_site']
        var = r.get('final_variance', None)

        # Look up reference
        Lx, Ly = map(int, lattice.split('x'))
        ref_key = (Lx, Ly, u_over_t)
        ref = ED_REFERENCE.get(ref_key) or AFQMC_REFERENCE.get(ref_key)

        ref_str = f'{ref:.4f}' if ref else '--'
        if ref:
            err = abs(e_n - ref) / abs(ref) * 100
            err_str = f'{err:.2f}%'
        else:
            err_str = '--'
        var_str = f'{var:.4e}' if var is not None else '--'

        print(f"{backbone:<12} {init:<8} {lattice:<8} {u_over_t:<6.1f} "
              f"{e_n:<12.6f} {ref_str:<12} {err_str:<8} {var_str:<10}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Lattice-JEPA Experiment Runner')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['pretrain', 'vmc', 'observables', 'all', 'summary'],
                       help='Which phase to run')
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['transformer'],
                       help='Specific backbone (default: both)')
    parser.add_argument('--init', type=str, default=None,
                       choices=['random', 'jepa'],
                       help='Specific init type (default: both)')
    parser.add_argument('--U', type=float, default=None,
                       help='Specific U/t value (default: 4 and 8)')
    parser.add_argument('--lattice', type=str, default=None,
                       help='Specific lattice size, e.g. "4x4" (default: all)')
    parser.add_argument('--scale', type=str, default='small',
                       choices=['small', 'medium', 'full'],
                       help='Lattice size scale')
    parser.add_argument('--n-seeds', type=int, default=1,
                       help='Number of seeds to run (default: 1)')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--save-root', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--symmetry', action='store_true',
                       help='Use translation symmetry projection')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Determine what to run
    backbones = [args.backbone] if args.backbone else BACKBONES
    init_types = [args.init] if args.init else INIT_TYPES
    u_values = [args.U] if args.U else U_VALUES
    seeds = SEEDS[:args.n_seeds]

    if args.lattice:
        Lx, Ly = map(int, args.lattice.split('x'))
        lattice_sizes = [(Lx, Ly)]
    else:
        lattice_sizes = LATTICE_SIZES[args.scale]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    print(f"Lattice-JEPA Experiment Runner")
    print(f"  Phase: {args.phase}")
    print(f"  Backbones: {backbones}")
    print(f"  Init types: {init_types}")
    print(f"  U/t values: {u_values}")
    print(f"  Lattice sizes: {lattice_sizes}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {args.device}")
    print(f"  Symmetry: {args.symmetry}")

    # Phase 1: Pretraining
    if args.phase in ('pretrain', 'all'):
        for Lx, Ly in lattice_sizes:
            for backbone in backbones:
                # Only pretrain if JEPA init is requested
                if 'jepa' in init_types:
                    pretrained_path = get_pretrained_path(
                        args.save_root, backbone, Lx, Ly
                    )
                    if os.path.exists(pretrained_path):
                        print(f"\nSkipping pretrain (exists): {backbone}, {Lx}x{Ly}")
                        continue
                    try:
                        run_pretrain_phase(
                            cfg, backbone, Lx, Ly,
                            args.save_root, args.device
                        )
                    except Exception as e:
                        print(f"ERROR in pretrain {backbone} {Lx}x{Ly}: {e}")
                        import traceback
                        traceback.print_exc()

    # Phase 2: VMC fine-tuning
    if args.phase in ('vmc', 'all'):
        for Lx, Ly in lattice_sizes:
            for backbone in backbones:
                for init_type in init_types:
                    for U in u_values:
                        for seed in seeds:
                            try:
                                result = run_vmc_phase(
                                    cfg, backbone, init_type, U,
                                    Lx, Ly, seed,
                                    args.save_root, args.device,
                                    use_symmetry=args.symmetry,
                                )
                                if result:
                                    all_results.append(result)
                            except Exception as e:
                                print(f"ERROR in VMC {backbone}+{init_type} "
                                      f"U={U} {Lx}x{Ly} seed={seed}: {e}")
                                import traceback
                                traceback.print_exc()

    # Phase 3: Observables
    if args.phase in ('observables', 'all'):
        for Lx, Ly in lattice_sizes:
            for backbone in backbones:
                for init_type in init_types:
                    for U in u_values:
                        for seed in seeds:
                            try:
                                run_observables(
                                    cfg, backbone, init_type, U,
                                    Lx, Ly, seed,
                                    args.save_root, args.device,
                                )
                            except Exception as e:
                                print(f"ERROR in observables: {e}")

    # Summary
    if args.phase in ('summary', 'all'):
        results = collect_results(args.save_root)
        print_results_table(results)

        # Save summary CSV
        if results:
            summary_path = os.path.join(args.save_root, f'summary_{timestamp}.json')
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nFull summary saved to: {summary_path}")

    # Print any results from this run
    if all_results:
        print_results_table(all_results)


if __name__ == '__main__':
    main()
