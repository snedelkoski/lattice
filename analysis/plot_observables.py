"""
Plot physical observables computed from NQS wave functions.

Generates figures showing:
- Spin structure factor S(q) heatmaps
- Double occupancy vs U/t
- Spin-spin correlations in real space
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

COLORS = {
    ('transformer', 'random'): '#2ca02c',
    ('transformer', 'jepa'): '#d62728',
}

MARKERS = {
    ('transformer', 'random'): '^',
    ('transformer', 'jepa'): 'D',
}

LABELS = {
    ('transformer', 'random'): 'Transformer (random)',
    ('transformer', 'jepa'): 'Transformer + JEPA',
}


def load_observables(save_root: str, backbone: str, init: str,
                     U: float, lattice: str, seed: int) -> dict:
    """Load observables for a specific run."""
    run_name = f'{backbone}_{init}_U{U:.0f}_seed{seed}'
    obs_path = os.path.join(
        save_root, 'observables', lattice, run_name, 'observables.json'
    )
    if not os.path.exists(obs_path):
        return None
    with open(obs_path) as f:
        return json.load(f)


def plot_structure_factor_heatmap(
    save_root: str,
    output_dir: str,
    lattice: str = '4x4',
    U: float = 4.0,
    seed: int = 42,
):
    """Plot S(q) heatmaps for all backbone/init combinations."""
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ('transformer', 'random'), ('transformer', 'jepa'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    Lx, Ly = map(int, lattice.split('x'))

    for ax, (backbone, init) in zip(axes, configs):
        obs = load_observables(save_root, backbone, init, U, lattice, seed)
        if obs is None:
            ax.set_title(f'{backbone}+{init}\n(no data)')
            continue

        sf = obs.get('spin_structure_factor', {})
        sq_full = sf.get('full')
        if sq_full is None:
            ax.set_title(f'{backbone}+{init}\n(no S(q))')
            continue

        sq = np.array(sq_full)

        # Plot
        qx = 2 * np.pi * np.arange(Lx) / Lx
        qy = 2 * np.pi * np.arange(Ly) / Ly
        im = ax.imshow(
            sq, origin='lower', cmap='hot', aspect='equal',
            extent=[qx[0], qx[-1] + 2*np.pi/Lx, qy[0], qy[-1] + 2*np.pi/Ly],
        )
        ax.set_xlabel('$q_x$')
        ax.set_ylabel('$q_y$')
        label = LABELS.get((backbone, init), f'{backbone}+{init}')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'Spin Structure Factor S(q): {lattice}, U/t={U:.0f}', fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(output_dir, f'structure_factor_{lattice}_U{U:.0f}.png')
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_double_occupancy_vs_U(
    save_root: str,
    output_dir: str,
    lattice: str = '4x4',
    seed: int = 42,
):
    """Plot double occupancy as a function of U/t."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    u_values = [1.0, 2.0, 4.0, 6.0, 8.0]

    for backbone, init in [('transformer', 'random'), ('transformer', 'jepa')]:
        d_occ = []
        u_plot = []
        for U in u_values:
            obs = load_observables(save_root, backbone, init, U, lattice, seed)
            if obs is not None and 'double_occupancy' in obs:
                d_occ.append(obs['double_occupancy'])
                u_plot.append(U)

        if d_occ:
            key = (backbone, init)
            ax.plot(u_plot, d_occ,
                    color=COLORS.get(key, '#333'),
                    marker=MARKERS.get(key, 'o'),
                    label=LABELS.get(key, f'{backbone}+{init}'),
                    linewidth=1.5, markersize=6)

    # Non-interacting reference: D = n_up * n_down = 0.25 at half-filling
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5,
               label='Non-interacting (0.25)')

    ax.set_xlabel('U/t')
    ax.set_ylabel('Double Occupancy $\\langle D \\rangle$')
    ax.set_title(f'Double Occupancy vs Coupling: {lattice}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f'double_occupancy_{lattice}.png')
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_spin_correlations(
    save_root: str,
    output_dir: str,
    lattice: str = '4x4',
    U: float = 4.0,
    seed: int = 42,
):
    """Plot real-space spin-spin correlations relative to a reference site."""
    os.makedirs(output_dir, exist_ok=True)

    Lx, Ly = map(int, lattice.split('x'))

    configs = [
        ('transformer', 'random'), ('transformer', 'jepa'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (backbone, init) in zip(axes, configs):
        obs = load_observables(save_root, backbone, init, U, lattice, seed)
        if obs is None:
            ax.set_title(f'{backbone}+{init}\n(no data)')
            continue

        corr = obs.get('spin_correlation')
        if corr is None:
            ax.set_title(f'{backbone}+{init}\n(no data)')
            continue

        corr = np.array(corr)

        # Correlations relative to site (0,0)
        ref_site = 0
        corr_map = np.zeros((Ly, Lx))
        for j in range(Lx * Ly):
            x = j % Lx
            y = j // Lx
            corr_map[y, x] = corr[ref_site, j]

        vmax = max(abs(corr_map.min()), abs(corr_map.max()))
        im = ax.imshow(
            corr_map, origin='lower', cmap='RdBu_r', aspect='equal',
            vmin=-vmax, vmax=vmax,
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        label = LABELS.get((backbone, init), f'{backbone}+{init}')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Spin Correlations $\\langle S_0^z S_j^z \\rangle$: {lattice}, U/t={U:.0f}',
        fontsize=14,
    )
    fig.tight_layout()
    save_path = os.path.join(output_dir, f'spin_correlations_{lattice}_U{U:.0f}.png')
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_af_signal_vs_U(
    save_root: str,
    output_dir: str,
    lattice: str = '4x4',
    seed: int = 42,
):
    """Plot antiferromagnetic signal S(pi,pi) vs U/t."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    u_values = [1.0, 2.0, 4.0, 6.0, 8.0]

    for backbone, init in [('transformer', 'random'), ('transformer', 'jepa')]:
        s_pi_pi = []
        u_plot = []
        for U in u_values:
            obs = load_observables(save_root, backbone, init, U, lattice, seed)
            if obs is not None and 'S_pi_pi' in obs:
                s_pi_pi.append(obs['S_pi_pi'])
                u_plot.append(U)

        if s_pi_pi:
            key = (backbone, init)
            ax.plot(u_plot, s_pi_pi,
                    color=COLORS.get(key, '#333'),
                    marker=MARKERS.get(key, 'o'),
                    label=LABELS.get(key, f'{backbone}+{init}'),
                    linewidth=1.5, markersize=6)

    ax.set_xlabel('U/t')
    ax.set_ylabel('$S(\\pi, \\pi) / N$')
    ax.set_title(f'Antiferromagnetic Signal: {lattice}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f'af_signal_{lattice}.png')
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot observables')
    parser.add_argument('--save-root', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='paper/figures')
    parser.add_argument('--lattice', type=str, default='4x4')
    parser.add_argument('--U', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    plot_structure_factor_heatmap(args.save_root, args.output_dir, args.lattice, args.U, args.seed)
    plot_spin_correlations(args.save_root, args.output_dir, args.lattice, args.U, args.seed)
    plot_double_occupancy_vs_U(args.save_root, args.output_dir, args.lattice, args.seed)
    plot_af_signal_vs_U(args.save_root, args.output_dir, args.lattice, args.seed)
