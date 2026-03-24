"""
Plot energy convergence curves for VMC training.

Generates figures showing:
- Energy vs training step for all configurations
- Comparison of Transformer backbone configurations
- Effect of JEPA pretraining
- Convergence at different U/t values
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# Consistent style
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

# Color scheme
COLORS = {
    ('transformer', 'random'): '#2ca02c',
    ('transformer', 'jepa'): '#d62728',
}

LABELS = {
    ('transformer', 'random'): 'Transformer (random)',
    ('transformer', 'jepa'): 'Transformer + JEPA',
}

LINESTYLES = {
    'random': '-',
    'jepa': '--',
}


def load_history(run_dir: str) -> dict:
    """Load training history from a run directory."""
    history_path = os.path.join(run_dir, 'history.json')
    if not os.path.exists(history_path):
        return None
    with open(history_path) as f:
        return json.load(f)


def load_summary(run_dir: str) -> dict:
    """Load run summary."""
    summary_path = os.path.join(run_dir, 'summary.json')
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f)


def find_runs(save_root: str, lattice: str = None, U: float = None) -> list[dict]:
    """Find all VMC run directories matching criteria."""
    runs = []
    vmc_dir = os.path.join(save_root, 'vmc')
    if not os.path.exists(vmc_dir):
        return runs

    for lattice_dir in sorted(os.listdir(vmc_dir)):
        if lattice and lattice_dir != lattice:
            continue
        lattice_path = os.path.join(vmc_dir, lattice_dir)
        if not os.path.isdir(lattice_path):
            continue

        for run_name in sorted(os.listdir(lattice_path)):
            run_path = os.path.join(lattice_path, run_name)
            if not os.path.isdir(run_path):
                continue

            summary = load_summary(run_path)
            history = load_history(run_path)
            if summary is None or history is None:
                continue

            if U is not None and summary.get('U_over_t') != U:
                continue

            runs.append({
                'path': run_path,
                'name': run_name,
                'lattice': lattice_dir,
                'summary': summary,
                'history': history,
            })

    return runs


def smooth(values: list, window: int = 50) -> np.ndarray:
    """Simple moving average smoothing."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_convergence_single(
    runs: list[dict],
    N: int,
    reference: float = None,
    title: str = '',
    save_path: str = None,
    window: int = 50,
):
    """Plot energy convergence for a set of runs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in runs:
        summary = run['summary']
        backbone = summary['backbone']
        init = summary['init']
        key = (backbone, init)

        energies = np.array(run['history']['energy']) / N
        smoothed = smooth(energies, window)
        steps = np.arange(len(smoothed))

        color = COLORS.get(key, '#333333')
        label = LABELS.get(key, f'{backbone}+{init}')
        ls = LINESTYLES.get(init, '-')

        ax.plot(steps, smoothed, color=color, linestyle=ls,
                label=label, linewidth=1.5, alpha=0.9)

        # Light band for raw data
        raw_steps = np.arange(len(energies))
        ax.fill_between(raw_steps, energies, alpha=0.05, color=color)

    if reference is not None:
        ax.axhline(reference, color='black', linestyle=':', linewidth=1.5,
                   label=f'Reference: {reference:.4f}', alpha=0.7)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Energy per site (E/N)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_convergence_comparison(
    save_root: str,
    output_dir: str,
    window: int = 50,
):
    """Generate all convergence comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # ED references
    ed_ref = {
        ('4x4', 4.0): -1.0959,
        ('4x4', 8.0): -1.0288,
    }
    afqmc_ref = {
        ('8x8', 4.0): -0.8603,
        ('8x8', 8.0): -0.5257,
    }

    # Plot per lattice and U/t
    for lattice in ['4x4', '6x6', '8x8']:
        for U in [4.0, 8.0]:
            runs = find_runs(save_root, lattice=lattice, U=U)
            if not runs:
                continue

            Lx, Ly = map(int, lattice.split('x'))
            N = Lx * Ly
            ref = ed_ref.get((lattice, U)) or afqmc_ref.get((lattice, U))

            plot_convergence_single(
                runs, N, reference=ref,
                title=f'{lattice} lattice, U/t = {U:.0f}',
                save_path=os.path.join(output_dir, f'convergence_{lattice}_U{U:.0f}.png'),
                window=window,
            )

    # Combined: JEPA benefit plot (for paper Fig 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, U in enumerate([4.0, 8.0]):
        ax = axes[ax_idx]
        runs_4x4 = find_runs(save_root, lattice='4x4', U=U)

        for run in runs_4x4:
            summary = run['summary']
            backbone = summary['backbone']
            init = summary['init']
            key = (backbone, init)

            energies = np.array(run['history']['energy']) / 16
            smoothed = smooth(energies, window)

            color = COLORS.get(key, '#333333')
            label = LABELS.get(key, f'{backbone}+{init}')
            ls = LINESTYLES.get(init, '-')

            ax.plot(np.arange(len(smoothed)), smoothed,
                    color=color, linestyle=ls, label=label, linewidth=1.5)

        ref = ed_ref.get(('4x4', U))
        if ref:
            ax.axhline(ref, color='black', linestyle=':', linewidth=1.5,
                       label=f'ED: {ref:.4f}', alpha=0.7)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('E/N')
        ax.set_title(f'4x4, U/t = {U:.0f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('JEPA Pretraining Effect on VMC Convergence', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'jepa_benefit_comparison.png'))
    print(f"Saved: {os.path.join(output_dir, 'jepa_benefit_comparison.png')}")
    plt.close(fig)


def plot_variance_convergence(save_root: str, output_dir: str, window: int = 50):
    """Plot energy variance convergence (should go to 0 for exact ground state)."""
    os.makedirs(output_dir, exist_ok=True)

    for lattice in ['4x4', '6x6', '8x8']:
        for U in [4.0, 8.0]:
            runs = find_runs(save_root, lattice=lattice, U=U)
            if not runs:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))

            for run in runs:
                summary = run['summary']
                backbone = summary['backbone']
                init = summary['init']
                key = (backbone, init)

                variance = np.array(run['history']['variance'])
                smoothed = smooth(variance, window)

                color = COLORS.get(key, '#333333')
                label = LABELS.get(key, f'{backbone}+{init}')
                ls = LINESTYLES.get(init, '-')

                ax.semilogy(np.arange(len(smoothed)), smoothed,
                           color=color, linestyle=ls, label=label, linewidth=1.5)

            ax.set_xlabel('Training Step')
            ax.set_ylabel('Energy Variance')
            ax.set_title(f'Variance: {lattice}, U/t = {U:.0f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig.savefig(os.path.join(output_dir, f'variance_{lattice}_U{U:.0f}.png'))
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot convergence curves')
    parser.add_argument('--save-root', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='paper/figures')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()

    plot_convergence_comparison(args.save_root, args.output_dir, args.window)
    plot_variance_convergence(args.save_root, args.output_dir, args.window)
