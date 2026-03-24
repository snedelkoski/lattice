"""
Generate comparison tables for the paper.

Creates LaTeX-formatted tables comparing:
- Energy per site across all configurations (Paper Table 1)
- Relative errors vs reference methods
- Aggregated statistics across seeds
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


# Reference energies per site
ED_REFERENCE = {
    ('4x4', 4.0): -1.0959,
    ('4x4', 8.0): -1.0288,
}

AFQMC_REFERENCE = {
    ('8x8', 4.0): -0.8603,
    ('8x8', 8.0): -0.5257,
}

LITERATURE = {
    ('8x8', 8.0, 'Transformer Backflow'): -0.5258,
    ('8x8', 8.0, 'HFDS'): -0.525,
    ('8x8', 8.0, 'RBM+PP (mVMC)'): -0.5246,
}


def collect_results(save_root: str) -> list[dict]:
    """Collect all VMC results."""
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
                results.append(summary)

    return results


def aggregate_by_config(results: list[dict]) -> dict:
    """
    Aggregate results across seeds for each (backbone, init, lattice, U/t) config.

    Returns dict mapping config key to {mean, std, min, max, n_seeds}.
    """
    groups = {}
    for r in results:
        key = (r['backbone'], r['init'], r['lattice'], r['U_over_t'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r['best_energy_per_site'])

    aggregated = {}
    for key, energies in groups.items():
        arr = np.array(energies)
        aggregated[key] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'n_seeds': len(energies),
        }
    return aggregated


def generate_main_table(results: list[dict], output_path: str = None):
    """
    Generate Paper Table 1: Energy comparison.

    Format:
    Method | 4x4 U/t=4 | 4x4 U/t=8 | 8x8 U/t=8
    """
    agg = aggregate_by_config(results)

    # Table header
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Ground state energy per site (E/N) comparison. '
                 'Values in parentheses are statistical uncertainties.}')
    lines.append('\\label{tab:energy_comparison}')
    lines.append('\\begin{tabular}{lccc}')
    lines.append('\\toprule')
    lines.append('Method & $4{\\times}4$ $U/t{=}4$ & $4{\\times}4$ $U/t{=}8$ '
                 '& $8{\\times}8$ $U/t{=}8$ \\\\')
    lines.append('\\midrule')

    # Reference methods
    lines.append(f'ED (exact) & $-1.0959$ & $-1.0288$ & --- \\\\')
    lines.append(f'AFQMC & --- & --- & $-0.5257(1)$ \\\\')
    lines.append(f'Transformer Backflow & --- & --- & $-0.5258$ \\\\')
    lines.append(f'HFDS & --- & --- & $-0.525$ \\\\')
    lines.append(f'RBM+PP (mVMC) & --- & --- & $-0.5246$ \\\\')
    lines.append(f'GRU (autoregressive) & converges & fails & --- \\\\')
    lines.append('\\midrule')

    # Our methods
    methods = [
        ('Transformer-NQS', 'transformer', 'random'),
        ('Transformer-NQS + JEPA', 'transformer', 'jepa'),
    ]

    for label, backbone, init in methods:
        cells = []
        for lattice, U in [('4x4', 4.0), ('4x4', 8.0), ('8x8', 8.0)]:
            key = (backbone, init, lattice, U)
            if key in agg:
                stats = agg[key]
                if stats['n_seeds'] > 1:
                    cells.append(f"${stats['mean']:.4f}({stats['std']:.4f})$")
                else:
                    cells.append(f"${stats['mean']:.4f}$")
            else:
                cells.append('---')
        lines.append(f'{label} & {" & ".join(cells)} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    table_str = '\n'.join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"Saved: {output_path}")

    return table_str


def generate_relative_error_table(results: list[dict], output_path: str = None):
    """Generate table of relative errors vs reference methods."""
    agg = aggregate_by_config(results)

    all_refs = {}
    all_refs.update({k: v for k, v in ED_REFERENCE.items()})
    all_refs.update({k: v for k, v in AFQMC_REFERENCE.items()})

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Relative error (\\%) vs reference energies.}')
    lines.append('\\label{tab:relative_error}')
    lines.append('\\begin{tabular}{lcccc}')
    lines.append('\\toprule')
    lines.append('Config & Backbone & Init & E/N & Rel. Error (\\%) \\\\')
    lines.append('\\midrule')

    for (backbone, init, lattice, U), stats in sorted(agg.items()):
        ref_key = (lattice, U)
        ref = all_refs.get(ref_key)
        if ref:
            err = abs(stats['mean'] - ref) / abs(ref) * 100
            err_str = f'{err:.2f}'
        else:
            err_str = '---'

        lines.append(
            f'{lattice} U/t={U:.0f} & {backbone} & {init} & '
            f"${stats['mean']:.4f}$ & {err_str} \\\\"
        )

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    table_str = '\n'.join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"Saved: {output_path}")

    return table_str


def generate_jepa_speedup_table(results: list[dict], output_path: str = None):
    """Generate table showing JEPA pretraining speedup."""
    agg = aggregate_by_config(results)

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Effect of JEPA pretraining on final energy. '
                 '$\\Delta E/N$ shows improvement from JEPA initialization.}')
    lines.append('\\label{tab:jepa_benefit}')
    lines.append('\\begin{tabular}{lcccc}')
    lines.append('\\toprule')
    lines.append('Config & Random E/N & JEPA E/N & $\\Delta$E/N & Improvement \\\\')
    lines.append('\\midrule')

    for backbone in ['transformer']:
        for lattice in ['4x4', '6x6', '8x8']:
            for U in [4.0, 8.0]:
                key_random = (backbone, 'random', lattice, U)
                key_jepa = (backbone, 'jepa', lattice, U)

                if key_random in agg and key_jepa in agg:
                    e_random = agg[key_random]['mean']
                    e_jepa = agg[key_jepa]['mean']
                    delta = e_jepa - e_random
                    better = 'JEPA' if e_jepa < e_random else 'Random'

                    lines.append(
                        f'{backbone}, {lattice}, U/t={U:.0f} & '
                        f'${e_random:.4f}$ & ${e_jepa:.4f}$ & '
                        f'${delta:+.4f}$ & {better} \\\\'
                    )

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    table_str = '\n'.join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"Saved: {output_path}")

    return table_str


def print_text_summary(results: list[dict]):
    """Print a plain-text summary to console."""
    agg = aggregate_by_config(results)

    if not agg:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f"{'RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Backbone':<12} {'Init':<8} {'Lattice':<8} {'U/t':<6} "
          f"{'E/N (mean)':<14} {'Std':<10} {'N seeds':<8}")
    print(f"{'-'*80}")

    for key in sorted(agg.keys()):
        backbone, init, lattice, U = key
        stats = agg[key]
        print(f"{backbone:<12} {init:<8} {lattice:<8} {U:<6.1f} "
              f"{stats['mean']:<14.6f} {stats['std']:<10.6f} {stats['n_seeds']:<8}")

    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate comparison tables')
    parser.add_argument('--save-root', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='paper/tables')
    args = parser.parse_args()

    results = collect_results(args.save_root)
    print_text_summary(results)

    if results:
        generate_main_table(
            results, os.path.join(args.output_dir, 'table_energy.tex')
        )
        generate_relative_error_table(
            results, os.path.join(args.output_dir, 'table_errors.tex')
        )
        generate_jepa_speedup_table(
            results, os.path.join(args.output_dir, 'table_jepa_benefit.tex')
        )
