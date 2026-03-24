# Lattice-JEPA: Implementation Program

## Project Overview

**Title**: Lattice-JEPA: Self-Supervised Pretraining of Recurrent Neural Quantum States for Strongly Correlated Electron Systems

**Target**: NeurIPS 2026 Main Track (Abstract May 4, Paper May 6)
**Hardware**: RTX 3050 4GB VRAM (CUDA CC 8.6), larger GPUs available if needed
**Framework**: PyTorch only, CUDA

## Contributions

1. **xLSTM-NQS**: First use of xLSTM (mLSTM with matrix memory) as neural quantum state ansatz. Bidirectional scanning, non-autoregressive.
2. **Lattice-JEPA**: First JEPA self-supervised pretraining for quantum lattice systems. Spatial block masking + SIGReg anti-collapse.
3. **Systematic comparison**: xLSTM vs Transformer backbones, with/without JEPA pretraining, benchmarked against AFQMC/ED on 2D Fermi-Hubbard.

---

## Architecture Specifications

### Shared Components

**Input encoding** (identical for xLSTM and Transformer):
- Input: sigma in {0,1,2,3}^N (0=empty, 1=up, 2=down, 3=doubly-occupied)
- Token embedding: nn.Embedding(4, d_model), d_model=128
- Position encoding: nn.Embedding(N_max, d_model), N_max=144 (up to 12x12)
- Input = TokenEmbed(sigma) + PosEmbed(site_indices)

**Output head** (identical for both):
- Pool: mean over sequence dim -> [batch, d_model]
- log|psi|: Linear(d_model, 1)
- phase: Linear(d_model, 1)
- log psi = log|psi| + i * phase

**Symmetry projection** (outside network):
- psi_sym(sigma) = (1/|G|) sum_g exp(log psi(g * sigma))
- G = lattice translations for PBC
- Implemented via log-sum-exp for numerical stability

### xLSTM-NQS Backbone (~600K params)

- Bidirectional mLSTM:
  - Forward: xLSTM stack (d_model=128, num_heads=4, num_blocks=4)
  - Backward: xLSTM stack (same config, input flipped)
  - Fusion: Linear(256, 128) + LayerNorm
- Use native PyTorch mLSTM implementation (parallel_stabilized) for portability
- Triton kernels optional if available

### Transformer-NQS Backbone (~550K params)

- Standard Transformer encoder (bidirectional/full attention):
  - num_layers=4, num_heads=4, d_model=128, d_ff=256
  - Activation: SiLU, no dropout
- Parameter count matched to xLSTM for fair comparison

### JEPA Pretraining Module

- Encoder: xLSTM or Transformer backbone (shared with NQS)
- Projector: MLP(d_model -> 512 -> d_embed) + BatchNorm1d, d_embed=64
- Predictor: Small Transformer (2 layers, 4 heads, dim=128)
- Pred Projector: MLP(d_model -> 512 -> d_embed) + BatchNorm1d
- Loss: MSE(pred, target) + 0.09 * SIGReg(embeddings)
- No target encoder, no EMA, no stop-gradient (following LeWorldModel)
- Masking: spatial block mask (2x2 for 4x4 lattice, 3x3 for 6x6+)

---

## Physics: Fermi-Hubbard Model

### Hamiltonian
H = -t sum_{<ij>,s} (c+_is c_js + h.c.) + U sum_i n_i_up n_i_down

- t = hopping amplitude (set to 1, energy unit)
- U = on-site Coulomb repulsion
- <ij> = nearest-neighbor pairs on square lattice
- s = spin (up, down)

### Local Energy Computation
E_loc(sigma) = U * (number of doubly-occupied sites)
             + sum over valid hops: (-t) * JW_sign * psi(sigma') / psi(sigma)

Jordan-Wigner sign for hopping from site b to site a:
sign = (-1)^(number of occupied sites between a and b in linear ordering)

### Key U/t Values
- U/t = 0: free fermions (exact solution known)
- U/t = 2: weak coupling
- U/t = 4: intermediate (standard benchmark)
- U/t = 8: strong coupling (where RNNs fail, Mott insulator regime)

### Exact Diagonalization Reference Energies (4x4, half-filling, PBC)
- U/t=2: E/N = -1.15596
- U/t=4: E/N = -1.09593
- U/t=6: E/N = -1.05625
- U/t=8: E/N = -1.02879

### AFQMC Reference (8x8, half-filling)
- U/t=4: E/N = -0.8603(1)
- U/t=8: E/N = -0.5257(1)

---

## VMC Training

### MCMC Sampler (Metropolis-Hastings)
- Proposal: pick random occupied site, pick random empty neighbor, hop electron
- Accept with prob min(1, |psi(sigma')/psi(sigma)|^2)
- Parallel chains on GPU (n_chains=256)
- Thermalization: 200 steps, sample every 10 steps

### Optimizer Strategy
- Start with AdamW (lr=1e-3, weight_decay=1e-4)
- If convergence is poor at U/t>=8, implement MinSR
- Gradient clipping: 1.0 (global norm)
- U-ramping: train at increasing U/t values

### Training Hyperparameters
- Batch size (n_chains): 256
- VMC steps: 5000-10000
- Precision: BF16 via torch.autocast
- MCMC sweeps between samples: 10
- Local energy clipping: 5 sigma

---

## JEPA Pretraining

### Data Generation
- Random valid fermion configurations with correct particle numbers
- 200K samples per lattice size
- Half-filling: N_up = N_down = N_sites / 2

### Hyperparameters
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Batch size: 256
- Epochs: 100
- Precision: BF16
- Gradient clipping: 1.0
- LR schedule: cosine with 5-epoch warmup
- SIGReg: lambda=0.09, num_proj=256, knots=17
- Mask: 2x2 block for 4x4, 3x3 block for 6x6+, ratio ~25-35%
- Predictor: 2-layer Transformer, 4 heads, dim=128

### Two-Stage Transfer
1. Pretrain encoder on JEPA task
2. Initialize NQS backbone from pretrained encoder weights
3. Add fresh output heads (log|psi|, phase)
4. Discard predictor
5. Fine-tune on VMC

---

## Experiment Matrix

### 8 Primary Configurations
| ID | Backbone | Init | What It Tests |
|----|----------|------|---------------|
| 1 | xLSTM | Random | xLSTM baseline |
| 2 | xLSTM | JEPA | JEPA benefit for xLSTM |
| 3 | Transformer | Random | Transformer baseline |
| 4 | Transformer | JEPA | JEPA benefit for Transformer |

Each at U/t = {4, 8}, lattice = {4x4, 6x6, 8x8}, 5 seeds = 120 VMC runs.

### Ablation Studies
- JEPA mask ratio: 15%, 25%, 35%, 50%
- Mask shape: block vs random
- Bidirectional vs unidirectional xLSTM
- With/without symmetry projection
- Model size: d=64, 128, 256
- AdamW vs MinSR

### Physical Observables
1. Spin structure factor S(q) - peak at (pi,pi) = antiferromagnetic order
2. Pairing correlations P(r) - d-wave superconductivity
3. Double occupancy <D> vs U/t
4. Momentum distribution n(k)

### Interpretability
- xLSTM matrix memory visualization
- Transformer attention maps
- Compare random-init vs JEPA-pretrained representations

---

## Benchmark Comparison Table (Paper Table 1)

| Method | Source | 4x4 U/t=4 | 4x4 U/t=8 | 8x8 U/t=8 |
|--------|--------|-----------|-----------|-----------|
| ED (exact) | VarBench | -1.0959 | -1.0288 | -- |
| AFQMC | Qin+ 2016 | -- | -- | -0.5257 |
| Transformer Backflow | Gu+ 2025 | -- | -- | -0.5258 |
| HFDS | Robledo Moreno+ 2022 | -- | -- | -0.525 |
| RBM+PP (mVMC) | Nomura & Imada | -- | -- | -0.5246 |
| GRU (autoregressive) | Ibarra-Garcia-Padilla+ 2024 | converges | FAILS | -- |
| xLSTM-NQS (ours) | This work | ? | ? | ? |
| xLSTM-NQS + JEPA (ours) | This work | ? | ? | ? |
| Transformer-NQS (ours) | This work | ? | ? | ? |
| Transformer-NQS + JEPA (ours) | This work | ? | ? | ? |

---

## File Structure

```
lattice-jepa/
├── program.md                 # This file - implementation plan
├── configs/
│   └── default.yaml           # Default hyperparameters
├── models/
│   ├── __init__.py
│   ├── base_nqs.py            # Abstract NQS base class
│   ├── xlstm_nqs.py           # Bidirectional xLSTM backbone
│   ├── transformer_nqs.py     # Transformer backbone
│   ├── jepa.py                # JEPA pretraining wrapper
│   ├── sigreg.py              # SIGReg anti-collapse loss
│   ├── masking.py             # Spatial block masking
│   └── predictors.py          # JEPA predictor network
├── physics/
│   ├── __init__.py
│   ├── hubbard.py             # Hubbard Hamiltonian + local energy
│   ├── exact_diag.py          # Exact diagonalization
│   ├── sampler.py             # MCMC Metropolis-Hastings
│   ├── vmc.py                 # VMC training loop
│   ├── observables.py         # Physical measurements
│   └── symmetry.py            # Lattice symmetry projection
├── training/
│   ├── __init__.py
│   ├── pretrain.py            # Stage 1: JEPA pretraining script
│   ├── finetune.py            # Stage 2: VMC fine-tuning script
│   └── optimizer.py           # AdamW + MinSR
├── experiments/
│   ├── run_experiment.py      # Main experiment runner
│   └── run_ablation.py        # Ablation study runner
├── analysis/
│   ├── plot_convergence.py    # Energy vs step plots
│   ├── plot_observables.py    # Structure factor, correlations
│   └── tables.py              # Generate comparison tables
├── tests/
│   ├── test_hubbard.py        # Validate Hamiltonian against ED
│   ├── test_nqs.py            # Test model forward/backward
│   ├── test_vmc.py            # Test VMC on 2x2
│   └── test_jepa.py           # Test JEPA training loop
└── requirements.txt           # Python dependencies
```

---

## Implementation Order

1. **Phase 0**: Create project structure, requirements.txt, configs
2. **Phase 1**: physics/hubbard.py, physics/exact_diag.py (validate against known energies)
3. **Phase 2**: models/base_nqs.py, models/xlstm_nqs.py, models/transformer_nqs.py
4. **Phase 3**: physics/sampler.py, physics/vmc.py, training/optimizer.py
5. **Phase 4**: models/sigreg.py, models/masking.py, models/predictors.py, models/jepa.py
6. **Phase 5**: training/pretrain.py, training/finetune.py
7. **Phase 6**: physics/observables.py, physics/symmetry.py
8. **Phase 7**: experiments/run_experiment.py, analysis scripts
9. **Phase 8**: tests/ (validate everything)
10. **Phase 9**: Run validation, fix bugs, iterate
