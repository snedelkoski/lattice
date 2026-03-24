"""
Variational Monte Carlo (VMC) training loop.

Implements the VMC optimization procedure:
1. Sample configurations from |psi(sigma)|^2 using MCMC
2. Compute local energies E_loc(sigma)
3. Estimate energy gradient
4. Update parameters (AdamW or Stochastic Reconfiguration)

The energy gradient estimator is:
  grad E = 2 Re[ <(E_loc - <E_loc>) * conj(O_k)> ]
where O_k = d(log psi) / d(theta_k)

Key features:
- Log-amplitude regularization to prevent wavefunction collapse
- Robust local energy clipping (median + MAD)
- U-ramping schedule for convergence at strong coupling
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from tqdm import tqdm
import time
import sys
from itertools import combinations

from .hubbard import FermiHubbardHamiltonian, SquareLattice
from .sampler import MetropolisSampler


class VMCTrainer:
    """
    Variational Monte Carlo trainer for Neural Quantum States.

    Handles the full VMC optimization loop including:
    - MCMC sampling
    - Local energy computation
    - VMC gradient estimation
    - Parameter updates with AdamW
    - Log-amplitude regularization (anti-collapse)
    - Energy/variance logging
    - U-ramping schedule
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        lattice: SquareLattice,
        t: float = 1.0,
        U: float = 4.0,
        n_up: Optional[int] = None,
        n_down: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_chains: int = 512,
        n_sweeps: int = 10,
        n_thermalize: int = 200,
        grad_clip: float = 1.0,
        e_loc_clip: float = 5.0,
        device: str = "cuda",
        use_amp: bool = False,
        # Regularization
        log_amp_reg: float = 0.0,
        log_amp_reg_decay_steps: int = 0,
        # Phase regularization: penalize large phases (for real ground states)
        phase_reg: float = 0.0,
        lr_schedule_mode: str = "global_cosine",
        min_lr: float = 1e-5,
        exact_sampling: bool = False,
        keep_best_state: bool = True,
        # Stochastic Reconfiguration
        use_sr: bool = False,
        sr_lr: float = 0.02,
        sr_diag_shift: float = 0.01,
        sr_n_chains: int = 128,  # fewer chains for SR (Jacobian is expensive)
        sr_min_U: float = 2.0,  # only use SR for U >= this (AdamW for easy stages)
        sr_optimizer: str = "minsr",  # "minsr" or "march"
        sr_momentum: float = 0.95,  # MARCH momentum (mu)
        sr_beta: float = 0.995,  # MARCH second moment decay
        sr_norm_decay_start: int = 8000,  # step after which MARCH norm constraint decays
        sr_chunk_size: int = 32,  # vmap chunk size for Jacobian
        # Optional: raw (unwrapped) model for SR Jacobian computation.
        # If model is wrapped (SymmetrizedNQS, MarshallSignNQS), pass the
        # innermost TransformerNQS here so Jacobian is computed efficiently.
        raw_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.raw_model = raw_model  # for SR Jacobian; None means use model directly
        self.lattice = lattice
        self.N = lattice.N
        self.t = t
        self.U = U
        self.n_up = n_up if n_up is not None else self.N // 2
        self.n_down = n_down if n_down is not None else self.N // 2
        self.device = device
        self.grad_clip = grad_clip
        self.e_loc_clip = e_loc_clip
        self.use_amp = use_amp
        self.log_amp_reg = log_amp_reg
        self.log_amp_reg_decay_steps = log_amp_reg_decay_steps
        self.phase_reg = phase_reg
        self.lr_schedule_mode = lr_schedule_mode
        self.min_lr = min_lr
        self.exact_sampling = exact_sampling
        self.keep_best_state = keep_best_state
        self.global_step = 0
        self.best_energy = float('inf')
        self.best_state_dict = None
        self.best_energy_by_U = {}
        self.best_state_by_U = {}

        # Hamiltonian
        self.hamiltonian = FermiHubbardHamiltonian(lattice, t=t, U=U)

        # Sampler — starts with n_chains for AdamW; will be rebuilt for SR
        self.n_chains_adamw = n_chains
        self.n_chains_sr = sr_n_chains
        self.n_sweeps = n_sweeps
        self.n_thermalize = n_thermalize
        self.sampler = MetropolisSampler(
            lattice=lattice,
            n_chains=n_chains,
            n_sweeps=n_sweeps,
            n_thermalize=n_thermalize,
            device=device,
        )

        # Optimizer
        self.use_sr = use_sr
        self.sr_min_U = sr_min_U
        self.sr_optimizer_type = sr_optimizer
        sr_target = raw_model if raw_model is not None else model
        self.sr_target = sr_target
        if use_sr:
            if sr_optimizer == "march":
                from training.optimizer import MARCH
                self.sr_optimizer = MARCH(
                    model=sr_target,
                    norm_constraint=sr_lr,  # MARCH uses norm_constraint, mapped from sr_lr
                    damping=sr_diag_shift,
                    mu=sr_momentum,
                    beta=sr_beta,
                    chunk_size=sr_chunk_size,
                    norm_decay_start=sr_norm_decay_start,
                )
            else:
                from training.optimizer import MinSR
                self.sr_optimizer = MinSR(
                    model=sr_target,
                    lr=sr_lr,
                    diag_shift=sr_diag_shift,
                    max_norm=grad_clip,
                    chunk_size=sr_chunk_size,
                )
        else:
            self.sr_optimizer = None
        # AdamW — always available (used for easy stages and as fallback)
        self.optimizer = torch.optim.AdamW(
            sr_target.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.base_lr = lr
        self.sr_base_lr = sr_lr

        # Track current optimizer mode
        self._current_mode = None  # 'adamw' or 'sr'

        # Learning rate scheduler (cosine annealing, set up in train())
        self.scheduler = None

        # Logging
        self.history = {
            'energy': [],
            'energy_std': [],
            'variance': [],
            'acceptance_rate': [],
            'grad_norm': [],
            'step_time': [],
            'log_amp_spread': [],
            'lr': [],
            'param_update_norm': [],
        }

        self.all_configs = None
        if self.exact_sampling:
            self.all_configs = self._enumerate_all_configs().to(self.device)

    def _enumerate_all_configs(self) -> torch.Tensor:
        """Enumerate all valid configs in the fixed (N_up, N_down) sector."""
        all_configs = []
        for up_sites in combinations(range(self.N), self.n_up):
            up = np.zeros(self.N, dtype=np.int64)
            up[list(up_sites)] = 1
            for down_sites in combinations(range(self.N), self.n_down):
                down = np.zeros(self.N, dtype=np.int64)
                down[list(down_sites)] = 1
                all_configs.append(up + 2 * down)
        return torch.tensor(np.array(all_configs), dtype=torch.long)

    def set_U(self, U: float):
        """Update the interaction strength U (for U-ramping)."""
        old_U = self.U
        self.U = U
        self.hamiltonian = FermiHubbardHamiltonian(self.lattice, t=self.t, U=U)
        # Reset MARCH momentum on U transitions to avoid stale gradients
        if (old_U != U and self.sr_optimizer is not None
                and hasattr(self.sr_optimizer, 'reset_state')):
            self.sr_optimizer.reset_state()

    def _get_reg_coeff(self) -> float:
        """Get the current log-amplitude regularization coefficient."""
        if self.log_amp_reg_decay_steps <= 0:
            return self.log_amp_reg
        # Linear decay
        progress = min(1.0, self.global_step / self.log_amp_reg_decay_steps)
        return self.log_amp_reg * (1.0 - progress)

    def clip_local_energies(self, e_loc: torch.Tensor) -> torch.Tensor:
        """Clip local energy outliers beyond k*sigma of the median.

        Uses median and MAD (median absolute deviation) as robust estimators
        to avoid outliers inflating the clipping threshold.
        """
        if self.e_loc_clip is None or self.e_loc_clip <= 0:
            return e_loc

        e_real = e_loc.real
        median_e = e_real.median()
        mad = (e_real - median_e).abs().median()
        # Convert MAD to std-equivalent (for normal distribution, std ≈ 1.4826 * MAD)
        std_e = 1.4826 * mad
        if std_e > 0:
            mask = (e_real - median_e).abs() < self.e_loc_clip * std_e
            # Replace outliers with clipped values
            e_clipped = torch.where(
                mask,
                e_loc,
                torch.complex(
                    median_e + self.e_loc_clip * std_e * torch.sign(e_real - median_e),
                    e_loc.imag,
                ),
            )
            return e_clipped
        return e_loc

    def compute_vmc_loss(
        self, configs: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute VMC loss and statistics.

        The VMC "loss" is constructed so that its gradient equals the
        energy gradient:
            L = 2 Re[ <(E_loc - <E_loc>)* * log_psi> ]

        Plus regularization terms:
        - Log-amplitude variance penalty (prevents wavefunction collapse)
        - Phase penalty (Hubbard ground state is real at half-filling)

        Args:
            configs: (B, N) sampled configurations

        Returns:
            loss: scalar tensor (for backprop)
            stats: dict with energy, variance, etc.
        """
        B = configs.shape[0]

        if self.exact_sampling:
            log_psi = self.model(configs)  # (B,) complex

            # p(sigma) = |psi|^2 / Z
            log_prob = 2.0 * log_psi.real
            log_prob = log_prob - log_prob.max()
            probs = torch.exp(log_prob)
            probs = probs / probs.sum()

            # Local energies
            with torch.no_grad():
                e_loc = self.hamiltonian.compute_local_energy_batch(self.model, configs)
                e_loc = self.clip_local_energies(e_loc)

            probs_detached = probs.detach()
            mean_e = (probs_detached * e_loc.real).sum()
            var_e = (probs_detached * (e_loc.real - mean_e) ** 2).sum()

            # Exact-sampling VMC estimator
            e_loc_centered = (e_loc - mean_e).detach().conj()
            vmc_loss = 2.0 * (probs_detached * e_loc_centered * log_psi).real.sum()

            log_amp = log_psi.real
            log_amp_mean = (probs_detached * log_amp).sum()
            log_amp_var = (probs_detached * (log_amp - log_amp_mean) ** 2).sum()
            reg_coeff = self._get_reg_coeff()

            phase = log_psi.imag
            phase_loss = (probs_detached * (phase ** 2)).sum()

            loss = vmc_loss + reg_coeff * log_amp_var + self.phase_reg * phase_loss

            stats = {
                'energy': mean_e.item(),
                'energy_std': 0.0,
                'variance': var_e.item(),
                'log_amp_spread': log_amp_var.item(),
            }
            return loss, stats

        # Compute local energies (no grad needed for this)
        with torch.no_grad():
            e_loc = self.hamiltonian.compute_local_energy_batch(self.model, configs)
            e_loc = self.clip_local_energies(e_loc)
            mean_e = e_loc.real.mean()
            var_e = ((e_loc.real - mean_e) ** 2).mean()
            std_e = e_loc.real.std() / np.sqrt(B)

        # Compute log psi WITH gradient
        log_psi = self.model(configs)  # (B,) complex

        # VMC loss: L = 2 Re[ mean( (E_loc - <E_loc>)* . log_psi ) ]
        # The .detach() on e_loc ensures gradients only flow through log_psi
        e_loc_centered = (e_loc - mean_e).detach().conj()
        vmc_loss = 2.0 * (e_loc_centered * log_psi).real.mean()

        # Log-amplitude regularization: penalize variance of log|psi|
        # This prevents the model from putting all probability on a few configs
        log_amp = log_psi.real
        log_amp_mean = log_amp.mean()
        log_amp_var = ((log_amp - log_amp_mean) ** 2).mean()
        reg_coeff = self._get_reg_coeff()

        # Phase regularization: penalize non-zero phases
        # The Hubbard model ground state at half-filling with real hopping t
        # can be chosen real (Marshall sign rule applies)
        phase = log_psi.imag
        phase_loss = (phase ** 2).mean()

        loss = vmc_loss + reg_coeff * log_amp_var + self.phase_reg * phase_loss

        stats = {
            'energy': mean_e.item(),
            'energy_std': std_e.item(),
            'variance': var_e.item(),
            'log_amp_spread': log_amp_var.item(),
        }

        return loss, stats

    def _should_use_sr(self) -> bool:
        """Determine if SR should be used at the current U value."""
        return self.use_sr and self.U >= self.sr_min_U and not self.exact_sampling

    def _switch_sampler_chains(self, n_chains: int):
        """Rebuild sampler with a different number of chains."""
        if self.sampler.n_chains == n_chains:
            return
        self.sampler = MetropolisSampler(
            lattice=self.lattice,
            n_chains=n_chains,
            n_sweeps=self.n_sweeps,
            n_thermalize=self.n_thermalize,
            device=self.device,
        )
        self.sampler.initialize_chains(n_up=self.n_up, n_down=self.n_down)

    def train_step(self) -> dict:
        """Perform one VMC training step (AdamW or MinSR)."""
        t_start = time.time()

        self.model.train()

        # Determine optimizer mode and adjust chain count
        use_sr_now = self._should_use_sr()
        target_chains = self.n_chains_sr if use_sr_now else self.n_chains_adamw
        if not self.exact_sampling:
            self._switch_sampler_chains(target_chains)

        # 1. Sample configurations
        if self.exact_sampling:
            configs = self.all_configs
        else:
            with torch.no_grad():
                self.model.eval()
                configs = self.sampler.sample(self.model)
                self.model.train()

        if use_sr_now:
            # ---- MinSR path ----
            stats = self._train_step_sr(configs)
        else:
            # ---- AdamW path (also used for exact sampling) ----
            stats = self._train_step_adamw(configs)

        self.global_step += 1

        # Log
        stats['acceptance_rate'] = (
            self.sampler.recent_acceptance_rate if not self.exact_sampling else 1.0
        )
        stats['step_time'] = time.time() - t_start
        stats['lr'] = (
            self.sr_optimizer.lr if use_sr_now and self.sr_optimizer
            else self.optimizer.param_groups[0]['lr']
        )
        stats['mode'] = 'SR' if use_sr_now else 'AdamW'

        for key in self.history:
            if key in stats:
                self.history[key].append(stats[key])

        if self.keep_best_state and stats['energy'] < self.best_energy:
            self.best_energy = stats['energy']
            self.best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }

        u_key = float(self.U)
        if stats['energy'] < self.best_energy_by_U.get(u_key, float('inf')):
            self.best_energy_by_U[u_key] = stats['energy']
            if self.keep_best_state:
                self.best_state_by_U[u_key] = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

        return stats

    def _train_step_sr(self, configs: torch.Tensor) -> dict:
        """MinSR update step."""
        B = configs.shape[0]

        # Compute local energies
        with torch.no_grad():
            e_loc = self.hamiltonian.compute_local_energy_batch(self.model, configs)
            e_loc = self.clip_local_energies(e_loc)
            mean_e = e_loc.real.mean()
            var_e = ((e_loc.real - mean_e) ** 2).mean()
            std_e = e_loc.real.std() / np.sqrt(B)

        # Compute log_psi for log_amp_spread logging
        with torch.no_grad():
            log_psi = self.model(configs)
            log_amp = log_psi.real
            log_amp_var = ((log_amp - log_amp.mean()) ** 2).mean()

        # MinSR/MARCH step (computes Jacobian internally, applies update)
        sr_stats = self.sr_optimizer.step(configs, e_loc)

        # Cosine anneal SR learning rate (MinSR only; MARCH handles its own decay)
        if (self.sr_optimizer_type != "march"
                and hasattr(self, '_sr_total_steps') and self._sr_total_steps > 0):
            self._sr_step_counter += 1
            progress = self._sr_step_counter / self._sr_total_steps
            min_sr_lr = self.min_lr
            new_lr = min_sr_lr + 0.5 * (self.sr_base_lr - min_sr_lr) * (
                1 + np.cos(np.pi * progress)
            )
            self.sr_optimizer.set_lr(new_lr)

        stats = {
            'energy': mean_e.item(),
            'energy_std': std_e.item(),
            'variance': var_e.item(),
            'log_amp_spread': log_amp_var.item(),
            'grad_norm': sr_stats.get('param_update_norm', 0.0),
            'param_update_norm': sr_stats.get('param_update_norm', 0.0),
        }
        return stats

    def _train_step_adamw(self, configs: torch.Tensor) -> dict:
        """Standard AdamW update step."""
        # Compute VMC loss
        if self.use_amp and self.device == "cuda":
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, stats = self.compute_vmc_loss(configs)
        else:
            loss, stats = self.compute_vmc_loss(configs)

        # Backprop and update
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )

        self.optimizer.step()

        # Step the LR scheduler if present
        if self.scheduler is not None:
            self.scheduler.step()

        stats['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        stats['param_update_norm'] = 0.0
        return stats

    def train(
        self,
        n_steps: int = 5000,
        log_interval: int = 50,
        u_ramp_schedule: Optional[list[dict]] = None,
        use_lr_schedule: bool = True,
    ) -> dict:
        """
        Run the full VMC training loop.

        Args:
            n_steps: total optimization steps
            log_interval: print stats every N steps
            u_ramp_schedule: list of {'U': float, 'steps': int} for U-ramping.
                            If None, trains at fixed U.
            use_lr_schedule: whether to use cosine annealing LR schedule

        Returns:
            history dict with all logged quantities
        """
        # Initialize MCMC chains (if not using exact sampling)
        if not self.exact_sampling:
            self.sampler.initialize_chains(
                n_up=self.n_up,
                n_down=self.n_down,
            )

        # Set up LR scheduler
        total_steps = n_steps
        if u_ramp_schedule is not None:
            total_steps = sum(stage['steps'] for stage in u_ramp_schedule)
        if use_lr_schedule and self.lr_schedule_mode == "global_cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.min_lr,
            )
            # Track total steps for SR lr cosine schedule
            self._sr_total_steps = total_steps
            self._sr_step_counter = 0
        else:
            self.scheduler = None
            self._sr_total_steps = total_steps
            self._sr_step_counter = 0

        if u_ramp_schedule is not None:
            # U-ramping: train at increasing U values
            step = 0
            for stage in u_ramp_schedule:
                U_val = stage['U']
                stage_steps = stage['steps']
                self.set_U(U_val)
                print(f"\n--- U-ramp: U/t = {U_val:.1f}, {stage_steps} steps ---", flush=True)

                if use_lr_schedule and self.lr_schedule_mode == "stage_cosine":
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.base_lr
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=stage_steps, eta_min=self.min_lr,
                    )
                elif not use_lr_schedule:
                    self.scheduler = None

                pbar = tqdm(range(stage_steps), desc=f"U/t={U_val:.1f}", file=sys.stderr)
                for i in pbar:
                    stats = self.train_step()
                    step += 1

                    if (i + 1) % log_interval == 0 or i == stage_steps - 1:
                        pbar.set_postfix({
                            'E/N': f"{stats['energy']/self.N:.6f}",
                            'var': f"{stats['variance']:.4f}",
                            'acc': f"{stats['acceptance_rate']:.3f}",
                            'lr': f"{stats['lr']:.2e}",
                            'opt': stats.get('mode', '?'),
                        })
                        print(
                            f"step={step:5d} U/t={U_val:3.1f} E/N={stats['energy']/self.N:+.6f} "
                            f"var={stats['variance']:.4f} acc={stats['acceptance_rate']:.3f} "
                            f"lr={stats['lr']:.2e} opt={stats.get('mode', '?')} "
                            f"t/step={stats.get('step_time', 0):.1f}s",
                            flush=True,
                        )
        else:
            # Fixed U training
            pbar = tqdm(range(n_steps), desc=f"VMC U/t={self.U:.1f}", file=sys.stderr)
            for i in pbar:
                stats = self.train_step()

                if (i + 1) % log_interval == 0 or i == n_steps - 1:
                    pbar.set_postfix({
                        'E/N': f"{stats['energy']/self.N:.6f}",
                        'var': f"{stats['variance']:.4f}",
                        'acc': f"{stats['acceptance_rate']:.3f}",
                        'lr': f"{stats['lr']:.2e}",
                        'opt': stats.get('mode', '?'),
                    })
                    print(
                        f"step={i+1:5d} U/t={self.U:3.1f} E/N={stats['energy']/self.N:+.6f} "
                        f"var={stats['variance']:.4f} acc={stats['acceptance_rate']:.3f} "
                        f"lr={stats['lr']:.2e} opt={stats.get('mode', '?')} "
                        f"t/step={stats.get('step_time', 0):.1f}s",
                        flush=True,
                    )

        return self.history

    def restore_best_state(self):
        """Restore model weights from the best observed energy step."""
        if self.keep_best_state and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

    def restore_best_state_for_U(self, U: float) -> bool:
        """Restore best model weights observed at a specific U stage."""
        if not self.keep_best_state:
            return False
        u_key = float(U)
        state = self.best_state_by_U.get(u_key)
        if state is None:
            return False
        self.model.load_state_dict(state)
        return True

    def get_best_energy_for_U(self, U: float) -> float:
        """Get best energy per site achieved for a specific U stage."""
        u_key = float(U)
        e = self.best_energy_by_U.get(u_key)
        if e is None:
            return float('inf')
        return e / self.N

    def get_best_energy(self) -> float:
        """Get the best (lowest) energy per site achieved during training."""
        if not self.history['energy']:
            return float('inf')
        # Use rolling average of last 10% of training for stability
        energies = self.history['energy']
        n_avg = max(1, len(energies) // 10)
        return min(np.mean(energies[-n_avg:]) / self.N, min(energies) / self.N)
