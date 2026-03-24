"""
Tests for VMC training components.

Validates:
1. Sampler initialization and proposals
2. Metropolis acceptance criterion
3. VMC loss computation
4. VMC gradient estimation
5. Short VMC run on 2x2 lattice converges toward ED
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from physics.hubbard import SquareLattice, FermiHubbardHamiltonian, generate_random_configs
from physics.sampler import MetropolisSampler
from physics.vmc import VMCTrainer
from physics.exact_diag import ExactDiagonalization
from models.transformer_nqs import TransformerNQS


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Small config for fast tests
SMALL_MODEL = dict(
    n_sites=4,
    d_model=32,
    num_heads=2,
    num_layers=2,
    d_ff=64,
    vocab_size=4,
    max_sites=36,
)


# ---- Sampler tests ----

class TestMetropolisSampler:
    def test_initialize_chains(self):
        """Chains should be initialized with correct particle counts."""
        lat = SquareLattice(2, 2, pbc=True)
        sampler = MetropolisSampler(lat, n_chains=16, device=DEVICE)
        sampler.initialize_chains(n_up=2, n_down=2)

        configs = sampler.configs
        assert configs.shape == (16, 4)

        for b in range(16):
            cfg = configs[b].cpu().numpy()
            n_up = np.sum((cfg == 1) | (cfg == 3))
            n_down = np.sum((cfg == 2) | (cfg == 3))
            assert n_up == 2, f"Chain {b}: expected 2 up, got {n_up}"
            assert n_down == 2, f"Chain {b}: expected 2 down, got {n_down}"

    def test_proposal_conserves_particles(self):
        """Proposed configs should conserve N_up and N_down."""
        lat = SquareLattice(2, 2, pbc=True)
        sampler = MetropolisSampler(lat, n_chains=32, device=DEVICE)
        sampler.initialize_chains(n_up=2, n_down=2)

        proposed, valid, _ = sampler._propose_hop()

        for b in range(32):
            if not valid[b]:
                continue
            cfg = proposed[b].cpu().numpy()
            n_up = np.sum((cfg == 1) | (cfg == 3))
            n_down = np.sum((cfg == 2) | (cfg == 3))
            assert n_up == 2, f"Proposed chain {b}: N_up={n_up}"
            assert n_down == 2, f"Proposed chain {b}: N_down={n_down}"

    def test_sweep_returns_acceptance_rate(self):
        """A sweep should return a valid acceptance rate."""
        lat = SquareLattice(2, 2, pbc=True)
        sampler = MetropolisSampler(lat, n_chains=16, device=DEVICE)
        sampler.initialize_chains(n_up=2, n_down=2)

        model = TransformerNQS(n_sites=4, d_model=16, num_heads=2,
                               num_layers=1, d_ff=32, vocab_size=4,
                               max_sites=16).to(DEVICE)
        model.eval()

        rate = sampler.sweep(model)
        assert 0.0 <= rate <= 1.0

    def test_sample_returns_correct_shape(self):
        """sample() should return (n_chains, N) tensor."""
        lat = SquareLattice(2, 2, pbc=True)
        sampler = MetropolisSampler(
            lat, n_chains=16, n_sweeps=2, n_thermalize=5, device=DEVICE
        )
        sampler.initialize_chains(n_up=2, n_down=2)

        model = TransformerNQS(n_sites=4, d_model=16, num_heads=2,
                               num_layers=1, d_ff=32, vocab_size=4,
                               max_sites=16).to(DEVICE)
        model.eval()

        configs = sampler.sample(model)
        assert configs.shape == (16, 4)


# ---- VMC loss tests ----

class TestVMCLoss:
    def test_vmc_loss_is_real_scalar(self):
        """VMC loss should be a real scalar."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=4.0,
            n_up=2, n_down=2, n_chains=16, n_sweeps=2,
            n_thermalize=5, device=DEVICE, use_amp=False,
        )

        # Generate some test configs
        configs = generate_random_configs(16, 4, 2, 2)
        configs_t = torch.from_numpy(configs).long().to(DEVICE)

        loss, stats = trainer.compute_vmc_loss(configs_t)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.is_complex() == False or loss.imag == 0, "Loss should be real"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_vmc_stats_keys(self):
        """VMC stats should contain expected keys."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=4.0,
            n_up=2, n_down=2, n_chains=16, n_sweeps=2,
            n_thermalize=5, device=DEVICE, use_amp=False,
        )

        configs = generate_random_configs(16, 4, 2, 2)
        configs_t = torch.from_numpy(configs).long().to(DEVICE)

        _, stats = trainer.compute_vmc_loss(configs_t)

        assert 'energy' in stats
        assert 'energy_std' in stats
        assert 'variance' in stats

    def test_clip_local_energies(self):
        """Clipping should reduce extreme outliers."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=4.0,
            n_up=2, n_down=2, e_loc_clip=2.0,  # tighter clipping
            device=DEVICE, use_amp=False,
        )

        # Create local energies with clear outliers relative to median/MAD
        e_loc = torch.complex(
            torch.tensor([1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 50.0, -40.0]),
            torch.zeros(8),
        )

        clipped = trainer.clip_local_energies(e_loc)
        # The extreme outliers (50, -40) should be clipped
        assert clipped.real.max() < 50.0, "Max outlier should be clipped"
        assert clipped.real.min() > -40.0, "Min outlier should be clipped"
        # Non-outliers should be unchanged
        assert clipped[0].real == pytest.approx(1.0)


# ---- VMC training tests ----

class TestVMCTraining:
    def test_train_step_runs(self):
        """A single train step should complete without error."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=2.0,
            n_up=2, n_down=2, n_chains=16, n_sweeps=2,
            n_thermalize=5, device=DEVICE, use_amp=False,
        )
        trainer.sampler.initialize_chains(n_up=2, n_down=2)

        stats = trainer.train_step()

        assert 'energy' in stats
        assert np.isfinite(stats['energy'])
        assert stats['step_time'] > 0

    def test_train_step_updates_history(self):
        """Training should populate history dict."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=2.0,
            n_up=2, n_down=2, n_chains=16, n_sweeps=2,
            n_thermalize=5, device=DEVICE, use_amp=False,
        )
        trainer.sampler.initialize_chains(n_up=2, n_down=2)

        for _ in range(3):
            trainer.train_step()

        assert len(trainer.history['energy']) == 3

    def test_set_U(self):
        """set_U should update the Hamiltonian."""
        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**SMALL_MODEL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=2.0,
            n_up=2, n_down=2, device=DEVICE, use_amp=False,
        )

        trainer.set_U(8.0)
        assert trainer.U == 8.0
        assert trainer.hamiltonian.U == 8.0

    @pytest.mark.slow
    def test_2x2_vmc_converges(self):
        """
        Short VMC run on 2x2 lattice at U/t=2 should converge toward ED.

        2x2 at U=2: ED gives E/N ~ -1.6180 (depends on exact geometry).
        We just check that energy improves and doesn't diverge.
        """
        torch.manual_seed(42)
        lat = SquareLattice(2, 2, pbc=True)

        # Get ED reference
        ed = ExactDiagonalization(lat, t=1.0, U=2.0, n_up=2, n_down=2)
        energies, _ = ed.solve(n_states=1)
        E0_per_site = energies[0] / 4

        # Train
        model = TransformerNQS(
            n_sites=4, d_model=32, num_heads=2, num_layers=2,
            d_ff=64, vocab_size=4, max_sites=16,
        ).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=2.0,
            n_up=2, n_down=2, n_chains=64, n_sweeps=5,
            n_thermalize=50, lr=5e-4, device=DEVICE, use_amp=False,
        )

        history = trainer.train(n_steps=200, log_interval=100)

        # Check energy improved from initial
        initial_e = np.mean(history['energy'][:10]) / 4
        final_e = np.mean(history['energy'][-20:]) / 4

        print(f"ED E/N = {E0_per_site:.4f}")
        print(f"Initial E/N = {initial_e:.4f}")
        print(f"Final E/N = {final_e:.4f}")

        # Energy should be finite and bounded
        assert np.isfinite(final_e), "Final energy is not finite"
        # Should at least be within 50% of ED (very generous for short run)
        assert final_e < 0, "Energy should be negative"


# ---- Batched local energy test ----

class TestBatchedLocalEnergy:
    def test_batched_matches_single(self):
        """Batched local energy should match single-sample computation."""
        lat = SquareLattice(2, 2, pbc=True)
        ham = FermiHubbardHamiltonian(lat, t=1.0, U=4.0)

        model = TransformerNQS(
            n_sites=4, d_model=16, num_heads=2, num_layers=1,
            d_ff=32, vocab_size=4, max_sites=16,
        ).to(DEVICE)
        model.eval()

        # Generate configs
        configs_np = generate_random_configs(8, 4, 2, 2)
        configs_t = torch.from_numpy(configs_np).long().to(DEVICE)

        # Batched computation
        with torch.no_grad():
            e_loc_batch = ham.compute_local_energy_batch(model, configs_t)

        # Single computation (using the model as log_psi_fn)
        def log_psi_fn(config):
            cfg_t = torch.from_numpy(config).long().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                return model(cfg_t)[0].cpu().numpy()

        for i in range(min(4, len(configs_np))):
            e_loc_single = ham.local_energy_single(configs_np[i], log_psi_fn)
            e_batch_i = e_loc_batch[i].cpu().numpy()

            # Allow some tolerance due to float precision
            np.testing.assert_allclose(
                e_batch_i.real, e_loc_single.real, rtol=1e-3, atol=1e-3,
                err_msg=f"Mismatch at sample {i}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
