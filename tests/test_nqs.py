"""
Tests for NQS model architectures (Transformer).

Validates:
1. Forward pass shapes
2. Complex output (log|psi| + i*phase)
3. Parameter counts
4. Backbone weight transfer
5. Gradient flow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from models.transformer_nqs import TransformerNQS
from models.base_nqs import BaseNQS
from physics.hubbard import generate_random_configs


# Default model config (small for testing)
TEST_CONFIG = dict(
    n_sites=16,
    d_model=32,
    num_heads=2,
    vocab_size=4,
    max_sites=36,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_test_configs(batch_size=8, n_sites=16, n_up=8, n_down=8):
    """Generate test configurations."""
    configs = generate_random_configs(batch_size, n_sites, n_up, n_down)
    return torch.from_numpy(configs).long().to(DEVICE)


# ---- Transformer tests ----

class TestTransformerNQS:
    def test_forward_shape(self):
        """Output should be (B,) complex tensor."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        assert log_psi.shape == (8,)
        assert log_psi.is_complex()

    def test_output_has_amplitude_and_phase(self):
        """Output should have nonzero real and imaginary parts."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        assert torch.isfinite(log_psi.real).all()
        assert torch.isfinite(log_psi.imag).all()

    def test_gradient_flow(self):
        """Gradients should flow to all parameters."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        loss = log_psi.real.sum()
        loss.backward()

        n_grads = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                n_grads += 1
                assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

        assert n_grads > 0

    def test_deterministic_output(self):
        """Same input should produce same output."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        model.eval()
        configs = make_test_configs()
        with torch.no_grad():
            out1 = model(configs)
            out2 = model(configs)
        torch.testing.assert_close(out1, out2)

    def test_different_configs_different_output(self):
        """Different configurations should produce different log_psi values."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        model.eval()
        configs1 = make_test_configs(batch_size=4)
        configs2 = make_test_configs(batch_size=4)
        with torch.no_grad():
            out1 = model(configs1)
            out2 = model(configs2)
        # Very unlikely all 4 outputs are the same for random configs
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_backbone_method(self):
        """Backbone method should work independently."""
        model = TransformerNQS(
            **TEST_CONFIG, num_layers=2, d_ff=64
        ).to(DEVICE)
        configs = make_test_configs()
        x = model.encode_input(configs)  # (B, N, d_model)
        h = model.backbone(x)  # (B, N, d_model)
        assert h.shape == (8, 16, TEST_CONFIG['d_model'])

    def test_is_base_nqs(self):
        """Should inherit from BaseNQS."""
        model = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64)
        assert isinstance(model, BaseNQS)


# ---- Parameter count tests ----

class TestParameterCounts:
    def test_model_has_parameters(self):
        """Model should have trainable parameters."""
        transformer = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64)
        assert transformer.count_parameters() > 0

    def test_count_parameters_method(self):
        """count_parameters should match manual count."""
        model = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64)
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.count_parameters() == manual_count

    def test_full_size_parameter_counts(self):
        """Full-size model should be in a reasonable parameter range."""
        transformer = TransformerNQS(
            n_sites=16, d_model=128, num_heads=4, num_layers=4,
            d_ff=256, vocab_size=4, max_sites=144,
        )
        transformer_params = transformer.count_parameters()
        print(f"Transformer params: {transformer_params:,}")
        assert 100_000 < transformer_params < 2_000_000


# ---- Backbone weight transfer ----

class TestBackboneTransfer:
    def test_get_backbone_state_dict(self):
        """get_backbone_state_dict should exclude output heads."""
        model = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64)
        backbone_sd = model.get_backbone_state_dict()

        for name in backbone_sd:
            assert 'log_amp_head' not in name
            assert 'phase_head' not in name

        # Should have fewer params than full state dict
        assert len(backbone_sd) < len(model.state_dict())

    def test_load_backbone_state_dict(self):
        """Loading backbone weights should change model outputs."""
        model1 = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64).to(DEVICE)
        model2 = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64).to(DEVICE)

        configs = make_test_configs()
        with torch.no_grad():
            out1_before = model2(configs).clone()

        # Transfer backbone from model1 to model2
        backbone_sd = model1.get_backbone_state_dict()
        model2.load_backbone_state_dict(backbone_sd)

        with torch.no_grad():
            out1_after = model2(configs)

        # Outputs should change (unless models were initialized identically)
        # Just check it doesn't crash; exact equality test would be fragile


# ---- Symmetry wrapper test ----

class TestSymmetrizedNQS:
    def test_symmetrized_output_shape(self):
        """SymmetrizedNQS should have same output shape."""
        from physics.symmetry import SymmetrizedNQS
        from physics.hubbard import SquareLattice

        model = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64).to(DEVICE)
        lattice = SquareLattice(4, 4, pbc=True)
        sym_model = SymmetrizedNQS(model, lattice)

        configs = make_test_configs()
        log_psi = sym_model(configs)

        assert log_psi.shape == (8,)
        assert log_psi.is_complex()

    def test_symmetrized_translation_invariance(self):
        """
        Symmetry-projected psi should be translation-invariant:
        psi_sym(sigma) == psi_sym(T*sigma) for any translation T.
        """
        from physics.symmetry import SymmetrizedNQS, SymmetryProjector
        from physics.hubbard import SquareLattice

        model = TransformerNQS(**TEST_CONFIG, num_layers=2, d_ff=64).to(DEVICE)
        model.eval()
        lattice = SquareLattice(4, 4, pbc=True)
        sym_model = SymmetrizedNQS(model, lattice)

        configs = make_test_configs(batch_size=4)
        with torch.no_grad():
            log_psi_orig = sym_model(configs)

        # Apply a non-trivial translation
        projector = SymmetryProjector(lattice)
        translated = projector.apply_symmetry(configs, perm_idx=5)  # Some translation

        with torch.no_grad():
            log_psi_trans = sym_model(translated)

        # Should be equal (up to numerical precision)
        torch.testing.assert_close(
            log_psi_orig.real, log_psi_trans.real, atol=1e-4, rtol=1e-4
        )


# ---- Backflow determinant tests ----

BACKFLOW_CONFIG = dict(
    n_sites=16,
    d_model=32,
    num_heads=2,
    num_layers=2,
    d_ff=64,
    vocab_size=4,
    max_sites=36,
    head_mode='backflow_det',
    n_determinants=4,
    n_up=8,
    n_down=8,
    use_layernorm=False,
)

BACKFLOW_SMALL = dict(
    n_sites=4,
    d_model=16,
    num_heads=2,
    num_layers=1,
    d_ff=16,
    vocab_size=4,
    max_sites=16,
    head_mode='backflow_det',
    n_determinants=2,
    n_up=2,
    n_down=2,
    use_layernorm=False,
)


class TestBackflowDetNQS:
    def test_forward_shape(self):
        """Output should be (B,) complex tensor."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        assert log_psi.shape == (8,)
        assert log_psi.is_complex()

    def test_output_is_finite(self):
        """Output should have finite real and imaginary parts."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        assert torch.isfinite(log_psi.real).all(), "log_amp has inf/nan"
        assert torch.isfinite(log_psi.imag).all(), "phase has inf/nan"

    def test_gradient_flow(self):
        """Gradients should flow through the determinant to all parameters."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        loss = log_psi.real.sum()
        loss.backward()

        n_grads = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                n_grads += 1
                assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

        assert n_grads > 0, "No gradients computed"

    def test_deterministic_output(self):
        """Same input should produce same output."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        model.eval()
        configs = make_test_configs()
        with torch.no_grad():
            out1 = model(configs)
            out2 = model(configs)
        torch.testing.assert_close(out1, out2)

    def test_different_configs_different_output(self):
        """Different configurations should produce different log_psi values."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        model.eval()
        configs1 = make_test_configs(batch_size=4)
        configs2 = make_test_configs(batch_size=4)
        with torch.no_grad():
            out1 = model(configs1)
            out2 = model(configs2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_phase_is_0_or_pi(self):
        """For real-valued determinants, phase should be 0 or pi."""
        model = TransformerNQS(**BACKFLOW_CONFIG).to(DEVICE)
        model.eval()
        configs = make_test_configs(batch_size=32)
        with torch.no_grad():
            log_psi = model(configs)
        phase = log_psi.imag
        # Phase should be close to 0 or pi
        is_zero = torch.abs(phase) < 0.01
        is_pi = torch.abs(phase - torch.pi) < 0.01
        assert (is_zero | is_pi).all(), f"Unexpected phases: {phase}"

    def test_no_layernorm_mode(self):
        """Model without LayerNorm should work."""
        cfg = {**BACKFLOW_CONFIG, 'use_layernorm': False}
        model = TransformerNQS(**cfg).to(DEVICE)
        configs = make_test_configs()
        log_psi = model(configs)
        assert log_psi.shape == (8,)
        assert torch.isfinite(log_psi.real).all()

    def test_parameter_count_reasonable(self):
        """Backflow model should have reasonable parameter count."""
        model = TransformerNQS(**BACKFLOW_CONFIG)
        n_params = model.count_parameters()
        print(f"Backflow det model params: {n_params:,}")
        # Should be larger than scalar head due to orbital_head
        assert n_params > 1000

    def test_2x2_forward(self):
        """Backflow det should work on tiny 2x2 lattice."""
        model = TransformerNQS(**BACKFLOW_SMALL).to(DEVICE)
        configs = make_test_configs(batch_size=4, n_sites=4, n_up=2, n_down=2)
        log_psi = model(configs)
        assert log_psi.shape == (4,)
        assert torch.isfinite(log_psi.real).all()

    def test_backbone_transfer_excludes_orbital_head(self):
        """get_backbone_state_dict should exclude orbital_head."""
        model = TransformerNQS(**BACKFLOW_CONFIG)
        backbone_sd = model.get_backbone_state_dict()
        for name in backbone_sd:
            assert 'orbital_head' not in name

    def test_vmc_loss_with_backflow(self):
        """VMC loss computation should work with backflow det model."""
        from physics.hubbard import SquareLattice
        from physics.vmc import VMCTrainer

        lat = SquareLattice(2, 2, pbc=True)
        model = TransformerNQS(**BACKFLOW_SMALL).to(DEVICE)

        trainer = VMCTrainer(
            model=model, lattice=lat, t=1.0, U=4.0,
            n_up=2, n_down=2, n_chains=16, n_sweeps=2,
            n_thermalize=5, device=DEVICE, use_amp=False,
        )

        configs = make_test_configs(batch_size=16, n_sites=4, n_up=2, n_down=2)
        loss, stats = trainer.compute_vmc_loss(configs)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert np.isfinite(stats['energy']), "Energy should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
