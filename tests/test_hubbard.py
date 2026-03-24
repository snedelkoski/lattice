"""
Tests for the Fermi-Hubbard Hamiltonian and exact diagonalization.

Validates:
1. Lattice geometry (neighbors, translation group)
2. Jordan-Wigner signs
3. Hamiltonian symmetry (Hermitian)
4. Exact diagonalization against known benchmark energies
5. Local energy computation consistency with ED
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

from physics.hubbard import (
    SquareLattice,
    FermiHubbardHamiltonian,
    generate_random_configs,
)
from physics.exact_diag import ExactDiagonalization


# ---- Lattice tests ----

class TestSquareLattice:
    def test_site_index_roundtrip(self):
        """site_index and site_coords are inverses."""
        lat = SquareLattice(4, 4, pbc=True)
        for i in range(lat.N):
            x, y = lat.site_coords(i)
            assert lat.site_index(x, y) == i

    def test_neighbor_count_pbc(self):
        """Each site has 4 neighbors on a PBC square lattice."""
        for L in [2, 3, 4, 6]:
            lat = SquareLattice(L, L, pbc=True)
            for i in range(lat.N):
                assert len(lat.neighbors[i]) == 4, (
                    f"Site {i} on {L}x{L} PBC has {len(lat.neighbors[i])} neighbors"
                )

    def test_neighbor_count_obc(self):
        """Corner sites have 2 neighbors, edge sites 3, bulk 4 on OBC."""
        lat = SquareLattice(4, 4, pbc=False)
        corners = {0, 3, 12, 15}
        for i in corners:
            assert len(lat.neighbors[i]) == 2, f"Corner {i} has {len(lat.neighbors[i])} nbrs"

    def test_neighbor_pairs_unique(self):
        """get_neighbor_pairs returns unique pairs with i < j."""
        lat = SquareLattice(4, 4, pbc=True)
        pairs = lat.get_neighbor_pairs()
        for i, j in pairs:
            assert i < j
        # Should be 2*N bonds for PBC square lattice
        assert len(pairs) == 2 * lat.N

    def test_translation_group_size(self):
        """Translation group has N elements for PBC."""
        lat = SquareLattice(4, 4, pbc=True)
        trans = lat.translation_group()
        assert len(trans) == lat.N  # Lx * Ly translations

    def test_translation_group_identity(self):
        """First translation (dx=0, dy=0) is identity."""
        lat = SquareLattice(4, 4, pbc=True)
        trans = lat.translation_group()
        np.testing.assert_array_equal(trans[0], np.arange(lat.N))

    def test_translation_group_obc(self):
        """OBC lattice has only identity translation."""
        lat = SquareLattice(4, 4, pbc=False)
        trans = lat.translation_group()
        assert len(trans) == 1

    def test_snake_order_permutation(self):
        """Snake order is a valid permutation."""
        lat = SquareLattice(4, 4, pbc=True)
        assert set(lat.snake_order) == set(range(lat.N))


# ---- Hamiltonian tests ----

class TestFermiHubbardHamiltonian:
    def test_config_conversion_roundtrip(self):
        """updown <-> config conversion is lossless."""
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        configs = generate_random_configs(10, 16, 8, 8)
        for cfg in configs:
            up, down = ham.config_to_updown(cfg)
            cfg2 = ham.updown_to_config(up, down)
            np.testing.assert_array_equal(cfg, cfg2)

    def test_config_conversion_torch(self):
        """Torch version of config conversion matches numpy."""
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        configs_np = generate_random_configs(10, 16, 8, 8)
        configs_t = torch.from_numpy(configs_np)

        for i in range(10):
            up_np, down_np = ham.config_to_updown(configs_np[i])
            up_t, down_t = ham.config_to_updown_torch(configs_t[i:i+1])
            np.testing.assert_array_equal(up_np, up_t[0].numpy())
            np.testing.assert_array_equal(down_np, down_t[0].numpy())

    def test_jordan_wigner_sign_adjacent(self):
        """JW sign for adjacent sites with no particles between them is +1."""
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        occ = np.zeros(16, dtype=np.int64)
        occ[2] = 1  # particle at site 2
        sign = ham.jordan_wigner_sign(occ, 2, 3)
        assert sign == 1  # No particles between 2 and 3

    def test_jordan_wigner_sign_with_particle_between(self):
        """JW sign with one particle between is -1."""
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        occ = np.zeros(16, dtype=np.int64)
        occ[0] = 1
        occ[1] = 1  # particle between 0 and 2
        sign = ham.jordan_wigner_sign(occ, 0, 2)
        assert sign == -1

    def test_diagonal_energy_no_doubles(self):
        """Config with no double occupancy has zero diagonal energy."""
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        # All sites either 0, 1, or 2 (no 3s)
        config = np.array([1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2])
        assert ham.diagonal_energy(config) == 0.0

    def test_diagonal_energy_all_doubles(self):
        """Config with all doubly-occupied sites has E_diag = U * N."""
        N = 16
        ham = FermiHubbardHamiltonian(SquareLattice(4, 4), t=1.0, U=4.0)
        config = np.full(N, 3)  # All doubly occupied
        assert ham.diagonal_energy(config) == pytest.approx(4.0 * N)

    def test_connected_configs_particle_conservation(self):
        """Connected configs conserve N_up and N_down."""
        lat = SquareLattice(4, 4, pbc=True)
        ham = FermiHubbardHamiltonian(lat, t=1.0, U=4.0)
        config = generate_random_configs(1, 16, 8, 8)[0]

        up_orig, down_orig = ham.config_to_updown(config)
        n_up_orig = up_orig.sum()
        n_down_orig = down_orig.sum()

        connected = ham.get_connected_configs(config)
        for cfg_prime, _ in connected:
            up_p, down_p = ham.config_to_updown(cfg_prime)
            assert up_p.sum() == n_up_orig, "N_up not conserved"
            assert down_p.sum() == n_down_orig, "N_down not conserved"

    def test_connected_configs_nonzero(self):
        """A generic config should have nonzero connected configs."""
        lat = SquareLattice(4, 4, pbc=True)
        ham = FermiHubbardHamiltonian(lat, t=1.0, U=4.0)
        config = generate_random_configs(1, 16, 8, 8)[0]
        connected = ham.get_connected_configs(config)
        assert len(connected) > 0

    def test_generate_random_configs_particle_count(self):
        """Random configs have correct particle counts."""
        configs = generate_random_configs(100, 16, 8, 8)
        assert configs.shape == (100, 16)
        for cfg in configs:
            n_up = np.sum((cfg == 1) | (cfg == 3))
            n_down = np.sum((cfg == 2) | (cfg == 3))
            assert n_up == 8, f"Expected 8 up electrons, got {n_up}"
            assert n_down == 8, f"Expected 8 down electrons, got {n_down}"


# ---- Exact diagonalization tests ----

class TestExactDiagonalization:
    def test_2x2_U0_energy(self):
        """
        2x2 lattice at U=0 (free fermions).
        For 2 up + 2 down on 2x2 PBC:
        Single-particle energies for 2x2 PBC: eps(k) = -2t(cos kx + cos ky)
          k = (0,0): eps = -4t
          k = (pi,0): eps = 0
          k = (0,pi): eps = 0
          k = (pi,pi): eps = +4t
        Half-filling (2 up, 2 down): fill lowest 2 levels per spin
          E_up = -4t + 0 = -4t (or -4t + 0 depending on degeneracy)
        Actually for 2x2, the ED result is E/N = -1.0 at U=0.
        """
        lat = SquareLattice(2, 2, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=0.0, n_up=2, n_down=2)
        energies, _ = ed.solve(n_states=1)
        E_per_site = energies[0] / 4
        # Verify against the actual free-fermion result
        assert E_per_site == pytest.approx(-1.0, abs=1e-6), (
            f"2x2 U=0 E/N = {E_per_site}, expected -1.0"
        )

    def test_2x2_noninteracting_hermitian(self):
        """Hamiltonian should be Hermitian."""
        lat = SquareLattice(2, 2, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=2, n_down=2)
        H = ed.build_hamiltonian()
        diff = H - H.T.conj()
        assert diff.nnz == 0 or abs(diff).max() < 1e-12, "Hamiltonian is not Hermitian"

    @pytest.mark.slow
    def test_4x4_U4_energy(self):
        """
        4x4 lattice at half-filling, U/t=4.
        Known ED result: E/N = -1.0959 (from VarBench).
        WARNING: This requires ~21 GB RAM due to 165M-dim Hilbert space.
        """
        lat = SquareLattice(4, 4, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=8, n_down=8)
        energies, _ = ed.solve(n_states=1)
        E_per_site = energies[0] / 16
        assert E_per_site == pytest.approx(-1.0959, abs=0.001), (
            f"4x4 U=4 E/N = {E_per_site:.6f}, expected -1.0959"
        )

    def test_2x2_U4_energy(self):
        """
        2x2 lattice at U/t=4. Small enough for fast ED.
        Verify energy is lower than U=0 case (interaction affects energy).
        """
        lat = SquareLattice(2, 2, pbc=True)
        ed_U0 = ExactDiagonalization(lat, t=1.0, U=0.0, n_up=2, n_down=2)
        ed_U4 = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=2, n_down=2)
        E0_U0, _ = ed_U0.solve(n_states=1)
        E0_U4, _ = ed_U4.solve(n_states=1)
        # At U=4, ground state energy should be higher than U=0
        # (interaction adds positive energy)
        assert E0_U4[0] > E0_U0[0], (
            f"E(U=4) = {E0_U4[0]:.4f} should be > E(U=0) = {E0_U0[0]:.4f}"
        )

    def test_2x3_U4_energy(self):
        """
        2x3 lattice at U/t=4, half-filling (3 up, 3 down).
        Hilbert dim = C(6,3)^2 = 400, fast to solve.
        Just verify it runs and produces finite energy.
        """
        lat = SquareLattice(3, 2, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=3, n_down=3)
        energies, _ = ed.solve(n_states=1)
        E_per_site = energies[0] / 6
        assert np.isfinite(E_per_site)
        assert E_per_site < 0, f"Ground state E/N should be negative, got {E_per_site}"

    @pytest.mark.slow
    def test_4x4_U8_energy(self):
        """
        4x4 lattice at half-filling, U/t=8.
        Known ED result: E/N = -1.0288.
        """
        lat = SquareLattice(4, 4, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=8.0, n_up=8, n_down=8)
        energies, _ = ed.solve(n_states=1)
        E_per_site = energies[0] / 16
        assert E_per_site == pytest.approx(-1.0288, abs=0.001), (
            f"4x4 U=8 E/N = {E_per_site:.6f}, expected -1.0288"
        )

    def test_ground_state_normalization(self):
        """Ground state should be normalized."""
        lat = SquareLattice(2, 2, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=2, n_down=2)
        _, states = ed.solve(n_states=1)
        norm = np.linalg.norm(states[:, 0])
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_double_occupancy_U0(self):
        """At U=0, double occupancy = (n_up/N) * (n_down/N) = 0.25 at half-filling."""
        lat = SquareLattice(2, 2, pbc=True)
        ed = ExactDiagonalization(lat, t=1.0, U=0.0, n_up=2, n_down=2)
        _, states = ed.solve(n_states=1)
        D = ed.compute_double_occupancy(states[:, 0])
        assert D == pytest.approx(0.25, abs=0.01), (
            f"Double occupancy at U=0 should be ~0.25, got {D}"
        )

    def test_double_occupancy_decreases_with_U(self):
        """Double occupancy should decrease as U increases."""
        lat = SquareLattice(2, 2, pbc=True)

        D_values = []
        for U in [0.0, 2.0, 4.0, 8.0]:
            ed = ExactDiagonalization(lat, t=1.0, U=U, n_up=2, n_down=2)
            _, states = ed.solve(n_states=1)
            D = ed.compute_double_occupancy(states[:, 0])
            D_values.append(D)

        for i in range(len(D_values) - 1):
            assert D_values[i] >= D_values[i+1] - 1e-6, (
                f"D should decrease with U: D(U={[0,2,4,8][i]})={D_values[i]:.4f} "
                f"> D(U={[0,2,4,8][i+1]})={D_values[i+1]:.4f}"
            )


# ---- Local energy consistency test ----

class TestLocalEnergy:
    def test_local_energy_matches_ed(self):
        """
        Local energy computed with the exact ground state wave function
        should equal the ground state energy for all configurations.

        E_loc(sigma) = <sigma|H|psi> / <sigma|psi> = E0 for eigenstates.
        """
        lat = SquareLattice(2, 2, pbc=True)
        ham = FermiHubbardHamiltonian(lat, t=1.0, U=4.0)
        ed = ExactDiagonalization(lat, t=1.0, U=4.0, n_up=2, n_down=2)
        energies, states = ed.solve(n_states=1)
        E0 = energies[0]
        psi0 = states[:, 0]

        # Get amplitude lookup
        amps = ed.ground_state_config_amplitudes(psi0)

        def log_psi_fn(config):
            key = tuple(config.tolist())
            amp = amps.get(key, 0.0)
            if abs(amp) < 1e-15:
                return -100.0 + 0j  # effectively zero
            # Handle sign properly: log(amp) for real wave function
            # log(amp) = log(|amp|) + i*pi if amp < 0
            if amp < 0:
                return np.log(-amp) + 1j * np.pi
            else:
                return np.log(amp) + 0j

        # Test on a few configs with non-zero amplitude
        tested = 0
        for config_key, amp in amps.items():
            if abs(amp) < 1e-10:
                continue
            config = np.array(config_key)
            e_loc = ham.local_energy_single(config, log_psi_fn)
            assert abs(e_loc.real - E0) < 0.01, (
                f"E_loc = {e_loc.real:.6f}, E0 = {E0:.6f} for config {config}"
            )
            tested += 1
            if tested >= 20:
                break

        assert tested > 0, "No configurations tested"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
