"""
Exact diagonalization for the Fermi-Hubbard model.

Uses scipy sparse matrices and Lanczos algorithm to find ground state
energies and wave functions for small lattices (up to 3x4 = 12 sites
at half-filling).

The Hilbert space is constructed in the occupation number basis with
fixed particle numbers (N_up, N_down), using the second-quantized
representation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from itertools import combinations
from typing import Optional

from .hubbard import SquareLattice, FermiHubbardHamiltonian


class ExactDiagonalization:
    """
    Exact diagonalization of the Fermi-Hubbard model in the
    (N_up, N_down) sector.

    The basis states are tensor products of up-spin and down-spin
    Fock states. Each spin sector has C(N_sites, N_particles) states.
    """

    def __init__(
        self,
        lattice: SquareLattice,
        t: float = 1.0,
        U: float = 4.0,
        n_up: Optional[int] = None,
        n_down: Optional[int] = None,
    ):
        self.lattice = lattice
        self.N = lattice.N
        self.t = t
        self.U = U
        self.n_up = n_up if n_up is not None else self.N // 2
        self.n_down = n_down if n_down is not None else self.N // 2

        # Build basis states for each spin sector
        self.up_states = self._build_sector_basis(self.n_up)
        self.down_states = self._build_sector_basis(self.n_down)
        self.dim_up = len(self.up_states)
        self.dim_down = len(self.down_states)
        self.dim = self.dim_up * self.dim_down

        # Build lookup tables for fast state -> index mapping
        self.up_lookup = {s: i for i, s in enumerate(self.up_states)}
        self.down_lookup = {s: i for i, s in enumerate(self.down_states)}

        print(f"ED: {self.N} sites, {self.n_up} up, {self.n_down} down")
        print(f"    Hilbert space: {self.dim_up} x {self.dim_down} = {self.dim}")

    def _build_sector_basis(self, n_particles: int) -> list[int]:
        """
        Build all Fock states with exactly n_particles in N sites.
        Each state is represented as an integer (bitstring).

        Returns sorted list of state integers.
        """
        states = []
        for sites in combinations(range(self.N), n_particles):
            state = 0
            for s in sites:
                state |= (1 << s)
            states.append(state)
        return sorted(states)

    def _state_to_occ(self, state: int) -> np.ndarray:
        """Convert bitstring integer to occupation array."""
        occ = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            if state & (1 << i):
                occ[i] = 1
        return occ

    def _hopping_matrix_elements(self, states: list[int], lookup: dict) -> sparse.csr_matrix:
        """
        Build the hopping (kinetic) Hamiltonian for one spin sector.

        H_kin = -t sum_{<ij>} (c+_i c_j + h.c.)

        Returns sparse matrix of dimension (dim_sector, dim_sector).
        """
        dim = len(states)
        rows, cols, vals = [], [], []

        neighbor_pairs = self.lattice.get_neighbor_pairs()

        for idx, state in enumerate(states):
            occ = self._state_to_occ(state)

            for i, j in neighbor_pairs:
                # Hop from j to i: c+_i c_j
                if occ[j] == 1 and occ[i] == 0:
                    new_state = state ^ (1 << j) ^ (1 << i)  # flip bits
                    # Jordan-Wigner sign
                    lo, hi = min(i, j), max(i, j)
                    n_between = 0
                    for k in range(lo + 1, hi):
                        if state & (1 << k):
                            n_between += 1
                    sign = (-1) ** n_between

                    new_idx = lookup.get(new_state)
                    if new_idx is not None:
                        rows.append(new_idx)
                        cols.append(idx)
                        vals.append(-self.t * sign)

                # Hop from i to j: c+_j c_i
                if occ[i] == 1 and occ[j] == 0:
                    new_state = state ^ (1 << i) ^ (1 << j)
                    lo, hi = min(i, j), max(i, j)
                    n_between = 0
                    for k in range(lo + 1, hi):
                        if state & (1 << k):
                            n_between += 1
                    sign = (-1) ** n_between

                    new_idx = lookup.get(new_state)
                    if new_idx is not None:
                        rows.append(new_idx)
                        cols.append(idx)
                        vals.append(-self.t * sign)

        return sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim))

    def build_hamiltonian(self) -> sparse.csr_matrix:
        """
        Build the full Hamiltonian as a sparse matrix in the
        (N_up, N_down) sector.

        H = H_kin_up (x) I_down + I_up (x) H_kin_down + H_int

        The combined basis index for (up_idx, down_idx) is:
            combined_idx = up_idx * dim_down + down_idx
        """
        print(f"Building Hamiltonian (dim = {self.dim})...")

        # Kinetic energy for spin-up
        H_kin_up = self._hopping_matrix_elements(self.up_states, self.up_lookup)
        # Kinetic energy for spin-down
        H_kin_down = self._hopping_matrix_elements(self.down_states, self.down_lookup)

        # Kronecker products: H_kin_up (x) I_down + I_up (x) H_kin_down
        I_up = sparse.eye(self.dim_up, format='csr')
        I_down = sparse.eye(self.dim_down, format='csr')

        H = sparse.kron(H_kin_up, I_down, format='csr') + \
            sparse.kron(I_up, H_kin_down, format='csr')

        # Interaction energy: U * sum_i n_i_up * n_i_down (diagonal)
        diag_vals = np.zeros(self.dim)
        for up_idx, up_state in enumerate(self.up_states):
            up_occ = self._state_to_occ(up_state)
            for down_idx, down_state in enumerate(self.down_states):
                down_occ = self._state_to_occ(down_state)
                combined_idx = up_idx * self.dim_down + down_idx
                n_double = np.sum(up_occ * down_occ)
                diag_vals[combined_idx] = self.U * n_double

        H += sparse.diags(diag_vals, format='csr')

        print(f"Hamiltonian built: {H.nnz} nonzero elements")
        return H

    def solve(self, n_states: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the lowest n_states eigenvalues and eigenvectors.

        Returns:
            energies: (n_states,) array of eigenvalues
            states: (dim, n_states) array of eigenvectors
        """
        H = self.build_hamiltonian()

        if self.dim <= 2000:
            # Full diagonalization for small systems
            print("Using full diagonalization...")
            H_dense = H.toarray()
            energies, states = np.linalg.eigh(H_dense)
            return energies[:n_states], states[:, :n_states]
        else:
            # Sparse Lanczos
            print(f"Using Lanczos for {n_states} lowest states...")
            energies, states = eigsh(H, k=n_states, which='SA')
            # Sort by energy
            order = np.argsort(energies)
            return energies[order], states[:, order]

    def ground_state_config_amplitudes(
        self, ground_state: np.ndarray
    ) -> dict[tuple, complex]:
        """
        Extract wave function amplitudes for each configuration
        from the ground state eigenvector.

        Returns dict mapping (config_tuple) -> amplitude.
        """
        amplitudes = {}
        for up_idx, up_state in enumerate(self.up_states):
            up_occ = self._state_to_occ(up_state)
            for down_idx, down_state in enumerate(self.down_states):
                down_occ = self._state_to_occ(down_state)
                combined_idx = up_idx * self.dim_down + down_idx
                amp = ground_state[combined_idx]
                if abs(amp) > 1e-15:
                    config = tuple((up_occ + 2 * down_occ).tolist())
                    amplitudes[config] = amp
        return amplitudes

    def compute_double_occupancy(self, state: np.ndarray) -> float:
        """Compute <n_i_up * n_i_down> averaged over sites."""
        D = 0.0
        for up_idx, up_state in enumerate(self.up_states):
            up_occ = self._state_to_occ(up_state)
            for down_idx, down_state in enumerate(self.down_states):
                down_occ = self._state_to_occ(down_state)
                combined_idx = up_idx * self.dim_down + down_idx
                prob = abs(state[combined_idx]) ** 2
                D += prob * np.sum(up_occ * down_occ)
        return D / self.N

    def compute_spin_correlation(self, state: np.ndarray) -> np.ndarray:
        """
        Compute spin-spin correlation matrix <S_i^z S_j^z>.

        Returns (N, N) correlation matrix.
        """
        corr = np.zeros((self.N, self.N))
        for up_idx, up_state in enumerate(self.up_states):
            up_occ = self._state_to_occ(up_state)
            for down_idx, down_state in enumerate(self.down_states):
                down_occ = self._state_to_occ(down_state)
                combined_idx = up_idx * self.dim_down + down_idx
                prob = abs(state[combined_idx]) ** 2

                # S_i^z = (n_i_up - n_i_down) / 2
                sz = 0.5 * (up_occ - down_occ).astype(float)
                corr += prob * np.outer(sz, sz)

        return corr


def run_ed_benchmark(
    Lx: int,
    Ly: int,
    U_values: list[float],
    t: float = 1.0,
    pbc: bool = True,
    n_up: Optional[int] = None,
    n_down: Optional[int] = None,
) -> dict:
    """
    Run exact diagonalization for multiple U/t values and return results.

    Returns dict with keys: energies, energies_per_site, double_occupancy, spin_correlations
    """
    lattice = SquareLattice(Lx, Ly, pbc=pbc)
    N = lattice.N

    if n_up is None:
        n_up = N // 2
    if n_down is None:
        n_down = N // 2

    results = {
        'Lx': Lx, 'Ly': Ly, 'N': N, 'pbc': pbc,
        'n_up': n_up, 'n_down': n_down,
        'U_values': U_values,
        'energies': [],
        'energies_per_site': [],
        'double_occupancy': [],
        'spin_correlations': [],
    }

    for U in U_values:
        print(f"\n{'='*50}")
        print(f"ED: {Lx}x{Ly} lattice, U/t = {U/t:.1f}")
        print(f"{'='*50}")

        ed = ExactDiagonalization(lattice, t=t, U=U, n_up=n_up, n_down=n_down)
        energies, states = ed.solve(n_states=1)

        E0 = energies[0]
        psi0 = states[:, 0]
        D = ed.compute_double_occupancy(psi0)
        spin_corr = ed.compute_spin_correlation(psi0)

        results['energies'].append(E0)
        results['energies_per_site'].append(E0 / N)
        results['double_occupancy'].append(D)
        results['spin_correlations'].append(spin_corr)

        print(f"E0 = {E0:.10f}")
        print(f"E0/N = {E0/N:.10f}")
        print(f"<D> = {D:.6f}")

    return results
