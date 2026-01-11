"""Ising model Hamiltonian implementation."""
try:
    import cudaq
except ImportError:
    import sys
    sys.path.insert(0, '..')
    import cudaq_mock as cudaq

import numpy as np
from typing import Optional


class IsingHamiltonian:
    """
    Transverse Field Ising Model Hamiltonian.

    H = -J * Σᵢ ZᵢZᵢ₊₁ - h * Σᵢ Xᵢ

    Where:
    - J: Coupling strength between neighboring spins
    - h: Transverse field strength
    """

    def __init__(
        self,
        num_qubits: int,
        J: float = 1.0,
        h: float = 0.5,
        periodic: bool = True
    ):
        """
        Initialize Ising Hamiltonian.

        Args:
            num_qubits: Number of qubits (spins)
            J: Coupling constant (positive = ferromagnetic, negative = antiferromagnetic)
            h: Transverse field strength
            periodic: Whether to use periodic boundary conditions
        """
        self.num_qubits = num_qubits
        self.J = J
        self.h = h
        self.periodic = periodic

        self.hamiltonian = self.build_hamiltonian()

    def build_hamiltonian(self) -> cudaq.SpinOperator:
        """
        Build the Ising Hamiltonian as a CUDA-Q SpinOperator.

        Returns:
            CUDA-Q SpinOperator representing the Hamiltonian
        """
        # Start with zero operator
        H = 0.0 * cudaq.spin.i(0)

        # ZZ interaction terms: -J * Σᵢ ZᵢZᵢ₊₁
        num_interactions = self.num_qubits if self.periodic else self.num_qubits - 1

        for i in range(num_interactions):
            j = (i + 1) % self.num_qubits
            # Add -J * Z_i * Z_j term
            H += -self.J * cudaq.spin.z(i) * cudaq.spin.z(j)

        # Transverse field terms: -h * Σᵢ Xᵢ
        for i in range(self.num_qubits):
            H += -self.h * cudaq.spin.x(i)

        return H

    def get_hamiltonian(self) -> cudaq.SpinOperator:
        """Get the Hamiltonian SpinOperator."""
        return self.hamiltonian

    def classical_ground_state_energy(self) -> float:
        """
        Compute classical ground state energy (all spins aligned).

        For ferromagnetic J > 0 and small h, the ground state
        has all spins aligned (|000...0> or |111...1>).

        Returns:
            Classical ground state energy approximation
        """
        # Energy from ZZ terms (all aligned)
        num_interactions = self.num_qubits if self.periodic else self.num_qubits - 1
        zz_energy = -self.J * num_interactions

        # Energy from X terms (expectation value in |0> state is 0)
        x_energy = 0.0

        return zz_energy + x_energy

    def exact_diagonalization_energy(self) -> float:
        """
        Compute exact ground state energy via diagonalization.

        Note: Only feasible for small systems (< 20 qubits).

        Returns:
            Exact ground state energy
        """
        # Build Hamiltonian matrix
        dim = 2 ** self.num_qubits
        H_matrix = np.zeros((dim, dim), dtype=complex)

        # ZZ terms
        num_interactions = self.num_qubits if self.periodic else self.num_qubits - 1
        for i in range(num_interactions):
            j = (i + 1) % self.num_qubits
            for state in range(dim):
                # Check if spins i and j are aligned
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                # Z eigenvalue is +1 for |0>, -1 for |1>
                z_i = 1 - 2 * bit_i
                z_j = 1 - 2 * bit_j
                H_matrix[state, state] += -self.J * z_i * z_j

        # X terms (flip bit i)
        for i in range(self.num_qubits):
            for state in range(dim):
                flipped_state = state ^ (1 << i)
                H_matrix[flipped_state, state] += -self.h

        # Diagonalize
        eigenvalues = np.linalg.eigvalsh(H_matrix)

        return eigenvalues[0]

    def __str__(self) -> str:
        """String representation."""
        return (f"IsingHamiltonian(num_qubits={self.num_qubits}, "
                f"J={self.J}, h={self.h}, periodic={self.periodic})")
