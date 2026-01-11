"""UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz."""
try:
    import cudaq
except ImportError:
    import sys
    sys.path.insert(0, '..')
    import cudaq_mock as cudaq

import numpy as np
from typing import List


class UCCSDAnsatz:
    """
    Simplified UCCSD ansatz for H2 molecule.

    For H2, we use a simplified version with single excitations.
    The full UCCSD would include singles and doubles excitations.
    """

    def __init__(self, num_qubits: int = 4, num_electrons: int = 2):
        """
        Initialize UCCSD ansatz.

        Args:
            num_qubits: Number of qubits (4 for H2)
            num_electrons: Number of electrons (2 for H2)
        """
        self.num_qubits = num_qubits
        self.num_electrons = num_electrons
        self.num_parameters = self.calculate_num_parameters()

    def calculate_num_parameters(self) -> int:
        """
        Calculate number of parameters for UCCSD.

        For H2 with 2 electrons and 2 orbitals (4 qubits):
        - Singles: transitions from occupied to virtual orbitals
        - Doubles: pair excitations

        Simplified version uses 1 parameter for the key excitation.

        Returns:
            Number of parameters
        """
        # Simplified: 1 parameter for dominant single excitation
        # Full UCCSD would have more
        return 1

    def build_kernel(self) -> callable:
        """
        Build CUDA-Q kernel for UCCSD ansatz.

        Returns:
            CUDA-Q kernel function
        """
        num_qubits = self.num_qubits

        @cudaq.kernel
        def kernel(theta: List[float]):
            """UCCSD ansatz kernel for H2."""
            qubits = cudaq.qvector(num_qubits)

            # Hartree-Fock initial state: |1100> (2 electrons in lowest orbitals)
            x(qubits[0])
            x(qubits[1])

            # Single excitation: rotate between occupied and virtual orbitals
            # This implements exp(theta * (a†_2 a_0 - a†_0 a_2))
            # Simplified using Givens rotation

            # Excitation from orbital 0 to orbital 2
            h(qubits[2])
            cx(qubits[2], qubits[0])
            ry(theta[0], qubits[2])
            cx(qubits[2], qubits[0])
            h(qubits[2])

        return kernel

    def initial_parameters(self, seed: int = 42) -> np.ndarray:
        """
        Generate initial parameters.

        Args:
            seed: Random seed

        Returns:
            Initial parameters (small random values)
        """
        np.random.seed(seed)
        # Start near HF solution with small perturbation
        return np.random.uniform(-0.1, 0.1, self.num_parameters)

    def __str__(self) -> str:
        """String representation."""
        return (f"UCCSDAnsatz(num_qubits={self.num_qubits}, "
                f"num_electrons={self.num_electrons}, "
                f"num_parameters={self.num_parameters})")


class SimplifiedH2Ansatz:
    """
    Simplified ansatz specifically designed for H2.

    Uses a single parameter to capture the dominant correlation.
    """

    def __init__(self):
        """Initialize simplified H2 ansatz."""
        self.num_qubits = 4
        self.num_parameters = 1

    def build_kernel(self) -> callable:
        """
        Build CUDA-Q kernel.

        Returns:
            CUDA-Q kernel function
        """
        @cudaq.kernel
        def kernel(theta: List[float]):
            """Simplified H2 ansatz."""
            qubits = cudaq.qvector(4)

            # Hartree-Fock state: |1100>
            x(qubits[0])
            x(qubits[1])

            # Parameterized rotation to capture correlation
            ry(theta[0], qubits[0])
            ry(-theta[0], qubits[2])

            # Entangling gates
            cx(qubits[0], qubits[2])

        return kernel

    def initial_parameters(self, seed: int = 42) -> np.ndarray:
        """Generate initial parameters."""
        np.random.seed(seed)
        return np.array([0.1])

    def __str__(self) -> str:
        """String representation."""
        return f"SimplifiedH2Ansatz(num_parameters={self.num_parameters})"


def create_h2_ansatz(ansatz_type: str = 'simplified'):
    """
    Factory function to create H2 ansatz.

    Args:
        ansatz_type: Type of ansatz ('uccsd' or 'simplified')

    Returns:
        Tuple of (ansatz_builder, num_parameters, initial_parameters)
    """
    if ansatz_type.lower() == 'uccsd':
        ansatz = UCCSDAnsatz()
    else:
        ansatz = SimplifiedH2Ansatz()

    def ansatz_builder(n_qubits: int, parameters: np.ndarray):
        """Build ansatz kernel with given parameters."""
        return ansatz.build_kernel()

    return ansatz_builder, ansatz.num_parameters, ansatz.initial_parameters()
