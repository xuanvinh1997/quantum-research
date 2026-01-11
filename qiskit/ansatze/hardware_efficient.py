"""Hardware-efficient ansatz for VQA using Qiskit."""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
from typing import List


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz with rotation and entangling layers.

    Structure:
    - Rotation layer: RY gates on all qubits
    - Entangling layer: CNOT gates between neighbors
    - Repeated for specified depth
    """

    def __init__(self, num_qubits: int, depth: int = 2):
        """
        Initialize hardware-efficient ansatz.

        Args:
            num_qubits: Number of qubits
            depth: Number of repetitions of rotation + entangling layers
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_parameters = self.calculate_num_parameters()

    def calculate_num_parameters(self) -> int:
        """
        Calculate total number of parameters.

        Returns:
            Number of parameters needed
        """
        # Each layer has num_qubits rotation parameters
        # depth layers of rotations
        return self.num_qubits * self.depth

    def build_kernel(self) -> QuantumCircuit:
        """
        Build quantum circuit for the ansatz.

        Returns:
            QuantumCircuit with parameters
        """
        # Create parameter vector
        params = ParameterVector('θ', self.num_parameters)

        # Create circuit
        circuit = QuantumCircuit(self.num_qubits)

        param_idx = 0

        for layer in range(self.depth):
            # Rotation layer: RY on each qubit
            for i in range(self.num_qubits):
                circuit.ry(params[param_idx], i)
                param_idx += 1

            # Entangling layer: CNOT between neighbors
            if layer < self.depth:  # Apply entangling on all layers
                for i in range(self.num_qubits - 1):
                    circuit.cx(i, i + 1)

        return circuit

    def initial_parameters(self, seed: int = 42) -> np.ndarray:
        """
        Generate random initial parameters.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Random parameters in [-π, π]
        """
        np.random.seed(seed)
        return np.random.uniform(-np.pi, np.pi, self.num_parameters)

    def __str__(self) -> str:
        """String representation."""
        return (f"HardwareEfficientAnsatz(num_qubits={self.num_qubits}, "
                f"depth={self.depth}, num_parameters={self.num_parameters})")


def create_ising_ansatz(num_qubits: int, depth: int = 2):
    """
    Factory function to create ansatz builder for Ising model.

    Args:
        num_qubits: Number of qubits
        depth: Ansatz depth

    Returns:
        Function that builds parameterized circuit
    """
    ansatz = HardwareEfficientAnsatz(num_qubits, depth)

    def ansatz_builder(n_qubits: int, parameters: np.ndarray):
        """Build ansatz circuit with given parameters."""
        return ansatz.build_kernel()

    return ansatz_builder, ansatz.num_parameters, ansatz.initial_parameters()
