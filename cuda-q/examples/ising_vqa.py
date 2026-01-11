"""Example script for VQA on Ising model."""
import cudaq
import numpy as np
import sys
sys.path.append('..')

from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA


def run_ising_vqa(
    num_qubits: int = 4,
    J: float = 1.0,
    h: float = 0.5,
    depth: int = 2,
    periodic: bool = True,
    method: str = 'COBYLA',
    max_iterations: int = 200
):
    """
    Run VQA for Ising model ground state.

    Args:
        num_qubits: Number of qubits (spins)
        J: Coupling constant
        h: Transverse field strength
        depth: Ansatz depth
        periodic: Use periodic boundary conditions
        method: Optimization method
        max_iterations: Maximum iterations

    Returns:
        Tuple of (optimal_energy, optimal_parameters)
    """
    print("=" * 60)
    print("VQA for Transverse Field Ising Model")
    print("=" * 60)

    # Build Ising Hamiltonian
    print(f"\nBuilding Ising Hamiltonian...")
    ising = IsingHamiltonian(num_qubits, J=J, h=h, periodic=periodic)
    print(ising)

    # Compute exact ground state energy for comparison (if feasible)
    if num_qubits <= 12:
        exact_energy = ising.exact_diagonalization_energy()
        print(f"\nExact ground state energy: {exact_energy:.8f}")
    else:
        exact_energy = None
        print(f"\nExact diagonalization not feasible for {num_qubits} qubits")

    # Classical approximation
    classical_energy = ising.classical_ground_state_energy()
    print(f"Classical ground state energy: {classical_energy:.8f}")

    # Build ansatz
    print(f"\nBuilding hardware-efficient ansatz (depth={depth})...")
    ansatz = HardwareEfficientAnsatz(num_qubits, depth)
    print(ansatz)

    # Initialize VQA
    print(f"\nInitializing VQA...")
    vqa = VQA(
        num_qubits=num_qubits,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    # Get initial parameters
    initial_params = ansatz.initial_parameters()
    print(f"Number of parameters: {len(initial_params)}")

    # Run optimization
    print(f"\n{'=' * 60}")
    optimal_energy, optimal_params = vqa.optimize(
        initial_parameters=initial_params,
        method=method,
        max_iterations=max_iterations
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"VQA ground state energy: {optimal_energy:.8f}")
    if exact_energy is not None:
        error = abs(optimal_energy - exact_energy)
        print(f"Exact ground state energy: {exact_energy:.8f}")
        print(f"Absolute error: {error:.8f}")
        print(f"Relative error: {error / abs(exact_energy) * 100:.4f}%")

    print(f"\nOptimal parameters:")
    print(optimal_params)

    return optimal_energy, optimal_params


if __name__ == "__main__":
    # Example 1: Small system (4 qubits)
    print("\n\nExample 1: 4-qubit Ising chain")
    print("-" * 60)
    energy, params = run_ising_vqa(
        num_qubits=4,
        J=1.0,
        h=0.5,
        depth=2,
        periodic=True,
        method='COBYLA',
        max_iterations=200
    )

    # Example 2: Larger system (6 qubits)
    print("\n\n" + "=" * 60)
    print("\nExample 2: 6-qubit Ising chain")
    print("-" * 60)
    energy, params = run_ising_vqa(
        num_qubits=6,
        J=1.0,
        h=0.3,
        depth=3,
        periodic=True,
        method='COBYLA',
        max_iterations=300
    )
