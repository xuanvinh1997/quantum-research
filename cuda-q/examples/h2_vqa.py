"""Example script for VQA on H2 molecule."""
import cudaq
import numpy as np
import sys
sys.path.append('..')

from models.h2_molecule import H2Hamiltonian
from ansatze.uccsd import SimplifiedH2Ansatz, UCCSDAnsatz
from vqa.base_vqa import VQA


def run_h2_vqa(
    bond_distance: float = 0.74,
    ansatz_type: str = 'simplified',
    method: str = 'COBYLA',
    max_iterations: int = 200
):
    """
    Run VQA for H2 molecule ground state.

    Args:
        bond_distance: H-H bond distance in Angstroms
        ansatz_type: Type of ansatz ('simplified' or 'uccsd')
        method: Optimization method
        max_iterations: Maximum iterations

    Returns:
        Tuple of (optimal_energy, optimal_parameters)
    """
    print("=" * 60)
    print("VQA for H2 Molecule Ground State")
    print("=" * 60)

    # Build H2 Hamiltonian
    print(f"\nBuilding H2 Hamiltonian...")
    h2 = H2Hamiltonian(bond_distance=bond_distance, use_pyscf=False)
    print(h2)

    # Get exact energy for comparison
    exact_energy = h2.exact_ground_state_energy()
    print(f"\nExact ground state energy: {exact_energy:.8f} Hartree")

    # Build ansatz
    print(f"\nBuilding {ansatz_type} ansatz...")
    if ansatz_type.lower() == 'uccsd':
        ansatz = UCCSDAnsatz(num_qubits=4, num_electrons=2)
    else:
        ansatz = SimplifiedH2Ansatz()
    print(ansatz)

    # Initialize VQA
    print(f"\nInitializing VQA...")
    vqa = VQA(
        num_qubits=4,
        hamiltonian=h2.get_hamiltonian(),
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
    print(f"VQA ground state energy: {optimal_energy:.8f} Hartree")
    print(f"Exact ground state energy: {exact_energy:.8f} Hartree")

    error = abs(optimal_energy - exact_energy)
    print(f"Absolute error: {error:.8f} Hartree")
    print(f"Error (mH): {error * 1000:.4f} mH")

    # Convert to more familiar units
    hartree_to_ev = 27.211386245988
    print(f"\nEnergy in eV: {optimal_energy * hartree_to_ev:.6f} eV")
    print(f"Error in eV: {error * hartree_to_ev:.6f} eV")

    print(f"\nOptimal parameters:")
    print(optimal_params)

    return optimal_energy, optimal_params


def h2_dissociation_curve(
    bond_distances: list = None,
    ansatz_type: str = 'simplified',
    method: str = 'COBYLA'
):
    """
    Compute H2 potential energy surface (dissociation curve).

    Args:
        bond_distances: List of bond distances to evaluate
        ansatz_type: Type of ansatz
        method: Optimization method

    Returns:
        Tuple of (distances, vqa_energies, exact_energies)
    """
    if bond_distances is None:
        bond_distances = np.linspace(0.5, 2.0, 10)

    print("=" * 60)
    print("H2 Dissociation Curve")
    print("=" * 60)
    print(f"Computing VQA energies at {len(bond_distances)} bond distances")
    print(f"Range: {bond_distances[0]:.2f} - {bond_distances[-1]:.2f} Angstrom")

    vqa_energies = []
    exact_energies = []

    for i, distance in enumerate(bond_distances):
        print(f"\n{'=' * 60}")
        print(f"Bond distance {i+1}/{len(bond_distances)}: {distance:.3f} Å")
        print(f"{'=' * 60}")

        energy, params = run_h2_vqa(
            bond_distance=distance,
            ansatz_type=ansatz_type,
            method=method,
            max_iterations=150
        )

        vqa_energies.append(energy)

        # Get exact energy
        h2 = H2Hamiltonian(bond_distance=distance, use_pyscf=False)
        exact_energies.append(h2.exact_ground_state_energy())

    # Print summary
    print(f"\n{'=' * 60}")
    print("DISSOCIATION CURVE SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Distance (Å)':<15} {'VQA Energy':<15} {'Exact Energy':<15} {'Error (mH)':<15}")
    print("-" * 60)

    for d, vqa_e, exact_e in zip(bond_distances, vqa_energies, exact_energies):
        error = abs(vqa_e - exact_e) * 1000
        print(f"{d:<15.3f} {vqa_e:<15.8f} {exact_e:<15.8f} {error:<15.4f}")

    return bond_distances, vqa_energies, exact_energies


if __name__ == "__main__":
    # Example 1: Single point calculation at equilibrium
    print("\n\nExample 1: H2 at equilibrium bond distance")
    print("-" * 60)
    energy, params = run_h2_vqa(
        bond_distance=0.74,
        ansatz_type='simplified',
        method='COBYLA',
        max_iterations=200
    )

    # Example 2: Dissociation curve
    print("\n\n" + "=" * 60)
    print("\nExample 2: H2 Dissociation Curve")
    print("-" * 60)
    distances = [0.5, 0.74, 1.0, 1.5, 2.0]
    bond_distances, vqa_energies, exact_energies = h2_dissociation_curve(
        bond_distances=distances,
        ansatz_type='simplified',
        method='COBYLA'
    )
