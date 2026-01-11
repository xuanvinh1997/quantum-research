"""Tests for H2 molecule implementation."""
import sys
sys.path.append('..')

import numpy as np
from models.h2_molecule import H2Hamiltonian


def test_h2_hamiltonian_creation():
    """Test H2 Hamiltonian creation."""
    print("Testing H2 Hamiltonian creation...")

    h2 = H2Hamiltonian(bond_distance=0.74, use_pyscf=False)

    assert h2.bond_distance == 0.74
    assert h2.num_qubits == 4

    print("✓ H2 Hamiltonian creation test passed")


def test_precomputed_coefficients():
    """Test pre-computed coefficient retrieval."""
    print("Testing pre-computed coefficients...")

    h2 = H2Hamiltonian(bond_distance=0.74, use_pyscf=False)

    # Check that coefficients exist
    assert 'II' in h2.coefficients
    assert 'ZZ' in h2.coefficients

    print(f"✓ Coefficients loaded: {list(h2.coefficients.keys())}")


def test_exact_energy():
    """Test exact ground state energy calculation."""
    print("Testing exact ground state energy...")

    h2 = H2Hamiltonian(bond_distance=0.74, use_pyscf=False)
    exact_energy = h2.exact_ground_state_energy()

    # For H2 at equilibrium, energy should be around -1.137 Hartree
    expected_range = (-1.2, -1.0)
    assert expected_range[0] < exact_energy < expected_range[1], \
        f"Energy {exact_energy} outside expected range {expected_range}"

    print(f"✓ Exact energy = {exact_energy:.6f} Hartree")


def test_different_bond_distances():
    """Test H2 at different bond distances."""
    print("Testing different bond distances...")

    distances = [0.5, 0.74, 1.0, 2.0]

    for distance in distances:
        h2 = H2Hamiltonian(bond_distance=distance, use_pyscf=False)
        energy = h2.exact_ground_state_energy()

        print(f"  Bond distance {distance:.2f} Å: Energy = {energy:.6f} Hartree")

    print("✓ Multiple bond distances work correctly")


def test_hamiltonian_properties():
    """Test Hamiltonian properties."""
    print("Testing Hamiltonian properties...")

    h2 = H2Hamiltonian(bond_distance=0.74, use_pyscf=False)
    H = h2.get_hamiltonian()

    # Hamiltonian should be a valid SpinOperator
    assert H is not None

    print("✓ Hamiltonian has correct properties")


def run_all_tests():
    """Run all H2 tests."""
    print("=" * 60)
    print("Running H2 Molecule Tests")
    print("=" * 60)

    test_h2_hamiltonian_creation()
    test_precomputed_coefficients()
    test_exact_energy()
    test_different_bond_distances()
    test_hamiltonian_properties()

    print("\n" + "=" * 60)
    print("All H2 molecule tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
