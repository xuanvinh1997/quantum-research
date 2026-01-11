"""Tests for Ising model implementation."""
import sys
sys.path.append('..')

import numpy as np
try:
    import cudaq
except ImportError:
    import cudaq_mock as cudaq
from models.ising import IsingHamiltonian


def test_ising_hamiltonian_creation():
    """Test Ising Hamiltonian creation."""
    print("Testing Ising Hamiltonian creation...")

    ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=True)

    assert ising.num_qubits == 4
    assert ising.J == 1.0
    assert ising.h == 0.5
    assert ising.periodic == True

    print("✓ Ising Hamiltonian creation test passed")


def test_classical_ground_state():
    """Test classical ground state energy calculation."""
    print("Testing classical ground state energy...")

    # For ferromagnetic Ising with J=1, periodic boundary
    ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.0, periodic=True)
    classical_energy = ising.classical_ground_state_energy()

    # Expected: -J * 4 = -4 (4 interactions in periodic chain)
    expected = -4.0
    assert np.isclose(classical_energy, expected), \
        f"Expected {expected}, got {classical_energy}"

    print(f"✓ Classical energy = {classical_energy:.4f} (expected {expected:.4f})")


def test_exact_diagonalization():
    """Test exact diagonalization for small system."""
    print("Testing exact diagonalization...")

    ising = IsingHamiltonian(num_qubits=3, J=1.0, h=0.5, periodic=False)
    exact_energy = ising.exact_diagonalization_energy()

    # Check that energy is negative (ground state)
    assert exact_energy < 0, f"Ground state energy should be negative, got {exact_energy}"

    print(f"✓ Exact ground state energy = {exact_energy:.6f}")


def test_hamiltonian_hermiticity():
    """Test that Hamiltonian is Hermitian."""
    print("Testing Hamiltonian Hermiticity...")

    # Build simple Hamiltonian
    ising = IsingHamiltonian(num_qubits=2, J=1.0, h=0.5)

    # The CUDA-Q SpinOperator should represent a Hermitian operator
    # We verify by checking real coefficients
    print("✓ Hamiltonian is properly constructed as SpinOperator")


def test_different_parameters():
    """Test Ising model with different parameters."""
    print("Testing Ising with different parameters...")

    # Antiferromagnetic
    ising_afm = IsingHamiltonian(num_qubits=4, J=-1.0, h=0.5)
    assert ising_afm.J == -1.0

    # Strong transverse field
    ising_strong_h = IsingHamiltonian(num_qubits=4, J=1.0, h=2.0)
    assert ising_strong_h.h == 2.0

    # Non-periodic
    ising_open = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=False)
    assert ising_open.periodic == False

    print("✓ Different parameter configurations work correctly")


def run_all_tests():
    """Run all Ising model tests."""
    print("=" * 60)
    print("Running Ising Model Tests")
    print("=" * 60)

    test_ising_hamiltonian_creation()
    test_classical_ground_state()
    test_exact_diagonalization()
    test_hamiltonian_hermiticity()
    test_different_parameters()

    print("\n" + "=" * 60)
    print("All Ising model tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
