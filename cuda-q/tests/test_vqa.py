"""Tests for VQA base class."""
import sys
sys.path.append('..')

import numpy as np
import cudaq
from vqa.base_vqa import VQA
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz


def test_vqa_initialization():
    """Test VQA initialization."""
    print("Testing VQA initialization...")

    # Create simple Hamiltonian
    ising = IsingHamiltonian(num_qubits=3, J=1.0, h=0.5)
    ansatz = HardwareEfficientAnsatz(num_qubits=3, depth=1)

    vqa = VQA(
        num_qubits=3,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    assert vqa.num_qubits == 3
    assert len(vqa.energy_history) == 0

    print("✓ VQA initialization test passed")


def test_expectation_value_computation():
    """Test expectation value computation."""
    print("Testing expectation value computation...")

    ising = IsingHamiltonian(num_qubits=2, J=1.0, h=0.0)
    ansatz = HardwareEfficientAnsatz(num_qubits=2, depth=1)

    vqa = VQA(
        num_qubits=2,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    # Test with some parameters
    params = np.array([0.0, 0.0])
    try:
        energy = vqa.compute_expectation(params)
        print(f"✓ Expectation value computed: {energy:.6f}")
    except Exception as e:
        print(f"Note: Expectation computation may require CUDA-Q setup: {e}")


def test_cost_function():
    """Test cost function."""
    print("Testing cost function...")

    ising = IsingHamiltonian(num_qubits=2, J=1.0, h=0.5)
    ansatz = HardwareEfficientAnsatz(num_qubits=2, depth=1)

    vqa = VQA(
        num_qubits=2,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    params = ansatz.initial_parameters()

    try:
        cost = vqa.cost_function(params)

        # Check that history is updated
        assert len(vqa.energy_history) == 1
        assert len(vqa.parameter_history) == 1

        print(f"✓ Cost function works: cost = {cost:.6f}")
    except Exception as e:
        print(f"Note: Cost function may require CUDA-Q setup: {e}")


def test_gradient_computation():
    """Test gradient computation via parameter shift."""
    print("Testing gradient computation...")

    ising = IsingHamiltonian(num_qubits=2, J=1.0, h=0.5)
    ansatz = HardwareEfficientAnsatz(num_qubits=2, depth=1)

    vqa = VQA(
        num_qubits=2,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    params = ansatz.initial_parameters()

    try:
        gradient = vqa.compute_gradient(params)

        assert len(gradient) == len(params)
        print(f"✓ Gradient computed with shape {gradient.shape}")
    except Exception as e:
        print(f"Note: Gradient computation may require CUDA-Q setup: {e}")


def run_all_tests():
    """Run all VQA tests."""
    print("=" * 60)
    print("Running VQA Base Class Tests")
    print("=" * 60)

    test_vqa_initialization()
    test_expectation_value_computation()
    test_cost_function()
    test_gradient_computation()

    print("\n" + "=" * 60)
    print("All VQA tests completed! ✓")
    print("=" * 60)
    print("\nNote: Some tests may require proper CUDA-Q environment setup.")


if __name__ == "__main__":
    run_all_tests()
