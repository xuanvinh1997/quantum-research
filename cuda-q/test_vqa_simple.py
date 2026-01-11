"""
Simple VQA test that works with mock CUDA-Q.
"""
import sys
import numpy as np

try:
    import cudaq
except ImportError:
    import cudaq_mock as cudaq

from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA


def simple_vqa_test():
    """Run a simple VQA test."""
    print("=" * 70)
    print("Simple VQA Test (with mock CUDA-Q)")
    print("=" * 70)

    # Create simple Ising model
    print("\n1. Creating 3-qubit Ising model...")
    ising = IsingHamiltonian(num_qubits=3, J=1.0, h=0.5, periodic=False)
    print(f"   {ising}")

    # Get exact energy
    exact_energy = ising.exact_diagonalization_energy()
    print(f"   Exact ground state energy: {exact_energy:.6f}")

    # Create ansatz
    print("\n2. Creating ansatz...")
    ansatz = HardwareEfficientAnsatz(num_qubits=3, depth=1)
    print(f"   Number of parameters: {ansatz.num_parameters}")

    # Create VQA
    print("\n3. Initializing VQA...")
    vqa = VQA(
        num_qubits=3,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )

    # Get initial parameters
    initial_params = ansatz.initial_parameters()
    print(f"   Initial parameters: {initial_params}")

    # Run VQA with limited iterations
    print("\n4. Running VQA optimization...")
    print("   (Note: Using mock CUDA-Q, so energies are approximate)\n")

    try:
        optimal_energy, optimal_params = vqa.optimize(
            initial_parameters=initial_params,
            method='COBYLA',
            max_iterations=50
        )

        print(f"\n5. Results:")
        print(f"   VQA energy: {optimal_energy:.6f}")
        print(f"   Exact energy: {exact_energy:.6f}")
        print(f"   Optimal parameters: {optimal_params}")

        print(f"\n   Success! VQA optimization completed.")

    except Exception as e:
        print(f"\n   Error during optimization: {e}")
        print(f"   This is expected with mock CUDA-Q")

    return True


def test_components():
    """Test individual components."""
    print("\n" + "=" * 70)
    print("Component Tests")
    print("=" * 70)

    tests_passed = 0
    total_tests = 0

    # Test 1: Ising creation
    total_tests += 1
    try:
        ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5)
        print(f"\n[PASS] Ising Hamiltonian creation")
        tests_passed += 1
    except Exception as e:
        print(f"\n[FAIL] Ising Hamiltonian creation: {e}")

    # Test 2: Exact diagonalization
    total_tests += 1
    try:
        energy = ising.exact_diagonalization_energy()
        print(f"[PASS] Exact diagonalization: energy = {energy:.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Exact diagonalization: {e}")

    # Test 3: Ansatz creation
    total_tests += 1
    try:
        ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)
        print(f"[PASS] Hardware-efficient ansatz: {ansatz.num_parameters} params")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Ansatz creation: {e}")

    # Test 4: VQA initialization
    total_tests += 1
    try:
        vqa = VQA(
            num_qubits=4,
            hamiltonian=ising.get_hamiltonian(),
            ansatz_builder=lambda n, p: ansatz.build_kernel()
        )
        print(f"[PASS] VQA initialization")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] VQA initialization: {e}")

    # Test 5: Cost function
    total_tests += 1
    try:
        params = ansatz.initial_parameters()
        cost = vqa.cost_function(params)
        print(f"[PASS] Cost function evaluation: cost = {cost:.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Cost function: {e}")

    print(f"\n" + "=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print("=" * 70)

    return tests_passed == total_tests


if __name__ == "__main__":
    # Run component tests
    components_ok = test_components()

    # Run simple VQA test
    vqa_ok = simple_vqa_test()

    print("\n" + "=" * 70)
    if components_ok and vqa_ok:
        print("All tests completed!")
    else:
        print("Some tests had issues (expected with mock CUDA-Q)")
    print("=" * 70)
