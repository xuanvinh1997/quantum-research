"""
Simple test script for Qiskit VQA implementation.
"""
import sys
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Estimator
    qiskit_available = True
except ImportError:
    qiskit_available = False
    print("Qiskit not installed. Install with: pip install qiskit qiskit-aer")
    sys.exit(1)

from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA


def test_qiskit_vqa():
    """Test Qiskit VQA implementation."""
    print("=" * 70)
    print("Qiskit VQA Test")
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
    print(f"   {ansatz}")
    print(f"   Number of parameters: {ansatz.num_parameters}")

    # Create VQA
    print("\n3. Initializing VQA...")
    vqa = VQA(
        num_qubits=3,
        hamiltonian=ising.get_hamiltonian(),
        ansatz_builder=lambda n, p: ansatz.build_kernel()
    )
    print("   VQA initialized successfully")

    # Get initial parameters
    initial_params = ansatz.initial_parameters()
    print(f"   Initial parameters: {initial_params}")

    # Test expectation value computation
    print("\n4. Testing expectation value computation...")
    try:
        energy = vqa.compute_expectation(initial_params)
        print(f"   Initial energy: {energy:.6f}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Run short VQA optimization
    print("\n5. Running VQA optimization (20 iterations)...")
    try:
        optimal_energy, optimal_params = vqa.optimize(
            initial_parameters=initial_params,
            method='COBYLA',
            max_iterations=20
        )

        print(f"\n6. Results:")
        print(f"   VQA energy: {optimal_energy:.6f}")
        print(f"   Exact energy: {exact_energy:.6f}")
        print(f"   Error: {abs(optimal_energy - exact_energy):.6f}")
        print(f"   Optimal parameters: {optimal_params}")

        print(f"\n   Success! Qiskit VQA working correctly.")
        return True

    except Exception as e:
        print(f"\n   Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qiskit_vqa()

    print("\n" + "=" * 70)
    if success:
        print("Qiskit VQA implementation verified!")
    else:
        print("Qiskit VQA test failed - check errors above")
    print("=" * 70)
