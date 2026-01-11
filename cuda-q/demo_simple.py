"""
Simple demo of VQA framework.
Works with or without CUDA-Q installed (uses mock if needed).
"""
import sys
import numpy as np

# Import with mock fallback
try:
    import cudaq
    using_mock = False
except ImportError:
    import cudaq_mock as cudaq
    using_mock = True

from models.ising import IsingHamiltonian
from models.h2_molecule import H2Hamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from ansatze.uccsd import SimplifiedH2Ansatz


def demo_ising_model():
    """Demo Ising model setup and exact diagonalization."""
    print("\n" + "=" * 70)
    print("DEMO 1: Ising Model Ground State")
    print("=" * 70)

    # Create 4-qubit Ising model
    print("\n1. Creating Transverse Field Ising Model...")
    ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=True)
    print(f"   {ising}")

    # Compute exact ground state
    print("\n2. Computing exact ground state energy...")
    exact_energy = ising.exact_diagonalization_energy()
    print(f"   Exact energy: {exact_energy:.8f}")

    # Classical approximation
    classical_energy = ising.classical_ground_state_energy()
    print(f"   Classical energy (all spins aligned): {classical_energy:.8f}")

    # Create ansatz
    print("\n3. Creating hardware-efficient ansatz...")
    ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)
    print(f"   Ansatz depth: 2")
    print(f"   Number of parameters: {ansatz.num_parameters}")

    # Show what VQA would do (without actually running optimization)
    print("\n4. VQA Workflow:")
    print("   a. Initialize random parameters")
    print("   b. Build parameterized quantum circuit")
    print("   c. Measure energy expectation value")
    print("   d. Classical optimizer adjusts parameters")
    print("   e. Repeat until convergence")

    print(f"\n   Target: Find energy close to {exact_energy:.8f}")

    return exact_energy


def demo_h2_molecule():
    """Demo H2 molecule setup."""
    print("\n" + "=" * 70)
    print("DEMO 2: H2 Molecule Ground State")
    print("=" * 70)

    # Create H2 molecule
    print("\n1. Creating H2 molecule Hamiltonian...")
    bond_distance = 0.74  # Angstroms (equilibrium)
    h2 = H2Hamiltonian(bond_distance=bond_distance, use_pyscf=False)
    print(f"   {h2}")
    print(f"   Bond distance: {bond_distance} Angstrom (equilibrium)")

    # Get exact energy
    print("\n2. Computing exact ground state energy...")
    exact_energy = h2.exact_ground_state_energy()
    print(f"   Exact energy: {exact_energy:.8f} Hartree")
    print(f"   Exact energy: {exact_energy * 27.211386:.8f} eV")

    # Create ansatz
    print("\n3. Creating simplified ansatz for H2...")
    ansatz = SimplifiedH2Ansatz()
    print(f"   Number of qubits: 4 (2 orbitals x 2 spins)")
    print(f"   Number of parameters: {ansatz.num_parameters}")

    # Show Hamiltonian structure
    print("\n4. H2 Hamiltonian structure:")
    print("   Pauli terms: II, IZ, ZI, ZZ, XX")
    print("   Mapped from electronic structure problem")

    print(f"\n   Target: Find energy close to {exact_energy:.8f} Hartree")

    return exact_energy


def demo_optimization_concepts():
    """Demo optimization concepts."""
    print("\n" + "=" * 70)
    print("DEMO 3: VQA Optimization Concepts")
    print("=" * 70)

    print("\n1. Gradient-Free Optimization (COBYLA, Nelder-Mead)")
    print("   - No gradient computation needed")
    print("   - Robust for noisy cost functions")
    print("   - Good default choice")

    print("\n2. Gradient-Based Optimization (Parameter Shift Rule)")
    print("   - Exact gradients via quantum circuit shifts")
    print("   - Potentially better convergence")
    print("   - 2N circuit evaluations per gradient (N parameters)")

    print("\n3. Adaptive Optimization")
    print("   - Try multiple methods automatically")
    print("   - Return best result")
    print("   - Useful when one method fails")

    print("\n4. Cost Function:")
    print("   E(theta) = <psi(theta)|H|psi(theta)>")
    print("   - Expectation value of Hamiltonian")
    print("   - Minimize to find ground state")


def demo_parameter_landscape():
    """Demo parameter landscape concept."""
    print("\n" + "=" * 70)
    print("DEMO 4: Energy Landscape")
    print("=" * 70)

    print("\n1. Parameter Space:")
    print("   - Each parameter controls a rotation angle")
    print("   - Energy landscape can be complex (local minima)")
    print("   - Good initialization is important")

    print("\n2. Convergence:")
    print("   - Iteration 0: Random initialization")
    print("   - Iterations 1-N: Optimizer explores landscape")
    print("   - Convergence: Energy stops improving")

    print("\n3. Monitoring:")
    print("   - Plot energy vs iteration")
    print("   - Check gradient norm (for gradient-based)")
    print("   - Compare with exact energy")


def main():
    """Run all demos."""
    print("=" * 70)
    print("VQA Framework Demo")
    print("=" * 70)

    if using_mock:
        print("\nNOTE: Using mock CUDA-Q (actual quantum simulation unavailable)")
        print("Install cudaq package for real quantum circuit simulation")
    else:
        print("\nUsing real CUDA-Q for quantum simulation")

    # Run demos
    ising_energy = demo_ising_model()
    h2_energy = demo_h2_molecule()
    demo_optimization_concepts()
    demo_parameter_landscape()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nIsing Model (4 qubits, J=1.0, h=0.5):")
    print(f"  Exact ground state: {ising_energy:.8f}")
    print(f"\nH2 Molecule (bond distance 0.74 A):")
    print(f"  Exact ground state: {h2_energy:.8f} Hartree")

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. To run actual VQA optimization:")
    print("   python run_examples.py ising")
    print("   python run_examples.py h2")

    print("\n2. To run with different parameters:")
    print("   python run_examples.py ising --qubits 6 --J 1.0 --h 0.3")

    print("\n3. To visualize results:")
    print("   See examples in examples/visualize.py")

    print("\n4. For full documentation:")
    print("   See GETTING_STARTED.md and USAGE_GUIDE.md")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
