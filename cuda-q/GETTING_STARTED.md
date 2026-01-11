# Getting Started with VQA

Quick start guide to get you running VQA simulations in minutes!

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- pip package manager
- (Optional) NVIDIA GPU with CUDA for acceleration

## Installation (5 minutes)

### Step 1: Clone/Download the Repository

If you're reading this, you likely already have the code. If not:
```bash
cd quantum-research
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- CUDA-Q (quantum simulation)
- NumPy (numerical computing)
- SciPy (optimization)
- Matplotlib (visualization)
- PySCF (quantum chemistry, optional)

### Step 3: Verify Installation

```bash
python -c "import cudaq; print('CUDA-Q version:', cudaq.__version__)"
```

You should see the CUDA-Q version printed (e.g., `0.6.0` or higher).

## Your First VQA Simulation (2 minutes)

### Option 1: Command Line (Easiest)

Run the Ising model VQA:
```bash
python run_examples.py ising
```

You should see output like:
```
============================================================
VQA for Transverse Field Ising Model
============================================================

Building Ising Hamiltonian...
Exact ground state energy: -5.12345678

Starting VQA optimization...
Iteration 0: Energy = -4.98765432
Iteration 10: Energy = -5.10123456
...
Iteration 50: Energy = -5.12345123

Optimization complete!
Final energy: -5.12345123
```

### Option 2: Python Script

Create a file `my_first_vqa.py`:

```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA

# Create a 4-qubit Ising model
ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=True)
print(f"Created: {ising}")

# Get exact solution for comparison
exact_energy = ising.exact_diagonalization_energy()
print(f"Exact ground state: {exact_energy:.6f}")

# Create ansatz
ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)
print(f"Ansatz has {ansatz.num_parameters} parameters")

# Run VQA
vqa = VQA(
    num_qubits=4,
    hamiltonian=ising.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel()
)

energy, params = vqa.optimize(
    initial_parameters=ansatz.initial_parameters(),
    method='COBYLA',
    max_iterations=100
)

print(f"\nVQA found energy: {energy:.6f}")
print(f"Error: {abs(energy - exact_energy):.8f}")
```

Run it:
```bash
python my_first_vqa.py
```

## Next Steps

### 1. Try H2 Molecule

```bash
python run_examples.py h2
```

This computes the ground state energy of the hydrogen molecule.

### 2. Experiment with Parameters

```bash
# Larger system
python run_examples.py ising --qubits 6 --depth 3

# Different Ising parameters
python run_examples.py ising --J 1.0 --h 2.0

# H2 at different bond distances
python run_examples.py h2 --distance 1.0
```

### 3. Run Tests

Verify everything works:
```bash
python run_examples.py test
```

### 4. Explore Visualizations

Create a visualization script `visualize_results.py`:

```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA
from examples.visualize import plot_convergence, create_summary_plot

# Run VQA
ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5)
ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)

vqa = VQA(
    num_qubits=4,
    hamiltonian=ising.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel()
)

energy, params = vqa.optimize(
    initial_parameters=ansatz.initial_parameters(),
    method='COBYLA',
    max_iterations=200
)

# Get exact energy
exact_energy = ising.exact_diagonalization_energy()

# Visualize
energy_history, param_history = vqa.get_history()

plot_convergence(
    energy_history,
    exact_energy,
    title="My First VQA",
    save_path="my_convergence.png"
)

print("Plot saved to my_convergence.png")
```

## Common Use Cases

### Use Case 1: Find Ising Ground State

```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA

# Ferromagnetic Ising chain
ising = IsingHamiltonian(num_qubits=6, J=1.0, h=0.3)
ansatz = HardwareEfficientAnsatz(num_qubits=6, depth=3)

vqa = VQA(6, ising.get_hamiltonian(), lambda n,p: ansatz.build_kernel())
energy, _ = vqa.optimize(ansatz.initial_parameters(), max_iterations=300)

print(f"Ground state energy: {energy}")
```

### Use Case 2: H2 Dissociation Curve

```python
from examples.h2_vqa import h2_dissociation_curve
import numpy as np

# Compute energy at multiple bond distances
distances = [0.5, 0.74, 1.0, 1.5, 2.0]
results = h2_dissociation_curve(bond_distances=distances)
```

### Use Case 3: Compare Optimizers

```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA

ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5)
ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)
vqa = VQA(4, ising.get_hamiltonian(), lambda n,p: ansatz.build_kernel())

# Try different optimizers
for method in ['COBYLA', 'Nelder-Mead', 'Powell']:
    energy, _ = vqa.optimize(
        ansatz.initial_parameters(),
        method=method,
        max_iterations=200
    )
    print(f"{method}: {energy:.6f}")
```

## Understanding the Output

When you run VQA, you'll see:

```
Iteration 0: Energy = -4.987654
Iteration 10: Energy = -5.101234
...
Iteration 50: Energy = -5.123451

Optimization complete!
Final energy: -5.12345123
Total iterations: 52
```

- **Iteration N**: Progress update every 10 iterations
- **Energy**: Current estimate of ground state energy
- **Final energy**: Best energy found
- **Total iterations**: Number of optimization steps

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'cudaq'"

**Solution**: Install CUDA-Q
```bash
pip install cuda-quantum
```

### Problem: "Convergence is slow"

**Solutions**:
- Increase max_iterations: `max_iterations=500`
- Try different optimizer: `method='Nelder-Mead'`
- Increase ansatz depth: `depth=3`

### Problem: "Energy not accurate"

**Solutions**:
- Increase ansatz depth (more expressive circuit)
- Try different random initialization
- Use gradient-based optimizer for fine-tuning

### Problem: "Script runs slowly"

**Solutions**:
- Reduce max_iterations
- Use shallower ansatz
- Ensure GPU is being used (CUDA-Q defaults to GPU if available)

## Tips for Success

1. **Start Small**: Begin with 4 qubits, depth 2
2. **Verify**: Compare with exact solutions when possible
3. **Visualize**: Always plot convergence to understand behavior
4. **Experiment**: Try different parameters to build intuition
5. **Read Docs**: Check USAGE_GUIDE.md for advanced features

## Learning Path

### Beginner (Week 1)
1.  Run provided examples
2.  Modify parameters
3.  Visualize results
4.  Understand output

### Intermediate (Week 2-3)
1. Create custom Hamiltonians
2. Experiment with ansatz depths
3. Compare optimizers
4. Analyze convergence

### Advanced (Month 2)
1. Implement custom ansatze
2. Add new molecules
3. Optimize performance
4. Contribute improvements

## Resources

- **USAGE_GUIDE.md**: Comprehensive usage documentation
- **README.md**: Project overview and features
- **IMPLEMENTATION_SUMMARY.md**: Technical details
- **examples/**: Working code examples
- **tests/**: Test suite for verification

## Getting Help

1. Check documentation files
2. Run tests to verify installation
3. Review example scripts
4. Consult CUDA-Q documentation: https://nvidia.github.io/cuda-quantum/

## What's Next?

Now that you're set up, try:

1. **Explore Parameters**
   ```bash
   python run_examples.py ising --qubits 5 --J 2.0 --h 1.0
   ```

2. **Study Convergence**
   - Run with different depths
   - Compare optimizers
   - Analyze parameter evolution

3. **Advanced Topics**
   - Read USAGE_GUIDE.md section on custom ansatze
   - Implement gradient-based optimization
   - Create custom Hamiltonians

4. **Research**
   - Investigate quantum phase transitions in Ising model
   - Study H2 dissociation curves
   - Compare with classical methods

Happy quantum computing! 
