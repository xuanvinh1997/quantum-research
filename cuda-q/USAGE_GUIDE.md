# VQA Implementation Guide

Comprehensive guide for using the VQA implementation with CUDA-Q for Ising model and H2 molecule.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Ising Model VQA](#ising-model-vqa)
4. [H2 Molecule VQA](#h2-molecule-vqa)
5. [Advanced Usage](#advanced-usage)
6. [Customization](#customization)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-Q (version 0.6.0 or later)
- NVIDIA GPU (recommended for performance)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install CUDA-Q

Follow the official CUDA-Q installation guide:
```bash
# For Linux
pip install cuda-quantum

# Verify installation
python -c "import cudaq; print(cudaq.__version__)"
```

---

## Quick Start

### Run Ising Model VQA

```bash
cd examples
python ising_vqa.py
```

### Run H2 Molecule VQA

```bash
cd examples
python h2_vqa.py
```

### Run Tests

```bash
cd tests
python run_all_tests.py
```

---

## Ising Model VQA

### Basic Usage

```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA

# Define system parameters
num_qubits = 4
J = 1.0  # Coupling strength (positive = ferromagnetic)
h = 0.5  # Transverse field strength

# Create Ising Hamiltonian
ising = IsingHamiltonian(num_qubits, J=J, h=h, periodic=True)

# Create ansatz
ansatz = HardwareEfficientAnsatz(num_qubits, depth=2)

# Initialize VQA
vqa = VQA(
    num_qubits=num_qubits,
    hamiltonian=ising.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel()
)

# Run optimization
initial_params = ansatz.initial_parameters()
energy, params = vqa.optimize(
    initial_parameters=initial_params,
    method='COBYLA',
    max_iterations=200
)

print(f"Ground state energy: {energy:.6f}")
```

### Understanding the Ising Model

The Transverse Field Ising Model is defined as:

**H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ**

Where:
- **J**: Coupling strength between neighboring spins
  - J > 0: Ferromagnetic (spins prefer to align)
  - J < 0: Antiferromagnetic (spins prefer opposite alignment)
- **h**: Transverse field strength
- **Zᵢ**: Pauli Z operator on qubit i
- **Xᵢ**: Pauli X operator on qubit i

### Parameter Tuning

```python
# Strong ferromagnetic coupling
ising_strong = IsingHamiltonian(num_qubits=4, J=2.0, h=0.1)

# Strong transverse field (quantum regime)
ising_quantum = IsingHamiltonian(num_qubits=4, J=1.0, h=2.0)

# Antiferromagnetic
ising_afm = IsingHamiltonian(num_qubits=4, J=-1.0, h=0.5)

# Open boundary conditions
ising_open = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=False)
```

---

## H2 Molecule VQA

### Basic Usage

```python
from models.h2_molecule import H2Hamiltonian
from ansatze.uccsd import SimplifiedH2Ansatz
from vqa.base_vqa import VQA

# Create H2 Hamiltonian at equilibrium distance
h2 = H2Hamiltonian(bond_distance=0.74)

# Create ansatz
ansatz = SimplifiedH2Ansatz()

# Initialize VQA
vqa = VQA(
    num_qubits=4,
    hamiltonian=h2.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel()
)

# Run optimization
initial_params = ansatz.initial_parameters()
energy, params = vqa.optimize(
    initial_parameters=initial_params,
    method='COBYLA',
    max_iterations=200
)

print(f"Ground state energy: {energy:.6f} Hartree")
```

### Dissociation Curve

Compute the potential energy surface:

```python
from examples.h2_vqa import h2_dissociation_curve
import numpy as np

# Define bond distances
distances = np.linspace(0.5, 2.0, 10)

# Compute dissociation curve
bond_distances, vqa_energies, exact_energies = h2_dissociation_curve(
    bond_distances=distances,
    ansatz_type='simplified',
    method='COBYLA'
)

# Visualize
from examples.visualize import plot_h2_dissociation_curve
plot_h2_dissociation_curve(
    bond_distances,
    vqa_energies,
    exact_energies,
    save_path='h2_dissociation.png'
)
```

### Energy Units

- **Hartree**: Atomic unit of energy (default output)
- **eV**: Multiply by 27.211386 to convert
- **kcal/mol**: Multiply by 627.509 to convert

```python
energy_hartree = -1.137
energy_ev = energy_hartree * 27.211386
energy_kcal = energy_hartree * 627.509
```

---

## Advanced Usage

### Custom Optimizers

#### Gradient-Free Optimization

```python
from optimizers.gradient_free import GradientFreeOptimizer

# Use specific method
optimizer = GradientFreeOptimizer(method='COBYLA')
result = optimizer.optimize(
    cost_function=vqa.cost_function,
    initial_parameters=initial_params,
    max_iterations=500,
    tolerance=1e-6
)
```

#### Gradient-Based Optimization

```python
from optimizers.parameter_shift import ParameterShiftOptimizer, AdamOptimizer

# Vanilla gradient descent
optimizer = ParameterShiftOptimizer(learning_rate=0.1)
result = optimizer.optimize(
    cost_function=vqa.cost_function,
    initial_parameters=initial_params,
    max_iterations=500
)

# Adam optimizer
adam = AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999)
result = adam.optimize(
    cost_function=vqa.cost_function,
    initial_parameters=initial_params,
    max_iterations=500
)
```

#### Adaptive Optimization

Try multiple methods and pick the best:

```python
from optimizers.gradient_free import AdaptiveOptimizer

optimizer = AdaptiveOptimizer(methods=['COBYLA', 'Nelder-Mead', 'Powell'])
result = optimizer.optimize(
    cost_function=vqa.cost_function,
    initial_parameters=initial_params,
    max_iterations=300
)
```

### Visualization

```python
from examples.visualize import (
    plot_convergence,
    plot_error_convergence,
    create_summary_plot
)

# Get optimization history
energy_history, param_history = vqa.get_history()

# Plot convergence
plot_convergence(
    energy_history,
    exact_energy=exact_energy,
    title="VQA Convergence",
    save_path="convergence.png"
)

# Plot error convergence
plot_error_convergence(
    energy_history,
    exact_energy=exact_energy,
    save_path="error.png"
)

# Create comprehensive summary
vqa_result = {
    'energy_history': energy_history,
    'parameter_history': param_history
}
create_summary_plot(
    vqa_result,
    exact_energy=exact_energy,
    save_path="summary.png"
)
```

---

## Customization

### Creating Custom Ansatze

```python
import cudaq
from typing import List

class CustomAnsatz:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.num_parameters = num_qubits * 2

    def build_kernel(self):
        @cudaq.kernel
        def kernel(params: List[float]):
            qubits = cudaq.qvector(self.num_qubits)

            # Your custom circuit here
            for i in range(self.num_qubits):
                ry(params[i], qubits[i])

            for i in range(self.num_qubits - 1):
                cx(qubits[i], qubits[i + 1])

            for i in range(self.num_qubits):
                rz(params[self.num_qubits + i], qubits[i])

        return kernel

    def initial_parameters(self):
        return np.random.uniform(-np.pi, np.pi, self.num_parameters)
```

### Custom Hamiltonians

```python
import cudaq

def custom_hamiltonian(num_qubits: int):
    """Create custom Hamiltonian."""
    H = 0.0 * cudaq.spin.i(0)

    # Add your terms
    # Example: X-Y model
    for i in range(num_qubits - 1):
        H += cudaq.spin.x(i) * cudaq.spin.x(i + 1)
        H += cudaq.spin.y(i) * cudaq.spin.y(i + 1)

    return H
```

### Benchmarking

```python
import time

# Time VQA execution
start_time = time.time()
energy, params = vqa.optimize(initial_params, max_iterations=200)
elapsed_time = time.time() - start_time

print(f"Optimization took {elapsed_time:.2f} seconds")
print(f"Time per iteration: {elapsed_time / len(vqa.energy_history):.3f} s")
```

---

## Tips and Best Practices

### 1. **Choosing Ansatz Depth**
- Start with depth=2 for Ising models
- Increase depth if convergence is poor
- Deeper circuits are more expressive but harder to optimize

### 2. **Optimization Method Selection**
- **COBYLA**: Good default, robust for most problems
- **Nelder-Mead**: Alternative when COBYLA struggles
- **Gradient-based**: More iterations but can find better minima

### 3. **Initial Parameters**
- Random initialization usually works well
- For H2, start near Hartree-Fock with small perturbations
- Use multiple random initializations to avoid local minima

### 4. **Convergence Issues**
- Increase max_iterations
- Try different optimization methods
- Reduce tolerance
- Use adaptive optimizer to try multiple methods

### 5. **Performance Optimization**
- Use GPU acceleration (CUDA-Q default)
- Reduce number of shots for faster (but noisier) gradients
- Parallelize multiple VQA runs for different parameters

---

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Ensure CUDA-Q is installed
pip install cuda-quantum
```

**2. GPU not detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA-Q can access GPU
python -c "import cudaq; cudaq.set_target('nvidia')"
```

**3. Slow convergence**
- Increase ansatz depth
- Try different optimizer
- Adjust learning rate (for gradient-based)

**4. Poor accuracy**
- Increase max_iterations
- Use more sophisticated ansatz (UCCSD for molecules)
- Verify Hamiltonian is correct

---

## References

1. **VQA Theory**: Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor", Nature Communications (2014)
2. **CUDA-Q**: NVIDIA CUDA-Q Documentation
3. **Ising Model**: Sachdev, "Quantum Phase Transitions" (2011)
4. **Quantum Chemistry**: Szabo & Ostlund, "Modern Quantum Chemistry" (1996)

---

## Support

For issues or questions:
- Check the examples directory
- Run tests to verify installation
- Consult CUDA-Q documentation
