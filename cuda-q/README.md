# VQA with CUDA-Q: Ising Model and H2 Molecule

This project implements Variational Quantum Algorithms (VQA) using CUDA-Q for:
- **Transverse Field Ising Model** - Finding ground states of quantum spin systems
- **H2 Molecule** - Computing ground state energy and dissociation curves

![VQA Workflow](https://img.shields.io/badge/VQA-CUDA--Q-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

 **Key Capabilities**
- GPU-accelerated quantum simulation with CUDA-Q
- Multiple ansatz options (hardware-efficient, UCCSD, custom)
- Classical optimizers (COBYLA, Nelder-Mead, Powell, gradient-based with parameter shift)
- Visualization tools for energy landscapes and convergence
- Comprehensive testing suite
- Exact diagonalization for verification

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- CUDA-Q 0.6.0+

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install CUDA-Q

```bash
pip install cuda-quantum
```

Verify installation:
```bash
python -c "import cudaq; print(cudaq.__version__)"
```

## Quick Start

### Run Examples via CLI

```bash
# Run Ising model VQA
python run_examples.py ising --qubits 4 --J 1.0 --h 0.5

# Run H2 molecule VQA
python run_examples.py h2 --distance 0.74

# Run both examples
python run_examples.py both

# Run tests
python run_examples.py test
```

### Command-Line Options

```bash
python run_examples.py --help

Options:
  --qubits N        Number of qubits for Ising (default: 4)
  --J FLOAT         Coupling strength (default: 1.0)
  --h FLOAT         Transverse field (default: 0.5)
  --distance FLOAT  H2 bond distance in Å (default: 0.74)
  --depth N         Ansatz depth (default: 2)
  --method STR      Optimizer (COBYLA, Nelder-Mead, Powell)
  --iterations N    Max iterations (default: 200)
```

### Python API Usage

#### Ising Model
```python
from models.ising import IsingHamiltonian
from ansatze.hardware_efficient import HardwareEfficientAnsatz
from vqa.base_vqa import VQA

# Create Hamiltonian
ising = IsingHamiltonian(num_qubits=4, J=1.0, h=0.5, periodic=True)

# Create ansatz
ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=2)

# Run VQA
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

print(f"Ground state energy: {energy:.6f}")
```

#### H2 Molecule
```python
from models.h2_molecule import H2Hamiltonian
from ansatze.uccsd import SimplifiedH2Ansatz
from vqa.base_vqa import VQA

# Create H2 Hamiltonian
h2 = H2Hamiltonian(bond_distance=0.74)

# Create ansatz
ansatz = SimplifiedH2Ansatz()

# Run VQA
vqa = VQA(
    num_qubits=4,
    hamiltonian=h2.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel()
)

energy, params = vqa.optimize(
    initial_parameters=ansatz.initial_parameters(),
    method='COBYLA'
)

print(f"Ground state energy: {energy:.6f} Hartree")
```

## Project Structure

```
quantum-research/
├── vqa/                    # Core VQA implementation
│   ├── __init__.py
│   └── base_vqa.py        # Base VQA class
├── models/                 # Hamiltonian definitions
│   ├── __init__.py
│   ├── ising.py           # Ising model Hamiltonian
│   └── h2_molecule.py     # H2 molecule Hamiltonian
├── ansatze/                # Quantum circuit ansatze
│   ├── __init__.py
│   ├── hardware_efficient.py  # Hardware-efficient ansatz
│   └── uccsd.py           # UCCSD ansatz for molecules
├── optimizers/             # Classical optimization
│   ├── __init__.py
│   ├── gradient_free.py   # COBYLA, Nelder-Mead, etc.
│   └── parameter_shift.py # Gradient-based optimizers
├── examples/               # Example scripts
│   ├── ising_vqa.py       # Ising model examples
│   ├── h2_vqa.py          # H2 molecule examples
│   └── visualize.py       # Visualization tools
├── tests/                  # Unit tests
│   ├── test_ising.py
│   ├── test_h2.py
│   ├── test_vqa.py
│   └── run_all_tests.py
├── run_examples.py         # Main CLI interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── USAGE_GUIDE.md         # Comprehensive usage guide
```

## Advanced Features

### Visualization

```python
from examples.visualize import plot_convergence, create_summary_plot

# Plot optimization convergence
energy_history, param_history = vqa.get_history()
plot_convergence(energy_history, exact_energy, save_path="convergence.png")

# Create comprehensive summary
vqa_result = {
    'energy_history': energy_history,
    'parameter_history': param_history
}
create_summary_plot(vqa_result, exact_energy, save_path="summary.png")
```

### Custom Optimizers

```python
from optimizers.parameter_shift import AdamOptimizer
from optimizers.gradient_free import AdaptiveOptimizer

# Use Adam optimizer with parameter shift gradients
adam = AdamOptimizer(learning_rate=0.1)
result = adam.optimize(vqa.cost_function, initial_params)

# Try multiple optimizers automatically
adaptive = AdaptiveOptimizer(methods=['COBYLA', 'Nelder-Mead'])
result = adaptive.optimize(vqa.cost_function, initial_params)
```

### H2 Dissociation Curve

```python
from examples.h2_vqa import h2_dissociation_curve
import numpy as np

distances = np.linspace(0.5, 2.0, 10)
bond_distances, vqa_energies, exact_energies = h2_dissociation_curve(
    bond_distances=distances
)
```

## Testing

Run all tests:

```bash
cd tests
python run_all_tests.py
```

Run specific test modules:

```bash
python test_ising.py
python test_h2.py
python test_vqa.py
```

## Documentation

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)

Topics covered:
- Installation and setup
- Ising model VQA with examples
- H2 molecule VQA and dissociation curves
- Advanced optimization techniques
- Custom ansatz creation
- Visualization tools
- Troubleshooting

## Theory Background

### Variational Quantum Eigensolver (VQE)

VQE is a hybrid quantum-classical algorithm for finding ground states:

1. **Prepare** parameterized quantum state |ψ(θ)⟩
2. **Measure** expectation value E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimize** parameters θ using classical optimizer
4. **Repeat** until convergence

### Ising Model

The Transverse Field Ising Model:

**H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ**

- Models magnetic interactions and quantum phase transitions
- J > 0: Ferromagnetic coupling
- h: Transverse field inducing quantum fluctuations

### H2 Molecule

Electronic structure Hamiltonian mapped to qubits:

**H = Σᵢ cᵢ Pᵢ**

Where Pᵢ are Pauli strings and cᵢ are coefficients from quantum chemistry.

## Performance

Typical performance (on NVIDIA GPU):
- 4-qubit Ising: ~10-30 seconds for 200 iterations
- H2 molecule: ~15-40 seconds for 200 iterations
- Gradient computation: 2N circuit evaluations (N parameters)

## Limitations

- Small system sizes (< 20 qubits) for simulation
- Simplified H2 model (full UCCSD requires more parameters)
- Classical optimization can get stuck in local minima

## Future Enhancements

- [ ] Support for larger molecules (H2O, LiH)
- [ ] Noise simulation and error mitigation
- [ ] Hardware backend integration
- [ ] Advanced ansatz (ADAPT-VQE, QAOA)
- [ ] Parallel multi-start optimization

## References

1. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor", Nature Communications (2014)
2. McClean et al., "The theory of variational hybrid quantum-classical algorithms", New Journal of Physics (2016)
3. NVIDIA CUDA-Q Documentation: https://nvidia.github.io/cuda-quantum/

## License

MIT
