# VQA with PennyLane: Ising Model and H2 Molecule

This project implements Variational Quantum Algorithms (VQA) using PennyLane for:
- **Transverse Field Ising Model** - Finding ground states of quantum spin systems
- **H2 Molecule** - Computing ground state energy and dissociation curves

This is a PennyLane adaptation of the CUDA-Q VQA framework, maintaining identical APIs and functionality.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PennyLane (quantum simulation framework)
- NumPy (numerical computing)
- SciPy (optimization)
- Matplotlib (visualization)

### Verify Installation

```bash
python -c "import pennylane as qml; print('PennyLane version:', qml.__version__)"
```

## Quick Start

### Simple Test

```bash
python test_simple.py
```

This runs a 3-qubit Ising model VQA optimization to verify the installation.

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

## Project Structure

```
pennylane/
├── vqa/
│   ├── __init__.py
│   └── base_vqa.py              # Core VQA framework
├── models/
│   ├── __init__.py
│   ├── ising.py                 # Ising Hamiltonian
│   └── h2_molecule.py           # H2 Hamiltonian (TODO)
├── ansatze/
│   ├── __init__.py
│   ├── hardware_efficient.py   # Hardware-efficient ansatz
│   └── uccsd.py                 # UCCSD ansatz (TODO)
├── optimizers/
│   ├── __init__.py
│   ├── gradient_free.py        # COBYLA, Nelder-Mead, Powell
│   └── parameter_shift.py      # Gradient-based optimizers
├── examples/                    # Example scripts (TODO)
├── tests/                       # Test suite (TODO)
├── test_simple.py              # Simple verification script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Features

- PennyLane quantum simulation (CPU/GPU)
- Multiple ansatz options (hardware-efficient, UCCSD)
- Classical optimizers (COBYLA, Nelder-Mead, Powell, gradient-based)
- Parameter shift rule for gradients
- Exact diagonalization for verification
- Visualization tools

## PennyLane-Specific Features

### Device Selection

The default device is 'default.qubit' (CPU simulator). You can specify different devices:

```python
vqa = VQA(
    num_qubits=4,
    hamiltonian=ising.get_hamiltonian(),
    ansatz_builder=lambda n, p: ansatz.build_kernel(),
    device_name='default.qubit'  # or 'lightning.qubit' for GPU
)
```

### Available Devices
- `default.qubit` - CPU simulator (default)
- `lightning.qubit` - Fast CPU/GPU simulator
- Hardware backends via plugins

## Differences from CUDA-Q Implementation

### Hamiltonian Construction
```python
# CUDA-Q
H = cudaq.spin.z(0) * cudaq.spin.z(1)

# PennyLane
H = qml.Hamiltonian([coeff], [qml.PauliZ(0) @ qml.PauliZ(1)])
```

### Circuit Definition
```python
# CUDA-Q
@cudaq.kernel
def circuit(params):
    qubits = cudaq.qvector(n)
    ry(params[0], qubits[0])

# PennyLane
def circuit(*params):
    qml.RY(params[0], wires=0)
```

### Expectation Value
```python
# CUDA-Q
expectation = cudaq.observe(kernel, H, *params).expectation()

# PennyLane
@qml.qnode(device)
def circuit(*params):
    # apply ansatz
    return qml.expval(H)

expectation = circuit(*params)
```

## Performance

Typical performance on CPU:
- 4-qubit Ising: ~15-45 seconds for 200 iterations
- GPU acceleration available with `lightning.qubit` device

## Theory Background

### Variational Quantum Eigensolver (VQE)

VQE is a hybrid quantum-classical algorithm:

1. **Prepare** parameterized quantum state |ψ(θ)⟩
2. **Measure** expectation value E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimize** parameters θ using classical optimizer
4. **Repeat** until convergence

### Ising Model

Transverse Field Ising Model:

**H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ**

- J > 0: Ferromagnetic coupling
- h: Transverse field strength
- Models magnetic interactions and quantum phase transitions

## Documentation

For comprehensive documentation, see the main quantum-research repository:
- Installation guides
- Usage examples
- API reference
- Troubleshooting

## Comparison with Other Frameworks

| Feature | PennyLane | CUDA-Q | Qiskit |
|---------|-----------|--------|--------|
| Ease of Use | High | Medium | Medium |
| Device Support | Broad | GPU-focused | Broad |
| Differentiation | Automatic | Manual | Manual |
| Hardware Access | Via plugins | NVIDIA | IBM, others |

## License

MIT

## References

1. PennyLane Documentation: https://pennylane.ai/
2. VQE Original Paper: Peruzzo et al., Nature Communications (2014)
3. Variational Algorithms: McClean et al., New Journal of Physics (2016)
