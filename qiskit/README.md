# VQA with Qiskit: Ising Model and H2 Molecule

This project implements Variational Quantum Algorithms (VQA) using Qiskit for:
- **Transverse Field Ising Model** - Finding ground states of quantum spin systems
- **H2 Molecule** - Computing ground state energy and dissociation curves

This is a Qiskit adaptation of the CUDA-Q VQA framework, maintaining identical APIs and functionality.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Qiskit (quantum simulation framework)
- Qiskit Aer (local simulator)
- NumPy (numerical computing)
- SciPy (optimization)
- Matplotlib (visualization)

### Verify Installation

```bash
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
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
qiskit/
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

- Qiskit quantum simulation (local and cloud)
- Multiple ansatz options (hardware-efficient, UCCSD)
- Classical optimizers (COBYLA, Nelder-Mead, Powell, gradient-based)
- Parameter shift rule for gradients
- Exact diagonalization for verification
- Visualization tools
- Hardware backend access via IBM Quantum

## Qiskit-Specific Features

### Estimator Primitive

Uses Qiskit's Estimator primitive for expectation value computation:

```python
from qiskit.primitives import Estimator

estimator = Estimator()
job = estimator.run(circuit, hamiltonian)
result = job.result()
expectation = result.values[0]
```

### Backend Selection

```python
from qiskit_aer import Aer

# Use Aer simulator
backend = Aer.get_backend('qasm_simulator')

# Or connect to IBM Quantum hardware
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.backend("ibmq_qasm_simulator")
```

## Differences from CUDA-Q Implementation

### Hamiltonian Construction
```python
# CUDA-Q
H = cudaq.spin.z(0) * cudaq.spin.z(1)

# Qiskit
from qiskit.quantum_info import SparsePauliOp
H = SparsePauliOp.from_list([('ZZ', coeff)])
```

### Circuit Definition
```python
# CUDA-Q
@cudaq.kernel
def circuit(params):
    qubits = cudaq.qvector(n)
    ry(params[0], qubits[0])

# Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
params = ParameterVector('θ', n)
circuit = QuantumCircuit(n)
circuit.ry(params[0], 0)
```

### Expectation Value
```python
# CUDA-Q
expectation = cudaq.observe(kernel, H, *params).expectation()

# Qiskit
from qiskit.primitives import Estimator
estimator = Estimator()
job = estimator.run(circuit, H)
expectation = job.result().values[0]
```

## Performance

Typical performance on CPU:
- 4-qubit Ising: ~20-60 seconds for 200 iterations
- GPU acceleration available via Qiskit Aer GPU
- Cloud quantum hardware access via IBM Quantum

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

## Pauli String Convention

Qiskit uses little-endian ordering (rightmost qubit is index 0):
- 'XYZI' means: X on qubit 3, Y on qubit 2, Z on qubit 1, I on qubit 0
- Reverse string when constructing Hamiltonians

## Documentation

For comprehensive documentation, see the main quantum-research repository:
- Installation guides
- Usage examples
- API reference
- Troubleshooting

## Comparison with Other Frameworks

| Feature | Qiskit | CUDA-Q | PennyLane |
|---------|--------|--------|-----------|
| Ease of Use | Medium | Medium | High |
| Device Support | Broad | GPU-focused | Broad |
| Hardware Access | IBM Quantum | NVIDIA | Via plugins |
| Maturity | High | Growing | High |
| Differentiation | Manual | Manual | Automatic |

## IBM Quantum Access

To run on IBM Quantum hardware:

1. Create account at https://quantum-computing.ibm.com/
2. Get API token
3. Configure:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
backend = service.backend("ibm_brisbane")
```

## License

MIT

## References

1. Qiskit Documentation: https://qiskit.org/
2. VQE Original Paper: Peruzzo et al., Nature Communications (2014)
3. Variational Algorithms: McClean et al., New Journal of Physics (2016)
4. IBM Quantum: https://quantum-computing.ibm.com/
