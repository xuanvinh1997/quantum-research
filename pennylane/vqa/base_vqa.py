"""Base VQA class for variational quantum algorithms using PennyLane."""
import pennylane as qml
import numpy as np
from typing import Callable, List, Tuple, Optional


class VQA:
    """
    Base class for Variational Quantum Algorithms using PennyLane.

    This class provides a framework for VQA with:
    - Parameterized quantum circuits (ansatz)
    - Hamiltonian expectation value calculation
    - Classical optimization interface
    """

    def __init__(
        self,
        num_qubits: int,
        hamiltonian,  # qml.Hamiltonian
        ansatz_builder: Callable,
        optimizer: Optional[object] = None,
        device_name: str = 'default.qubit'
    ):
        """
        Initialize VQA.

        Args:
            num_qubits: Number of qubits in the system
            hamiltonian: PennyLane Hamiltonian observable
            ansatz_builder: Function that builds the parameterized quantum circuit
            optimizer: Classical optimizer object (optional)
            device_name: PennyLane device name (default: 'default.qubit')
        """
        self.num_qubits = num_qubits
        self.hamiltonian = hamiltonian
        self.ansatz_builder = ansatz_builder
        self.optimizer = optimizer
        self.device_name = device_name

        # Create PennyLane device
        self.device = qml.device(device_name, wires=num_qubits)

        self.iteration = 0
        self.energy_history = []
        self.parameter_history = []

    def compute_expectation(self, parameters: np.ndarray) -> float:
        """
        Compute expectation value of Hamiltonian with given parameters.

        Args:
            parameters: Circuit parameters

        Returns:
            Expectation value <ψ(θ)|H|ψ(θ)>
        """
        # Create QNode that returns expectation value
        @qml.qnode(self.device)
        def circuit(*params):
            # Build ansatz circuit
            ansatz_circuit = self.ansatz_builder(self.num_qubits, params)
            # Apply the circuit (it's a callable)
            ansatz_circuit(*params)
            # Return expectation value of Hamiltonian
            return qml.expval(self.hamiltonian)

        # Compute expectation value
        expectation = circuit(*parameters)

        return float(expectation)

    def cost_function(self, parameters: np.ndarray) -> float:
        """
        Cost function to minimize (energy expectation value).

        Args:
            parameters: Circuit parameters

        Returns:
            Energy expectation value
        """
        energy = self.compute_expectation(parameters)

        # Store history
        self.energy_history.append(energy)
        self.parameter_history.append(parameters.copy())

        if self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: Energy = {energy:.6f}")

        self.iteration += 1

        return energy

    def optimize(
        self,
        initial_parameters: np.ndarray,
        method: str = 'COBYLA',
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Tuple[float, np.ndarray]:
        """
        Run VQA optimization to find ground state.

        Args:
            initial_parameters: Starting parameters
            method: Optimization method ('COBYLA', 'Nelder-Mead', 'gradient')
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (optimal_energy, optimal_parameters)
        """
        from scipy.optimize import minimize

        print(f"Starting VQA optimization with {len(initial_parameters)} parameters")
        print(f"Method: {method}")

        # Reset history
        self.iteration = 0
        self.energy_history = []
        self.parameter_history = []

        # Run optimization
        if method.lower() == 'gradient':
            # Use custom gradient-based optimizer
            if self.optimizer is None:
                raise ValueError("Gradient-based optimization requires an optimizer")
            result = self.optimizer.optimize(
                self.cost_function,
                initial_parameters,
                max_iterations=max_iterations,
                tolerance=tolerance
            )
            optimal_params = result['x']
            optimal_energy = result['fun']
        else:
            # Use scipy optimizer
            result = minimize(
                self.cost_function,
                initial_parameters,
                method=method,
                options={'maxiter': max_iterations, 'ftol': tolerance}
            )
            optimal_params = result.x
            optimal_energy = result.fun

        print(f"\nOptimization complete!")
        print(f"Final energy: {optimal_energy:.8f}")
        print(f"Total iterations: {self.iteration}")

        return optimal_energy, optimal_params

    def get_history(self) -> Tuple[List[float], List[np.ndarray]]:
        """
        Get optimization history.

        Returns:
            Tuple of (energy_history, parameter_history)
        """
        return self.energy_history, self.parameter_history

    def compute_gradient(
        self,
        parameters: np.ndarray,
        shift: float = np.pi / 2
    ) -> np.ndarray:
        """
        Compute gradient using parameter shift rule.

        Args:
            parameters: Current parameters
            shift: Shift value for parameter shift rule

        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(parameters)

        for i in range(len(parameters)):
            params_plus = parameters.copy()
            params_plus[i] += shift

            params_minus = parameters.copy()
            params_minus[i] -= shift

            energy_plus = self.compute_expectation(params_plus)
            energy_minus = self.compute_expectation(params_minus)

            gradient[i] = (energy_plus - energy_minus) / 2

        return gradient
