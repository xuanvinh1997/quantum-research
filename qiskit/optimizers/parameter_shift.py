"""Parameter shift rule based gradient optimizer."""
import numpy as np
from typing import Callable, Dict, Any


class ParameterShiftOptimizer:
    """
    Gradient-based optimizer using parameter shift rule.

    The parameter shift rule allows exact computation of gradients
    for variational quantum circuits.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        shift: float = np.pi / 2
    ):
        """
        Initialize parameter shift optimizer.

        Args:
            learning_rate: Step size for gradient descent
            shift: Shift value for parameter shift rule (default π/2)
        """
        self.learning_rate = learning_rate
        self.shift = shift
        self.gradient_history = []

    def compute_gradient(
        self,
        cost_function: Callable,
        parameters: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient using parameter shift rule.

        For a parameter θᵢ, the partial derivative is:
        ∂f/∂θᵢ = [f(θ + s*eᵢ) - f(θ - s*eᵢ)] / 2

        where s is the shift value and eᵢ is the unit vector.

        Args:
            cost_function: Function to differentiate
            parameters: Current parameters

        Returns:
            Gradient vector
        """
        gradient = np.zeros_like(parameters)

        for i in range(len(parameters)):
            # Shift parameter i forward
            params_plus = parameters.copy()
            params_plus[i] += self.shift

            # Shift parameter i backward
            params_minus = parameters.copy()
            params_minus[i] -= self.shift

            # Compute finite difference
            f_plus = cost_function(params_plus)
            f_minus = cost_function(params_minus)

            gradient[i] = (f_plus - f_minus) / 2.0

        return gradient

    def optimize(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run gradient descent optimization.

        Args:
            cost_function: Function to minimize
            initial_parameters: Starting parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        parameters = initial_parameters.copy()
        energy_history = []
        self.gradient_history = []

        for iteration in range(max_iterations):
            # Compute energy and gradient
            energy = cost_function(parameters)
            gradient = self.compute_gradient(cost_function, parameters)

            energy_history.append(energy)
            self.gradient_history.append(gradient)

            # Check convergence
            gradient_norm = np.linalg.norm(gradient)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                      f"Energy = {energy:.8f}, "
                      f"Gradient norm = {gradient_norm:.6e}")

            if gradient_norm < tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

            # Update parameters
            parameters = parameters - self.learning_rate * gradient

        final_energy = cost_function(parameters)

        return {
            'x': parameters,
            'fun': final_energy,
            'success': gradient_norm < tolerance,
            'nit': iteration + 1,
            'gradient_norm': gradient_norm,
            'energy_history': energy_history,
            'message': 'Optimization converged' if gradient_norm < tolerance
                      else 'Maximum iterations reached'
        }


class AdamOptimizer(ParameterShiftOptimizer):
    """
    Adam optimizer with parameter shift rule for gradients.

    Adaptive moment estimation (Adam) is a popular optimizer
    that combines momentum and adaptive learning rates.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        shift: float = np.pi / 2
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            shift: Shift value for parameter shift rule
        """
        super().__init__(learning_rate=learning_rate, shift=shift)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Adam optimization.

        Args:
            cost_function: Function to minimize
            initial_parameters: Starting parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        parameters = initial_parameters.copy()
        energy_history = []
        self.gradient_history = []

        # Initialize moment estimates
        m = np.zeros_like(parameters)  # First moment
        v = np.zeros_like(parameters)  # Second moment

        for iteration in range(max_iterations):
            # Compute energy and gradient
            energy = cost_function(parameters)
            gradient = self.compute_gradient(cost_function, parameters)

            energy_history.append(energy)
            self.gradient_history.append(gradient)

            # Update biased moment estimates
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** (iteration + 1))
            v_hat = v / (1 - self.beta2 ** (iteration + 1))

            # Update parameters
            parameters = parameters - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Check convergence
            gradient_norm = np.linalg.norm(gradient)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                      f"Energy = {energy:.8f}, "
                      f"Gradient norm = {gradient_norm:.6e}")

            if gradient_norm < tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

        final_energy = cost_function(parameters)

        return {
            'x': parameters,
            'fun': final_energy,
            'success': gradient_norm < tolerance,
            'nit': iteration + 1,
            'gradient_norm': gradient_norm,
            'energy_history': energy_history,
            'message': 'Optimization converged' if gradient_norm < tolerance
                      else 'Maximum iterations reached'
        }
