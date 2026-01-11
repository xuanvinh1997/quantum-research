"""Gradient-free optimizers for VQA."""
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, Any


class GradientFreeOptimizer:
    """
    Wrapper for gradient-free optimization methods.

    Supports:
    - COBYLA (Constrained Optimization BY Linear Approximation)
    - Nelder-Mead simplex method
    - Powell's method
    """

    def __init__(self, method: str = 'COBYLA'):
        """
        Initialize optimizer.

        Args:
            method: Optimization method ('COBYLA', 'Nelder-Mead', 'Powell')
        """
        self.method = method
        self.result = None

    def optimize(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            cost_function: Function to minimize
            initial_parameters: Starting parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            **kwargs: Additional arguments for scipy.optimize.minimize

        Returns:
            Dictionary with optimization results
        """
        options = {
            'maxiter': max_iterations,
        }

        if self.method == 'COBYLA':
            options['tol'] = tolerance
        elif self.method in ['Nelder-Mead', 'Powell']:
            options['ftol'] = tolerance

        # Merge with user-provided options
        if 'options' in kwargs:
            options.update(kwargs['options'])
            del kwargs['options']

        # Run optimization
        self.result = minimize(
            cost_function,
            initial_parameters,
            method=self.method,
            options=options,
            **kwargs
        )

        return {
            'x': self.result.x,
            'fun': self.result.fun,
            'success': self.result.success,
            'nit': self.result.nit if hasattr(self.result, 'nit') else None,
            'message': self.result.message
        }

    def get_result(self):
        """Get the optimization result object."""
        return self.result


class AdaptiveOptimizer:
    """
    Adaptive optimizer that tries multiple methods.

    Useful when one method fails or gets stuck.
    """

    def __init__(self, methods: list = None):
        """
        Initialize adaptive optimizer.

        Args:
            methods: List of methods to try (default: COBYLA, Nelder-Mead)
        """
        if methods is None:
            methods = ['COBYLA', 'Nelder-Mead']
        self.methods = methods
        self.results = {}

    def optimize(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Try multiple optimization methods and return best result.

        Args:
            cost_function: Function to minimize
            initial_parameters: Starting parameters
            max_iterations: Maximum iterations per method
            tolerance: Convergence tolerance

        Returns:
            Best optimization result
        """
        best_result = None
        best_energy = float('inf')

        for method in self.methods:
            print(f"\nTrying method: {method}")

            optimizer = GradientFreeOptimizer(method=method)

            try:
                result = optimizer.optimize(
                    cost_function,
                    initial_parameters,
                    max_iterations=max_iterations,
                    tolerance=tolerance
                )

                self.results[method] = result

                if result['fun'] < best_energy:
                    best_energy = result['fun']
                    best_result = result

                print(f"Method {method}: energy = {result['fun']:.8f}, "
                      f"success = {result['success']}")

            except Exception as e:
                print(f"Method {method} failed: {str(e)}")
                continue

        if best_result is None:
            raise RuntimeError("All optimization methods failed")

        print(f"\nBest method: energy = {best_energy:.8f}")

        return best_result
