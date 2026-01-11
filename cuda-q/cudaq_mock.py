"""
Mock CUDA-Q module for testing without actual CUDA-Q installation.
This allows testing the VQA logic without GPU/CUDA requirements.
"""
import numpy as np
from typing import List, Callable


class SpinOperator:
    """Mock SpinOperator class."""

    def __init__(self, coefficient=1.0, terms=None):
        self.coefficient = coefficient
        self.terms = terms or []

    def __add__(self, other):
        result = SpinOperator()
        result.terms = self.terms + (other.terms if hasattr(other, 'terms') else [])
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = SpinOperator(self.coefficient * other, self.terms)
            return result
        result = SpinOperator()
        result.terms = self.terms + (other.terms if hasattr(other, 'terms') else [])
        return result

    def __rmul__(self, other):
        return self.__mul__(other)


class SpinOps:
    """Mock spin operators."""

    @staticmethod
    def i(qubit):
        return SpinOperator(1.0, [('I', qubit)])

    @staticmethod
    def x(qubit):
        return SpinOperator(1.0, [('X', qubit)])

    @staticmethod
    def y(qubit):
        return SpinOperator(1.0, [('Y', qubit)])

    @staticmethod
    def z(qubit):
        return SpinOperator(1.0, [('Z', qubit)])


spin = SpinOps()


class ObserveResult:
    """Mock observe result."""

    def __init__(self, energy):
        self._energy = energy

    def expectation(self):
        return self._energy


def observe(kernel, hamiltonian, *args):
    """
    Mock observe function.
    Returns a simple energy value based on parameters.
    """
    # Simple mock: return sum of squared parameters (convex function)
    params = args
    if len(params) > 0 and hasattr(params[0], '__iter__'):
        params = params[0]

    # Mock energy calculation - just a simple quadratic
    energy = sum(p**2 for p in params) - 5.0

    return ObserveResult(energy)


def kernel(func):
    """Kernel decorator - just returns the function."""
    return func


class QVector:
    """Mock quantum vector."""

    def __init__(self, size):
        self.size = size
        self.qubits = list(range(size))

    def __getitem__(self, idx):
        return self.qubits[idx]


def qvector(size):
    """Create quantum vector."""
    return QVector(size)


# Mock quantum gates (no-ops for simulation)
def x(qubit):
    pass

def y(qubit):
    pass

def z(qubit):
    pass

def h(qubit):
    pass

def cx(control, target):
    pass

def cy(control, target):
    pass

def cz(control, target):
    pass

def rx(angle, qubit):
    pass

def ry(angle, qubit):
    pass

def rz(angle, qubit):
    pass


# Add to module globals
import sys
current_module = sys.modules[__name__]
for gate in ['x', 'y', 'z', 'h', 'cx', 'cy', 'cz', 'rx', 'ry', 'rz']:
    setattr(current_module, gate, globals()[gate])


__version__ = "0.0.0-mock"


print("=" * 70)
print("WARNING: Using mock CUDA-Q module for testing!")
print("This is NOT the real CUDA-Q - energies will be approximate.")
print("Install real cudaq package for accurate quantum simulations.")
print("=" * 70)
