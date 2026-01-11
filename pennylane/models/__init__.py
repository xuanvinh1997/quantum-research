"""Hamiltonian models module for PennyLane."""
from .ising import IsingHamiltonian
from .h2_molecule import H2Hamiltonian

__all__ = ['IsingHamiltonian', 'H2Hamiltonian']
