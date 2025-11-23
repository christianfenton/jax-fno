"""
JAX Fourier Neural Operators

A JAX/Flax implementation of Fourier Neural Operators for solving PDEs,
with high-performance time integration methods.

Main components:
- operators: Fourier neural operator implementations 
- solver: Time integration methods for data generation
"""

from .operators import FNO1D, FourierLayer1D

# Time integration solvers
from .solver import integrate, solve_ivp, RK4, ForwardEuler, BackwardEuler

__all__ = [
    # FNO operators
    "FNO1D",
    "FourierLayer1D",
    # Time integration
    "integrate",
    "solve_ivp",
    "RK4",
    "ForwardEuler",
    "BackwardEuler",
]
