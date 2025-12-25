"""
JAX Fourier Neural Operators

A JAX/Flax implementation of Fourier Neural Operators for solving PDEs,
with high-performance time integration methods.

Main components:
- operators: Fourier neural operator implementations 
- solver: Time integration methods for data generation
"""

from .learn import FNO1D, FourierLayer1D

# Time integration solvers
from .integrate import solve_ivp, solve_with_history, RK4, ForwardEuler, BackwardEuler

__all__ = [
    # FNO layers and architectures
    "FNO1D",
    "FourierLayer1D",

    # ODE integration methods
    "solve_ivp",
    "solve_with_history",
    "RK4",
    "ForwardEuler",
    "BackwardEuler",
]
