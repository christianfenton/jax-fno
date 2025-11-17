"""
JAX-compatible time integration methods for initial value problems.

This module provides:
- Solver interfaces (solve_ivp, integrate)
- Time-stepping schemes (ForwardEuler, RK4, BackwardEuler)
- Setup utilities for spatial discretisation (grids, boundary conditions, operators)
"""

# Core integration functions
from .solve_ivp import solve_ivp, integrate

# Time-stepping schemes
from .timesteppers import ForwardEuler, RK4, BackwardEuler

from . import setup

__all__ = [
    # Core functions
    'integrate',
    'solve_ivp',
    # Time-stepping methods
    'ForwardEuler',
    'RK4',
    'BackwardEuler',
    # Setup submodule
    'setup',
]
