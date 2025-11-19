"""
Time integration schemes for initial value problems written in JAX.

This module provides:
- Solver interfaces (solve_ivp, integrate)
- Time-stepping schemes (ForwardEuler, RK4, BackwardEuler)
- Setup utilities for spatial discretisation
"""

# Core integration functions
from .solve_ivp import solve_ivp, integrate

# Time-stepping schemes
from .timesteppers import ForwardEuler, RK4, BackwardEuler

__all__ = [
    # Core functions
    'integrate',
    'solve_ivp',
    # Time-stepping methods
    'ForwardEuler',
    'RK4',
    'BackwardEuler',
]
