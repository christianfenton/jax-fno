"""
Time integration schemes for initial value problems written in JAX.
"""

# Solver interfaces
from .solve import solve_with_history, solve_ivp

# Time-stepping schemes
from .timesteppers import ForwardEuler, RK4, BackwardEuler, AbstractStepper

# Root-finding algorithms
from .rootfinders import AbstractRootFinder, NewtonRaphson

# Linear solvers
from .linsolvers import (
    AbstractLinearSolver,
    GMRES,
    CG,
    BiCGStab,
    Direct,
)

__all__ = [
    # Solver interfaces
    'solve_ivp',
    'solve_with_history',

    # Time-stepping methods
    'AbstractStepper',
    'ForwardEuler',
    'RK4',
    'BackwardEuler',

    # Root-finding algorithms
    'AbstractRootFinder',
    'NewtonRaphson',

    # Linear solvers
    'AbstractLinearSolver',
    'GMRES',
    'CG',
    'BiCGStab',
    'Direct',
]
