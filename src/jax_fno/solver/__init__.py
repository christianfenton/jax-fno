"""
Time integration schemes for initial value problems written in JAX.
"""

# Solver interfaces
from .solve import solve_ivp, integrate

# Time-stepping schemes
from .timesteppers import ForwardEuler, RK4, BackwardEuler, AbstractStepper

# Root-finding algorithms
from .newtonraphson import NewtonRaphson, RootFindingProtocol

# Linear solvers
from .linearsolver import (
    GMRES,
    CG,
    BiCGStab,
    DirectSolve,
    LinearSolverProtocol,
)

__all__ = [
    # Solver interfaces
    'integrate',
    'solve_ivp',

    # Time-stepping methods
    'AbstractStepper',
    'ForwardEuler',
    'RK4',
    'BackwardEuler',

    # Root-finding algorithms
    'NewtonRaphson',
    'RootFindingProtocol',

    # Linear solvers
    'GMRES',
    'CG',
    'BiCGStab',
    'DirectSolve',
    'LinearSolverProtocol',
]
