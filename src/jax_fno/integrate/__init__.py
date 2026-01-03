"""Time integration schemes for initial value problems written in JAX."""

# Solver interfaces
from .solve import solve_with_history, solve_ivp

# Time-stepping schemes
from .timesteppers import (
    StepperProtocol,
    ForwardEuler,
    RK4,
    BackwardEuler,
    IMEX,
)

# Root-finding algorithms
from .rootfinders import RootFinderProtocol, NewtonRaphson

# Linear solvers
from .linsolvers import (
    LinearSolverProtocol,
    # Direct solvers
    DirectDense,
    # Krylov solvers
    GMRES,
    CG,
    BiCGStab,
    # Spectral solvers
    Spectral,
)

__all__ = [
    # Solver interfaces
    "solve_ivp",
    "solve_with_history",
    # Protocols
    "StepperProtocol",
    "RootFinderProtocol",
    "LinearSolverProtocol",
    # Time-stepping methods
    "ForwardEuler",
    "RK4",
    "BackwardEuler",
    "IMEX",
    # Root-finding algorithms
    "NewtonRaphson",
    # Linear solvers
    "DirectDense",
    "GMRES",
    "CG",
    "BiCGStab",
    "Spectral",
]
