"""Linear solvers used in root-finding and implicit time stepping schemes."""

from .protocol import LinearSolverProtocol
from .direct import DirectDense
from .krylov import GMRES, CG, BiCGStab
from .spectral import Spectral, dst1, idst1


__all__ = [
    # Protocol
    "LinearSolverProtocol",

    # Direct solvers
    "DirectDense",

    # Krylov methods
    "GMRES",
    "CG",
    "BiCGStab",

    # Spectral methods
    "Spectral",

    # Transforms
    "dst1",
    "idst1",
]