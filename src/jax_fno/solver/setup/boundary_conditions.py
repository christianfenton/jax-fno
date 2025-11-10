from dataclasses import dataclass
from enum import Enum


class BCType(Enum):
    PERIODIC = 0
    DIRICHLET = 1
    NEUMANN = 2


@dataclass(frozen=True)
class BoundaryCondition:
    bc_type: BCType


@dataclass(frozen=True)
class PeriodicBC(BoundaryCondition):
    def __init__(self):
        object.__setattr__(self, 'bc_type', BCType.PERIODIC)


@dataclass(frozen=True)
class DirichletBC(BoundaryCondition):
    left: float = 0.0
    right: float = 0.0

    def __init__(self, left: float = 0.0, right: float = 0.0):
        object.__setattr__(self, 'bc_type', BCType.DIRICHLET)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)


@dataclass(frozen=True)
class NeumannBC(BoundaryCondition):
    left: float = 0.0
    right: float = 0.0

    def __init__(self, left: float = 0.0, right: float = 0.0):
        object.__setattr__(self, 'bc_type', BCType.NEUMANN)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)
