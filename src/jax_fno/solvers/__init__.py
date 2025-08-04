"""
Non-linear PDE/IVP Solvers

JIT-compiled JAX solvers for generating training data
and providing ground truth solutions for comparison with neural operators.
"""

from .ivp import solve, BCType, newton_raphson_step, implicit_euler_step
from .derivatives import (
    d__dx_c_periodic, d2__dx2_c_periodic,
    d__dx_c_dirichlet, d2__dx2_c_dirichlet
)
from .grid import create_uniform_grid
from .equations import (
    burgers_residual_1d, burgers_jvp_1d,
    heat_residual_1d, heat_jvp_1d
)

__all__ = [
    # Main solver
    "solve",
    "BCType",
    "newton_raphson_step", 
    "implicit_euler_step",
    
    # Finite difference operators
    "d__dx_c_periodic",
    "d2__dx2_c_periodic",
    "d__dx_c_dirichlet", 
    "d2__dx2_c_dirichlet",
    
    # Grid utilities
    "create_uniform_grid",
    
    # Built-in equations
    "burgers_residual_1d",
    "burgers_jvp_1d",
    "heat_residual_1d",
    "heat_jvp_1d",
]