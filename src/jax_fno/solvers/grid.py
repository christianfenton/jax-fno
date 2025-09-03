import jax.numpy as jnp
from enum import IntEnum


class BCType(IntEnum):
    """
    Boundary condition types.
    
    Currently supported boundary conditions:
        PERIODIC
        DIRICHLET
    """
    PERIODIC = 0
    DIRICHLET = 1


def create_uniform_grid(L, nx: int, bc_type: BCType, return_spacing=False):
    """
    Create a uniform grid based on the boundary condition type.
    
    Args:
        L: Domain length
        nx: Number of grid points
        bc: BCType object

    Returns: 
        x: Grid
        dx: Grid spacing
    """
    if bc_type not in (BCType.DIRICHLET, BCType.PERIODIC):
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    if bc_type == BCType.DIRICHLET:
        # For Dirichlet BC, grid contains only interior points
        # Total domain is [0, L] but we only discretize interior
        dx = L / (nx + 1)  # nx interior points, 2 boundary points
        x = jnp.linspace(dx, L - dx, nx, endpoint=True)
    else:
        # For periodic BC, grid covers [0, L) with nx points
        dx = L / nx
        x = jnp.linspace(0, L, nx, endpoint=False)
    
    if return_spacing:
        return x, dx
    else:
        return x
    