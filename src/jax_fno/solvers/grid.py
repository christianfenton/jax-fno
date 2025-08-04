import jax.numpy as jnp
from .ivp import BCType

def create_uniform_grid(L, nx: int, bc_type: BCType):
    """
    Create a uniform grid based on the boundary condition type.
    
    Args:
        L: Domain length
        nx: Number of grid points
        bc: BCType object
    """
    if bc_type == BCType.DIRICHLET:
        # For Dirichlet BC, grid contains only interior points
        # Total domain is [0, L] but we only discretize interior
        dx = L / (nx + 1)  # nx interior points, 2 boundary points
        x = jnp.linspace(dx, L - dx, nx, endpoint=True)
        return x
    elif bc_type == BCType.PERIODIC:
        # For periodic BC, grid covers [0, L) with nx points
        dx = L / nx
        x = jnp.linspace(0, L, nx, endpoint=False)
        return x
    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")
    