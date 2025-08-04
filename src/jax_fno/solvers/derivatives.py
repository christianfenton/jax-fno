"""
Convenience functions for common finite difference operations
"""

import jax
import jax.numpy as jnp


def d__dx_c_periodic(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Approximate the first derivative using central differences with periodic boundary conditions.

    Accuracy: Second-order
    """
    u_plus = jnp.roll(u, -1)  # u[i+1]
    u_minus = jnp.roll(u, 1)  # u[i-1]
    return (u_plus - u_minus) / (2.0 * dx)


def d2__dx2_c_periodic(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Approximate the second derivative using central differences with periodic boundary conditions.

    Accuracy: Second-order
    """
    u_plus = jnp.roll(u, -1)   # u[i+1]
    u_minus = jnp.roll(u, 1)   # u[i-1]
    return (u_plus - 2.0 * u + u_minus) / (dx**2)


def d__dx_c_dirichlet(u: jnp.ndarray, dx: float, u_left: float, u_right: float) -> jnp.ndarray:
    """
    Approximate the first derivative using central differences with Dirichlet boundary conditions.

    Accuracy: Second-order
    """
    nx = len(u)
    du_dx = jnp.zeros_like(u)
    
    # Interior points: central difference
    if nx > 2:
        du_dx = du_dx.at[1:-1].set((u[2:] - u[:-2]) / (2.0 * dx))
    
    # Boundary points
    if nx >= 1:
        du_dx = du_dx.at[0].set((u[1] - u_left) / (2.0 * dx) if nx > 1 else 0.0)
        du_dx = du_dx.at[-1].set((u_right - u[-2]) / (2.0 * dx) if nx > 1 else 0.0)
    
    return du_dx


def d2__dx2_c_dirichlet(u: jnp.ndarray, dx: float, u_left: float, u_right: float) -> jnp.ndarray:
    """
    Approximate the second derivative using central differences with Dirichlet boundary conditions.

    Accuracy: Second-order
    """
    nx = len(u)
    d2u_dx2 = jnp.zeros_like(u)
    
    # Interior points: central difference
    if nx > 2:
        d2u_dx2 = d2u_dx2.at[1:-1].set((u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2))
    
    # Boundary points
    if nx >= 1:
        d2u_dx2 = d2u_dx2.at[0].set((u_left - 2.0 * u[0] + u[1]) / (dx**2) if nx > 1 else 0.0)
        d2u_dx2 = d2u_dx2.at[-1].set((u[-2] - 2.0 * u[-1] + u_right) / (dx**2) if nx > 1 else 0.0)
    
    return d2u_dx2