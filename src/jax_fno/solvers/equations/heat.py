import jax
import jax.numpy as jnp
from typing import Dict, Any
from ..grid import BCType
from ..derivatives import *


@jax.jit
def heat_residual_1d(u_new: jnp.ndarray, u_old: jnp.ndarray, dt: float, dx: float, params: Dict[str, Any]) -> jnp.ndarray:
    """
    Heat equation in 1D: ∂u/∂t = D∂²u/∂x²
    
    Args:
        u_new: Solution at new time step
        u_old: Solution at old time step
        dt: Time step size  
        dx: Spatial grid spacing
        params: Dictionary with keys 'D', 'bc_type', and boundary values if needed
    """
    D = params['D']
    bc_type = params['bc_type']
    
    # Time derivative term
    time_term = (u_new - u_old) / dt
    
    # Compute spatial derivatives using JAX-compatible control flow
    def periodic_derivatives(_):
        return d2__dx2_c_periodic(u_new, dx)
    
    def dirichlet_derivatives(_):
        bc_left = params.get('bc_left', 0.0)  # Use default if not provided
        bc_right = params.get('bc_right', 0.0)  # Use default if not provided
        return d2__dx2_c_dirichlet(u_new, dx, bc_left, bc_right)
    
    # Use jax.lax.cond for JAX-compatible branching
    d2u_dx2 = jax.lax.cond(
        bc_type == BCType.PERIODIC,
        periodic_derivatives,
        dirichlet_derivatives,
        None
    )
    
    # Heat equation residual
    return time_term - D * d2u_dx2


@jax.jit
def heat_jvp_1d(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """Jacobian-vector product `J(u) @ v` for heat equation in 1D."""
    bc_type = params['bc_type']
    periodic = lambda _ : heat_jvp_periodic(u, dt, dx, params, v)
    dirichlet = lambda _ : heat_jvp_dirichlet(u, dt, dx, params, v)
    return jax.lax.cond(bc_type == BCType.PERIODIC, periodic, dirichlet, None)


@jax.jit
def heat_jvp_periodic(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """Analytical Jacobian-vector product for the heat equation with periodic BCs."""
    nu = params['D']
    
    # Time derivative contribution
    time_jvp = v / dt
    
    # Diffusion term contribution
    d2v_dx2 = d2__dx2_c_periodic(v, dx)
    diffusion_jvp = -nu * d2v_dx2
    
    return time_jvp + diffusion_jvp


@jax.jit  
def heat_jvp_dirichlet(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """Analytical Jacobian-vector product for heat equation with Dirichlet BCs."""
    nu = params['D']
    
    # Time derivative contribution
    time_jvp = v / dt
    
    # Diffusion term contribution
    d2v_dx2 = d2__dx2_c_dirichlet(v, dx, 0.0, 0.0)  # BCs on perturbuation should not affect BCs on u
    diffusion_jvp = -nu * d2v_dx2
    
    return time_jvp + diffusion_jvp