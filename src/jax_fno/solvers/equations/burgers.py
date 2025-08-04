import jax
import jax.numpy as jnp
from typing import Dict, Any
from functools import partial
from ..ivp import BCType
from ..derivatives import *


# Residuals for the burgers equation

@jax.jit
def burgers_residual_1d(u_new: jnp.ndarray, u_old: jnp.ndarray, dt: float, dx: float, params: Dict[str, Any]) -> jnp.ndarray:
    """
    Burgers equation in 1D: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Args:
        u_new: Solution at new time step
        u_old: Solution at old time step  
        dt: Time step size
        dx: Spatial grid spacing
        params: Dictionary with keys 'nu', 'bc_type', 'bc_left', 'bc_right'
    """
    nu = params['nu']
    bc_type = params['bc_type']
    
    # Time derivative term
    time_term = (u_new - u_old) / dt
    
    # Compute spatial derivatives using JAX-compatible control flow
    def periodic_derivatives(_):
        du_dx = d__dx_c_periodic(u_new, dx)
        d2u_dx2 = d2__dx2_c_periodic(u_new, dx)
        return du_dx, d2u_dx2
    
    def dirichlet_derivatives(_):
        bc_left = params['bc_left']
        bc_right = params['bc_right']
        du_dx = d__dx_c_dirichlet(u_new, dx, bc_left, bc_right)
        d2u_dx2 = d2__dx2_c_dirichlet(u_new, dx, bc_left, bc_right)
        return du_dx, d2u_dx2
    
    # Use jax.lax.cond for JAX-compatible branching
    du_dx, d2u_dx2 = jax.lax.cond(
        bc_type == BCType.PERIODIC,
        periodic_derivatives,
        dirichlet_derivatives,
        None
    )
    
    # Nonlinear advection term: u * du/dx
    advection_term = u_new * du_dx
    
    # Viscous diffusion term: ν * d²u/dx²
    diffusion_term = nu * d2u_dx2
    
    # PDE residual
    return time_term + advection_term - diffusion_term


# Matrix-free Jacobian-vector products for Burgers equation.


# Automatic BC selection
@jax.jit
def burgers_jvp_1d(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the Jacobian-vector product `J(r) @ v` for Burgers' equation in 1D,
    where `r = (u - u_old)/dt + u * du/dx - nu * d²u/dx²` is the 
    residual at the new time step.
    """
    bc_type = params['bc_type']
    periodic = lambda _ : burgers_jvp_periodic(u, dt, dx, params, v)
    dirichlet = lambda _ : burgers_jvp_dirichlet(u, dt, dx, params, v)
    return jax.lax.cond(bc_type == BCType.PERIODIC, periodic, dirichlet, None)


@jax.jit
def burgers_jvp_periodic(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """Analytical Jacobian-vector product for Burgers equation with periodic BCs."""
    nu = params['nu']
    nx = len(u)
    
    # Time derivative contribution: ∂/∂u_new[(u_new - u_old)/dt] @ v = v/dt
    time_jvp = v / dt
    
    # Advection term contribution: ∂/∂u_new[u_new * du_new/dx] @ v
    # This has two parts:
    # 1. ∂u_new/∂u_new * du_new/dx @ v = v * du_new/dx  
    # 2. u_new * ∂(du_new/dx)/∂u_new @ v = u_new * dv/dx
    du_dx = d__dx_c_periodic(u, dx)
    dv_dx = d__dx_c_periodic(v, dx)
    advection_jvp = v * du_dx + u * dv_dx
    
    # Diffusion term contribution: ∂/∂u_new[-nu * d²u_new/dx²] @ v = -nu * d²v/dx²
    d2v_dx2 = d2__dx2_c_periodic(v, dx)
    diffusion_jvp = -nu * d2v_dx2
    
    return time_jvp + advection_jvp + diffusion_jvp


@jax.jit  
def burgers_jvp_dirichlet(
    u: jnp.ndarray, 
    dt: float, 
    dx: float, 
    params: Dict[str, Any], 
    v: jnp.ndarray
) -> jnp.ndarray:
    """Analytical Jacobian-vector product for Burgers equation with Dirichlet BCs."""
    nu = params['nu']
    bc_left = params['bc_left']
    bc_right = params['bc_right']
    
    # Time derivative contribution
    time_jvp = v / dt
    
    # Advection term contribution
    du_dx = d__dx_c_dirichlet(u, dx, bc_left, bc_right)
    dv_dx = d__dx_c_dirichlet(v, dx, 0.0, 0.0)  # BCs on perturbuation should not affect BCs on u
    advection_jvp = v * du_dx + u * dv_dx
    
    # Diffusion term contribution
    d2v_dx2 = d2__dx2_c_dirichlet(v, dx, 0.0, 0.0)  # BCs on perturbuation should not affect BCs on u
    diffusion_jvp = -nu * d2v_dx2
    
    return time_jvp + advection_jvp + diffusion_jvp