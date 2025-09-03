"""
Generic JAX-based functional IVP solver.

This module provides a fully general functional interface for solving IVPs with
JAX optimisation, where users only need to provide residual and jacobian (optional) functions.
"""

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Tuple, Dict, Any, Optional
from functools import partial
from .grid import create_uniform_grid


@partial(jax.jit, static_argnames=['residual_fn', 'jvp_fn'])
def newton_raphson_step(
    u_new: jnp.ndarray,
    u_old: jnp.ndarray,
    dt: float,
    dx: float,
    parameters: Dict[str, Any],
    residual_fn: Callable,
    jvp_fn: Optional[Callable] = None,
    maxiter: int = 50,
    tol: float = 1e-6
) -> jnp.ndarray:
    """
    Perform a single step of the Newton-Raphson method using matrix-free GMRES.
    
    Args:
        u_new: Current estimate of solution at new time step
        u_old: Solution at old time step
        dt: Time step size
        dx: Spatial grid spacing
        parameters: Dictionary of PDE parameters
        residual_fn: Function computing PDE residual
        jvp_fn: Function computing Jacobian-vector products, or None for autodiff JVP
        maxiter: Maximum GMRES iterations
        tol: GMRES tolerance
        
    Returns:
        Updated solution estimate
    """
    # Compute residual
    residual = residual_fn(u_new, u_old, dt, dx, parameters)
    
    if jvp_fn is not None:
        # Use user-provided analytical Jacobian-vector product
        matvec = lambda v: jvp_fn(u_new, dt, dx, parameters, v)
    else:
        # Use automatic differentiation for Jacobian-vector product
        r = lambda u: residual_fn(u, u_old, dt, dx, parameters)
        matvec = lambda v : jax.jvp(r, (u_new,), (v,))[1]  
    
    # Solve J @ delta_u = -residual using GMRES
    delta_u, info = jax.scipy.sparse.linalg.bicgstab(
        A=matvec, b=-residual, maxiter=maxiter, tol=tol
    )
    
    return u_new + delta_u


@partial(jax.jit, static_argnames=['residual_fn', 'jvp_fn'])
def implicit_euler_step(
    u_old: jnp.ndarray,
    dt: float,
    dx: float,
    parameters: Dict[str, Any],
    residual_fn: Callable,
    jvp_fn: Optional[Callable] = None,
    tol: float = 1e-6,
    maxiter: int = 10,
    tol_gmres: float = 1e-6,
    maxiter_gmres: int = 50
) -> jnp.ndarray:
    """
    Perform a single implicit (backward) Euler step using a matrix-free Newton-Raphson method.
    
    Args:
        u_old: Solution at previous time step
        dt: Time step size
        dx: Spatial grid spacing
        parameters: Dictionary of PDE parameters
        residual_fn: Function computing PDE residual
        jvp_fn: Function computing Jacobian-vector products, or None for autodiff JVP
        tol: Convergence tolerance for Newton-Raphson method
        maxiter: Maximum number of Newton-Raphson iterations
        tol_gmres: GMRES tolerance
        maxiter_gmres: Maximum GMRES iterations per Newton step
        
    Returns:
        Solution at new time step
    """
    
    # Condition for exiting `while` loop
    def cond(state):
        iter_count, u_new, u_prev = state
        # Compute residual norm for proper Newton-Raphson convergence check
        residual = residual_fn(u_new, u_old, dt, dx, parameters)
        residual_norm = jnp.linalg.norm(residual)
        return jnp.logical_and(iter_count < maxiter, residual_norm > tol)

    # `while` loop body in Newton-Raphson method
    def body(state):
        iter_count, u_new, _ = state
        u_prev = u_new
        u_new = newton_raphson_step(
            u_new, u_old, dt, dx, parameters, residual_fn, jvp_fn, maxiter_gmres, tol_gmres
        )
        return iter_count + 1, u_new, u_prev

    # Initial guess
    u0 = u_old
    init_state = (0, u0, u0)

    # Iterate the Newton-Raphson method
    _, u_final, _ = jax.lax.while_loop(cond, body, init_state)

    return u_final


def solve(
    initial_condition: Callable,
    t_span: Tuple[float, float],
    residual_fn: Callable,
    parameters: Dict[str, Any],
    L: float,
    nx: int,
    jvp_fn: Optional[Callable] = None,
    dt: float = 0.01,
    tol: float = 1e-6,
    maxiter: int = 10,
    tol_gmres: float = 1e-6,
    maxiter_gmres: int = 20,
    save_every: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve a non-linear initial value problem with an implicit Euler method.

    The non-linear system at each time step is solved with a Newton-Raphson method.

    For optimal performance, it is recommended explicitly pass a function for computing 
    the Jacobian vector product (JVP) of the system (see the `README.md` for further details).
    
    Args:
        initial_condition: Callable function u0(x) that defines the initial condition
        t_span: (t_start, t_end) time interval
        residual_fn: Function with signature residual_fn(u_new, u_old, dt, dx, params) -> residual
        parameters: Dictionary of PDE parameters used in residual_fn and jvp_fn
        L: Domain length
        nx: Number of grid points
        jvp_fn: Optional function computing Jacobian-vector products:
                jvp_fn(u_new, u_old, dt, dx, params, v) -> J @ v
                If None, uses JAX automatic differentiation
        dt: Time step size
        tol: Convergence tolerance for Newton-Raphson method
        maxiter: Maximum Newton-Raphson iterations per time step
        tol_gmres: GMRES solver tolerance
        maxiter_gmres: Maximum GMRES iterations per Newton step
        save_every: Save solution every N time steps (1 = save all, default 1)
        
    Returns:
        Tuple of (time_array, spatial_grid, solution_array)
    """
    t_start, t_end = t_span
    
    # Determine boundary condition type
    bc_type = parameters.get('bc_type', None)
    if bc_type is None:
        raise ValueError("parameters must include 'bc_type' to determine proper grid spacing")
    
    # Create grid and evaluate initial condition
    x, dx = create_uniform_grid(L, nx, bc_type, return_spacing=True)
    u0 = initial_condition(x)
    
    # Set up time array
    nt = int((t_end - t_start) / dt) + 1
    t_full = jnp.linspace(t_start, t_end, nt)
    
    # Initialise storage lists
    u_saved = [u0]
    t_saved = [t_start]
    
    u_current = u0
    
    for step in range(1, nt):
        u_current = implicit_euler_step(
            u_current, dt, dx, parameters, residual_fn, jvp_fn,
            tol, maxiter, tol_gmres, maxiter_gmres
        )
        
        # Save solution every save_every steps or at the final step
        if step % save_every == 0 or step == nt - 1:
            u_saved.append(u_current)
            t_saved.append(t_full[step])
    
    # Convert lists to JAX arrays
    u_array = jnp.stack(u_saved, axis=0)
    t_array = jnp.array(t_saved)

    return t_array, x, u_array


# @partial(jax.jit, static_argnames=['residual_fn', 'jvp_fn', 'nt'])
# def _solve_jit(
#     initial_condition: jnp.ndarray,
#     dt: float,
#     dx: float,
#     parameters: Dict[str, Any],
#     residual_fn: Callable,
#     jvp_fn: Optional[Callable],
#     tol: float,
#     maxiter: int,
#     tol_gmres: float,
#     maxiter_gmres: int,
#     nt: int
# ) -> jnp.ndarray:
#     """
#     JIT-compiled function barrier for time-stepping loop in 'solve'.
    
#     The function barrier allows the scan loop to be compiled once per 
#     (nt, residual_fn, jvp_fn) combination, reducing recompilation overhead.
#     """
#     # Functional form in JAX 'scan' loop
#     def scan_step(u_curr, _):
#         """Single time step for lax.scan."""
#         u_new = implicit_euler_step(
#             u_curr, dt, dx, parameters, residual_fn, jvp_fn,
#             tol, maxiter, tol_gmres, maxiter_gmres
#         )
#         return u_new, u_new
    
#     # Loop over time steps
#     _, u_history = jax.lax.scan(scan_step, initial_condition, jnp.arange(nt - 1))
    
#     # Stack initial condition with computed history
#     u = jnp.concatenate([initial_condition[None, :], u_history], axis=0)
    
#     return u