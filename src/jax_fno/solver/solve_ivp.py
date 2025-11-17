import time
from functools import singledispatch

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional

from .timesteppers.base import AbstractStepper
from .timesteppers.explicit import ExplicitStepper
from .timesteppers.implicit import ImplicitStepper
from .linearsolver import default_linear_solver


def _integrate_explicit(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: ExplicitStepper,
) -> Tuple[jnp.ndarray, float]:
    """
    Time-integration with explicit stepping schemes.

    Args:
        f: RHS function: du/dt = f(u, t)
        u: Current solution at time t
        t: Current time
        dt: Time step size
    """

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    def body_fn(i, carry):
        u, t = carry
        u_next = stepper.step(f, u, t, dt)
        t_next = t + dt
        return (u_next, t_next)

    u_final, t_final = jax.lax.fori_loop(0, n_steps, body_fn, (u0, t_start))
    return u_final, t_final


def _integrate_implicit(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: ImplicitStepper,
    **stepper_kwargs,
) -> Tuple[jnp.ndarray, float]:
    """
    Time-integration with implicit stepping schemes.

    Args:
        f: RHS function: du/dt = f(u, t)
        u: Current solution at time t
        t: Current time
        dt: Time step size

    Kwargs:
        tol: Convergence tolerance in Newton-Raphson iterations. 
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. 
            Default: 50.
        df: Jacobian-vector product of f.
            df(u, t) must return a function with signature v -> (∂f/∂u)*v.
            Defaults to using JAX's automatic differentiation ('jax.jvp').
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres') 
                with tol=1e-6 and maxiter=100.
    """
    allowed_kwargs = ["tol", "maxiter", "linsolver", "df"]
    for k in stepper_kwargs.keys():
        if k not in allowed_kwargs:
            raise ValueError(
                f"Unexpected keyword argument '{k}'. "
                f"Valid options are: {', '.join(allowed_kwargs)}."
            )

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    # Get optional arguments
    tol = stepper_kwargs.get("tol", 1e-6)
    maxiter = stepper_kwargs.get("maxiter", 50)
    linsolver = stepper_kwargs.get("linsolver", default_linear_solver())

    # TODO: fix bug here
    def df_default(u: jnp.ndarray, t: float):
        return lambda v: jax.jvp(f, (u, t), (v,))[1]

    df = stepper_kwargs.get("df", df_default)

    def body_fn(i, carry):
        u, t = carry
        u_next = stepper.step(
            f, u, t, dt, df=df, linsolver=linsolver, tol=tol, maxiter=maxiter
        )
        t_next = t + dt
        return (u_next, t_next)

    u_final, t_final, _ = jax.lax.fori_loop(0, n_steps, body_fn, (u0, t_start))

    return u_final, t_final


# Use single-dispatch to route to the appropriate integration scheme 
# based on the stepper type (explicit vs implicit)
@singledispatch
def integrate(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: AbstractStepper,
    **kwargs,
) -> Tuple[jnp.ndarray, float]:
    """
    Integrate du/dt = f(u, t) from t=t_start to t=t_end.

    Args:
        f: Callable right-hand side of IVP du/dt = f(u, t)
        u0: Initial condition
        t_span: (t_start, t_end) time interval
        dt: Time step size
        stepper: Time-stepping scheme instance. Default: BackwardEuler()

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in the Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. Default: 50.
        df: Jacobian-vector product of f.
            df(u, t) must return a function with signature v -> (∂f/∂u)*v.
            Defaults to using JAX's automatic differentiation ('jax.jvp').
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres')
                with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        u_final: Solution at t_end
        t_final: Final time
    """
    raise ValueError(
        f"Unknown stepper type: {type(stepper)}. "
        f"Expected ExplicitStepper or ImplicitStepper instance."
    )


@integrate.register
def _(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: ExplicitStepper,
    **kwargs,
) -> Tuple[jnp.ndarray, float]:
    """Dispatch to explicit integration routine."""
    return _integrate_explicit(f, u0, t_span, dt, stepper)


@integrate.register
def _(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: ImplicitStepper,
    **kwargs,
) -> Tuple[jnp.ndarray, float]:
    """Dispatch to implicit integration routine."""
    return _integrate_implicit(f, u0, t_span, dt, stepper, **kwargs)


def solve_ivp(
    f: Callable[[jnp.ndarray, float], jnp.ndarray],
    u0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    stepper: AbstractStepper,
    save_every: Optional[int] = None,
    verbose: bool = False,
    **stepper_kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve an initial value problem of the form du/dt = f(u, t).

    Args:
        f: Right-hand side function with signature (u, t) -> dudt
        u0: Initial condition
        t_span: (t_start, t_end) time interval
        dt: Time step size
        method: Time-stepping method object (e.g., RK4(), BackwardEuler())
        save_every: Save every N steps. 
            If None, saves only initial and final states.
        verbose: Print progress information

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in Newton-Raphson iterations. 
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. 
            Default: 50.
        df: Jacobian-vector product of f.
            df(u, t) must return a function with signature v -> (∂f/∂u)*v.
            Defaults to using JAX's automatic differentiation ('jax.jvp').
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres') 
                with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        t: Array of snapshot times, shape (n_snapshots,)
        u: Array of solution snapshots, shape (n_snapshots, *u0.shape)
    """
    t_start, t_end = t_span
    n_steps_total = int((t_end - t_start) / dt)

    if save_every is None:
        save_every = n_steps_total

    if verbose:
        method_name = type(stepper).__name__
        print(f"Solving with {method_name}")
        print(
            f"Time: [{t_start}, {t_end}], dt={dt}, {n_steps_total} total steps"
        )
        print(f"Saving every {save_every} steps")

    start_time = time.time()

    # Initialise storage
    t_save = [t_start]
    u_save = [u0]

    # Integrate in chunks
    n_chunks = n_steps_total // save_every
    u, t = u0, t_start

    for _ in range(n_chunks):
        # Compute chunk time span
        t_chunk_end = t + save_every * dt
        u, t = integrate(f, u, (t, t_chunk_end), dt, stepper, **stepper_kwargs)
        t_save.append(t)
        u_save.append(u)

    # Convert to arrays
    t_arr = jnp.array(t_save)
    u_arr = jnp.stack(u_save, axis=0)

    elapsed_time = time.time() - start_time

    if verbose:
        print(
            f"Completed in {elapsed_time:.3f}s"
            +
            f"at ({n_steps_total / elapsed_time:.1f} steps/s)"
        )

    return t_arr, u_arr
