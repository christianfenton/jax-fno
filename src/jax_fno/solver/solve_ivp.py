import time

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
        u0: Initial condition
        t_span: (t_start, t_end) time interval
        dt: Time step size
        stepper: Explicit time-stepping scheme

    Returns:
        u_final: Solution at t_end
        t_final: Final time
    """
    t_start, t_end = t_span

    def cond_fn(carry):
        u, t = carry
        return t < t_end

    def body_fn(carry):
        u, t = carry
        # Adjust final step to hit t_end exactly
        dt_step = jnp.maximum(0.0, jnp.minimum(dt, t_end - t))
        u_next = stepper.step(f, u, t, dt_step)
        t_next = t + dt_step
        return (u_next, t_next)

    u_final, t_final = jax.lax.while_loop(cond_fn, body_fn, (u0, t_start))
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
        u0: Initial condition
        t_span: (t_start, t_end) time interval
        dt: Time step size
        stepper: Implicit time-stepping scheme

    Kwargs:
        tol: Convergence tolerance in Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations.
            Default: 50.
        jvp: Jacobian-vector product with signature (u, t, v) -> (∂f/∂u)*v.
            Computes the action of the Jacobian ∂f/∂u on vector v.
            Defaults to using JAX's automatic differentiation.
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres')
                with tol=1e-6 and maxiter=100.

    Returns:
        u_final: Solution at t_end
        t_final: Final time
    """
    allowed_kwargs = ["tol", "maxiter", "linsolver", "jvp"]
    for k in stepper_kwargs.keys():
        if k not in allowed_kwargs:
            raise ValueError(
                f"Unexpected keyword argument '{k}'. "
                f"Valid options are: {', '.join(allowed_kwargs)}."
            )

    t_start, t_end = t_span

    # Get optional arguments
    tol = stepper_kwargs.get("tol", 1e-6)
    maxiter = stepper_kwargs.get("maxiter", 50)
    linsolver = stepper_kwargs.get("linsolver", default_linear_solver())

    # Default JVP using JAX automatic differentiation
    def jvp_default(u: jnp.ndarray, t: float, v: jnp.ndarray) -> jnp.ndarray:
        """Compute (∂f/∂u)*v using automatic differentiation."""
        return jax.jvp(lambda u_: f(u_, t), (u,), (v,))[1]

    jvp = stepper_kwargs.get("jvp", jvp_default)

    def cond_fn(carry):
        u, t = carry
        return t < t_end

    def body_fn(carry):
        u, t = carry
        # Adjust final step to hit t_end exactly
        # max(0, ...) prevents negative dt from floating-point errors
        dt_step = jnp.maximum(0.0, jnp.minimum(dt, t_end - t))
        u_next = stepper.step(
            f, u, t, dt_step,
            jvp=jvp, linsolver=linsolver, tol=tol, maxiter=maxiter
        )
        t_next = t + dt_step
        return (u_next, t_next)

    u_final, t_final = jax.lax.while_loop(cond_fn, body_fn, (u0, t_start))

    return u_final, t_final


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
        stepper: Time-stepping scheme instance

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in the Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. Default: 50.
        jvp: Jacobian-vector product function (u, t, v) -> (∂f/∂u)*v.
            Computes the action of the Jacobian ∂f/∂u on vector v.
            Defaults to using JAX's automatic differentiation.
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres')
                with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        u_final: Solution at t_end
        t_final: Final time
    """
    if isinstance(stepper, ExplicitStepper):
        return _integrate_explicit(f, u0, t_span, dt, stepper)
    elif isinstance(stepper, ImplicitStepper):
        return _integrate_implicit(f, u0, t_span, dt, stepper, **kwargs)
    else:
        raise ValueError(
            f"Unknown stepper type: {type(stepper)}. "
            f"Expected ExplicitStepper or ImplicitStepper instance."
        )


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
            Intermediate saves trigger recompilation and may 
                significantly worsen performance.
        verbose: Print progress information

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations.
            Default: 50.
        jvp: Jacobian-vector product function (u, t, v) -> (∂f/∂u)*v.
            Computes the action of the Jacobian ∂f/∂u on vector v.
            Defaults to using JAX's automatic differentiation.
        linsolver: Linear solver used inside the Newton-Raphson method.
            Must be a matrix-free iterative method.
            Default: GMRES ('jax.scipy.sparse.linalg.gmres')
                with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        u: Array of solution snapshots, shape (n_snapshots, *u0.shape)
        t: Array of snapshot times, shape (n_snapshots,)

    Example usage with custom Jacobian-vector product:
    ```python
    import jax.numpy as jnp
    from jax_fno.solver import solve_ivp, BackwardEuler

    # Define PDE right-hand side
    def heat_eq(u, t):
        dx = 0.01
        diffusivity = 0.1
        # Laplacian with periodic BC
        d2u = (jnp.roll(u, -1) - 2*u + jnp.roll(u, 1)) / dx**2
        return diffusivity * d2u

    # Define Jacobian-vector product of right-hand side
    def heat_jvp(u, t, v):
        '''Compute (∂f/∂u)*v for heat equation.'''
        dx = 0.01
        diffusivity = 0.1
        d2v = (jnp.roll(v, -1) - 2*v + jnp.roll(v, 1)) / dx**2
        return diffusivity * d2v

    # Solve
    u0 = jnp.sin(2 * jnp.pi * jnp.linspace(0, 1, 100))
    u, t = solve_ivp(
        heat_eq, u0, (0, 1), dt=0.01,
        stepper=BackwardEuler(),
        jvp=heat_jvp
    )
    ```
    """
    t_start, t_end = t_span

    # Compute minimum number of time steps required
    n_steps_total = int(jnp.ceil((t_end - t_start) / dt))

    if save_every is None:
        save_every = n_steps_total

    # Compute times where we save intermediate states (including start time)
    n_saves = max(2, int(jnp.ceil(n_steps_total / save_every)) + 1)
    t_saves = jnp.linspace(t_start, t_end, n_saves)

    if verbose:
        method_name = type(stepper).__name__
        print(f"Solving with {method_name}")
        print(
            f"Time: [{t_start}, {t_end}], dt={dt}, "
            f"~{n_steps_total} total steps"
        )
        print(f"Saving at {len(t_saves)} time points")

    start_time = time.time()

    # Integrate between consecutive save points
    u_save = [u0]  # initialise storage
    u = u0
    for i in range(len(t_saves) - 1):
        t_chunk_start = t_saves[i]
        t_chunk_end = t_saves[i + 1]
        u, _ = integrate(
            f, u, (t_chunk_start, t_chunk_end), dt, stepper, **stepper_kwargs
        )
        u_save.append(u)

    u_arr = jnp.stack(u_save, axis=0)  # convert to arrays

    elapsed_time = time.time() - start_time

    if verbose:
        print(
            f"Completed in {elapsed_time:.3f}s "
            f"({n_steps_total / elapsed_time:.1f} steps/s)"
        )

    return u_arr, t_saves
