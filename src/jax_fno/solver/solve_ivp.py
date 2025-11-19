import time

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional

from .timesteppers.base import AbstractStepper
from .timesteppers.explicit import ExplicitStepper
from .timesteppers.implicit import ImplicitStepper
from .linearsolver import default_linear_solver


def _integrate_explicit(
    fun: Callable[[float, jnp.ndarray], jnp.ndarray],
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    stepper: ExplicitStepper,
    dt: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Time-integration with explicit stepping schemes.

    Args:
        fun: Right-hand side of the system dy/dt = fun(t, y)
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        stepper: Integration method to use (ExplicitStepper instance)
        dt: Time step size

    Returns:
        y_final: Final solution
        t_final: Final time
    """
    t_start, t_end = t_span

    def cond_fn(carry):
        t, y = carry
        return t < t_end

    def body_fn(carry):
        t, y = carry
        # Adjust final step to hit t_end exactly
        dt_step = jnp.maximum(0.0, jnp.minimum(dt, t_end - t))
        y_next = stepper.step(fun, t, y, dt_step)
        t_next = t + dt_step
        return (t_next, y_next)

    t_final, y_final = jax.lax.while_loop(cond_fn, body_fn, (t_start, y0))
    return y_final, t_final


def _integrate_implicit(
    fun: Callable[[float, jnp.ndarray], jnp.ndarray],
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    stepper: ImplicitStepper,
    dt: float,
    **stepper_kwargs,
) -> Tuple[float, jnp.ndarray]:
    """
    Time-integration with implicit stepping schemes.

    Args:
        fun: Right-hand side of the system dy/dt = fun(t, y)
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        stepper: Implicit time-stepping scheme
        dt: Time step size

    Kwargs:
        tol: Convergence tolerance in Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations.
            Default: 50.
        jvp: Jacobian-vector product with signature (t, y, v) -> (∂f/∂y)*v.
            Computes the action of the Jacobian ∂f/∂y on a vector v.
            If None, defaults to JAX's automatic differentiation (`jax.jvp`). 
        jac: Evaluates the Jacobian at (t, y), returning a dense matrix.
            If provided, systems are solved directly 
            with `jax.numpy.linalg.solve`)
        linsolver: Linear solver used inside the Newton-Raphson method.
            Required for matrix-free (jvp) mode.
            Default: GMRES with tol=1e-6 and maxiter=100.

    Returns:
        t_final: Final time
        y_final: Solution at t_end
    """
    allowed_kwargs = ["tol", "maxiter", "linsolver", "jvp", "jac"]
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
    jac = stepper_kwargs.get("jac", None)
    jvp = stepper_kwargs.get("jvp", None)

    # Set up Jacobian (either dense or matrix-free)
    if jac is None and jvp is None:
        # Default: use JAX automatic differentiation for JVP
        def jvp_default(
            t: float, y: jnp.ndarray, v: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute (∂f/∂y)*v using automatic differentiation."""
            return jax.jvp(lambda y_: fun(t, y_), (y,), (v,))[1]

        jvp = jvp_default

    # Get linear solver (only needed for matrix-free mode)
    linsolver = stepper_kwargs.get("linsolver", None)
    if jac is None and linsolver is None:
        linsolver = default_linear_solver()

    def cond_fn(carry):
        t, y = carry
        return t < t_end

    def body_fn(carry):
        t, y = carry

        # Adjust final step to hit t_end exactly
        dt_step = jnp.maximum(0.0, jnp.minimum(dt, t_end - t))

        y_next = stepper.step(
            fun,
            t,
            y,
            dt_step,
            jvp=jvp,
            jac=jac,
            linsolver=linsolver,
            tol=tol,
            maxiter=maxiter,
        )

        t_next = t + dt_step

        return (t_next, y_next)

    t_final, y_final = jax.lax.while_loop(cond_fn, body_fn, (t_start, y0))

    return t_final, y_final


def integrate(
    fun: Callable[[float, jnp.ndarray], jnp.ndarray],
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    stepper: AbstractStepper,
    dt: float,
    **kwargs,
) -> Tuple[float, jnp.ndarray]:
    """
    Integrate dy/dt = fun(t, y) from t=t_start to t=t_end.

    Args:
        fun: Callable right-hand side of system dy/dt = fun(t, y)
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        stepper: Time-stepping scheme instance
        dt: Time step size

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in the Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. Default: 50.
        jvp: Jacobian-vector product with signature (t, y, v) -> (∂f/∂y)*v.
            Computes the action of the Jacobian ∂f/∂y on a vector v.
            If None, defaults to JAX's automatic differentiation (`jax.jvp`). 
        jac: Evaluates the Jacobian at (t, y), returning a dense matrix.
            If provided, systems are solved directly 
            with `jax.numpy.linalg.solve`)
        linsolver: Linear solver used inside the Newton-Raphson method.
            Required for matrix-free (jvp) mode.
            Default: GMRES with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        t_final: Final time
        y_final: Solution at t_end
    """
    if isinstance(stepper, ExplicitStepper):
        return _integrate_explicit(fun, t_span, y0, stepper, dt)
    elif isinstance(stepper, ImplicitStepper):
        return _integrate_implicit(fun, t_span, y0, stepper, dt, **kwargs)
    else:
        raise ValueError(
            f"Unknown stepper type: {type(stepper)}. "
            f"Expected ExplicitStepper or ImplicitStepper instance."
        )


def solve_ivp(
    fun: Callable[[float, jnp.ndarray], jnp.ndarray],
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    stepper: AbstractStepper,
    t_eval: Optional[jnp.ndarray] = None,
    dt: float = 0.01,
    verbose: bool = False,
    **options,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve an initial value problem for a system of ODEs.

    Solves the initial value problem dy/dt = fun(t, y) with y(t0) = y0.

    Args:
        fun: Right-hand side function with signature (t, y) -> dydt
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        stepper: Time-stepping method (e.g., RK4(), BackwardEuler())
        t_eval: Times at which to store the computed solution.
            If None, returns only the initial and final states.
            Must be sorted and lie within t_span.
        dt: Time step size for integration. Default: 0.01
        verbose: Print progress information

    For implicit methods, the following keyword arguments are available:
        tol: Convergence tolerance in Newton-Raphson iterations.
            Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations.
            Default: 50.
        jvp: Jacobian-vector product with signature (t, y, v) -> (∂f/∂y)*v.
            Computes the action of the Jacobian ∂f/∂y on a vector v.
            If None, defaults to JAX's automatic differentiation (`jax.jvp`). 
        jac: Evaluates the Jacobian at (t, y), returning a dense matrix.
            If provided, systems are solved directly 
            with `jax.numpy.linalg.solve`)
        linsolver: Linear solver used inside the Newton-Raphson method.
            Required for matrix-free (jvp) mode.
            Default: GMRES with tol=1e-6 and maxiter=100.

    Keyword arguments passed to explicit time-steppers are ignored.

    Returns:
        t: Array of time points, shape (n_points,)
        y: Array of solution values at times t, shape (n_points, *y0.shape)

    Example usage:
    ```python
    import jax.numpy as jnp
    from jax_fno.solver import solve_ivp, RK4

    # Define ODE: dy/dt = -y
    def fun(t, y):
        return -y

    # Solve
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    t_eval = jnp.linspace(0, 2, 5)

    t, y = solve_ivp(fun, t_span, y0, method=RK4(), t_eval=t_eval, dt=0.01)
    ```

    Example with implicit method and dense Jacobian:
    ```python
    from jax_fno.solver import BackwardEuler

    # Define Jacobian matrix
    def jac(t, y):
        return jnp.array([[-1.0]])  # ∂f/∂y for f(t,y) = -y

    t, y = solve_ivp(
        fun, t_span, y0,
        method=BackwardEuler(),
        t_eval=t_eval,
        dt=0.01,
        jac=jac  # Use dense Jacobian with direct solve
    )
    ```
    """
    t_start, t_end = t_span

    # Set up evaluation times
    if t_eval is None:
        # Only save initial and final states
        t_eval = jnp.array([t_start, t_end])
    else:
        # Validate t_eval
        t_eval = jnp.asarray(t_eval)
        if jnp.any(t_eval < t_start) or jnp.any(t_eval > t_end):
            raise ValueError("All values in t_eval must be within t_span")
        if jnp.any(jnp.diff(t_eval) < 0):
            raise ValueError("t_eval must be sorted in increasing order")

        # Ensure t_start is included
        if t_eval[0] != t_start:
            t_eval = jnp.concatenate([jnp.array([t_start]), t_eval])

    n_steps_total = int(jnp.ceil((t_end - t_start) / dt))

    if verbose:
        method_name = type(stepper).__name__
        print(f"Solving with {method_name}")
        print(
            f"Time: [{t_start}, {t_end}], dt={dt}, "
            f"~{n_steps_total} total steps"
        )
        print(f"Evaluating at {len(t_eval)} time points")

    start_time = time.time()

    # Integrate between consecutive evaluation points
    y_save = [y0]
    t_save = [t_start]
    y = y0

    for i in range(len(t_eval) - 1):
        t_chunk_start = t_eval[i]
        t_chunk_end = t_eval[i + 1]
        t, y = integrate(
            fun, (t_chunk_start, t_chunk_end), y, stepper, dt, **options
        )
        t_save.append(t)
        y_save.append(y)

    t_arr = jnp.stack(t_save)
    y_arr = jnp.stack(y_save, axis=0)

    elapsed_time = time.time() - start_time

    if verbose:
        print(
            f"Completed in {elapsed_time:.3f}s "
            f"({n_steps_total / elapsed_time:.1f} steps/s)"
        )

    return t_arr, y_arr
