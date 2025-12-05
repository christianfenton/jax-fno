import time
from typing import Callable, Tuple, Optional

import jax
from jax import Array
import jax.numpy as jnp

from .timesteppers.base import AbstractStepper

def integrate(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Array,
    method: AbstractStepper,
    dt: float,
    args: tuple = ()
) -> Tuple[float, Array]:
    """
    Integrate dy/dt = fun(t, y, *args) over the time interval t_span.

    Args:
        fun: Callable right-hand side of system dy/dt = fun(t, y, *args)
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        method: Time-stepping method instance (e.g., RK4(), BackwardEuler())
        dt: Time step size
        args: Additional arguments to pass to fun (and jvp/jac if provided)

    Returns:
        t_final: Final time
        y_final: Solution at t_end

    Example usage:
    ```python
    import jax.numpy as jnp
    from jax_fno.solver import integrate, RK4

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    # Solve with args
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    k = 0.5

    t, y = integrate(fun, t_span, y0, RK4(), dt=0.01, args=(k,))
    ```

    Example usage with an implicit method and user-defined parameters:
    ```python
    import jax.numpy as jnp
    from jax_fno.solver import integrate, NewtonRaphson, GMRES, BackwardEuler

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    # Create linear solver
    linsolver = GMRES(tol=1e-6, maxiter=20)

    # Create non-linear solver
    root_finder = NewtonRaphson(linsolver=linsolver, tol=1e-5, maxiter=50)

    # Choose integration method
    method = BackwardEuler(root_finder=root_finder)

    # Solve
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    k = 0.5
    t, y = integrate(fun, t_span, y0, method, dt=0.01, args=(k,))
    ```
    """
    t_start, t_end = t_span

    def cond_fn(carry):
        t, y = carry
        return t < t_end

    def body_fn(carry):
        t, y = carry

        # Adjust final step to hit t_end exactly
        dt_step = jax.lax.max(0.0, jax.lax.min(dt, t_end - t))

        y_next = method.step(fun, t, y, dt_step, args)

        t_next = t + dt_step

        return (t_next, y_next)

    t_final, y_final = jax.lax.while_loop(cond_fn, body_fn, (t_start, y0))

    return t_final, y_final


def solve_ivp(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Array,
    method: AbstractStepper,
    t_eval: Optional[Array] = None,
    dt: float = 0.01,
    args: tuple = (),
    verbose: bool = False
) -> Tuple[Array, Array]:
    """
    Integrate dy/dt = fun(t, y, *args) over the time interval t_span.

    Args:
        fun: Right-hand side function with signature (t, y, *args) -> dydt
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        method: Time-stepping method instance (e.g., RK4(), BackwardEuler())
        t_eval: Times at which to store the computed solution.
            If None, returns only the initial and final states.
            Must be sorted and lie within t_span.
            Warning: Storing states at intermediate steps triggers
                recompilation and worsens performance.
        dt: Time step size for integration. Default: 0.01
        args: Additional arguments to pass to fun (and jvp/jac if provided)
        verbose: Print progress information

    Returns:
        t: Array of time points, shape (n_points,)
        y: Array of solution values at times t, shape (n_points, *y0.shape)

    Example usage:
    ```python
    import jax.numpy as jnp
    from jax_fno.solver import solve_ivp, RK4

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    # Solve with args
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    t_eval = jnp.linspace(0, 2, 5)
    k = 0.5

    t, y = solve_ivp(fun, t_span, y0, RK4(), t_eval=t_eval, dt=0.01, args=(k,))
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
        method_name = type(method).__name__
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
        t_i = float(t_eval[i])
        t_ip1 = float(t_eval[i + 1])
        t, y = integrate(fun, (t_i, t_ip1), y, method, dt, args)
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
