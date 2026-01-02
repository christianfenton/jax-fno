import time
from typing import Callable, Tuple, Optional

import jax
from jax import Array
import jax.numpy as jnp

from .timesteppers import StepperProtocol


def solve_ivp(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Array,
    method: StepperProtocol,
    step_size: float,
    args: tuple = ()
) -> Tuple[float, Array]:
    """
    Integrate dy/dt = fun(t, y, *args) over the time interval t_span.

    Args:
        fun: Callable right-hand side of system dy/dt = fun(t, y, *args)
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        method: Time-stepping method instance (e.g., RK4(), BackwardEuler())
        step_size: Time step size
        args: Additional arguments to pass to fun (and jvp/jac if provided)

    Returns:
        t_final: Final time
        y_final: Solution at t_end

    Example usage:
    ```python
    import jax.numpy as jnp
    from jax_fno.integrate import solve_ivp, RK4

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    # Solve with args
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    k = 0.5

    t, y = solve_ivp(fun, t_span, y0, RK4(), step_size=0.01, args=(k,))
    ```

    Example usage with an implicit method and user-defined parameters:
    ```python
    import jax.numpy as jnp
    from jax_fno.integrate import solve_ivp, NewtonRaphson, GMRES, BackwardEuler

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
    t, y = solve_ivp(fun, t_span, y0, method, step_size=0.01, args=(k,))
    ```
    """
    t_start, t_end = t_span

    def cond_fn(carry):
        t, y, _ = carry
        return t < t_end

    def body_fn(carry):
        t, y, m = carry

        # Adjust final step to hit t_end exactly
        h = jax.lax.max(0.0, jax.lax.min(step_size, t_end - t))

        y_next = m.step(fun, t, y, h, args)

        t_next = t + h

        return (t_next, y_next, m)

    t_final, y_final, _ = jax.lax.while_loop(cond_fn, body_fn, (t_start, y0, method))

    return t_final, y_final


def solve_with_history(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Array,
    method: StepperProtocol,
    step_size: float,
    t_eval: Optional[Array] = None,
    args: tuple = (),
    verbose: bool = False
) -> Tuple[Array, Array]:
    """
    Integrate dy/dt = fun(t, y, *args) over the time interval t_span.

    This function allows users to return intermediate states at times `t_eval`,
    but is not compatible with JAX transformations. The integration is done in 
    chunks by calling `solve_ivp`, where `solve_ivp` has been JIT-compiled.

    Args:
        fun: Right-hand side function with signature (t, y, *args) -> dydt
        t_span: (t_start, t_end) time interval
        y0: Initial condition
        method: Time-stepping method instance (e.g., RK4(), BackwardEuler())
        step_size: Time step size for integration.
        t_eval: Times at which to store the computed solution.
            If None, returns only the initial and final states.
            Must be sorted and lie within t_span.
        args: Additional arguments to pass to fun (and jvp/jac if provided)
        verbose: Print progress information

    Returns:
        t: Array of time points, shape (n_points,)
        y: Array of solution values at times t, shape (n_points, *y0.shape)

    Example usage:
    ```python
    import jax.numpy as jnp
    from jax_fno.integrate import solve_with_history, RK4

    # Define ODE: dy/dt = -k*y
    def fun(t, y, k):
        return -k * y

    # Solve with args
    y0 = jnp.array([1.0])
    t_span = (0.0, 2.0)
    t_eval = jnp.linspace(0, 2, 5)
    k = 0.5

    t, y = solve_with_history(
        fun, t_span, y0, RK4(), step_size=0.01, t_eval=t_eval, args=(k,)
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

    n_steps_total = int(jnp.ceil((t_end - t_start) / step_size))

    if verbose:
        method_name = type(method).__name__
        print(f"Solving with {method_name}")
        print(
            f"Time: [{t_start}, {t_end}], dt={step_size}, "
            f"~{n_steps_total} total steps"
        )
        print(f"Evaluating at {len(t_eval)} time points")

    integrate_jit = jax.jit(solve_ivp, static_argnames=['fun', 'method'])

    # Integrate between consecutive evaluation points:
    y_save = [y0]
    t_save = [t_start]
    y = y0

    start_wallclock = time.time()

    for i in range(len(t_eval) - 1):
        t_i = float(t_eval[i])
        t_ip1 = float(t_eval[i + 1])
        t, y = integrate_jit(fun, (t_i, t_ip1), y, method, step_size, args)
        t_save.append(t)
        y_save.append(y)

    t_arr = jnp.stack(t_save)
    y_arr = jnp.stack(y_save, axis=0)

    elapsed_wallclock = time.time() - start_wallclock

    if verbose:
        print(
            f"Completed in {elapsed_wallclock:.3f}s "
            f"({n_steps_total / elapsed_wallclock:.1f} steps/s)"
        )

    return t_arr, y_arr
