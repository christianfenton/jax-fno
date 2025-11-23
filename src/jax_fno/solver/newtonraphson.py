from typing import Callable, Optional

import jax
import jax.numpy as jnp


def newton_raphson(
    y0: jnp.ndarray,
    R: Callable[[jnp.ndarray], jnp.ndarray],
    jac_or_jvp: Callable,
    linsolver: Optional[Callable[[Callable, jnp.ndarray], jnp.ndarray]],
    tol: float,
    maxiter: int,
    dense: bool = False,
) -> jnp.ndarray:
    """
    Solve the non-linear system R(y) = 0 using a Newton-Raphson method.

    Iterative update:
    $$y \\leftarrow y - J^{-1}(y) * R(y)$$

    Supports both dense Jacobian (direct solve) and matrix-free JVP 
    (iterative solve).

    Args:
        y0: Initial guess
        R: A function returning the residual R(y)
        jac_or_jvp: Can be either:
            - Dense mode: A function with signature y -> J(y)
            - Matrix-free mode: A function with signature (y, v) -> J(y) * v
        linsolver: Linear solver for matrix-free mode: (J_op, rhs) -> solution
            Required when dense=False, ignored when dense=True
        tol: Convergence tolerance
        maxiter: Maximum number of iterations
        dense: If True, use dense Jacobian with direct solve.
               If False, use JVP with iterative solve.

    Returns:
        Final solution y
    """

    y_k = y0
    r_k = R(y_k)
    state0 = (y_k, r_k, 0)

    if dense:
        # Dense Jacobian path - use direct solve
        def body_fun(state):
            y_k, r_k, k = state
            J_k = jac_or_jvp(y_k)  # Get dense Jacobian matrix
            delta = jnp.linalg.solve(J_k, -r_k)  # Direct solve
            y_kp1 = y_k + delta
            r_kp1 = R(y_kp1)
            return (y_kp1, r_kp1, k + 1)
    else:
        # Matrix-free JVP path - use iterative solve
        if linsolver is None:
            raise ValueError("linsolver must be provided when dense=False")

        def body_fun(state):
            y_k, r_k, k = state
            delta = linsolver(lambda v: jac_or_jvp(y_k, v), -r_k)
            y_kp1 = y_k + delta
            r_kp1 = R(y_kp1)
            return (y_kp1, r_kp1, k + 1)

    def cond_fun(state):
        _, r_k, k = state
        return (jnp.linalg.norm(r_k) > tol) & (k < maxiter)

    y_final, _, niters = jax.lax.while_loop(cond_fun, body_fun, state0)

    def callback(iters, maxiter):
        if iters >= maxiter:
            print(
                "Newton-Raphson method did not converge "
                + f"within {int(maxiter)} iterations."
            )
        return None

    jax.pure_callback(callback, None, niters, maxiter)

    return y_final
