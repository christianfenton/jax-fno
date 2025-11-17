from typing import Callable

import jax
import jax.numpy as jnp


def newton_raphson(
    u0: jnp.ndarray,
    R: Callable[[jnp.ndarray], jnp.ndarray],
    jvp: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    linsolver: Callable[[Callable, jnp.ndarray], jnp.ndarray],
    tol: float,
    maxiter: int,
) -> jnp.ndarray:
    """
    Solve the non-linear system R(u) = 0 using a Newton-Raphson method.

    Iterative update: u <-- u - J^{-1}(u) * R(u)

    Args:
        u0: Initial guess
        R: A function returning the residual R(u)
        jvp: Jacobian-vector product with signature (u, v) -> J(u) * v
        linsolver_fn: Linear solver: (J_op, rhs) -> solution
        tol: Convergence tolerance. Default: 1e-6.
        maxiter: Maximum number of iterations. Default: 50.

    Returns:
        Final solution u
    """

    u_k = u0
    r_k = R(u_k)
    state0 = (u_k, r_k, 0)

    def body_fun(state):
        """
        Update the solution by solving linear system: J(u_k) * delta = -r_k
        """
        u_k, r_k, k = state
        delta = linsolver(lambda v: jvp(u_k, v), -r_k)
        u_kp1 = u_k + delta
        r_kp1 = R(u_kp1)
        return (u_kp1, r_kp1, k + 1)

    def cond_fun(state):
        _, r_k, k = state
        return (jnp.linalg.norm(r_k) > tol) & (k < maxiter)

    u_final, _, niters = jax.lax.while_loop(cond_fun, body_fun, state0)

    def callback(iters, maxiter):
        if iters >= maxiter:
            print(
                "Newton-Raphson method did not converge"
                + f"within {int(maxiter)} iterations."
            )
        return None

    jax.pure_callback(callback, None, niters, maxiter)

    return u_final
