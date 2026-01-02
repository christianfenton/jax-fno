"""Newton-Raphson method for root finding."""

from typing import Optional

from flax import nnx
import jax
from jax import Array
import jax.numpy as jnp

from ..linsolvers import LinearSolverProtocol, GMRES
from ..custom_types import LinearMap, JVPConstructor, JacobianConstructor


class NewtonRaphson(nnx.Module):
    """
    Newton-Raphson root-finding algorithm.

    Iterative update: $y \\leftarrow y - J^{-1}(y) R(y)$

    Implements: RootFinderProtocol

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of Newton-Raphson iterations
        linsolver: Linear solver for inner iterations (default: GMRES)
    """

    def __init__(
        self,
        tol: float = 1e-6,
        maxiter: int = 50,
        linsolver: LinearSolverProtocol = GMRES()
    ):
        self.tol = tol
        self.maxiter = maxiter
        self.linsolver = linsolver

    def __call__(
        self,
        residual_fn: LinearMap,
        y_guess: Array,
        jvp_fn: Optional[JVPConstructor] = None,
        jac_fn: Optional[JacobianConstructor] = None,
    ) -> Array:
        """
        Find the root of residual_fn(y) = 0 using Newton-Raphson method.

        Args:
            residual_fn: Residual function R(y)
            y_guess: Initial guess
            jvp_fn: Matrix-free Jacobian-vector product with
                signature (y, v) -> J(y)*v
            jac_fn: Function returning a dense Jacobian matrix with
                signature y -> J(y)

        Returns:
            Solution y
        """
        if jac_fn is not None and jvp_fn is not None:
            raise ValueError("Provide either jac_fn OR jvp_fn, not both.")

        # State carried through Newton iterations
        y_k = y_guess
        r_k = residual_fn(y_k)
        state0 = (y_k, r_k, 0)

        # Define body for a single Newton iteration
        if jac_fn is not None:
            # Dense mode
            def body_fun(state):
                y_k, r_k, k = state
                J = jac_fn(y_k)  # construct dense matrix
                delta = self.linsolver(J, -r_k)
                y_kp1 = y_k + delta
                r_kp1 = residual_fn(y_kp1)
                return (y_kp1, r_kp1, k + 1)
        elif jvp_fn is not None:
            # Matrix-free mode
            def body_fun(state):
                y_k, r_k, k = state
                jvp = lambda v : jvp_fn(y_k, v)  # define matrix-free operator
                delta = self.linsolver(jvp, -r_k)
                y_kp1 = y_k + delta
                r_kp1 = residual_fn(y_kp1)
                return (y_kp1, r_kp1, k + 1)
        else:
            raise ValueError("Must provide either jvp_fn or jac_fn")

        # Convergence condition
        def cond_fun(state):
            _, r_k, k = state
            return (jnp.linalg.norm(r_k) > self.tol) & (k < self.maxiter)

        y_final, r_final, niters = jax.lax.while_loop(cond_fun, body_fun, state0)

        # Runtime warning
        def warn_callback(iters, maxiter, residual_norm, tol):
            if iters >= maxiter and residual_norm > tol:
                s1 = f"WARNING: Newton-Raphson did not converge within {int(maxiter)} iterations."
                s2 = f"Final residual norm: {float(residual_norm):.2e}."
                print(s1 + "\n" + s2)

        jax.debug.callback(
            warn_callback, 
            niters, self.maxiter, 
            jnp.linalg.norm(r_final), self.tol
        )

        return y_final
