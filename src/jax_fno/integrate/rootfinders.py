"""
Root-finding algorithms used in implicit time-stepping methods.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
from jax import Array
import jax.numpy as jnp

from .linsolvers import AbstractLinearSolver, GMRES


@dataclass(frozen=True)
class AbstractRootFinder(ABC):
    """
    Base class for root finders.
    
    Inherited classes must be frozen dataclasses (JAX-compatible pytrees).
    """

    def solve(
        self,
        residual_fn: Callable[[Array], Array],
        y_guess: Array,
        jvp_fn: Optional[Callable[[Array, Array], Array]] = None,
        jac_fn: Optional[Callable[[Array], Array]] = None,
    ) -> Array:
        """
        Find the root of residual_fn(y) = 0.

        Args:
            residual_fn: Residual function R(y) to find root of.
            y_guess: Initial guess for the solution.
            jvp_fn: Optional Jacobian-vector product with
                signature (y, v) -> J(y)*v. Used in matrix-free mode.
            jac_fn: Optional dense Jacobian function with
                signature y -> J(y). Used in direct solve mode.

        Returns:
            Solution y.
        """
        ...


@dataclass(frozen=True)
class NewtonRaphson(AbstractRootFinder):
    """
    Newton-Raphson root-finding algorithm.

    Iterative update: $y \\leftarrow y - J^{-1}(y) R(y)$

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of Newton-Raphson iterations
        linsolver: Linear solver for inner iterations
            Default: GMRES (iterative)
    """

    tol: float = 1e-6
    maxiter: int = 50
    linsolver: AbstractLinearSolver = field(default_factory=GMRES)

    def solve(
        self,
        residual_fn: Callable[[Array], Array],
        y_guess: Array,
        jvp_fn: Optional[Callable[[Array, Array], Array]] = None,
        jac_fn: Optional[Callable[[Array], Array]] = None,
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
        if (jac_fn is not None) and (jvp_fn is not None):
            raise ValueError(
                """
                Both `jvp_fn` and `jac_fn` were provided.
                Only one of these arguments can be given at a time.
                """
            )

        y_k = y_guess
        r_k = residual_fn(y_k)
        state0 = (y_k, r_k, 0)

        if jac_fn is not None:
            # Dense mode
            def body_fun(state):
                y_k, r_k, k = state
                delta = self.linsolver.solve(jac_fn(y_k), -r_k)
                y_kp1 = y_k + delta
                r_kp1 = residual_fn(y_kp1)
                return (y_kp1, r_kp1, k + 1)
        elif jvp_fn is not None:
            # Matrix-free mode
            def body_fun(state):
                y_k, r_k, k = state
                delta = self.linsolver.solve(lambda v: jvp_fn(y_k, v), -r_k)
                y_kp1 = y_k + delta
                r_kp1 = residual_fn(y_kp1)
                return (y_kp1, r_kp1, k + 1)
        else:
            raise ValueError("Must provide either jvp_fn or jac_fn")

        def cond_fun(state):
            _, r_k, k = state
            return (jnp.linalg.norm(r_k) > self.tol) & (k < self.maxiter)

        y_final, r_final, niters = jax.lax.while_loop(
            cond_fun, body_fun, state0
        )

        def warn_callback(iters, maxiter, residual_norm):
            if iters >= maxiter:
                print(
                    f"""WARNING: Newton-Raphson did not converge 
                    within {int(maxiter)} iterations. """
                    + f"Final residual norm: {float(residual_norm):.2e}"
                )
            return None
        
        jax.debug.callback(
            warn_callback, niters, self.maxiter, jnp.linalg.norm(r_final)
        )

        return y_final
