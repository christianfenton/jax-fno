from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse


@dataclass(frozen=True)
class LinearSolverConfig:
    """
    Configuration for linear solver used within Newton-Raphson iterations.

    Attributes:
        method: Linear solver method (iterative only, for matrix-free Jacobians)
            - 'gmres': Generalised minimal residuals (default, robust)
            - 'cg': Conjugate gradients (for symmetric positive definite systems)
            - 'bicgstab': Stabilised biconjugate gradients (for non-symmetric systems)
        tol: Convergence tolerance for iterative solver
        maxiter: Maximum iterations for iterative solver
    """

    method: str = "gmres"
    tol: float = 1e-6
    maxiter: int = 100

    def __post_init__(self):
        valid_methods = {"gmres", "cg", "bicgstab"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid linear solver method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )


# TODO: add callbacks in iterative solvers to warn about non-convergence
def dispatch_linear_solver(
    config: LinearSolverConfig,
) -> Callable[[Callable, jnp.ndarray], jnp.ndarray]:
    """
    Function factory for a matrix-free iterative linear solver.

    Args:
        config: Linear solver configuration

    Returns:
        Solver function with signature: (jvp_operator, rhs) -> solution
    """
    if config.method == "gmres":

        def gmres_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.gmres(
                jvp_fn, rhs, tol=config.tol, maxiter=config.maxiter
            )
            return solution

        return gmres_solve

    elif config.method == "cg":

        def cg_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.cg(
                jvp_fn, rhs, tol=config.tol, maxiter=config.maxiter
            )
            return solution

        return cg_solve

    elif config.method == "bicgstab":

        def bicgstab_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.bicgstab(
                jvp_fn, rhs, tol=config.tol, maxiter=config.maxiter
            )
            return solution

        return bicgstab_solve

    else:
        raise ValueError(f"Unknown linear solver method: {config.method}")


def default_linear_solver():
    """
    Default linear solver for use inside Newton-Raphson iterations.

    Default Settings:
        method: 'gmres'
        tol: 1e-6
        maxiter: 100
    """
    return dispatch_linear_solver(LinearSolverConfig())
