"""
Linear solvers used in implicit time-stepping methods.
"""

from dataclasses import dataclass
from typing import Protocol, Callable, Union

from jax import Array
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse


class LinearSolverProtocol(Protocol):
    """
    Protocol for linear solvers.

    Any JAX-friendly object with a .solve(A, b) method matching this
    signature can be used as a linear solver in implicit methods.
    """

    def solve(
        self, A: Union[Callable[[Array], Array], Array], b: Array
    ) -> Array:
        """
        Solve the linear system A*x = b.

        Args:
            A: Either a dense matrix OR a linear operator (x -> A*x)
            b: Right-hand side vector

        Returns:
            Solution vector x such that A*x ≈ b
        """
        ...


@dataclass(frozen=True)
class GMRES:
    """
    Generalised Minimal Residual (GMRES).

    Dispatches to `jax.scipy.sparse.linalg.gmres`.
    Suitable for general non-symmetric systems.

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """

    tol: float = 1e-6
    maxiter: int = 100

    def solve(
        self, A: Union[Callable[[Array], Array], Array], b: Array
    ) -> Array:
        """
        Solve A*x = b using GMRES.

        Args:
            A: Either a dense matrix or linear operator (x -> A*x)
            b: Right-hand side vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.gmres(
            A, b, tol=self.tol, maxiter=self.maxiter
        )
        return solution


@dataclass(frozen=True)
class CG:
    """
    Conjugate Gradients (CG).

    Dispatches to `jax.scipy.sparse.linalg.cg`.
    Only suitable for symmetric and positive-definite systems.

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """

    tol: float = 1e-6
    maxiter: int = 100

    def solve(
        self, A: Union[Callable[[Array], Array], Array], b: Array
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Either a dense matrix or linear operator (x -> A*x)
            b: Right-hand side vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.cg(
            A, b, tol=self.tol, maxiter=self.maxiter
        )
        return solution


@dataclass(frozen=True)
class BiCGStab:
    """
    Stabilised Biconjugate Gradients (BiCGStab).

    Dispatches to `jax.scipy.sparse.linalg.bicgstab`.
    Suitable for non-symmetric systems.

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """

    tol: float = 1e-6
    maxiter: int = 100

    def solve(
        self, A: Union[Callable[[Array], Array], Array], b: Array
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Either a dense matrix or linear operator (x -> A*x)
            b: Right-hand side vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.bicgstab(
            A, b, tol=self.tol, maxiter=self.maxiter
        )
        return solution


@dataclass(frozen=True)
class DirectSolve:
    """
    Direct solver for linear systems.

    Dispatches to `jax.numpy.linalg.solve`.
    Only suitable for small systems where the Jacobian is provided explicitly.
    """

    def solve(
        self, A: Union[Callable[[Array], Array], Array], b: Array
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense Jacobian matrix
            b: Right-hand side vector

        Returns:
            Solution x

        Raises:
            TypeError: If A is a callable (linear operator) instead of a matrix
        """
        if callable(A):
            raise TypeError(
                "DirectSolve requires a dense matrix, not a linear operator. "
                "Please provide the Jacobian as a dense matrix (jac_fn), "
                "or use an iterative solver (GMRES, CG, BiCGStab)."
            )

        return jnp.linalg.solve(A, b)
