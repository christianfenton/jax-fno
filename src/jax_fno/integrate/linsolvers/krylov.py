"""Linear solvers based on Krylov subspaces."""

from typing import Union, Optional
from flax import nnx
from jax import Array
import jax.scipy.sparse.linalg as jax_sparse

from ..custom_types import LinearMap


class GMRES(nnx.Module):
    """
    Generalised Minimal Residual (GMRES).

    Dispatches to `jax.scipy.sparse.linalg.gmres`.
    Suitable for general non-symmetric systems.
    """

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[LinearMap, Array], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b using GMRES.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.gmres(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution


class CG(nnx.Module):
    """
    Conjugate Gradients (CG).

    Dispatches to `jax.scipy.sparse.linalg.cg`.
    Only suitable for symmetric and positive-definite systems.

    Implements: LinearSolverProtocol

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[LinearMap, Array], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.cg(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution


class BiCGStab(nnx.Module):
    """
    Stabilised Biconjugate Gradients (BiCGStab).

    Dispatches to `jax.scipy.sparse.linalg.bicgstab`.
    Suitable for non-symmetric systems.

    Implements: LinearSolverProtocol

    Attributes:
        tol: Convergence tolerance for residual norm
        maxiter: Maximum number of iterations
    """

    def __init__(self, tol: float = 1e-6, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, 
        A: Union[LinearMap, Array], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Approximate solution x
        """
        solution, info = jax_sparse.bicgstab(
            A, b, x0=x0, tol=self.tol, maxiter=self.maxiter
        )
        return solution