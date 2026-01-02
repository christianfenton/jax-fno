"""Direct linear solvers."""


from typing import Union, Optional
from flax import nnx
from jax import Array
import jax.numpy as jnp

from ..custom_types import LinearMap


class DirectDense(nnx.Module):
    """
    Direct solver for dense linear systems.

    Dispatches to `jax.numpy.linalg.solve`.
    Only suitable for small systems where the Jacobian is provided explicitly.
    """

    def __call__(
        self, 
        A: Union[LinearMap, Array], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Dense matrix
            b: Right-hand side vector
            x0: Ignored (kept for interface compatibility). Can be None.

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