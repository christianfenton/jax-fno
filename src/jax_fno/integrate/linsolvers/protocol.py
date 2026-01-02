"""Protocol for linear solvers used in implicit time-stepping methods."""

from typing import Protocol, runtime_checkable, Union, Optional

from jax import Array

from ..custom_types import LinearMap


@runtime_checkable
class LinearSolverProtocol(Protocol):
    """
    Protocol for linear solvers.

    Defines the interface for solving linear systems of the form A*x = b.
    Any class implementing a __call__() method with this signature can be used
    as a linear solver in implicit integration schemes.
    """

    def __call__(
        self, 
        A: Union[LinearMap, Array], 
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve the linear system A*x = b.

        Args:
            A: Dense matrix or linear operator with signature x -> A*x
            b: Right-hand side vector
            x0: Initial guess vector

        Returns:
            Solution vector x such that A*x ≈ b
        """
        ...
