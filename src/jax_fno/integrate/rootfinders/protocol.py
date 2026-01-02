"""Protocol for root-finding algorithms."""

from typing import Optional, Protocol, runtime_checkable

from jax import Array

from ..custom_types import LinearMap, JVPConstructor, JacobianConstructor


@runtime_checkable
class RootFinderProtocol(Protocol):
    """
    Protocol for root-finding algorithms.

    Defines the interface for finding roots of nonlinear equations.
    Used by implicit time-stepping schemes to solve the nonlinear systems
    that arise from implicit discretization.
    """

    def __call__(
        self,
        residual_fn: LinearMap,
        y_guess: Array,
        jvp_fn: Optional[JVPConstructor] = None,
        jac_fn: Optional[JacobianConstructor] = None,
    ) -> Array:
        """
        Find the root of residual_fn(y) = 0.

        Args:
            residual_fn: Function mapping y -> R(y), where we seek R(y) = 0
            y_guess: Initial guess for the solution
            jvp_fn: Optional Jacobian-vector product function (y, v) -> J*v
            jac_fn: Optional dense Jacobian function y -> J

        Returns:
            Solution y such that residual_fn(y) ≈ 0
        """
        ...