"""Protocols for time-stepping schemes."""

from typing import Protocol, runtime_checkable, Callable
from jax import Array


@runtime_checkable
class StepperProtocol(Protocol):
    """
    Protocol for time-stepping schemes.

    Defines the interface for advancing an ODE one time step.
    Any class implementing a step() method with this signature can be used
    as a time-stepping method in solve_ivp.
    """

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Take a single time step.

        Args:
            fun: Right-hand side function.
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        ...