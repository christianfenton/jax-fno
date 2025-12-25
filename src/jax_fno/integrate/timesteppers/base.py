"""Abstract base class for time-stepping schemes."""

from abc import ABC
from dataclasses import dataclass
from typing import Callable

import jax


@dataclass(frozen=True)
class AbstractStepper(ABC):
    """Base class for time-stepping schemes."""
    
    def step(
        self, 
        fun: Callable, 
        t: jax.Array, 
        y: jax.Array, 
        h: jax.Array, 
        args: tuple = ()
    ) -> jax.Array:
        """
        Take a single time step.

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args).
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        ...