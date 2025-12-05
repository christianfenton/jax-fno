"""Explicit time-stepping schemes."""

from dataclasses import dataclass
from typing import Callable

from jax import Array

from .base import AbstractStepper


@dataclass(frozen=True)
class ForwardEuler(AbstractStepper):
    """
    Forward Euler method.

    Discretisation:
    $$
    \\frac{\\partial y}{\\partial t} \\rightarrow
    (y_{n+1} - y_n) / dt = f(t_n, y_n)
    $$
    """

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        dt: Array,
        args: tuple = ()
    ) -> Array:
        """
        Perform a single Forward Euler step.

        Computes
        $$
        y_{n+1} = y_n + dt f(t_n, y_n, *args).
        $$

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args).
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution.
            dt: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + dt.
        """
        return y + dt * fun(t, y, *args)


@dataclass(frozen=True)
class RK4(AbstractStepper):
    """Fourth (4th) order Runge-Kutta method."""

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        dt: Array,
        args: tuple = ()
    ) -> Array:
        """
        Perform a single RK4 step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args).
            t: Current time.
            y: Current solution.
            dt: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + dt.
        """
        k1 = fun(t, y, *args)
        k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
        k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
        k4 = fun(t + dt, y + dt * k3, *args)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
