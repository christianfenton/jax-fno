"""Explicit time-stepping schemes."""

from typing import Callable

from flax import nnx
from jax import Array


class ForwardEuler(nnx.Module):
    """
    Forward Euler method.

    Discretisation:
        $$ \\frac{\\partial y}{\\partial t} \\rightarrow
        \\frac{(y_{n+1} - y_n)}{h} = f(t_n, y_n) $$
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
        Perform a single Forward Euler step.

        Computes $$ y_{n+1} = y_n + h f(t_n, y_n, *args). $$

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args).
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        return y + h * fun(t, y, *args)


class RK4(nnx.Module):
    """
    Fourth (4th) order Runge-Kutta method.

    Implements: StepperProtocol
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
        Perform a single RK4 step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args).
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        k1 = fun(t, y, *args)
        k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, *args)
        k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, *args)
        k4 = fun(t + h, y + h * k3, *args)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
