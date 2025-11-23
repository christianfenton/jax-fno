"""Explicit time-stepping schemes."""

from abc import abstractmethod

import jax.numpy as jnp
from typing import Callable

from .base import AbstractStepper


class ExplicitStepper(AbstractStepper):
    """Base class for explicit time-stepping schemes."""

    @staticmethod
    @abstractmethod
    def step(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t: float,
        y: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """
        Advance the solution y from t to t+dt.

        Args:
            fun: Right-hand side of system dydt = f(t, y)
            t: Current time
            y: Current solution at time t
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        pass


class ForwardEuler(ExplicitStepper):
    """
    Forward Euler method.

    Discretisation:
    $$
    \\frac{\\partial y}{\\partial t} \\rightarrow 
    (y_{n+1} - y_n) / dt = f(t_n, y_n)
    $$
    """

    @staticmethod
    def step(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t: float,
        y: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """
        Perform a single Forward Euler step.

        Computes 
        $$
        y_{n+1} = y_n + dt * f(t_n, y_n).
        $$

        Args:
            fun: Right-hand side of system dydt = f(t, y)
            t: Current time
            y: Current solution
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        return y + dt * fun(t, y)


class RK4(ExplicitStepper):
    """Fourth (4th) order Runge-Kutta method."""

    @staticmethod
    def step(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t: float,
        y: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """
        Perform a single RK4 step.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y)
            t: Current time
            y: Current solution
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        k1 = fun(t, y)
        k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = fun(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
