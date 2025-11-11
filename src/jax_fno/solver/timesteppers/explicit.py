"""Explicit time-stepping schemes."""

from dataclasses import dataclass
import jax.numpy as jnp
from typing import Callable
from .stepper import Stepper


class ForwardEuler(Stepper):
    """
    Forward Euler method.
    """
    @staticmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float,
    ) -> jnp.ndarray:
        """
        Perform a single Forward Euler step.

        Computes u_{n+1} = u_n + dt * f(u_n, t_n).

        Args:
            f: Right-hand side of PDE
            u: Current solution
            t: Current time
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        return u + dt * f(u, t)


class RK4(Stepper):
    """
    Fourth (4th) order Runge-Kutta method.
    """
    @staticmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float,
    ) -> jnp.ndarray:
        """
        Perform a single RK4 step.

        Args:
            f: Right-hand side of PDE
            u: Current solution
            t: Current time
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        k1 = f(u, t)
        k2 = f(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(u + dt * k3, t + dt)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
