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
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float
    ) -> jnp.ndarray:
        """
        Advance the solution u from t to t+dt.
        
        Args:
            f: RHS function: du/dt = f(u, t)
            u: Current solution at time t
            t: Current time
            dt: Time step size

        Returns:
            Solution at t + dt
        """
        pass


class ForwardEuler(ExplicitStepper):
    """
    Forward Euler method.
    
    Discretisation:
        du/dt = f(u, t) --> (u_{n+1} - u_n) / dt = f(u_n, t_n)
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


class RK4(ExplicitStepper):
    """Fourth (4th) order Runge-Kutta method."""

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
