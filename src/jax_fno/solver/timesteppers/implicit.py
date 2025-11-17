"""
Implicit time-stepping schemes.

This module implements implicit time integration methods that require solving
non-linear systems via Newton-Raphson iterations.

A linear solver is used inside each Newton-Raphson iteration.

Each implicit scheme provides:
    - make_residual: Creates residual function R(u_{n+1}) for the scheme
    - make_jvp: Creates matrix-free Jacobian-vector product function
    - step: Advance the solution by one time step

Note: Only matrix-free (JVP) Jacobians are currently supported.
"""

from abc import abstractmethod
from typing import Callable

import jax.numpy as jnp

from .base import AbstractStepper
from ..newtonraphson import newton_raphson


class ImplicitStepper(AbstractStepper):
    """Base class for implicit time-stepping schemes."""

    @staticmethod
    @abstractmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float,
        *,
        tol: float,
        maxiter: int,
        df: Callable[
            [jnp.ndarray, float], 
            Callable[[jnp.ndarray], jnp.ndarray]
        ],
        linsolver: Callable[[Callable, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Advance the solution u from t to t+dt.
        
        Args:
            f: RHS function: du/dt = f(u, t)
            u: Current solution at time t
            t: Current time
            dt: Time step size

        Kwargs:
            tol: Convergence tolerance for Newton-Raphson method
            maxiter: Maximum number of Newton-Raphson iterations
            df: Jacobian-vector product of f.
                df(u, t) must return a function with signature v -> (∂f/∂u)*v
            linsolver: Iterative linear solver
        """
        pass


class BackwardEuler(ImplicitStepper):
    """
    Backward Euler time-stepping scheme.

    Discretisation:
        du/dt = f(u, t) --> (u_{n+1} - u_n) / dt = f(u_{n+1}, t_{n+1})

    Residual:
        R(u_{n+1}) = u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1})

    Jacobian:
        J = ∂R/∂u_{n+1} = I - dt * ∂f/∂u(u_{n+1}, t_{n+1})
    """

    @staticmethod
    def make_residual(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u_prev: jnp.ndarray,
        t_prev: float,
        dt: float,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Create residual function for a backward Euler scheme.

        Args:
            f: RHS of ODE du/dt = f(u, t)
            u_prev: Solution at previous time step t_n
            t_prev: Time at previous step t_n
            dt: Time step size

        Returns:
            A function R(u_{n+1}) = u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1})
        """
        return lambda u_np1: u_np1 - u_prev - dt * f(u_np1, t_prev + dt)

    @staticmethod
    def make_jvp(
        df: Callable[
            [jnp.ndarray, float], Callable[[jnp.ndarray], jnp.ndarray]
        ],
        t_prev: float,
        dt: float,
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Function factory for a matrix-free Jacobian-vector product.

        Jacobian of residual: J = I - dt * ∂f/∂u

        Args:
            df: Jacobian-vector product of f.
                df(u, t) returns a function v -> (∂f/∂u)*v
            t_prev: Time at previous step
            dt: Time step size

        Returns:
            Function (u, v) -> J(u) * v = v - dt * (∂f/∂u)*v
        """
        return lambda u, v: v - dt * df(u, t_prev + dt)(v)

    @staticmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float,
        *,
        tol: float,
        maxiter: int,
        df: Callable[
            [jnp.ndarray, float], 
            Callable[[jnp.ndarray], jnp.ndarray]
        ],
        linsolver: Callable[[Callable, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Perform a backward Euler step.

        Solves: u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1}) = 0
        for u_{n+1} using Newton-Raphson method with a matrix-free Jacobian.

        Args:
            f: RHS function: du/dt = f(u, t)
            u: Current solution at time t
            t: Current time
            dt: Time step size

        Kwargs:
            tol: Convergence tolerance for Newton-Raphson method
            maxiter: Maximum number of Newton-Raphson iterations
            df: Jacobian-vector product of f.
                df(u, t) must return a function with signature v -> (∂f/∂u)*v
            linsolver: Iterative linear solver

        Returns:
            Solution at time t + dt
        """
        # Define residual function for this step
        residual_fn = BackwardEuler.make_residual(f, u, t, dt)

        # Define Jacobian-vector product
        jvp_fn = BackwardEuler.make_jvp(df, t, dt)

        # Initial guess (forward Euler step)
        u_guess = u + dt * f(u, t)

        # Solve system with Newton-Raphson method
        u_next = newton_raphson(
            u_guess, residual_fn, jvp_fn, linsolver, tol=tol, maxiter=maxiter
        )

        return u_next
