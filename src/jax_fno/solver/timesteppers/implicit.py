"""
Implicit time-stepping schemes.

This module implements implicit time integration methods that require solving
non-linear systems via Newton-Raphson iterations.

Each implicit scheme provides:
    - make_residual: Creates residual function R(y_{n+1}) for the scheme
    - make_jvp: Creates matrix-free Jacobian-vector product function
    - make_jacobian: Creates dense Jacobian matrix function
    - step: Advance the solution by one time step

Supports both matrix-free (JVP for large systems) and dense Jacobian (direct solve for small systems).
"""

from abc import abstractmethod
from typing import Callable, Optional

import jax.numpy as jnp

from .base import AbstractStepper
from ..newtonraphson import newton_raphson


class ImplicitStepper(AbstractStepper):
    """Base class for implicit time-stepping schemes."""

    @staticmethod
    @abstractmethod
    def step(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t: float,
        y: jnp.ndarray,
        dt: float,
        *,
        tol: float,
        maxiter: int,
        jvp: Optional[
            Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
        jac: Optional[
            Callable[[float, jnp.ndarray], jnp.ndarray]
        ] = None,
        linsolver: Optional[
            Callable[[Callable, jnp.ndarray], jnp.ndarray]
        ] = None,
    ) -> jnp.ndarray:
        """
        Advance the solution y from t to t+dt.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y)
            t: Current time
            y: Current solution at time t
            dt: Time step size

        Kwargs:
            tol: Convergence tolerance for Newton-Raphson method
            maxiter: Maximum number of Newton-Raphson iterations
            jvp: Jacobian-vector product function (t, y, v) -> (∂f/∂y)*v (matrix-free)
            jac: Jacobian matrix function (t, y) -> ∂f/∂y (dense matrix)
            linsolver: Linear solver (iterative for JVP, direct for JAC)
        """
        pass


class BackwardEuler(ImplicitStepper):
    """
    Backward Euler time-stepping scheme.

    Discretisation:
    $$
    \\frac{\\partial y}{\\partial t} \\rightarrow 
    \\frac{y_{n+1} - y_n}{\\delta t} = f(t_{n+1}, y_{n+1})
    $$

    Residual:
    $$
    R(y_{n+1}) = y_{n+1} - y_n - \\delta t f(t_{n+1}, y_{n+1})
    $$

    Jacobian:
    $$
    J = \\frac{\\partial R}{\\partial y_{n+1}} 
    = I - \\delta t \\frac{\\partial f(t_{n+1}, y_{n+1})}{\\partial y}
    $$
    """

    @staticmethod
    def make_residual(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t_prev: float,
        y_prev: jnp.ndarray,
        dt: float,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Create residual function for a backward Euler scheme.

        Args:
            fun: Right-hand side of system dy/dt = f(t, y)
            t_prev: Time at previous step t_n
            y_prev: Solution at previous time step t_n
            dt: Time step size

        Returns:
            A function R(y_{n+1}) = y_{n+1} - y_n - dt * f(t_{n+1}, y_{n+1})
        """
        t_next = t_prev + dt
        return lambda y_np1: y_np1 - y_prev - dt * fun(t_next, y_np1)

    @staticmethod
    def make_jvp(
        jvp: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        t_prev: float,
        dt: float,
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Function factory for a matrix-free Jacobian-vector product.

        Jacobian: $J = I - \\delta t * \\frac{\\partial f}{\\partial y}$

        Args:
            jvp: Jacobian-vector product function (t, y, v) -> (dfdy)*v
            t_prev: Time at previous step
            dt: Time step size

        Returns:
            A function with signature (y, v) -> J_y * v
        """
        t_next = t_prev + dt
        return lambda y, v: v - dt * jvp(t_next, y, v)

    @staticmethod
    def make_jacobian(
        jac: Callable[[float, jnp.ndarray], jnp.ndarray],
        t_prev: float,
        dt: float,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Function factory for dense Jacobian matrix.

        Jacobian: $J = I - \\delta t * \\frac{\\partial f}{\\partial y}$

        Args:
            jac: Jacobian matrix function (t, y) -> ∂f/∂y
            t_prev: Time at previous step
            dt: Time step size

        Returns:
            A function with signature y -> J_y
        """
        t_next = t_prev + dt
        return lambda y: jnp.eye(y.size) - dt * jac(t_next, y)

    @staticmethod
    def step(
        fun: Callable[[float, jnp.ndarray], jnp.ndarray],
        t: float,
        y: jnp.ndarray,
        dt: float,
        *,
        tol: float,
        maxiter: int,
        jvp: Optional[
            Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
        jac: Optional[
            Callable[[float, jnp.ndarray], jnp.ndarray]
        ] = None,
        linsolver: Optional[Callable] = None,
    ) -> jnp.ndarray:
        """
        Perform a backward Euler step.

        Solves
        $$
        y_{n+1} - y_n - \\delta t f(t_{n+1}, y_{n+1}) = 0
        $$
        for $y_{n+1}$ using a Newton-Raphson method.

        Args:
            fun: Right-hand side of system dydt = f(t, y)
            t: Current time
            y: Current solution at time t
            dt: Time step size

        Kwargs:
            tol: Convergence tolerance for Newton-Raphson method
            maxiter: Maximum number of Newton-Raphson iterations
            jvp: Jacobian-vector product function (t, y, v) -> dfdy*v
            jac: Jacobian matrix function (t, y) -> dfdy (dense)
            linsolver: Linear solver

        Returns:
            Solution at time t + dt
        """
        # Define residual function for this step
        residual_fn = BackwardEuler.make_residual(fun, t, y, dt)

        # Initial guess (forward Euler step)
        y_guess = y + dt * fun(t, y)

        # Choose between dense and matrix-free Jacobian
        if jac is not None:
            # Dense Jacobian path
            jac_fn = BackwardEuler.make_jacobian(jac, t, dt)
            y_next = newton_raphson(
                y_guess, residual_fn, jac_fn, None,
                tol=tol, maxiter=maxiter, dense=True
            )
        elif jvp is not None:
            # Matrix-free JVP path
            jvp_fn = BackwardEuler.make_jvp(jvp, t, dt)
            y_next = newton_raphson(
                y_guess, residual_fn, jvp_fn, linsolver,
                tol=tol, maxiter=maxiter, dense=False
            )
        else:
            raise ValueError("Must provide either 'jvp' or 'jac'")

        return y_next
