"""
Implicit time-stepping schemes.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
from jax import Array
import jax.numpy as jnp

from .base import AbstractStepper
from ..rootfinders import AbstractRootFinder, NewtonRaphson


@dataclass(frozen=True)
class BackwardEuler(AbstractStepper):
    """
    Backward Euler time-stepping scheme.

    Discretisation:
    $$ \\frac{\\partial y}{\\partial t} \\rightarrow
    \\frac{y_{n+1} - y_n}{h} = f(t_{n+1}, y_{n+1}) $$

    Residual:
    $$ R(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}) $$

    Jacobian:
    $$ J = \\frac{\\partial R}{\\partial y_{n+1}}
    = I - h \\frac{\\partial f(t_{n+1}, y_{n+1})}{\\partial y} $$

    Attributes:
        root_finder: Root-finding algorithm for implicit equations.
            Default: NewtonRaphson.
        jvp: Optional user-provided Jacobian-vector product with
            signature (t, y, v, *args) -> J*v.
        jac: Optional user-provided dense Jacobian with
            signature (t, y, *args) -> J.

    If neither jvp nor jac are provided, the `step` method defaults to 
    automatic differentiation with `jax.jvp`.
    """

    root_finder: AbstractRootFinder = field(default_factory=NewtonRaphson)
    jvp: Optional[Callable] = None
    jac: Optional[Callable] = None

    @staticmethod
    def make_residual(
        fun: Callable,
        t_prev: Array,
        y_prev: Array,
        h: Array,
        args: tuple = (),
    ) -> Callable[[Array], Array]:
        """
        Create residual function for a backward Euler scheme.

        Residual: $R(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}, \\cdot)$

        Args:
            fun: Right-hand side of system dy/dt = f(t, y, *args).
            t_prev: Time at previous step. Type: 0-dimensional JAX array.
            y_prev: Solution at previous time step.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun.

        Returns:
            A function with signature y -> R(y)
        """
        t_next = t_prev + h
        return lambda y_np1: y_np1 - y_prev - h * fun(t_next, y_np1, *args)

    @staticmethod
    def make_jvp(
        jvp: Callable, t_prev: Array, h: Array, args: tuple = ()
    ) -> Callable[[Array, Array], Array]:
        """
        Function factory for a matrix-free Jacobian-vector product.

        Jacobian: $J = I - h \\frac{\\partial f}{\\partial y}$

        Args:
            jvp: Jacobian-vector product, signature: (t, y, v, *args) -> (dfdy)*v
            t_prev: Time at previous step. Type: 0-dimensional JAX array.
            h: Time step size. Type: 0-dimensional JAX array
            args: Additional arguments to pass to jvp

        Returns:
            A function with signature (y, v) -> J_y * v
        """
        t_next = t_prev + h
        return lambda y, v: v - h * jvp(t_next, y, v, *args)

    @staticmethod
    def make_jacobian(
        jac: Callable, t_prev: Array, h: Array, args: tuple = ()
    ) -> Callable[[Array], Array]:
        """
        Function factory for dense Jacobian matrix.

        Jacobian: $J = I - h \\frac{\\partial f}{\\partial y}$

        Args:
            jac: Jacobian matrix function (t, y, *args) -> ∂f/∂y
            t_prev: Time at previous step. Type: 0-dimensional JAX array.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to jac

        Returns:
            A function with signature y -> J_y.
        """
        t_next = t_prev + h
        return lambda y: jnp.eye(y.size) - h * jac(t_next, y, *args)

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = (),
    ) -> Array:
        """
        Perform a backward Euler step.

        Solves $$ y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}, \\cdot) = 0 $$
        for $y_{n+1}$ using a root-finding algorithm.

        Args:
            fun: Right-hand side of system dydt = f(t, y, *args).
            t: Current time. Type: 0-dimensional JAX array.
            y: Current solution at time t.
            h: Time step size. Type: 0-dimensional JAX array.
            args: Additional arguments to pass to fun, jvp, and jac.

        Returns:
            Solution at time t + h.
        """
        # Define residual function for this step
        residual_fn = self.make_residual(fun, t, y, h, args)

        # Initial guess (forward Euler step)
        y_guess = y + h * fun(t, y, *args)

        # Prepare Jacobian functions
        jac = self.jac
        jvp = self.jvp

        # Default: use JAX's autodiff for JVP if neither jac nor jvp provided
        if jac is None and jvp is None:

            def jvp_autodiff(
                t_inner: Array, y_inner: Array, v: Array, *args_inner
            ) -> Array:
                """Compute (∂f/∂y)*v using automatic differentiation."""
                return jax.jvp(
                    lambda y_: fun(t_inner, y_, *args_inner), (y_inner,), (v,)
                )[1]

            jvp = jvp_autodiff

        # Build Jacobian functions for the residual
        jac_fn = None
        jvp_fn = None

        if jac is not None:
            jac_fn = self.make_jacobian(jac, t, h, args)

        if jvp is not None:
            jvp_fn = self.make_jvp(jvp, t, h, args)

        # Solve nonlinear system using configured root-finding algorithm
        y_next = self.root_finder.solve(
            residual_fn, y_guess, jvp_fn=jvp_fn, jac_fn=jac_fn
        )

        return y_next
