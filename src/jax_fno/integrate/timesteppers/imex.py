"""Implicit-Explicit (IMEX) time-stepping schemes."""

from typing import Callable, Union, Dict

from flax import nnx
from jax import Array

from .protocol import StepperProtocol


class IMEX(nnx.Module):
    """
    Implicit-Explicit (IMEX) time-stepping scheme.

    Splits the ODE into stiff (implicit) and non-stiff (explicit) parts:
        dy/dt = f_explicit(t, y) + f_implicit(t, y)

    The scheme advances the solution in two steps:
        1. Explicit step:  u* = u^n + h * f_explicit(t^n, u^n)
        2. Implicit step:  u^{n+1} = u* + h * f_implicit(t^{n+1}, u^{n+1})

    The implicit step is solved via root-finding:
        R(u^{n+1}) = u^{n+1} - u* - h * f_implicit(t^{n+1}, u^{n+1}) = 0

    This formulation allows you to:
    - Use high-order explicit methods (RK4, etc.) for the non-stiff terms
    - Use implicit methods (BackwardEuler, etc.) for the stiff terms
    - Avoid overly restrictive time step constraints from stiff terms

    Example:
        ```python
        from jax_fno.integrate import (
            solve_ivp, IMEX, RK4, BackwardEuler,
            NewtonRaphson, Spectral
        )

        def explicit_term(t, u, ...):
            return ...

        def implicit_term(t, u, ...):
            return ...

        # Instantiate solver
        solver = IMEX(implicit=BackwardEuler(), explicit=RK4())

        # Define ODE as a dict
        ode = {'implicit': implicit_term, 'explicit': explicit_term}

        # Solve
        t, y = solve_ivp(ode, t_span, y0, solver, step_size, args)
        ```
    """

    def __init__(
        self,
        implicit: StepperProtocol,
        explicit: StepperProtocol,
    ):
        self.implicit = implicit
        self.explicit = explicit

    def step(
        self,
        fun: Union[Callable, Dict[str, Callable]],
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """
        Advance one IMEX step.

        Args:
            fun: Either a dict with keys 'implicit' and 'explicit', or a callable.
                If a dict, fun['explicit'](t, y, *args) gives the non-stiff term
                and fun['implicit'](t, y, *args) gives the stiff term.
                If a callable, it's treated as the implicit term with zero explicit term.
            t: Current time.
            y: Current solution.
            h: Time step size.
            args: Additional arguments to pass to fun.

        Returns:
            Solution at t + h.
        """
        # Handle dict-based interface
        if isinstance(fun, dict):
            if 'explicit' not in fun or 'implicit' not in fun:
                raise ValueError(
                    "IMEX requires fun to be a dict with 'explicit' and 'implicit' keys, "
                    f"but got keys: {list(fun.keys())}"
                )
            fun_explicit = fun['explicit']
            fun_implicit = fun['implicit']
        else:
            # If fun is a callable, treat it as implicit-only
            fun_explicit = lambda t, y, *args: 0.0
            fun_implicit = fun

        # Step 1: Explicit advance to get intermediate state u*
        # u* = u^n + h * f_explicit(t^n, u^n)
        u_star = self.explicit.step(fun_explicit, t, y, h, args)

        # Step 2: Implicit solve from u* to get u^{n+1}
        # Solve: u^{n+1} = u* + h * f_implicit(t^{n+1}, u^{n+1})
        #
        # The implicit stepper expects to solve:
        #   u^{n+1} = u^n + h * f(t^{n+1}, u^{n+1})
        #
        # We substitute u^n -> u* to get the correct equation
        u_next = self.implicit.step(fun_implicit, t, u_star, h, args)

        return u_next
