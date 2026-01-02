from typing import Callable, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from jax import Array

from ..rootfinders import RootFinderProtocol, NewtonRaphson


class BackwardEuler(nnx.Module):
    """
    Backward Euler time stepper.

    Discretisation:
        $$ \\frac{\\partial y}{\\partial t} \\rightarrow
        \\frac{(y_{n+1} - y_n)}{h} = f(t_{n+1}, y_{n+1}) $$
        
    If neither jvp nor jac are provided, defaults to root-finding using
    automatic differentiation with `jax.jvp`.
    """

    def __init__(
        self,
        root_finder: RootFinderProtocol = NewtonRaphson(),
        jvp: Optional[Callable] = None,
        jac: Optional[Callable] = None,
    ):
        self.root_finder = root_finder
        self.jvp = jvp
        self.jac = jac

    def _residual(self, fun, t, y, h, args):
        return lambda y_np1: y_np1 - y - h * fun(t + h, y_np1, *args)

    def _build_linearisation(self, fun, t, h, args):
        """Return (jac_fn, jvp_fn) for the Jacobian J_f = I - h * df/dy."""

        # Case 1: explicit dense Jacobian supplied
        if self.jac is not None:
            def jac_fn(y):
                return jnp.eye(y.size) - h * self.jac(t + h, y, *args)
            return jac_fn, None

        # Case 2: matrix-free JVP supplied (preferred for large systems)
        if self.jvp is not None:
            def jvp_fn(y, v):
                return v - h * self.jvp(t + h, y, v, *args)
            return None, jvp_fn

        # Case 3: fall back to automatic differentiation
        def jvp_autodiff(y, v):
            _, df_v = jax.jvp(lambda y_: fun(t + h, y_, *args), (y,), (v,))
            return v - h * df_v

        return None, jvp_autodiff

    def step(self, fun, t: Array, y: Array, h: Array, args=()):
        """Advance one backward Euler step."""

        # If using a spectral solver, construct diagonal symbol (1 - h s(k))
        if hasattr(self.root_finder, 'linsolver'):
            linsolver = self.root_finder.linsolver
            if hasattr(linsolver, 'eigvals'):
                linsolver.set_symbol(1.0 - h * linsolver.eigvals)

        residual = self._residual(fun, t, y, h, args)

        # cheap forward Euler guess
        y0 = y + h * fun(t, y, *args)

        jac_fn, jvp_fn = self._build_linearisation(fun, t, h, args)

        return self.root_finder(residual, y0, jac_fn=jac_fn, jvp_fn=jvp_fn)