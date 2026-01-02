"""Linear solvers using spectral methods."""

from typing import Callable, Union, Optional

from flax import nnx
from jax import Array
import jax.numpy as jnp

from ..custom_types import LinearMap


class Spectral(nnx.Module):
    """Spectral linear solver used in implicit time-stepping schemes."""

    def __init__(
        self,
        eigvals: Array,
        forward_transform: Callable[[Array], Array],
        backward_transform: Callable[[Array], Array],
        constraint: Callable[[Array], Array] = lambda f: f,
    ):
        """
        Create a spectral linear solver for implicit time-stepping.

        The solver stores the eigenvalues s(k) of a linear operator A.
        When used in a time-stepping scheme, the time stepper computes
        inv_symbol = f(eigvals, h) at each step and updates the solver 
        before solving.

        Args:
            base_symbol: Eigenvalues s(k) of a linear operator A.
            forward_transform: Transform to eigenvector basis.
            backward_transform: Transform back to original domain.
            constraint: Optional constraint applied to RHS before transforming.

        Example (1D heat equation with Backward Euler):
            ```python
            import jax.numpy as jnp
            from jax_fno.integrate import solve_ivp, BackwardEuler, NewtonRaphson
            from jax_fno.integrate.linsolvers import Spectral

            # Heat equation: du/dt = D * laplacian(u)
            n = 128
            dx = 1.0 / n
            k = 2 * jnp.pi * jnp.fft.rfftfreq(n, d=dx)

            # Compute discrete Laplacian eigenvalues
            eigvals = -(4 * D / dx**2) * jnp.sin(k * dx / 2)**2

            # Create solver
            solver = Spectral(
                eigvals=eigvals,
                forward_transform=jnp.fft.rfft,
                backward_transform=lambda x: jnp.fft.irfft(x, n=n),
            )

            # Time-stepper automatically updates inv_symbol at each step
            method = BackwardEuler(root_finder=NewtonRaphson(linsolver=solver))
            t, y = solve_ivp(heat_rhs, (0, 1), y0, method, step_size=0.01)
            ```
        """
        self.forward_transform = forward_transform
        self.backward_transform = backward_transform
        self.constraint = constraint
        self._eigvals = nnx.Variable(eigvals)
        self._symbol = nnx.Variable(jnp.full(eigvals.shape, jnp.nan))

    @property
    def eigvals(self) -> Array:
        """Get eigenvalues s(k)."""
        return self._eigvals.get_value()

    def set_symbol(self, symbol: Array):
        """
        Update the symbol.

        Called by time-steppers to update operator for current timestep.

        Args:
            symbol: New inverse symbol to use for solving
        """
        self._symbol.set_value(symbol)

    def __call__(
        self,
        A: Union[LinearMap, Array, None],
        b: Array,
        x0: Optional[Array] = None
    ) -> Array:
        """
        Solve A*x = b.

        Args:
            A: Ignored (for interface compatibility)
            b: Right-hand side array with shape matching inv_symbol
            x0: Ignored (for interface compatibility)

        Returns:
            Solution x with same shape as b

        Raises:
            ValueError: If inverse symbol has not been set via set_inv_symbol()
        """
        # Invert symbol
        symbol = self._symbol.get_value()
        Ainv = jnp.where(symbol > 1e-15, 1.0 / symbol, 0.0)

        # Apply constraint to RHS
        b_constrained = self.constraint(b)

        # Transform to frequency domain
        b_hat = self.forward_transform(b_constrained)

        # Multiply by inverse symbol
        x_hat = Ainv * b_hat

        # Transform back to physical domain
        x = self.backward_transform(x_hat)

        return jnp.real(x)


def dst1(x):
    """Discrete sine transform (type 1)."""
    N = len(x)
    extended = jnp.concatenate([jnp.array([0.0]), x, jnp.array([0.0]), -x[::-1]])
    rfft_result = jnp.fft.rfft(extended)
    X = -rfft_result[1:N+1].imag / 2
    return X


def idst1(X):
    """Inverse discrete sine transform (type 1)."""
    N = len(X)
    extended = jnp.concatenate([jnp.array([0.0]), X, jnp.array([0.0]), -X[::-1]])
    rfft_result = jnp.fft.rfft(extended)
    x = -rfft_result[1:N+1].imag / (N + 1)
    return x