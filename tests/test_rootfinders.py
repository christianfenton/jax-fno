"""Unit tests for the root-finding algorithms."""

import pytest
import jax.numpy as jnp

from jax_fno.integrate.rootfinders import NewtonRaphson
from jax_fno.integrate.linsolvers import GMRES, Direct


@pytest.fixture
def simple_nonlinear_system():
    """
    Non-linear system: 2y (sqrt(2) - y) = 0

    This has an attractive fixed point at y=sqrt(2) and an unstable fixed point
    at y=0.

    Initial condition: y(t=0) = 0.5
    Expected solution: y(t=t') = sqrt(2), where t' >> 0.
    """
    R = lambda y: 2.0 * y * (jnp.sqrt(2.0) - y)  # ODE
    jvp = lambda y, v: 2.0 * (jnp.sqrt(2.0) - 2.0 * y) * v  # Jacobian-vector product
    jac = lambda y: jnp.diag(2.0 * (jnp.sqrt(2.0) - 2.0 * y))  # Jacobian
    y0 = jnp.ones((4,))  # Initial condition
    soln = jnp.full_like(y0, jnp.sqrt(2.0))  # Solution
    return R, jvp, jac, y0, soln


class TestRootFinders:

    def test_newton_raphson_matrix_free(self, simple_nonlinear_system):
        root_finder = NewtonRaphson(tol=1e-6, maxiter=50, linsolver=GMRES())
        residual, jvp, _, y0, expected = simple_nonlinear_system
        soln = root_finder.solve(residual, y0, jvp_fn=jvp)
        assert jnp.allclose(soln, expected, atol=1e-5, rtol=1e-5)

    def test_newton_raphson_dense_matrix(self, simple_nonlinear_system):
        root_finder = NewtonRaphson(tol=1e-6, maxiter=50, linsolver=Direct())
        residual, _, jac, y0, expected = simple_nonlinear_system
        soln = root_finder.solve(residual, y0, jac_fn=jac)
        assert jnp.allclose(soln, expected, atol=1e-5, rtol=1e-5)
