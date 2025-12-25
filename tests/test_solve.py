"""Unit/integration tests for jax_fno.integrate."""

import pytest
import jax.numpy as jnp

from jax_fno.integrate import (
    solve_ivp, solve_with_history, BackwardEuler, NewtonRaphson
)


@pytest.fixture
def simple_dynamical_system():
    """
    ODE: dy/dt = 2y (sqrt(2) - y)

    This has an attractive fixed point at y=sqrt(2) and an unstable fixed point
    at y=0.

    Initial condition: y(t=0) = 0.5
    Expected solution: y(t=t') = sqrt(2), where t' >> 0.
    """
    fun = lambda t, y: 2.0 * y * (jnp.sqrt(2.0) - y)  # ODE
    jvp = lambda t, y, v: 2.0 * (jnp.sqrt(2.0) - 2.0 * y) * v  # Jacobian-vector product
    jac = lambda t, y: jnp.diag(2.0 * (jnp.sqrt(2.0) - 2.0 * y))  # Jacobian
    y0 = jnp.ones((4,))  # Initial condition
    soln = jnp.full_like(y0, jnp.sqrt(2.0))  # Solution
    return fun, jvp, jac, y0, soln


class TestODEIntegrator:

    def test_implicit_jvp(self, simple_dynamical_system):
        """
        Test implicit time stepping using a matrix-free Jacobian.
        """
        fun, jvp, _, y0, expected = simple_dynamical_system
        method = BackwardEuler(jvp=jvp, root_finder=NewtonRaphson(tol=1e-10))
        _, y_final = solve_ivp(fun, (0.0, 10.0), y0, method, step_size=0.1)
        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_implicit_autodiff(self, simple_dynamical_system):
        """
        Test implicit time stepping using automatic differentiation.

        Test case: dy/dt = 2y (sqrt(2) - y),    y(t=0) = 0.5
        This case has a fixed point at y = sqrt(2).
        """
        fun, _, _, y0, expected = simple_dynamical_system
        method = BackwardEuler(root_finder=NewtonRaphson(tol=1e-10))
        _, y_final = solve_ivp(fun, (0.0, 10.0), y0, method, step_size=0.1)
        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_implicit_jac(self, simple_dynamical_system):
        """
        Test implicit time stepping with a dense Jacobian.
        
        Test case: dy/dt = 2y (sqrt(2) - y),    y(t=0) = 0.5
        This case has a fixed point at y = sqrt(2).
        """
        fun, _, jac, y0, expected = simple_dynamical_system
        method = BackwardEuler(root_finder=NewtonRaphson(tol=1e-10), jac=jac)
        _, y_final = solve_ivp(fun, (0.0, 10.0), y0, method, step_size=0.1)
        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_ode_eval(self):
        """Test solver, storing intermediate steps."""

        y0 = jnp.array([1.0])
        t_eval = jnp.array([0.0, 1.0, 2.0])
        h = 1e-2

        t, y = solve_with_history(
            lambda t, y: -y,  # dy/dt = -y
            (0.0, 2.0),
            y0,
            BackwardEuler(root_finder=NewtonRaphson(tol=1e-10)),
            t_eval=t_eval,
            step_size=h,
        )

        # Analytical solution: y(t) = exp(-t)
        expected = jnp.exp(-t_eval)

        assert jnp.allclose(t, t_eval, atol=1e-10)
        assert jnp.allclose(y[:, 0], expected, atol=h)
