"""Unit tests for the initial value problem solver."""

import pytest
import jax
import jax.numpy as jnp

from jax_fno.solver import solve_ivp
from jax_fno.solver.timesteppers.implicit import BackwardEuler


class TestSolver:
    """Test the main solver function."""

    def test_solver_simple_ode(self):
        """
        Test solver with a simple ODE: dy/dt = 2y(sqrt(2) - y).

        This ODE has equilibrium at y = sqrt(2). Starting from y(0) = 0.5,
        the solution should converge to sqrt(2).
        """

        def fun(t, y):
            """dy/dt = 2y(sqrt(2) - y)"""
            return 2.0 * y * (jnp.sqrt(2.0) - y)

        def jvp(t, y, v):
            """d/dy[2y(sqrt(2) - y)] * v = 2(sqrt(2) - 2y) * v"""
            return 2.0 * (jnp.sqrt(2.0) - 2.0 * y) * v

        # Initial condition
        y0 = jnp.array([0.5, 0.5, 0.5, 0.5])

        # Solve with BackwardEuler and custom JVP
        t, y = solve_ivp(
            fun, (0.0, 10.0), y0, BackwardEuler(), dt=0.1, tol=1e-10, jvp=jvp
        )

        y_final = y[-1]
        expected = jnp.full_like(y0, jnp.sqrt(2.0))

        print(f"Final solution: {y_final}")
        print(f"Expected: {expected}")
        print(f"Error: {jnp.linalg.norm(y_final - expected):.2e}")

        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_solver_simple_ode_autodiff(self):
        """
        Test solver with a simple ODE using automatic differentiation for JVP.
        """

        def fun(t, y):
            """dy/dt = 2y(sqrt(2) - y)"""
            return 2.0 * y * (jnp.sqrt(2.0) - y)

        # Initial condition
        y0 = jnp.array([0.5, 0.5, 0.5, 0.5])

        # Solve with BackwardEuler and autodiff (no explicit JVP)
        t, y = solve_ivp(
            fun, (0.0, 10.0), y0, BackwardEuler(), dt=0.1, tol=1e-10
        )

        y_final = y[-1]
        expected = jnp.full_like(y0, jnp.sqrt(2.0))

        print(f"Final solution: {y_final}")
        print(f"Expected: {expected}")
        print(f"Error: {jnp.linalg.norm(y_final - expected):.2e}")

        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_solver_simple_ode_dense_jacobian(self):
        """Test solver with dense Jacobian matrix."""

        def fun(t, y):
            """dy/dt = 2y(sqrt(2) - y)"""
            return 2.0 * y * (jnp.sqrt(2.0) - y)

        def jac(t, y):
            """d/dy[2y(sqrt(2) - y)] = diag(2(sqrt(2) - 2y))"""
            diag_vals = 2.0 * (jnp.sqrt(2.0) - 2.0 * y)
            return jnp.diag(diag_vals)

        # Initial condition
        y0 = jnp.array([0.5, 0.5, 0.5, 0.5])

        # Solve with BackwardEuler and dense Jacobian
        t, y = solve_ivp(
            fun, (0.0, 10.0), y0, BackwardEuler(), dt=0.1, tol=1e-10, jac=jac
        )

        y_final = y[-1]
        expected = jnp.full_like(y0, jnp.sqrt(2.0))

        print(f"Final solution: {y_final}")
        print(f"Expected: {expected}")
        print(f"Error: {jnp.linalg.norm(y_final - expected):.2e}")

        assert jnp.allclose(y_final, expected, atol=1e-6)

    def test_solver_with_t_eval(self):
        """Test solver with specified evaluation times."""

        def fun(t, y):
            """Simple exponential decay: dy/dt = -y."""
            return -y

        # Initial condition
        y0 = jnp.array([1.0])

        # Evaluation times
        t_eval = jnp.array([0.0, 1.0, 2.0])

        dt = 1e-2

        # Solve
        t, y = solve_ivp(
            fun,
            (0.0, 2.0),
            y0,
            BackwardEuler(),
            t_eval=t_eval,
            dt=dt,
            tol=1e-10,
        )

        # Analytical solution: y(t) = exp(-t)
        expected = jnp.exp(-t_eval)

        print(f"Times: {t}")
        print(f"Solution: {y[:, 0]}")
        print(f"Expected: {expected}")
        print(f"Max error: {jnp.max(jnp.abs(y[:, 0] - expected)):.2e}")

        assert jnp.allclose(t, t_eval, atol=1e-10)
        assert jnp.allclose(y[:, 0], expected, atol=dt)
