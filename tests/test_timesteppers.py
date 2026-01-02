"""Unit tests for the time stepping schemes."""

import pytest
import jax.numpy as jnp

from jax_fno.integrate import (
    ForwardEuler,
    RK4,
    BackwardEuler,
    NewtonRaphson,
    GMRES,
    Spectral,
    solve_ivp,
)

from jax_fno.integrate.linsolvers import dst1, idst1

def laplacian_dirichlet_1d(
    u: jnp.ndarray,
    bc_left: float,
    bc_right: float,
    dx: float
) -> jnp.ndarray:
    """
    Compute the Laplacian (second derivative) using finite differences.
    Assumes ghost points at the boundaries with Dirichlet conditions.

    Args:
        u: Solution at interior grid points
        bc_left: Boundary condition value at left boundary (x=0)
        bc_right: Boundary condition value at right boundary (x=L)
        dx: Grid spacing

    Returns:
        Second derivative d²u/dx² at interior points
    """
    dudx = jnp.diff(u, prepend=bc_left, append=bc_right)
    return jnp.diff(dudx) / dx**2


def heat_rhs_dirichlet(
    t: float,
    u: jnp.ndarray,
    diffusivity: float,
    bc_left: float,
    bc_right: float,
    dx: float,
) -> jnp.ndarray:
    """
    Right-hand side of heat equation: du/dt = D d²u/dx²

    Args:
        t: Current time
        u: Solution at interior grid points
        diffusivity: Thermal diffusivity D
        bc_left: Boundary condition at x=0
        bc_right: Boundary condition at x=L
        dx: Grid spacing

    Returns:
        Time derivative du/dt
    """
    d2udx2 = laplacian_dirichlet_1d(u, bc_left, bc_right, dx)
    return diffusivity * d2udx2


def gaussian_ic(x: jnp.ndarray, t: float, D: float, L: float) -> jnp.ndarray:
    """
    Gaussian initial condition for the heat equation.

    Analytical solution: u(t, x) = (1/√(4πDt)) exp(-(x-L/2)²/(4Dt))

    Args:
        x: Spatial coordinates
        t: Time
        D: Diffusivity
        L: Domain length

    Returns:
        Temperature distribution at time t
    """
    k = 1.0 / jnp.sqrt(4.0 * jnp.pi * D * t)
    return k * jnp.exp(-((x - L / 2.0)**2) / (4.0 * D * t))


@pytest.fixture
def heat_equation_setup():
    """
    Setup for 1D heat equation with Dirichlet BCs.

    Problem: du/dt = D d²u/dx² on [0, L] with u(0,t) = u(L,t) = 0
    Initial condition: Gaussian centered at L/2
    """
    # Physical parameters
    D = 2.0  # diffusivity
    L = 100.0  # domain length
    n = 32  # number of interior grid points
    dx = L / (n + 1)  # grid spacing
    bc_values = (0.0, 0.0)  # Dirichlet boundary condition values

    # Spatial grid (interior points only)
    x = jnp.linspace(dx, L - dx, n, endpoint=True)

    return {
        'D': D,
        'L': L,
        'n': n,
        'dx': dx,
        'x': x,
        'bc_values': bc_values,
    }


class TestExplicitMethods:
    """Test explicit time-stepping methods on the heat equation."""

    def test_forward_euler(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        # Time span
        t_start = 1.0
        t_end = 1.5

        # Compute time step size
        cfl = 0.2  # CFL number
        dt = cfl * dx**2 / D

        # Initial condition
        y0 = gaussian_ic(x, t_start, D, L)

        # Solve using Forward Euler
        method = ForwardEuler()
        t_final, y_final = solve_ivp(
            heat_rhs_dirichlet,
            (t_start, t_end),
            y0,
            method,
            step_size=dt,
            args=(D, bc_left, bc_right, dx)
        )

        # Compare with analytical solution
        y_exact = gaussian_ic(x, t_end, D, L)

        # Check that solution is close to analytical
        assert jnp.allclose(y_final, y_exact, atol=9*dt)

    def test_rk4(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        # Time span
        t_start = 1.0
        t_end = 5.0

        # Compute time step size
        cfl = 0.2  # CFL number
        dt = cfl * dx**2 / D

        # Initial condition
        y0 = gaussian_ic(x, t_start, D, L)

        # Solve using RK4
        method = RK4()
        t_final, y_final = solve_ivp(
            heat_rhs_dirichlet,
            (t_start, t_end),
            y0,
            method,
            step_size=dt,
            args=(D, bc_left, bc_right, dx)
        )

        # Compare with analytical solution
        y_exact = gaussian_ic(x, t_end, D, L)

        # Check that solution is close to analytical
        assert jnp.allclose(y_final, y_exact, atol=9*(dt**2))


class TestImplicitMethods:
    """Test implicit time-stepping methods on the heat equation."""

    def test_backward_euler_gmres(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        # Time span
        t_start = 1.0
        t_end = 5.0

        # Compute time step size
        cfl = 1.5  # CFL number
        dt = cfl * dx**2 / D

        # Initial condition
        y0 = gaussian_ic(x, t_start, D, L)

        # Configure implicit method
        linsolver = GMRES(tol=1e-8, maxiter=100)
        root_finder = NewtonRaphson(linsolver=linsolver, tol=1e-8, maxiter=20)
        method = BackwardEuler(root_finder=root_finder)

        # Solve using Backward Euler
        t_final, y_final = solve_ivp(
            heat_rhs_dirichlet,
            (t_start, t_end),
            y0,
            method,
            step_size=dt,
            args=(D, bc_left, bc_right, dx)
        )

        # Compare with analytical solution
        y_exact = gaussian_ic(x, t_end, D, L)

        # Check that solution is close to analytical
        assert jnp.allclose(y_final, y_exact, atol=9*dt)

    def test_backward_euler_spectral(self, heat_equation_setup):
        setup = heat_equation_setup
        D = setup['D']
        L = setup['L']
        dx = setup['dx']
        x = setup['x']
        bc_left, bc_right = setup['bc_values']

        # Time span
        t_start = 1.0
        t_end = 5.0

        # Compute time step size
        cfl = 1.5  # CFL number
        dt = cfl * dx**2 / D

        # Initial condition
        y0 = gaussian_ic(x, t_start, D, L)

        # Configure implicit method
        n = len(x)
        k = jnp.arange(1, n+1)
        eigvals = -D * (4 / dx**2) * jnp.sin(jnp.pi * k / (2 * (n + 1)))**2
        linsolver = Spectral(eigvals=eigvals, forward_transform=dst1, backward_transform=idst1)
        root_finder = NewtonRaphson(linsolver=linsolver, tol=1e-8, maxiter=1)
        method = BackwardEuler(root_finder=root_finder)

        # Solve using Backward Euler
        t_final, y_final = solve_ivp(
            heat_rhs_dirichlet,
            (t_start, t_end),
            y0,
            method,
            step_size=dt,
            args=(D, bc_left, bc_right, dx)
        )

        # Compare with analytical solution
        y_exact = gaussian_ic(x, t_end, D, L)

        # Check that solution is close to analytical
        assert jnp.allclose(y_final, y_exact, atol=9*dt)