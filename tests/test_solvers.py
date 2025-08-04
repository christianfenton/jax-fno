"""Unit tests for the initial value problem solver."""

import pytest
import jax.numpy as jnp
import jax_fno.solvers as solve_ivp


class TestFiniteDifferences:
    """Test finite difference operators."""
        
    def test_dirichlet_first_derivative(self):
        """Test first derivative with Dirichlet BC."""
        L = 1.0
        nx = 6 
        dx = L / (nx + 1) 
        
        # Interior grid points
        x = jnp.linspace(dx, L - dx, nx)
        
        # Test with linear function: u(x) = x, so du/dx = 1 everywhere
        u = x.copy()
        
        # Compute derivative using Dirichlet BCs (boundaries at 0 and L)
        du_dx = solve_ivp.d__dx_c_dirichlet(u, dx, u_left=0.0, u_right=L)
        
        # For a linear function, derivative should be constant = 1
        expected = jnp.ones_like(u)
        
        assert du_dx.shape == u.shape
        assert jnp.allclose(du_dx, expected, atol=dx**2)

    def test_dirichlet_second_derivative(self):
        """Test second derivative with Dirichlet BCs."""
        L = 1.0
        nx = 6  # Interior points only
        dx = L / (nx + 1)
        
        # Interior grid points  
        x = jnp.linspace(dx, L - dx, nx)
        
        # Test with quadratic function: u(x) = x^2, so d2u/dx2 = 2 everywhere
        u = x**2
        
        # Compute second derivative using Dirichlet BCs
        d2u_dx2 = solve_ivp.d2__dx2_c_dirichlet(u, dx, u_left=0.0, u_right=L**2)
        
        # For a quadratic function, second derivative should be constant = 2
        expected = jnp.full_like(u, 2.0)
        
        assert d2u_dx2.shape == u.shape
        assert jnp.allclose(d2u_dx2, expected, atol=dx**2)
        
    def test_periodic_first_derivative(self):
        """Test first derivative with periodic BC."""
        L = 1.0
        nx = 16
        dx = L / nx
        
        # Periodic grid (no endpoint, wraparound assumed)
        x = jnp.linspace(0, L, nx, endpoint=False)
        
        # Test with sine function: u(x) = sin(2πx/L)
        # du/dx = (2π/L) * cos(2πx/L)
        u = jnp.sin(2 * jnp.pi * x / L)
        
        # Compute derivative using periodic BCs
        du_dx = solve_ivp.d__dx_c_periodic(u, dx)
        
        # Expected analytical derivative
        expected = (2 * jnp.pi / L) * jnp.cos(2 * jnp.pi * x / L)

        # Use theoretical error bound: (dx²/6) * |f'''|_max
        theoretical_tol = (dx**2 / 6) * (2 * jnp.pi / L)**3
        
        assert du_dx.shape == u.shape
        assert jnp.allclose(du_dx, expected, atol=theoretical_tol)
        
    def test_periodic_second_derivative(self):
        """Test second derivative with periodic BC."""
        L = 1.0
        nx = 16
        dx = L / nx
        
        # Periodic grid
        x = jnp.linspace(0, L, nx, endpoint=False)
        
        # Test with sine function: u(x) = sin(2πx/L)
        # d2u/dx2 = -(2π/L)^2 * sin(2πx/L)
        u = jnp.sin(2 * jnp.pi * x / L)
        
        # Compute second derivative using periodic BCs
        d2u_dx2 = solve_ivp.d2__dx2_c_periodic(u, dx)
        
        # Expected analytical second derivative
        expected = -(2 * jnp.pi / L)**2 * jnp.sin(2 * jnp.pi * x / L)

        # Use theoretical error bound
        theoretical_tol = (dx**2 / 12) * (2 * jnp.pi / L)**4
        
        assert d2u_dx2.shape == u.shape
        assert jnp.allclose(d2u_dx2, expected, atol=theoretical_tol)


class TestSolver:
    """Test the main solver function."""
    
    def test_solver_simple_pde(self):
        """Test solver with a simple algebraic PDE: u^2 = 2 (so u = sqrt(2))."""
        
        def simple_residual(u_new, u_old, dt, dx, params):
            """Simple PDE: u^2 - 2 = 0, solution should be u = sqrt(2)."""
            # Ignore time stepping for this algebraic equation
            return u_new**2 - 2.0
        
        def simple_jvp(u, dt, dx, params, v):
            """Jacobian vector product: d/du([u^2 - 2]v) = 2uv."""
            return 2.0 * u * v
        
        # Problem setup (interior points for Dirichlet-like setup)
        L = 1.0
        nx = 4
        
        # Initial condition (far from solution)
        u0 = jnp.ones(nx) * 0.5
        
        # Parameters (empty for this simple case)
        params = {}
        
        # Solve for steady state (large dt, single step)
        t, u = solve_ivp.solve(
            u0, (0.0, 0.1), L,
            simple_residual,
            params,
            jvp_fn=simple_jvp,
            dt=0.1,
            tol=1e-10
        )
        
        # Check final solution
        u_final = u[-1]  # Last time step
        expected = jnp.full(nx, jnp.sqrt(2.0))
        
        print(f"Final solution: {u_final}")
        print(f"Expected: {expected}")
        print(f"Error: {jnp.linalg.norm(u_final - expected):.2e}")
        
        assert jnp.allclose(u_final, expected, atol=1e-6)

    def test_solver_simple_pde_autodiff(self):
        """Test solver with a simple algebraic PDE: u^2 = 2 (so u = sqrt(2))."""
        
        def simple_residual(u_new, u_old, dt, dx, params):
            """Simple PDE: u^2 - 2 = 0, solution should be u = sqrt(2)."""
            # Ignore time stepping for this algebraic equation
            return u_new**2 - 2.0
        
        # Problem setup (interior points for Dirichlet-like setup)
        L = 1.0
        nx = 4
        
        # Initial condition (far from solution)
        u0 = jnp.ones(nx) * 0.5
        
        # Parameters (empty for this simple case)
        params = {}
        
        # Solve for steady state (large dt, single step)
        t, u = solve_ivp.solve(
            u0, (0.0, 0.1), L,
            simple_residual,
            params,
            jvp_fn=None,
            dt=0.1,
            tol=1e-10
        )
        
        # Check final solution
        u_final = u[-1]  # Last time step
        expected = jnp.full(nx, jnp.sqrt(2.0))
        
        print(f"Final solution: {u_final}")
        print(f"Expected: {expected}")
        print(f"Error: {jnp.linalg.norm(u_final - expected):.2e}")
        
        assert jnp.allclose(u_final, expected, atol=1e-6)


class TestBoundaryConditions:
    """Test different boundary condition implementations."""
    
    def test_periodic_grid_setup(self):
        L = 2.0
        nx = 8
        dx = L / nx
        bc_type = solve_ivp.BCType.PERIODIC
        x = solve_ivp.create_uniform_grid(L, nx, bc_type)
        x_true = jnp.linspace(0, L, nx, endpoint=False)
        assert jnp.allclose(jnp.diff(x), dx)
        assert jnp.allclose(x, x_true)
        
    def test_dirichlet_grid_setup(self):
        L = 2.0
        nx = 8  # Interior points
        dx = L / (nx + 1)
        bc_type = solve_ivp.BCType.DIRICHLET
        x = solve_ivp.create_uniform_grid(L, nx, bc_type)
        x_true = jnp.linspace(dx, L - dx, nx)  # Interior points only
        assert jnp.allclose(jnp.diff(x), dx)
        assert jnp.allclose(x, x_true)


class TestEquations:
    """Test the standard equation implementations."""

    def test_heat_equation(self):
        """Solve the diffusion equation in 1D and compare with an analytical solution."""

        def analytical_solution(x, t, D, L):
            """
            Analytical solution for 1D diffusion equation with:
            - Initial condition: u(x, 0) = exp(-(x-L/2)**2 / (4 * D * t0)) / j(4 * pi * D * t0) 
            - Boundary conditions: u(0, t) = u(L, t) = 0
            - Solution: u(x, t) = exp(-(x-L/2)**2 / (4 * D * t)) / (4 * pi * D * t) 
            """
            return jnp.exp(-(x - L/2)**2 / (4 * D * t)) / jnp.sqrt(4 * jnp.pi * D * t) 

        # Set up system
        L = 100.0  # domain length
        nx = 32  # number of grid points
        D = 2.0  # diffusivity
        bc_type = solve_ivp.BCType.DIRICHLET  # boundary condition type 
        bc_values = (0.0, 0.0)  # boundary condition values
        x = solve_ivp.create_uniform_grid(L, nx, bc_type)  # grid

        # Store PDE parameters in a dictionary
        params = {
            'D': D,
            'bc_type': bc_type,
            'bc_left': bc_values[0],
            'bc_right': bc_values[1]
            }

        # Load pre-defined functions for the heat equation
        # Users have to write their own functions to use the solver for other equations
        residual_fn = solve_ivp.heat_residual_1d

        # Time parameters
        t_span = (1.0, 10.0)

        # Create Gaussian initial condition
        u0 = analytical_solution(x, t_span[0], D, L)

        # Solve the equation
        t, u_numerical = solve_ivp.solve(u0, t_span, L, residual_fn, params, jvp_fn=solve_ivp.heat_jvp_1d, dt=0.1)

        # Compute the analytical solution
        u_analytical = [analytical_solution(x, t[i], D, L) for i in range(len(t))]
                
        # Compare at several time points
        for i in range(0, len(t), len(t)//5):  # Check 5 time points
            u_analytical = analytical_solution(x, t[i], D, L)
            error = jnp.linalg.norm(u_numerical[i, :] - u_analytical) / jnp.linalg.norm(u_analytical)
            assert error < 1e-1