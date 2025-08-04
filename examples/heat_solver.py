import jax.numpy as jnp
from matplotlib import pyplot as plt
import jax_fno.solvers as solve_ivp

def analytical_diffusion_solution(x, t, D, L):
    """
    Analytical solution for 1D diffusion equation with:
    - Initial condition: u(x, 0) = exp(-(x-L/2)**2 / (4 * D * t0)) / j(4 * pi * D * t0) 
    - Boundary conditions: u(0, t) = u(L, t) = 0
    - Solution: u(x, t) = exp(-(x-L/2)**2 / (4 * D * t)) / (4 * pi * D * t) 
    """
    return jnp.exp(-(x - L/2)**2 / (4 * D * t)) / jnp.sqrt(4 * jnp.pi * D * t) 

def main(L=100.0, nx=100, D=2.0, t_span=(1.0, 10.0), dt=0.01):
    """
    Solve the diffusion equation in 1D and plot the initial and 
    final solutions found numerically and analytically.

    Arguments:
        L - Domain length (default 100.0)
        nx - Number of grid points (default 100)
        D - Diffusion coefficient (default 2.0)
        t_span - Simulation time (default (1.0, 10.0))
        dt - Time step size (default 0.01)
    """
    # Set up system
    L = 100.0  # domain length
    nx = 100  # number of grid points
    D = 2.0  # diffusivity
    bc_type = solve_ivp.BCType.DIRICHLET  # boundary condition type 
    bc_values = (0.0, 0.0)  # boundary condition values
    x = solve_ivp.create_uniform_grid(L, nx, bc_type)  # grid

    # Resolution
    dx = x[1] - x[0]

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
    u0 = analytical_diffusion_solution(x, t_span[0], D, L)

    # Solve the equation
    print("Solving...")
    t, u = solve_ivp.solve(u0, t_span, L, residual_fn, params, jvp_fn=solve_ivp.heat_jvp_1d, dt=dt)
    print("Solve finished.")

    # Compute the analytical solution
    u_analytical = [analytical_diffusion_solution(x, t[i], D, L) for i in range(len(t))]
            
    # Compare at several time points
    for i in range(0, len(t), len(t)//5):  # Check 5 time points
        time = t[i]
        error = jnp.linalg.norm(u[i] - u_analytical[i]) / jnp.linalg.norm(u_analytical[i])
        print(f"t={time:.3f}: relative error = {error:.3e}")

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(x, u[0, :], '-', marker='.', label=f"Numerical, $t={t[0]:.2f}$")
    ax.plot(x, u_analytical[0], '--', label=f"Analytical, $t={t[0]:.2f}$")
    ax.plot(x, u[-1, :], '-', marker='.', label=f"Numerical, $t={t[-1]:.2f}$")
    ax.plot(x, u_analytical[-1], '--', label=f"Analytical, $t={t[-1]:.2f}$")
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    plt.show()    

if __name__ == "__main__":
    main()