"""Time-step sensitivity analysis for Burgers' equation with the implicit Euler solver."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax_fno.solvers as solve_ivp


def sensitivity_analysis(nx, dt_values, make_plots=True):
    """
    Sensitivity analysis for Burgers' equation with different time step sizes.

    # Arguments
        nx: Spatial grid resolution
        dt_values: Time step sizes
    """
    
    print("Testing time step convergence for implicit solver...")
    
    # Problem setup
    L = 1.0
    t_span = (0.0, 0.5)
    bc_type = solve_ivp.BCType.PERIODIC
    params = {
        'nu': 0.1,
        'bc_type': bc_type,
        'bc_left': 0.0,  # dummy value
        'bc_right': 0.0  # dummy value
    }
    x = solve_ivp.create_uniform_grid(L, nx, bc_type)
    setup = {'grid': x, 'nu': params['nu'], 'L': L}

    # Initial condition
    u0 = jnp.sin(2 * jnp.pi * x / L) + 0.5 * jnp.sin(4 * jnp.pi * x / L)

    print("\nSolving with different time steps:")
    
    solutions = {}
    for dt in dt_values:
        print(f"\ndt = {dt:.0e}")
        
        t, u = solve_ivp.solve(
            u0, t_span, L,
            solve_ivp.burgers_residual_1d,
            params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=dt,
            tol=1e-8
        )
        
        solutions[dt] = (t, u)
        
        print(f"  Time steps: {len(t)}")
        print(f"  Final solution range: [{jnp.min(u[-1]):.3f}, {jnp.max(u[-1]):.3f}]")

    print("\n" + "="*50)
    print("Comparing results...")
    print("="*50)
    
    # Get smallest time step
    dt_values = list(solutions.keys())
    dt_fine = min(dt_values)  # Finest time step as reference
    t_ref, u_ref = solutions[dt_fine]
    
    print(f"Using dt = {dt_fine:.0e} as reference solution")
    
    # Compute relative error to finest time step
    errors = {}
    for dt in dt_values:
        if dt == dt_fine:
            continue
        t, u = solutions[dt]
        error = jnp.linalg.norm(u[-1] - u_ref[-1]) / jnp.linalg.norm(u_ref[-1])
        errors[dt] = error
        print(f"dt = {dt:.0e}: Relative error = {error:.2e}")

    if make_plots == True:  # Generate plots
        try:
            plot_solutions(setup, solutions, errors)
        except ImportError:
            print("Matplotlib not available - skipping plots")

    return setup, solutions, errors


def plot_solutions(setup, solutions, errors):
    """Plot solutions for visual comparison."""
    
    print("\nGenerating solution plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = setup['grid']
    dt_values = list(solutions.keys())
    dt_fine = min(dt_values)
    
    # Plot initial and final solutions for each time step
    for dt in sorted(solutions.keys()):
        t, u = solutions[dt]
        if dt == dt_fine:
            # Reference solution
            ax1.plot(x, u[0], 'k--', alpha=0.7, label='Initial condition')
            ax1.plot(x, u[-1], 'k-', linewidth=2, label=f'dt = {dt:.0e} (reference)')
        else:
            ax1.plot(x, u[-1], '--', alpha=0.8, label=f'dt = {dt:.0e}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x, T)')
    ax1.set_title('Final Solutions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot error vs time step    
    errs = [errors[dt] for (i, dt) in enumerate(dt_values)]
    ax2.loglog(dt_values, errs, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time step dt')
    ax2.set_ylabel('Relative error')
    ax2.set_title('Convergence vs Time Step')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timestep_convergence.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'timestep_convergence.png'")
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Time step sensitivity analysis for Burgers' Equation with 'solve_ivp'")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    
    # Run convergence test
    nx = 4096
    dt_values = [1e-1, 1e-2, 1e-3, 1e-4]
    solutions = sensitivity_analysis(nx, dt_values)