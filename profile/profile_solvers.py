"""
Simple profiling script for `solve_ivp` performance analysis.
"""

import time
import cProfile
import pstats
import io
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import jax_fno.solvers as solve_ivp


@contextmanager
def timer(description):
    """Simple timing context manager."""
    start = time.perf_counter()
    print(f"Starting: {description}")
    yield
    elapsed = time.perf_counter() - start
    print(f"Completed: {description} in {elapsed:.3f}s")


def create_test_problem(nx=256):
    """Create a simple test problem for profiling."""
    L = 1.0
    x = jnp.linspace(0, L, nx, endpoint=False)
    
    # Simple initial condition: sin wave
    u0 = jnp.sin(2 * jnp.pi * x)
    
    # Burgers equation parameters
    params = {
        'nu': 0.01,  # Lower viscosity for more challenging problem
        'bc_type': solve_ivp.BCType.PERIODIC,
        'bc_left': 0.0,
        'bc_right': 0.0
    }
    
    return u0, params, L


def profile_solve_ivp_cprofile(nx=256, n_runs=1):
    """Profile solve_ivp using cProfile."""
    print(f"\n{'='*60}")
    print(f"Profiling solve_ivp with cProfile (nx={nx}, runs={n_runs})")
    print(f"{'='*60}")
    
    u0, params, L = create_test_problem(nx)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    def run_solver():
        for _ in range(n_runs):
            t, u = solve_ivp.solve(
                initial_condition=u0,
                t_span=(0.0, 0.1),  # Short time span for profiling
                L=L,
                residual_fn=solve_ivp.burgers_residual_1d,
                parameters=params,
                jvp_fn=solve_ivp.burgers_jvp_1d,
                dt=1e-4,
                tol=1e-6,
                maxiter=10
            )
            # Block until computation is complete for accurate profiling
            u.block_until_ready()
    
    # Profile the solver
    profiler.enable()
    run_solver()
    profiler.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(s.getvalue())
    
    return profiler


def profile_compilation_overhead():
    print(f"\n{'='*60}")
    print("JAX Compilation Overhead Analysis")
    print(f"{'='*60}")
    
    u0, params, L = create_test_problem(64)  # Small problem for compilation test
    
    # Enable JAX compilation tracking
    jax.config.update('jax_log_compiles', True)
    
    # First call (includes compilation)
    print("=== FIRST CALL (expect compilation) ===")
    with timer("First call (with compilation)"):
        t1, u1 = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.01),  # Very short
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u1.block_until_ready()
    
    # Second call (should be compiled)
    print("\n=== SECOND CALL (should be fast, no compilation) ===")
    with timer("Second call (compiled)"):
        t2, u2 = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u2.block_until_ready()
    
    # Third call with different shape (will cause recompilation)
    print("\n=== THIRD CALL with different shape (expect recompilation) ===")
    u0_different = create_test_problem(128)[0]  # Different size
    params_different, L_different = params, L
    
    with timer("Third call (different shape)"):
        t3, u3 = solve_ivp.solve(
            initial_condition=u0_different,
            t_span=(0.0, 0.01),
            L=L_different,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params_different,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u3.block_until_ready()
    
    # Fourth call with original shape (should use cached compilation)
    print("\n=== FOURTH CALL back to original shape (should be fast again) ===")
    with timer("Fourth call (back to original shape)"):
        t4, u4 = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u4.block_until_ready()

    # Disable JAX compilation tracking
    jax.config.update('jax_log_compiles', False)


def test_recompilation_triggers():
    print(f"\n{'='*60}")
    print("Testing Recompilation Triggers")
    print(f"{'='*60}")
    
    # Base case
    u0, params, L = create_test_problem(64)

    # Enable JAX compilation tracking
    jax.config.update('jax_log_compiles', True)
    
    print("=== BASE CASE ===")
    with timer("Base case"):
        t, u = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u.block_until_ready()
    
    print("\n=== SAME INPUTS (should reuse compilation) ===")
    with timer("Same inputs"):
        t, u = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u.block_until_ready()
    
    print("\n=== DIFFERENT ARRAY SHAPE (triggers recompilation) ===")
    u0_bigger, _, _ = create_test_problem(128)
    with timer("Different array shape"):
        t, u = solve_ivp.solve(
            initial_condition=u0_bigger,
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u.block_until_ready()
    
    print("\n=== DIFFERENT SCALAR VALUES (should re-use compilation) ===")
    params_different = params.copy()
    params_different['nu'] = 0.02  # Different viscosity
    with timer("Different scalar parameter"):
        t, u = solve_ivp.solve(
            initial_condition=u0,  # Same shape as base case
            t_span=(0.0, 0.01),
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params_different,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u.block_until_ready()
    
    print("\n=== DIFFERENT TIME SPAN (triggers recompilation) ===")
    with timer("Different time span"):
        t, u = solve_ivp.solve(
            initial_condition=u0,
            t_span=(0.0, 0.02),  # Different end time
            L=L,
            residual_fn=solve_ivp.burgers_residual_1d,
            parameters=params,
            jvp_fn=solve_ivp.burgers_jvp_1d,
            dt=1e-4,
            tol=1e-6,
            maxiter=5
        )
        u.block_until_ready()

    # Disable JAX compilation tracking
    jax.config.update('jax_log_compiles', False)


def main():
    print("Profiling `solve_ivp`...")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Compilation overhead and recompilation triggers
    profile_compilation_overhead()
    test_recompilation_triggers()
    
    # cProfile analysis
    profile_solve_ivp_cprofile(nx=256, n_runs=3)
    
    print(f"\n{'='*60}")
    print("Profiling finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()