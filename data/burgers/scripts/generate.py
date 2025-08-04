"""
Data generation for Burgers equation following the FNO paper methodology.

Implements the data generation procedure from Li et al. (2020):
"Fourier Neural Operator for Parametric Partial Differential Equations"
https://arxiv.org/pdf/2010.08895

Initial conditions: u0 ~ N(0, 625(-Δ + 25I)^-2) with periodic BCs
Resolution: 2^10 = 1024 spatial points (original paper uses 2^13)
Viscosity: ν = 0.1
"""

import os
import time
import argparse
import jax
import jax.numpy as jnp
from typing import Tuple, Optional 
import jax_fno
import jax_fno.solvers as solve_ivp
import multiprocessing as mp


def configure_jax_threading():
    """Configure JAX for optimal CPU threading."""
    n_cores = mp.cpu_count()
    os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count={n_cores}'
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_cores)
    print(f"Configured JAX for {n_cores} CPU cores")


def sample_initial_condition(
    key,
    nx: int,
    L: float = 1.0
) -> jnp.ndarray:
    """
    Sample initial condition from Gaussian process prior N(0, 625(-Δ + 25I)^-2).
    
    This implements the exact covariance operator from the FNO paper.
    The operator (-Δ + 25I)^-2 corresponds to a Matérn covariance with specific
    smoothness properties.
    
    Args:
        key: JAX random key
        nx: Number of spatial points
        L: Domain length (default 1.0)
        
    Returns:
        Initial condition u0(x) with shape (nx,)
    """
    # Create spatial grid (periodic, no endpoint)
    dx = L / nx
    x = jnp.linspace(0, L, nx, endpoint=False)
    
    # Frequency grid for Fourier space operations
    k = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    
    # Covariance operator in Fourier space: 625 * (k^2 + 25)^-2
    # Note: k^2 corresponds to -Δ in Fourier space
    covariance_sqrt = 25.0 / (k**2 + 25.0)  # sqrt(625 * (k^2 + 25)^-2)
    
    # Handle the k=0 mode (constant component)
    covariance_sqrt = covariance_sqrt.at[0].set(25.0 / 25.0)  # = 1.0
    
    # Sample white noise in Fourier space
    # For real-valued output, we need to ensure Hermitian symmetry
    noise_real = jax.random.normal(key, (nx,))
    key, subkey = jax.random.split(key)
    noise_imag = jax.random.normal(subkey, (nx,))
    
    # Create complex noise with proper symmetry for real FFT
    noise_complex = noise_real + 1j * noise_imag
    
    # Apply symmetry constraints for real-valued result
    # For real-valued functions: F(k) = conj(F(-k))
    n_half = nx // 2
    noise_complex = noise_complex.at[n_half+1:].set(jnp.conj(noise_complex[1:n_half][::-1]))
    noise_complex = noise_complex.at[0].set(jnp.real(noise_complex[0]))  # DC component is real
    if nx % 2 == 0:
        noise_complex = noise_complex.at[n_half].set(jnp.real(noise_complex[n_half]))  # Nyquist is real
    
    # Apply covariance in Fourier space
    u0_fourier = covariance_sqrt * noise_complex
    
    # Transform back to physical space
    u0 = jnp.real(jnp.fft.ifft(u0_fourier))
    
    return u0


def generate_burgers_trajectory(
    initial_condition: jnp.ndarray,
    t_final: float = 1.0,
    nu: float = 0.1,
    L: float = 1.0,
    dt: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a single Burgers equation solution trajectory.
    
    Args:
        initial_condition: Initial condition u0(x)
        t_final: Final simulation time
        nu: Viscosity parameter (0.1 as in FNO paper)
        L: Domain length
        dt: Time step size
        
    Returns:
        Tuple of (time_array, solution_array)
    """
    # Set up parameters for solve_ivp
    params = {
        'nu': nu,
        'bc_type': solve_ivp.BCType.PERIODIC,
        'bc_left': 0.0,  # dummy value for solver
        'bc_right': 0.0  # dummy value for solver
    }
    
    # Solve
    t, u = solve_ivp.solve(
        initial_condition=initial_condition,
        t_span=(0.0, t_final),
        L=L,
        residual_fn=solve_ivp.burgers_residual_1d,
        parameters=params,
        jvp_fn=solve_ivp.burgers_jvp_1d,  # Use analytical Jacobian
        dt=dt,
        tol=1e-8,
        maxiter=20
    )
    
    return t, u


def solve_single_sample(args):
    """
    Solve a single sample (for multiprocessing).
    
    Args:
        args: Tuple of (sample_idx, sample_key, nx, L, t_final, nu, dt)
    
    Returns:
        Tuple of (sample_idx, initial_condition, final_solution, time_array)
    """
    sample_idx, sample_key, nx, L, t_final, nu, dt = args
    
    # Generate initial condition
    u0 = sample_initial_condition(sample_key, nx, L)
    
    # Solve trajectory
    t, u = generate_burgers_trajectory(u0, t_final, nu, L, dt)
    
    # Convert to numpy for multiprocessing
    import numpy as np
    return (sample_idx, np.array(u0), np.array(u[-1]), np.array(t))


def progress_callback(result):
    """Callback function to track progress of multiprocessing."""
    progress_callback.completed += 1
    total = progress_callback.total
    completed = progress_callback.completed
    
    # Calculate percentage
    percent = (completed / total) * 100
    
    # Print progress every 1% or at completion
    if completed == 1 or completed == total or percent >= progress_callback.next_milestone:
        print(f"Progress: {completed}/{total} ({percent:.1f}%)")
        progress_callback.next_milestone = int((percent // 1 + 1) * 1)  # Next 1% milestone


def generate_dataset(
    nx: int,
    n_samples: int,
    key,
    t_final: float = 1.0,
    nu: float = 0.1,
    L: float = 1.0,
    dt: float = 1e-4,
    n_processes: int = 4
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate dataset of Burgers equation solutions following FNO paper methodology.
    
    Args:
        nx: Spatial resolution
        n_samples: Number of trajectory samples to generate
        key: JAX random key
        t_final: Final simulation time
        nu: Viscosity parameter
        L: Domain length
        dt: Time step size
        n_processes: Number of parallel processes (default: 4)
        
    Returns:
        Tuple of (initial_conditions, final_solutions, time_array)
        - initial_conditions: (n_samples, nx) 
        - final_solutions: (n_samples, nx)
        - time_array: (n_time_steps,)
    """
    print(f"Generating {n_samples} samples using {n_processes} processes...")
    
    # Initialize progress tracking
    progress_callback.completed = 0
    progress_callback.total = n_samples
    progress_callback.next_milestone = 1.0
    
    # Generate random keys for all samples
    keys = jax.random.split(key, n_samples)
    
    # Create argument tuples for each sample
    args_list = [
        (i, keys[i], nx, L, t_final, nu, dt)
        for i in range(n_samples)
    ]
    
    # Run multiprocessing with progress tracking
    with mp.Pool(processes=n_processes) as pool:
        results = []
        # Use map_async with callback to track progress
        for args in args_list:
            result = pool.apply_async(solve_single_sample, (args,), callback=progress_callback)
            results.append(result)
        
        # Wait for all results and collect them
        completed_results = [result.get() for result in results]
    
    # Sort results by sample index and separate components
    completed_results.sort(key=lambda x: x[0])  # Sort by sample_idx
    
    initial_conditions = []
    final_solutions = []
    time_array = None
    
    for sample_idx, u0, u1, t in completed_results:
        initial_conditions.append(u0)
        final_solutions.append(u1)
        if time_array is None:
            time_array = t  # Save time array from first sample
    
    # Convert back to JAX arrays
    return (jnp.stack(initial_conditions),
            jnp.stack(final_solutions), 
            jnp.array(time_array))


# Convenience function for generating standard FNO paper dataset
def generate_train_test_data(
    nx: int = 256,
    n_train: int = 1000,
    n_test: int = 200,
    key = None,
    n_processes: int = 4
) -> dict:
    """
    Generate train/test split.
    
    Args:
        nx: Number of grid points
        n_train: Number of training samples
        n_test: Number of test samples  
        key: Random key (if None, uses default seed)
        
    Returns:
        Dictionary with keys: 'train_input', 'train_output', 'test_input', 'test_output', 'time'
    """
    if key is None:
        key = jax.random.key(42)
        
    # Generate training data
    print("Generating training data...")
    key_train, key_test = jax.random.split(key)
    train_u0, train_u1, t = generate_dataset(nx, n_train, key_train, n_processes=n_processes)
    
    # Generate test data
    print("Generating test data...")
    test_u0, test_u1, _ = generate_dataset(nx, n_test, key_test, n_processes=n_processes)
    
    return {
        'train_input': train_u0,
        'train_output': train_u1,
        'test_input': test_u0,
        'test_output': test_u1,
        'time': t
    }

def main():
    # Call this at the start of your main() function
    configure_jax_threading()

    # Parse command line interface (CLI)
    parser = argparse.ArgumentParser(description='Generate FNO training and testing data')
    parser.add_argument('--n_train', type=int, default=100, help='Number of training samples (default: 100)')
    parser.add_argument('--n_test', type=int, default=20,help='Number of test samples (default: 20)')
    parser.add_argument('--resolution', type=int, default=1024, help='Spatial resolution (default: 1024)')
    parser.add_argument('--output_dir', type=str, default='data/burgers/datasets', help='Output directory (default: data/burgers/datasets)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dt', type=float, default=1e-3, help='Time step size (default: 1e-3)')
    parser.add_argument('--nu', type=float, default=0.1, help='Viscosity parameter (default: 0.1)')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of parallel processes (default: 4)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FNO Data Generation")
    print("=" * 60)
    print(f"Training samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Spatial resolution: {args.resolution}")
    print(f"Viscosity (ν): {args.nu}")
    print(f"Time step (dt): {args.dt}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()
    
    # Generate dataset
    print("Starting data generation...")
    start_time = time.time()
    
    key = jax.random.key(args.seed)
    dataset = generate_train_test_data(
        nx=args.resolution,
        n_train=args.n_train,
        n_test=args.n_test,
        key=key,
        n_processes=args.n_processes
    )
    
    generation_time = time.time() - start_time
    print(f"\nData generation completed in {generation_time:.1f} seconds")
    
    # Prepare metadata
    metadata = {
        'resolution': args.resolution,
        'n_train': args.n_train,
        'n_test': args.n_test,
        'viscosity': args.nu,
        'dt': args.dt,
        'seed': args.seed,
        'generation_time': generation_time,
        'jax_backend': jax.default_backend()
    }
    
    # Create output filename
    resolution_str = f"{args.resolution}"
    filename = f"burgers_n{args.n_train + args.n_test}_res{resolution_str}.npz"
    save_path = os.path.join(args.output_dir, filename)
    
    # Save dataset
    jax_fno.data_utils.save_dataset(dataset, save_path, metadata)
    
    print(f"\n✓ Successfully generated and saved FNO dataset!")
    print(f"  Total samples: {args.n_train + args.n_test}")
    print(f"  File size: {os.path.getsize(save_path) / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()