"""
Data generation script for Burgers' equation following the FNO paper methodology.

Implements the data generation procedure from Li et al. (2020):
"Fourier Neural Operator for Parametric Partial Differential Equations"
https://arxiv.org/pdf/2010.08895

Initial conditions: u0 ~ N(0, 625(-Δ + 25I)^-2) with periodic BCs
Resolution: 2^10 = 1024 spatial points (original paper uses 2^13)
Viscosity: ν = 1e-1
"""

import os
import time
import argparse
import h5py
from typing import Tuple 

# Linear Algebra
import numpy as np
import jax
import jax.numpy as jnp

# PDE solvers
import jax_fno.solvers as solve_ivp

# Parallelisation
import multiprocessing as mp
from threading import Lock


class GaussianRandomField:
    def __init__(self, shape: tuple, alpha=2, tau=3):
        assert len(shape) == 1, "Only 1D is currently supported."
        assert len(set(shape)) == 1, "Expected all dimensions to have equal number of grid points"

        self.shape = shape
        self.ndims = len(shape)  # number of dimensions
        self.npoints = shape[0]  # number of grid points per dimension
    
        k_max = self.npoints // 2
        k = jnp.concatenate([jnp.arange(0, k_max), jnp.arange(-k_max, 0)])  # wavenumbers
        p = 2 * jnp.pi * k  # momenta

        sigma = tau**(0.5 * (2 * alpha - self.ndims))
        sqrt_eig = self.npoints * jnp.sqrt(2.0) * sigma * ((p**2 + tau**2)**(-alpha / 2.0))
        sqrt_eig = sqrt_eig.at[0].set(0.0)
        self.sqrt_eig = sqrt_eig

    def sample(self, key):
        coeff = jax.random.normal(key, (*self.shape, 2))
        coeff = (coeff[...,0] + 1j * coeff[...,1]) * self.sqrt_eig

        # Enforce conjugate symmetry for real output
        k_max = self.npoints // 2
        coeff = coeff.at[k_max+1:].set(jnp.conj(coeff[1:k_max][::-1]))
        coeff = coeff.at[0].set(jnp.real(coeff[0]))
        if self.npoints % 2 == 0:  # Nyquist frequency
            coeff = coeff.at[k_max].set(jnp.real(coeff[k_max]))
        
        u = jnp.real(jnp.fft.ifft(coeff))
        return u
    

class ProgressTracker:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed = mp.Value('i', 0)  # Use multiprocessing.Value for thread-safe counter ('i' = integer)
        self.lock = Lock()  # Extra safety for the print statement

        self.checkpoints = set()
        for percent in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            checkpoint = int((percent / 100) * total_tasks)
            self.checkpoints.add(checkpoint)
    
    def task_callback(self, result):
        self.completed.value += 1

        if self.completed.value in self.checkpoints:
            progress_pct = (self.completed.value / self.total_tasks) * 100
            with self.lock:
                print(f"Progress: {self.completed.value}/{self.total_tasks} ({progress_pct:.1f}%)")
    

def solve_sample(args):
    """
    Generate a single Burgers equation solution trajectory.
    
    Args:
        args: Tuple of (id, key, grf, nx, L, t_span, nu, dt)
    
    Returns:
        Tuple of (id, initial_condition, final_solution)
    """
    id, key, grf, nx, L, t_span, nu, dt = args

    params = {'nu': nu, 'bc_type': solve_ivp.BCType.PERIODIC}

    ic = lambda _ : grf.sample(key)

    t, x, u = solve_ivp.solve(
        initial_condition=ic,
        t_span=t_span,
        residual_fn=solve_ivp.burgers_residual_1d,
        parameters=params,
        L=L,
        nx=nx,
        jvp_fn=solve_ivp.burgers_jvp_1d,
        dt=dt,
        tol=1e-5,
        maxiter=20,
        tol_gmres=1e-5
    )

    return (id, x, u[0], u[-1])


def generate_dataset(
    nx: int,
    n_samples: int,
    key,
    t_span: Tuple = (0.0, 0.1),
    nu: float = 1e-1,
    L: float = 2 * jnp.pi,
    dt: float = 1e-2,
    n_processes: int = 1
) -> dict:
    """
    Generate dataset of Burgers equation solutions following FNO paper methodology.
    
    Args:
        nx: Spatial resolution
        n_samples: Number of trajectory samples to generate
        key: JAX random key
        t_span: A tuple of the start and final times
        nu: Viscosity parameter
        L: Domain length
        dt: Time step size
        n_processes: Number of parallel processes
        
    Returns:
        Tuple of (grids, initial_conditions, final_solutions)
        - inputs: `n_samples` of u(t_0, x) and x with shape (n_samples, nx, nx)
        - outputs: `n_samples` of u(t_f, x) with shape (n_samples, nx)
    """
    print(f"Generating {n_samples} samples using {n_processes} processes...")
    
    grf = GaussianRandomField((nx, ), alpha=2, tau=5)
    tracker = ProgressTracker(n_samples)
    keys = jax.random.split(key, n_samples)  # Generate keys for all samples
    args_list = [(i, keys[i], grf, nx, L, t_span, nu, dt) for i in range(n_samples)]
    
    with mp.Pool(processes=n_processes) as pool:
        results = []
        for args in args_list:
            result = pool.apply_async(solve_sample, (args,), callback=tracker.task_callback)
            results.append(result)
        completed_results = [result.get() for result in results]

    x_list = []
    u0_list = []
    uf_list = []

    for (_, x, u0, uf) in completed_results:
        x_list.append(x)
        u0_list.append(u0)
        uf_list.append(uf)

    x_arr = jnp.stack(x_list, axis=0)
    u0_arr = jnp.stack(u0_list, axis=0)
    uf_arr = jnp.stack(uf_list, axis=0)

    inputs = jnp.stack([u0_arr, x_arr], axis=-1)
    outputs = uf_arr[..., None]

    return inputs, outputs


def main():
    # Call this at the start of your main() function
    n_cores = mp.cpu_count()
    os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count={n_cores}'
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_cores)
    print(f"Configured JAX for {n_cores} CPU cores")


    # Parse command line interface (CLI)
    parser = argparse.ArgumentParser(description='Generate FNO training and testing data')
    parser.add_argument('--n_train', type=int, default=100, help='Number of training samples (default: 100)')
    parser.add_argument('--n_test', type=int, default=20,help='Number of test samples (default: 20)')
    parser.add_argument('--resolution', type=int, default=1024, help='Spatial resolution (default: 1024)')
    parser.add_argument('--output_dir', type=str, default='data/burgers/datasets', help='Output directory (default: data/burgers/datasets)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--dt', type=float, default=5e-3, help='Time step size (default: 1e-2)')
    parser.add_argument('--nu', type=float, default=1e-1, help='Viscosity (default: 1e-1)')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of parallel processes (default: 1)')
    
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
    key_train, key_test = jax.random.split(key)

    t_span = (0.0, 0.1)

    print("\tGenerating training data...")
    train_inputs, train_outputs = generate_dataset(
        args.resolution, args.n_train, key_train, 
        t_span=t_span, nu=args.nu, dt=args.dt, n_processes=args.n_processes
    )
    
    print("\tGenerating test data...")
    test_inputs, test_outputs = generate_dataset(
        args.resolution, args.n_test, key_test, 
        t_span=t_span, nu=args.nu, dt=args.dt, n_processes=args.n_processes
    )
    
    t_elapsed = time.time() - start_time
    print(f"\nData generation completed in {t_elapsed:.1f} seconds")
    
    # Prepare metadata
    metadata = {
        'resolution': args.resolution,
        'viscosity': args.nu,
        't_start': t_span[0],
        't_end': t_span[1],
        'dt': args.dt,
        'seed': args.seed,
        'n_train': args.n_train,
        'n_test': args.n_test,
        'generation_time': t_elapsed,
        'jax_backend': jax.default_backend()
    }
    
    # Save dataset
    resolution_str = f"{args.resolution}"
    filename = f"burgers_res{resolution_str}.h5"
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, filename)
    print(f"Saving dataset to {save_path}...")
    with h5py.File(save_path, "w") as hf:
        # train group
        train_group = hf.create_group("train")
        train_group.create_dataset("inputs", data=train_inputs)
        train_group.create_dataset("outputs", data=train_outputs)
        
        # test group  
        test_group = hf.create_group("test")
        test_group.create_dataset("inputs", data=test_inputs)
        test_group.create_dataset("outputs", data=test_outputs)
        
        # Save metadata as attributes
        for key, value in metadata.items():
            hf.attrs[key] = value
    
    print(f"\nSuccessfully generated and saved FNO dataset!")
    print(f"  Total samples: {args.n_train + args.n_test}")
    print(f"  File size: {os.path.getsize(save_path) / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()