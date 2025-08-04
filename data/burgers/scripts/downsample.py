#!/usr/bin/env python3
"""
Downsample Burgers equation datasets to lower spatial resolution.

This script loads a high-resolution dataset and creates a downsampled version
using linear interpolation. Useful for creating training/testing datasets
at multiple resolutions.
"""

import jax.numpy as jnp
import numpy as np
from pathlib import Path
import argparse
import multiprocessing as mp
from functools import partial
from jax_fno import load_dataset, save_dataset 


def downsample(data: jnp.ndarray, nx_coarse: int, n_jobs: int = 1) -> jnp.ndarray:
    """
    Downsample high-resolution data to target resolution.
    
    Args:
        data: High-resolution data with shape (n_samples, nx_high)
        nx_coarse: Target number of spatial points
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        
    Returns:
        Subsampled data with shape (n_samples, nx_coarse)
    """
    n_samples, nx_fine = data.shape
    
    # Create interpolation indices
    fine_indices = jnp.linspace(0, nx_fine - 1, nx_fine)
    coarse_indices = jnp.linspace(0, nx_fine - 1, nx_coarse)
    
    def downsample_single(sample_data):
        """Downsample a single sample."""
        return jnp.interp(coarse_indices, fine_indices, sample_data)
    
    # Determine number of processes
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    if n_samples < 100 or n_jobs == 1:
        # For small datasets, use sequential processing anyway
        coarse_samples = []
        for i in range(n_samples):
            coarse_sample = downsample_single(data[i])
            coarse_samples.append(coarse_sample)
        return jnp.stack(coarse_samples)
    else:
        with mp.Pool(processes=n_jobs) as pool:
            data_np = np.array(data)
            coarse_samples = pool.map(downsample_single, [data_np[i] for i in range(n_samples)])
        return jnp.stack(coarse_samples)


def downsample_dataset(
    input_path: str,
    output_path: str, 
    target_resolution: int,
    n_jobs: int = 1
) -> None:
    """
    Load a dataset and create a downsampled version.
    
    Args:
        input_path: Path to input .npz dataset file
        output_path: Path for output downsampled dataset
        target_resolution: Target spatial resolution
        n_jobs: Number of parallel jobs for downsampling (-1 for all CPUs)
    """
    # Load the dataset using jax_fno utilities
    print("Loading dataset...")
    dataset, metadata = load_dataset(input_path, convert_to_jax=True)
    
    # Extract data arrays
    train_input = dataset['train_input']
    train_output = dataset['train_output']
    test_input = dataset['test_input']  # Fixed typo
    test_output = dataset['test_output']  # Fixed typo

    original_resolution = metadata['resolution']
    n_train = metadata['n_train']
    n_test = metadata['n_test']  # Fixed typo

    grid = dataset.get('grid', jnp.linspace(0, 1, original_resolution, endpoint=False))
    
    print(f"Original resolution: {original_resolution}")
    print(f"Target resolution: {target_resolution}")
    print(f"Number of samples: {n_train + n_test}")
    print(f"Using {n_jobs if n_jobs != -1 else 'all available'} CPU cores")
    
    print("Downsampling training data...")
    train_input_coarse = downsample(train_input, target_resolution, n_jobs)
    train_output_coarse = downsample(train_output, target_resolution, n_jobs)

    print("Downsampling test data...")
    test_input_coarse = downsample(test_input, target_resolution, n_jobs)
    test_output_coarse = downsample(test_output, target_resolution, n_jobs)
    
    grid_coarse = jnp.linspace(grid[0], grid[-1], target_resolution)
    
    # Prepare downsampled dataset
    downsampled_dataset = {
        'train_input': train_input_coarse,
        'train_output': train_output_coarse,
        'test_input': test_input_coarse,
        'test_output': test_output_coarse,
        'grid': grid_coarse,
    }
    
    # Add any other data arrays (not metadata)
    for key, value in dataset.items():
        if key not in ['train_input', 'train_output', 'test_input', 'test_output', 'grid']:
            downsampled_dataset[key] = value
    
    # Prepare enhanced metadata
    metadata_coarse = metadata.copy()
    metadata_coarse.update({
        'original_resolution': original_resolution,
        'resolution': target_resolution,  # Update main resolution field
        'downsampled_resolution': target_resolution,
        'downsampling_method': 'linear_interpolation'
    })
    
    # Save using jax_fno utilities
    save_dataset(downsampled_dataset, output_path, metadata_coarse)


def main():
    """Command line interface for downsampling datasets."""
    parser = argparse.ArgumentParser(
        description="Downsample Burgers equation datasets to lower spatial resolution"
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input .npz dataset file'
    )
    
    parser.add_argument(
        'target_resolution',
        type=int,
        help='Target spatial resolution (number of grid points)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path (default: auto-generate based on input and resolution)'
    )
    
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=-1,
        help='Number of parallel jobs for downsampling (-1 for all CPUs, 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.input_path)
        stem = input_path.stem
        suffix = input_path.suffix
        
        # Extract info from filename to create new name
        # e.g., burgers_n1200_res1024.npz -> burgers_n1200_res128.npz
        if '_res' in stem:
            base_name = stem.split('_res')[0]
            new_name = f"{base_name}_res{args.target_resolution}{suffix}"
        else:
            # Fallback naming
            new_name = f"{stem}_res{args.target_resolution}{suffix}"
        
        output_path = input_path.parent / new_name
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Downsample the dataset
    downsample_dataset(
        input_path=args.input_path,
        output_path=str(output_path),
        target_resolution=args.target_resolution,
        n_jobs=args.jobs
    )


if __name__ == "__main__":
    main()