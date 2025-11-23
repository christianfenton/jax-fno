"""
Script for downsampling a dataset with periodic boundary conditions to a target resolution.

Data with resolution 2^p is downsampled to resolution 2^t 
by taking every (2^(p-t))-th point starting from index 0.
"""

import argparse
import os
import h5py
import jax.numpy as jnp
from typing import Optional


def downsample_dataset(input_path: str, output_path: str, target_resolution: Optional[int] = None):
    """
    Downsample a dataset with periodic boundary conditions to a target resolution.

    Data with resolution 2^p is downsampled to resolution 2^t 
    by taking every (2^(p-t))-th point starting from index 0.
    
    Args:
        input_path: Path to input HDF5 dataset
        output_path: Path to output downsampled HDF5 dataset  
        target_resolution: Target resolution (must be power of 2). If None, halve the original resolution.
    """
    print(f"Loading dataset from {input_path}...")
    
    with h5py.File(input_path, 'r') as input_file:
        # Load training data
        train_inputs = jnp.array(input_file['train/inputs'])
        train_outputs = jnp.array(input_file['train/outputs'])
        
        # Load test data
        test_inputs = jnp.array(input_file['test/inputs'])
        test_outputs = jnp.array(input_file['test/outputs'])
        
        # Get metadata
        metadata = dict(input_file.attrs)
        
        original_resolution = train_inputs.shape[1]
        print(f"Original resolution: {original_resolution}")
        
        # Determine target resolution
        if target_resolution is None:
            target_resolution = original_resolution // 2
        
        # Validate that both resolutions are powers of 2
        if not (original_resolution & (original_resolution - 1)) == 0:
            raise ValueError(f"Original resolution {original_resolution} is not a power of 2")
        if not (target_resolution & (target_resolution - 1)) == 0:
            raise ValueError(f"Target resolution {target_resolution} is not a power of 2")
        if target_resolution > original_resolution:
            raise ValueError(f"Target resolution {target_resolution} cannot be larger than original {original_resolution}")
        
        # Calculate downsampling stride
        stride = original_resolution // target_resolution
        print(f"Downsampling stride: {stride}")
        
        # Downsample by taking every stride-th point
        train_inputs_ds = train_inputs[:, ::stride, :]  # (batch, target_resolution, features)
        train_outputs_ds = train_outputs[:, ::stride, :]   # (batch, target_resolution)
        
        test_inputs_ds = test_inputs[:, ::stride, :]
        test_outputs_ds = test_outputs[:, ::stride, :]
        
        actual_resolution = train_inputs_ds.shape[1]
        print(f"Target resolution: {target_resolution}")
        print(f"Actual resolution: {actual_resolution}")
        
        if actual_resolution != target_resolution:
            raise ValueError(f"Downsampling failed: got {actual_resolution}, expected {target_resolution}")
        
        # Update metadata
        metadata['resolution'] = target_resolution
        metadata['downsampled_from'] = original_resolution
        
        # Save downsampled dataset
        print(f"Saving downsampled dataset to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as output_file:
            # Train group
            train_group = output_file.create_group('train')
            train_group.create_dataset('inputs', data=train_inputs_ds)
            train_group.create_dataset('outputs', data=train_outputs_ds)
            
            # Test group
            test_group = output_file.create_group('test')
            test_group.create_dataset('inputs', data=test_inputs_ds)
            test_group.create_dataset('outputs', data=test_outputs_ds)
            
            # Save metadata
            for key, value in metadata.items():
                output_file.attrs[key] = value
    
    input_size = os.path.getsize(input_path) / 1024**2
    output_size = os.path.getsize(output_path) / 1024**2
    
    print(f"Downsampling completed!")
    print(f"  Output file size: {output_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Downsample Burgers dataset to target resolution')
    parser.add_argument('input_path', help='Path to input HDF5 dataset')
    parser.add_argument('output_path', help='Path to output dataset')
    parser.add_argument('--resolution', type=int, help='Target resolution (must be power of 2)')
    args = parser.parse_args()
    downsample_dataset(args.input_path, args.output_path, args.resolution)


if __name__ == '__main__':
    main()