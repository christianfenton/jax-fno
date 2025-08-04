#!/usr/bin/env python3
"""
Test the dataset loading utilities.
"""

import os
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax_fno import data_utils

def test_loading_utilities():
    """Test the dataset loading utilities."""
    
    print("\n" + "="*60)
    print("Testing Dataset Loading Utilities")
    print("="*60)
    
    # Create test dataset if needed
    test_file = 'data/burgers/burgers_n3_res16_nu0.1.npz'
    
    print("\n" + "-"*40)
    print("1. Testing inspect_dataset()")
    print("-"*40)
    data_utils.inspect_dataset(test_file)
    
    print("\n" + "-"*40)
    print("2. Testing load_dataset()")
    print("-"*40)
    dataset, metadata = data_utils.load_dataset(test_file, convert_to_jax=True)
    
    print("\n" + "-"*40)
    print("3. Verifying data integrity")
    print("-"*40)
    
    # Check data shapes and types
    expected_keys = ['train_input', 'train_output', 'test_input', 'test_output', 'time']
    for key in expected_keys:
        if key in dataset:
            print(f"✓ {key}: {dataset[key].shape} ({type(dataset[key])})")
        else:
            print(f"✗ Missing key: {key}")
    
    # Check metadata
    print(f"\nMetadata keys: {list(metadata.keys())}")
    print(f"Training samples: {metadata.get('n_train', 'unknown')}")
    print(f"Test samples: {metadata.get('n_test', 'unknown')}")
    print(f"Resolution: {metadata.get('resolution', 'unknown')}")
    
    # Test array operations (should work if JAX arrays)
    if 'train_input' in dataset:
        sample_input = dataset['train_input'][0]
        sample_mean = jnp.mean(sample_input)
        print(f"✓ JAX operations work: mean of first sample = {sample_mean:.4f}")
    
    print("\n" + "-"*40)
    print("4. Testing list_dataset_files()")
    print("-"*40)
    
    files = data_utils.list_dataset_files("data/test")
    
    print("\n" + "-"*40)
    print("5. Testing load with NumPy arrays")
    print("-"*40)
    
    dataset_np, metadata_np = data_utils.load_dataset(test_file, convert_to_jax=False)
    
    if 'train_input' in dataset_np:
        print(f"NumPy array type: {type(dataset_np['train_input'])}")
        print(f"JAX array type: {type(dataset['train_input'])}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}")
    
    # Run tests
    test_loading_utilities()