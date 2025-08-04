#!/usr/bin/env python3
"""
Utilities for saving FNO train/test data.
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional  


def save_dataset(dataset: dict, save_path: str, metadata: Optional[dict] = None):
    """
    Save dataset to NPZ file with metadata.
    
    Args:
        dataset: Dictionary containing train/test data
        save_path: Path to save the NPZ file
        metadata: Additional metadata to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare data for saving (convert JAX arrays to NumPy)
    save_data = {}
    for key, value in dataset.items():
        if isinstance(value, jnp.ndarray):
            save_data[key] = np.array(value)
        else:
            save_data[key] = value
    
    # Add metadata
    if metadata:
        for key, value in metadata.items():
            save_data[f"meta_{key}"] = value
    
    # Save to NPZ file
    np.savez_compressed(save_path, **save_data)
    print(f"Dataset saved to: {save_path}")
    
    # Print dataset info
    print("\nDataset summary:")
    for key, value in save_data.items():
        if key.startswith('meta_'):
            print(f"  {key}: {value}")
        elif hasattr(value, 'shape'):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")


def load_dataset(file_path: str, convert_to_jax: bool = True) -> Tuple[dict, dict]:
    """
    Load dataset from NPZ file.
    
    Args:
        file_path: Path to the NPZ file
        convert_to_jax: Whether to convert arrays to JAX arrays (default: True)
        
    Returns:
        Tuple of (dataset, metadata) dictionaries
        - dataset: Contains train/test data
        - metadata: Contains metadata with 'meta_' prefix removed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load NPZ file
    data = np.load(file_path)
    
    # Separate dataset and metadata
    dataset = {}
    metadata = {}
    
    for key in data.files:
        value = data[key]
        
        if key.startswith('meta_'):
            # Remove 'meta_' prefix and add to metadata
            meta_key = key[5:]  # Remove 'meta_' prefix
            metadata[meta_key] = value.item() if value.ndim == 0 else value
        else:
            # Add to dataset, converting to JAX arrays if requested
            if convert_to_jax and hasattr(value, 'shape'):
                dataset[key] = jnp.array(value)
            else:
                dataset[key] = value
    
    data.close()
    
    # Print summary
    print(f"Dataset loaded from: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / 1024**2:.1f} MB")
    
    print("\nDataset contents:")
    for key, value in dataset.items():
        if hasattr(value, 'shape'):
            array_type = "JAX" if convert_to_jax else "NumPy"
            print(f"  {key}: shape {value.shape}, dtype {value.dtype} ({array_type})")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return dataset, metadata


def list_dataset_files(directory: str, pattern: str = "*.npz") -> list:
    """
    List dataset files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "*.npz")
        
    Returns:
        List of dataset file paths
    """
    import glob
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    files.sort()  # Sort for consistent ordering
    
    if files:
        print(f"Found {len(files)} dataset files in {directory}:")
        for file in files:
            file_size = os.path.getsize(file) / 1024**2
            print(f"  {os.path.basename(file)} ({file_size:.1f} MB)")
    else:
        print(f"No dataset files found in {directory}")
    
    return files


def inspect_dataset(file_path: str):
    """
    Quick inspection of dataset contents without loading into memory.
    
    Args:
        file_path: Path to the NPZ file
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    data = np.load(file_path)
    
    print(f"Dataset: {os.path.basename(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / 1024**2:.1f} MB")
    print(f"Arrays: {len([k for k in data.files if not k.startswith('meta_')])}")
    print(f"Metadata: {len([k for k in data.files if k.startswith('meta_')])}")
    
    print("\nContents:")
    for key in sorted(data.files):
        value = data[key]
        if key.startswith('meta_'):
            meta_key = key[5:]
            meta_value = value.item() if value.ndim == 0 else value
            print(f"  meta_{meta_key}: {meta_value}")
        else:
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    
    data.close()