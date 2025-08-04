"""
JAX Fourier Neural Operators

A JAX/Flax implementation of Fourier Neural Operators for solving PDEs,
with high-performance traditional solvers for data generation.
"""

# Main exports
from .operators import FNO1D, FourierLayer1D
from .solvers import solve, BCType, burgers_residual_1d, burgers_jvp_1d, heat_residual_1d, heat_jvp_1d
from .data_utils import save_dataset, load_dataset, list_dataset_files, inspect_dataset

__version__ = "0.1.0"
__all__ = [
    # FNO operators
    "FNO1D",
    "FourierLayer1D",
    
    # PDE solvers
    "solve",
    "BCType",
    
    # Built-in equations
    "burgers_residual_1d",
    "burgers_jvp_1d",
    "heat_residual_1d", 
    "heat_jvp_1d",
    
    # Data utilities
    "save_dataset",
    "load_dataset",
    "list_dataset_files", 
    "inspect_dataset",
]