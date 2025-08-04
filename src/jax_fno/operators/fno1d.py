import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Optional
from .fourier_layer import FourierLayer1D


class FNO1D(nnx.Module):
    """
    1D Fourier Neural Operator as described by Li et al. (2020).
    
    Architecture:
    1. Lifting FCNN: Map input from input_dim → lifting_hidden → width
    2. Fourier Layers: Apply n_layers Fourier layers in width-dimensional space
    3. Projection FCNN: Map width → projection_hidden → projection_hidden → output_dim
    
    Args:
        input_dim: Input feature dimension (typically 2: u(x) + x coordinate)
        output_dim: Output feature dimension (typically 1: u(x) at final time)
        width: Hidden dimension for Fourier layers
        n_modes: Number of Fourier modes to keep in each layer
        n_layers: Number of Fourier layers (typically 4)
        activation: Activation function (default: GELU)
        lifting_hidden: Hidden dimension for lifting FCNN (default: width)
        projection_hidden: Hidden dimension for projection FCNN (default: width)
        rngs: Random number generators
    
    Input/Output:
        Input: (batch, n_points, input_dim) 
        Output: (batch, n_points, output_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int, 
        width: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        lifting_hidden: Optional[int] = None,
        projection_hidden: Optional[int] = None,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.activation = jax.nn.gelu
        
        # Default hidden dimensions
        if lifting_hidden is None:
            lifting_hidden = width
        if projection_hidden is None:
            projection_hidden = width
            
        self.lifting_hidden = lifting_hidden
        self.projection_hidden = projection_hidden
        
        # Lifting FCNN: input_dim -> hidden -> width
        self.lifting = nnx.Sequential(
            nnx.Linear(
                input_dim, lifting_hidden,
                rngs=rngs,
                kernel_init=jax.nn.initializers.glorot_uniform()
            ),
            self.activation,
            nnx.Linear(
                lifting_hidden, width,
                rngs=rngs,
                kernel_init=jax.nn.initializers.glorot_uniform()
            )
        )
        
        # Fourier layers
        self.fourier_layers = []
        for i in range(n_layers):
            layer = FourierLayer1D(
                d_v=width,
                k_max=n_modes,
                rngs=rngs
            )
            self.fourier_layers.append(layer)
        
        # Projection FCNN: width -> hidden -> hidden -> output_dim
        self.projection = nnx.Sequential(
            nnx.Linear(
                width, projection_hidden,
                rngs=rngs,
                kernel_init=jax.nn.initializers.glorot_uniform()
            ),
            self.activation,
            nnx.Linear(
                projection_hidden, projection_hidden,
                rngs=rngs,
                kernel_init=jax.nn.initializers.glorot_uniform()
            ),
            self.activation,
            nnx.Linear(
                projection_hidden, output_dim,
                rngs=rngs,
                kernel_init=jax.nn.initializers.glorot_uniform()
            )
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the FNO.
        
        Args:
            x: Input tensor of shape (batch, n_points, input_dim)
            
        Returns:
            Output tensor of shape (batch, n_points, output_dim)
        """
        # Lifting: project to hidden dimension
        x = self.lifting(x)
        
        # Apply Fourier layers sequentially
        for fourier_layer in self.fourier_layers:
            x = fourier_layer(x)
        
        # Projection: map to output dimension
        x = self.projection(x)
        
        return x