import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable
from .initializers import glorot_uniform_complex


class FourierLayer1D(nnx.Module):
    """
    A Fourier layer for a 1D Fourier Neural Operator as described by
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" (2020).

    Args:
        d_v: Embedding dimension
        k_max: Number of Fourier modes to keep
        rngs: Keys for pseudo-random initialisation (default: nnx.Rngs(0))

    Details:
        The layer computes

        v_out = σ(W * v_in + F^{-1}[R * F(v_in)]),

        where σ is a non-linear activation function, W is a trainable linear 
        transformation, F is the Fast Fourier Transform, R is a tensor of trainable 
        complex weights for Fourier modes, and * denotes element-wise multiplication.

        Fourier modes with k > k_max are filtered out during the convolution F^{-1}[R * F(v)].

    List of Terms:
        --------------------------------------
        Term | Dimensions          | Data Type
        --------------------------------------
        v    | [batch, n, d_v]     | Real
        W    | [d_v, d_v]          | Real
        R    | [k_max, d_v, d_v]   | Complex (with conjugate symmetry)

        where n is the number of grid points and d_v is 
        the dimensionality of the space that v is embedded in.
    """

    def __init__(
        self, 
        d_v: int, 
        k_max: int, 
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.d_v = d_v
        self.k_max = k_max
        self.activation = jax.nn.gelu
        
        # Initialise Fourier weights R with shape (k_max, d_v, d_v)
        key = rngs.params()
        shape = (k_max, d_v, d_v)
        fan_in = d_v
        fan_out = d_v
        R = glorot_uniform_complex(key, shape, fan_in, fan_out)
        self.R = nnx.Param(R)
        
        # Linear transformation for skip connection
        self.linear = nnx.Linear(d_v, d_v, rngs=rngs, kernel_init=jax.nn.initializers.glorot_uniform())
    
    def spectral_conv1d(self, v: jax.Array) -> jax.Array:
        """
        Perform 1D spectral convolution using FFT.
        
        Args:
            x: Input tensor of shape (batch_size, spatial_points, channels)
            
        Returns:
            Output tensor of same shape after spectral convolution
        """
        batch_size, n, d_v = v.shape
        
        # Transform to spectral space
        Fv = jnp.fft.rfft(v, axis=1)  # Shape: (batch, n//2+1, d_v)
        
        # Filter high-frequency modes out of F(v)
        k_max = self.k_max
        Fv_filtered = Fv[:, :k_max, :]  # Shape: (batch, k_max, d_v)

        # Compute batched matrix multiplication for each mode, i.e.
        # Compute 'R_ij [F(v)]_j' for each mode and batch
        # Shape of R: (k_max, d_v, d_v)
        # Einsum indices:
        #   'm': mode index
        #   'i, j': matrix indices
        #   'b': batch index
        RFv_filtered = jnp.einsum('mij,bmj->bmi', self.R, Fv_filtered)

        # Pad result to get correct shape (batch, n//2+1, d_v)
        padding = (n//2 + 1) - k_max
        RFv = jnp.pad(RFv_filtered, ((0,0), (0, padding), (0, 0)))
        
        # Transform back to physical space
        out = jnp.fft.irfft(RFv, n=n, axis=1)
        
        return out

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass of the Fourier layer.
        
        Args:
            x: Input tensor of shape (batch, spatial_points, channels)
            
        Returns:
            Output tensor of same shape
        """
        # Spectral convolution branch
        x1 = self.spectral_conv1d(x)
        
        # Linear transformation branch (skip connection)
        x2 = self.linear(x)
        
        # Combine and apply activation
        return self.activation(x1 + x2)