import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


class SpectralConv1D(nnx.Module):
    """
    A spectral convolution layer used in the 1D Fourier Neural Operator as described by
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" (2020).

    Args:
        key: Key for pseudo-random initialisation 
        channels_in: Number of channels in input
        channels_out: Number of channels in output
        n_modes: Maximum number of Fourier modes
    """

    def __init__(self, key, channels_in, channels_out, n_modes):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_modes = n_modes

        # Initialise complex weights
        key1, key2 = jax.random.split(key)
        initializer = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)
        real_part = initializer(key1, (self.channels_out, self.channels_in, self.n_modes))
        imag_part = initializer(key2, (self.channels_out, self.channels_in, self.n_modes))
        self.weights = nnx.Param((real_part + 1j * imag_part))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Perform 1D spectral convolution using FFT.
        
        Args:
            v: Input tensor of shape (batch_size, channels, spatial_points)
            
        Returns:
            Output tensor of same shape after spectral convolution
        """
        batch_size, d_v, n = x.shape
        
        # Transform to spectral space
        Fx = jnp.fft.rfft(x, n=n, axis=2, norm='ortho')  # Shape: (batch, d_v, n//2+1)
        
        # Filter out high-frequency modes
        k_max = min(self.n_modes, Fx.shape[2])
        Fx_filtered = Fx[:, :, :k_max]  # Shape: (batch, d_v, k_max)

        # Multiply with weights
        # Einsum indices:
        #   'o': output channel
        #   'i': input channel  
        #   'm': mode index
        #   'b': batch index
        RFv_filtered = jnp.einsum('oim,bim->bom', self.weights, Fx_filtered)

        # Pad result to get correct shape (batch, d_v, n//2+1)
        padding = (n//2 + 1) - k_max
        RFv = jnp.pad(RFv_filtered, ((0,0), (0, 0), (0, padding)))
        
        # Transform back to physical space
        out = jnp.fft.irfft(RFv, n=n, axis=2, norm='ortho')
        
        return out
    

class Linear1D(nnx.Module):
    """
    Linear layer that operates on tensors (batch, channels, spatial) ordering.
    """

    def __init__(self, key, channels_in: int, channels_out: int):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.weight = nnx.Param(jax.nn.initializers.glorot_uniform()(key, (channels_out, channels_in)))
        self.bias = nnx.Param(jnp.zeros(channels_out))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape (batch, in_features, spatial)
        Returns:
            Output tensor of shape (batch, out_features, spatial)
        """
        return jnp.einsum('ij,bjn->bin', self.weight, x) + self.bias[None, :, None]
    

class FourierLayer1D(nnx.Module):
    """
    A Fourier layer for a 1D Fourier Neural Operator as described by
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" (2020).

    The layer computes

    v_out = W * v_in + F^{-1}[R * F(v_in)],

    where W is a trainable linear transformation, F is the Fast Fourier Transform, 
    R is a tensor of trainable complex weights for Fourier modes, 
    and * denotes element-wise multiplication.

    Fourier modes with k > k_max are filtered out during the convolution F^{-1}[R * F(v)].

    Args:
        key: Key for pseudo-random initialisation 
        channels_in: Number of channels in input
        channels_out: Number of channels in output
        n_modes: Maximum number of Fourier modes
    """

    def __init__(self, key = jax.random.key(0), channels_in: int = 64, channels_out: int = 64, n_modes: int = 16):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_modes = n_modes

        key1, key2 = jax.random.split(key)
        self.spectral = SpectralConv1D(key1, channels_in, channels_out, n_modes)
        self.linear = Linear1D(key2, channels_in, channels_out)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        x1 = self.spectral(x)
        x2 = self.linear(x)
        return x1 + x2


class FNO1D(nnx.Module):
    """
    1D Fourier Neural Operator as described by Li et al. (2020).
    
    Architecture:
    1. Lifting FCNN: Map input from `input_dim` --> `width`
    2. Fourier Layers: Apply `n_layers` Fourier layers in `width`-dimensional space
    3. Projection FCNN: Map `width` --> `projection_hidden` --> `projection_hidden` --> `output_dim`
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        width: Hidden dimension for Fourier layers
        n_modes: Number of Fourier modes to keep in each layer
        n_layers: Number of Fourier layers
        activation: Activation function (default: GELU)
        projection_hidden: Hidden dimension for projection FCNN (default: width)
        rngs: Random number generators
    
    Input/Output:
        Input: (batch, input_dim, n_points) 
        Output: (batch, output_dim, n_points)
    """
    
    def __init__(
        self,
        key,
        input_dim: int,
        output_dim: int, 
        width: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        projection_hidden: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.activation = jax.nn.gelu
        
        if projection_hidden is None:
            projection_hidden = width
            
        self.projection_hidden = projection_hidden
        
        # Lifting FCNN: input_dim -> width
        key, subkey = jax.random.split(key, 2)
        self.lift = Linear1D(subkey, input_dim, width)
        
        # Fourier layers
        self.fourier_layers = []
        for i in range(n_layers):
            key, subkey = jax.random.split(key)
            layer = FourierLayer1D(key=subkey, channels_in=width, channels_out=width, n_modes=n_modes)
            self.fourier_layers.append(layer)
        
        # Projection FCNN: width -> hidden -> output_dim
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.proj1 = Linear1D(subkey1, width, projection_hidden)
        self.proj2 = Linear1D(subkey2, projection_hidden, output_dim)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the FNO.
        
        Args:
            x: Input tensor of shape (batch, input_dim, n_points)
            
        Returns:
            Output tensor of shape (batch, output_dim, n_points)
        """
        # (batch, input_dim, n_points) -> (batch, width, n_points)
        x = self.lift(x)

        # (batch, width, n_points) -> (batch, width, n_points)
        for fourier_layer in self.fourier_layers:
            x = self.activation(fourier_layer(x))
        
        # (batch, width, n_points) -> (batch, output_dim, n_points)
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        
        return x