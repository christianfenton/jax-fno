import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple

def glorot_uniform_complex(
        key: nnx.Rngs, 
        shape: Tuple[int], 
        n_in: int, 
        n_out: int
    ):
    """
    Initialize a complex-valued tensor with Glorot uniform distribution.
    
    Args:
        key: RNG key
        shape: Shape of the output tensor
        n_in: Number of input features
        n_out: Number of output features
        
    Returns:
        Complex tensor of given shape with Glorot uniform initialization
    """
    # Procedure:
    # 1. Draw magnitudes from a real-valued Glorot distribution
    # 2. Draw angles from a real-valued distribution
    # 3. Use the angles to rotate to the complex plane, i.e. output = r * e^(i * theta)
    limit = jnp.sqrt(6 / (n_in + n_out))
    key_r, key_theta = jax.random.split(key)
    r = jax.random.uniform(key_r, shape, minval=0, maxval=limit)  
    theta = jax.random.uniform(key_theta, shape, minval=0, maxval=2*jnp.pi)
    return r * jnp.exp(1j * theta)
