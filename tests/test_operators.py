import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from jax_fno.operators import FourierLayer1D, FNO1D


class TestFourierLayer1D:
    """Test 1D Fourier Layer"""

    def test_forward_pass(self):
        """Test basic forward pass and shape consistency."""

        # Parameters
        batch_size = 4
        n_points = 64
        d_v = 16
        k_max = 12
        
        # Create layer
        rngs = nnx.Rngs(42)
        layer = FourierLayer1D(d_v=d_v, k_max=k_max,  rngs=rngs)
        
        # Create test input
        x = jax.random.normal(jax.random.key(123), (batch_size, n_points, d_v))
        
        # Forward pass
        output = layer(x)
        
        # Check shape consistency
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
            
        # Check output is finite
        assert jnp.isfinite(output).all(), "Output contains NaN or Inf"

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""

        batch_size = 3
        n_points = 32
        d_v = 4
        k_max = 8
        
        rngs = nnx.Rngs(42)
        layer = FourierLayer1D(d_v=d_v, k_max=k_max, rngs=rngs)
        
        # Create input
        x = jax.random.normal(jax.random.key(789), (batch_size, n_points, d_v))
        
        # Define a simple loss function
        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)
        
        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(layer, x)
        
        # Check that gradients exist and are non-zero for all parameters
        grad_R = grads['R']
        grad_linear = grads['linear']
        assert jnp.linalg.norm(grad_R.value) > 1e-8, "R gradients are too small"
        assert jnp.linalg.norm(grad_linear['kernel'].value) > 1e-8, "Linear gradients are too small"


    def test_different_configurations(self):
        """Test layer with different configurations."""

        configurations = [
            {"d_v": 4, "k_max": 2, "n_points": 16},
            {"d_v": 32, "k_max": 20, "n_points": 128},
            {"d_v": 1, "k_max": 5, "n_points": 64},  # Single channel
        ]
        
        for i, config in enumerate(configurations):
            
            rngs = nnx.Rngs(42 + i)
            layer = FourierLayer1D(
                d_v=config['d_v'], 
                k_max=config['k_max'], 
                rngs=rngs
            )
            
            x = jax.random.normal(
                jax.random.key(100 + i), 
                (2, config['n_points'], config['d_v'])
            )
            
            output = layer(x)
            assert output.shape == x.shape
            assert jnp.isfinite(output).all()


class TestFNO1D:
    """Test 1D FNO architecture."""

    def test_fno1d_basic(self):
        """Test basic FNO1D functionality."""
        
        # Parameters
        batch_size = 4
        n_points = 64
        input_dim = 2
        output_dim = 1
        width = 32
        modes = 8
        n_layers = 2  # Small for testing
        
        # Create model
        rngs = nnx.Rngs(42)
        model = FNO1D(
            input_dim=input_dim,
            output_dim=output_dim,
            width=width,
            n_modes=modes,
            n_layers=n_layers,
            rngs=rngs
        )
        
        # Create test input
        x = jax.random.normal(jax.random.key(123), (batch_size, n_points, input_dim))
        
        # Forward pass
        output = model(x)

        # Check shape
        expected_shape = (batch_size, n_points, output_dim)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"

        # Check values are finite
        assert jnp.isfinite(output).all(), "Output contains NaN or Inf"
        
        # Check output statistics
        output_mean = jnp.mean(output)
        output_std = jnp.std(output)
        print(f"  Output mean: {output_mean:.6f}")
        print(f"  Output std: {output_std:.6f}")
