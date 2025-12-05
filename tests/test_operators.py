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
        channels = 16
        n_modes = 12

        # Create layer
        key = jax.random.key(42)
        layer = FourierLayer1D(
            key=key,
            channels_in=channels,
            channels_out=channels,
            n_modes=n_modes,
        )

        # Create test input (batch, channels, spatial)
        x = jax.random.normal(
            jax.random.key(123), (batch_size, channels, n_points)
        )

        # Forward pass
        output = layer(x)

        # Check shape consistency
        assert output.shape == x.shape, (
            f"Shape mismatch: {output.shape} != {x.shape}"
        )

        # Check output is finite
        assert jnp.isfinite(output).all(), "Output contains NaN or Inf"

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""

        batch_size = 3
        n_points = 32
        channels = 4
        n_modes = 8

        key = jax.random.key(42)
        layer = FourierLayer1D(
            key=key,
            channels_in=channels,
            channels_out=channels,
            n_modes=n_modes,
        )

        # Create input (batch, channels, spatial)
        x = jax.random.normal(
            jax.random.key(789), (batch_size, channels, n_points)
        )

        # Define a simple loss function
        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(layer, x)

        # Check that gradients exist and are non-zero for all parameters
        grad_spectral = grads["spectral"]
        grad_linear = grads["linear"]
        assert jnp.linalg.norm(grad_spectral["weights"].value) > 1e-8, (
            "Spectral gradients are too small"
        )
        assert jnp.linalg.norm(grad_linear["weight"].value) > 1e-8, (
            "Linear gradients are too small"
        )

    def test_different_configurations(self):
        """Test layer with different configurations."""

        configurations = [
            {"channels": 4, "n_modes": 2, "n_points": 16},
            {"channels": 32, "n_modes": 20, "n_points": 128},
            {"channels": 1, "n_modes": 5, "n_points": 64},
        ]

        for i, config in enumerate(configurations):
            key = jax.random.key(42 + i)
            layer = FourierLayer1D(
                key=key,
                channels_in=config["channels"],
                channels_out=config["channels"],
                n_modes=config["n_modes"],
            )

            x = jax.random.normal(
                jax.random.key(100 + i),
                (2, config["channels"], config["n_points"]),
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
        n_modes = 8
        n_layers = 2  # Small for testing

        # Create model
        key = jax.random.key(42)
        model = FNO1D(
            key=key,
            input_dim=input_dim,
            output_dim=output_dim,
            width=width,
            n_modes=n_modes,
            n_layers=n_layers,
        )

        # Create test input (batch, input_dim, spatial)
        x = jax.random.normal(
            jax.random.key(123), (batch_size, input_dim, n_points)
        )

        # Forward pass
        output = model(x)

        # Check shape
        expected_shape = (batch_size, output_dim, n_points)
        assert output.shape == expected_shape, (
            f"Shape mismatch: {output.shape} != {expected_shape}"
        )

        # Check values are finite
        assert jnp.isfinite(output).all(), "Output contains NaN or Inf"

        # Check output statistics
        output_mean = jnp.mean(output)
        output_std = jnp.std(output)
        print(f"  Output mean: {output_mean:.6f}")
        print(f"  Output std: {output_std:.6f}")
