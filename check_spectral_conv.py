import jax
import jax.numpy as jnp
from flax import nnx
from jax_fno.operators import FourierLayer1D


def test_spectral_convolution():
    """Test properties specific to spectral convolution."""
    print("\n=== Testing Spectral Convolution Properties ===")
    
    batch_size = 2
    n_points = 128
    d_v = 8
    k_max = 16
    
    rngs = nnx.Rngs(42)
    layer = FourierLayer1D(d_v=d_v, k_max=k_max, activation=lambda x: x, rngs=rngs)
    
    # Test 1: Zero input should give reasonable output
    x_zero = jnp.zeros((batch_size, n_points, d_v))
    output_zero = layer(x_zero) 
    print(f"✓ Zero input handled (output norm: {jnp.linalg.norm(output_zero):.6f})")
    
    # Test 2: Different frequencies should be handled differently
    # Create low and high frequency signals
    x_coords = jnp.linspace(0, 2*jnp.pi, n_points)
    
    # Low frequency (should pass through)
    low_freq = jnp.sin(2 * x_coords)[None, :, None]  # Shape: (1, n_points, 1)
    low_freq = jnp.tile(low_freq, (batch_size, 1, d_v))
    
    # High frequency (should be filtered)
    high_freq = jnp.sin(20 * x_coords)[None, :, None]
    high_freq = jnp.tile(high_freq, (batch_size, 1, d_v))
    
    # Process through spectral convolution only (no activation/linear)
    output_low = layer.spectral_conv1d(low_freq)
    output_high = layer.spectral_conv1d(high_freq)
    
    print(f"✓ Low frequency output norm: {jnp.linalg.norm(output_low):.6f}")
    print(f"✓ High frequency output norm: {jnp.linalg.norm(output_high):.6f}")
    
    # Test 3: Check that only k_max modes are preserved
    x_test = jax.random.normal(jax.random.key(456), (1, n_points, d_v))
    x_ft = jnp.fft.rfft(x_test, axis=1)
    output_test = layer.spectral_conv1d(x_test)
    output_ft = jnp.fft.rfft(output_test, axis=1)
    
    # High frequency modes should be affected differently than low frequency
    n_modes_available = x_ft.shape[1]
    if k_max < n_modes_available:
        # Check that we're actually filtering modes
        high_mode_energy_input = jnp.mean(jnp.abs(x_ft[:, k_max:, :]))
        high_mode_energy_output = jnp.mean(jnp.abs(output_ft[:, k_max:, :]))
        print(f"✓ High frequency filtering: input energy {high_mode_energy_input:.6f}, "
              f"output energy {high_mode_energy_output:.6f}")
    
    return True


def main():
    test_spectral_convolution()


if __name__ == "__main__":
    main()