import jax
import jax.numpy as jnp
import jax_fno
from flax import nnx
import optax
import numpy as np
import orbax.checkpoint as ocp

import os
from typing import Callable, Optional
from IPython.display import clear_output
import matplotlib.pyplot as plt


class FNOBurgers1D(nnx.Module):
    """
    FNO specifically configured for 1D Burgers equation.
    
    Input: Initial condition u0(x) + spatial coordinates x
    Output: Solution u(x, T) at final time T
    """
    
    def __init__(
        self, 
        width: int = 64,
        n_modes: int = 16, 
        n_layers: int = 4,
        lifting_hidden: Optional[int] = None,
        projection_hidden: Optional[int] = None,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.fno = jax_fno.FNO1D(
            input_dim=2,    # u0(x) + x coordinate
            output_dim=1,   # u(x, T)
            width=width,
            n_modes=n_modes,
            n_layers=n_layers,
            lifting_hidden=lifting_hidden,
            projection_hidden=projection_hidden,
            rngs=rngs
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass for Burgers equation.
        
        Args:
            x: Input tensor of shape (batch, n_points, 2)
               x[..., 0] = initial condition u0(x)
               x[..., 1] = spatial coordinate x
               
        Returns:
            Predicted solution u(x, T) of shape (batch, n_points, 1)
        """
        return self.fno(x)
    

# Convenience function for model creation
def create_fno_burgers(
    width: int = 64,
    n_modes: int = 16,
    n_layers: int = 4,
    lifting_hidden: Optional[int] = None,
    projection_hidden: Optional[int] = None,
    seed: int = 42
) -> FNOBurgers1D:
    """Default FNO configuration for 1D Burgers' equation."""
    rngs = nnx.Rngs(seed)
    return FNOBurgers1D(
        width=width,
        n_modes=n_modes, 
        n_layers=n_layers,
        lifting_hidden=lifting_hidden,
        projection_hidden=projection_hidden,
        rngs=rngs
    )


def create_learning_rate_schedule(
    initial_lr: float = 0.001,
    decay_steps: int = 100,
    decay_rate: float = 0.5
) -> optax.Schedule:
    """Create exponential decay learning rate schedule."""
    return optax.exponential_decay(
        init_value=initial_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )


def mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean-squared error loss function."""
    return jnp.mean((predictions - targets) ** 2)


def loss_fn(
    model: FNOBurgers1D, 
    batch_input: jax.Array, 
    batch_output: jax.Array
):
    """
    Loss function for the model.
    
    Args:
        model: FNO model
        batch_input: Input batch of shape (batch_size, n_points, 2)
        batch_output: Target batch of shape (batch_size, n_points, 1)
    """
    predictions = model(batch_input)
    return mse_loss(predictions, batch_output)


@nnx.jit
def train_step(
        model: FNOBurgers1D,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch_input: jax.Array,
        batch_output: jax.Array
    ):
    """Single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch_input, batch_output)
    metrics.update(loss=loss)  # In-place updates
    optimizer.update(model, grads)  # In-place updates


@nnx.jit
def eval_step(
        model: FNOBurgers1D,
        metrics: nnx.MultiMetric,
        batch_input: jax.Array,
        batch_output: jax.Array
    ):
    loss = loss_fn(model, batch_input, batch_output)
    metrics.update(loss=loss)  # In-place updates


@nnx.jit
def pred_step(model: FNOBurgers1D, batch_input: jax.Array):
  return model(batch_input)


def save_model(model: FNOBurgers1D, checkpoint_dir: str):
    """Save the trained FNO model."""
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    checkpointer.save(os.path.join(checkpoint_dir, 'state'), state)
    print(f"Model saved to: {checkpoint_dir}")


def plot_test_examples(
        model: FNOBurgers1D,
        data: dict,
        n_examples: int = 3,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
    """Make some plots to verify predictions."""
    test_input = data['test_input'] 
    test_output = data['test_output']

    resolution = test_output.shape[-1]
    grid = data.get('grid', jnp.linspace(0, 1, resolution, endpoint=False))
    
    # Select random examples
    n_test = test_input.shape[0]
    example_indices = np.random.choice(n_test, size=min(n_examples, n_test), replace=False)
    
    # Get predictions for selected examples
    selected_inputs = test_input[example_indices]
    selected_outputs = test_output[example_indices]
    predictions = model(selected_inputs)
    
    # Create subplots
    fig, axes = plt.subplots(n_examples, 1, figsize=figsize)
    if n_examples == 1:
        axes = [axes]
    
    for i, (ax, idx) in enumerate(zip(axes, example_indices)):
        # Extract data for this example
        u0 = selected_inputs[i, :, 0]  # Initial condition
        u_true = selected_outputs[i, :, 0]  # Ground truth final state
        u_pred = predictions[i, :, 0]  # Model prediction
        
        # Plot
        ax.plot(grid, u0, 'b-', label='Initial condition u₀(x)', linewidth=2, alpha=0.8)
        ax.plot(grid, u_true, 'r-', label='Ground truth u(x,T)', linewidth=2, alpha=0.8)
        ax.plot(grid, u_pred, 'g--', label='FNO prediction', linewidth=2, alpha=0.8)
        
        # Compute error metrics
        mse_error = float(jnp.mean((u_pred - u_true) ** 2))
        l2_error = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))
        
        ax.set_title(f'Test Example {idx+1} - MSE: {mse_error:.6f}, L2 Error: {l2_error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def train_fno(
        model: FNOBurgers1D,
        data: dict,
        n_epochs: int = 500,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        seed: int = 42
    ) -> tuple[FNOBurgers1D, dict]:
    """
    Train the FNO model.
        
    Returns: (trained_model, metrics_history)
    """
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=learning_rate), wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'),)
    
    # Training data
    train_input = data['train_input']  # shape: (n_samples, n_points, 2)
    train_output = data['train_output']  # shape: (n_samples, n_points, 1)
    n_train = train_input.shape[0]

    # Testing data
    test_input = data['test_input']  # shape: (n_samples, n_points, 2)
    test_output = data['test_output']  # shape: (n_samples, n_points, 1)
    n_test = test_input.shape[0]
    
    metrics_history = {
        'train_loss': [],
        'test_loss': [],
    }

    key = jax.random.key(seed)
    
    print(f"Starting training for {n_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Training samples: {n_train}, Batch size: {batch_size}")
    
    for epoch in range(n_epochs):
        print(f"    Epoch: {epoch}")
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)
        train_input_shuffled = train_input[perm]
        train_output_shuffled = train_output[perm]
        
        # Training epoch
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            
            batch_input = train_input_shuffled[start_idx:end_idx]
            batch_output = train_output_shuffled[start_idx:end_idx]
            
            # Training step (in-place updates)
            train_step(model, optimizer, metrics, batch_input, batch_output)

        if epoch > 0 and (int(100 * (epoch / n_epochs) % 10) == 0 or epoch == n_epochs - 1):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
                metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_train)
                
                batch_input = test_input[start_idx:end_idx]
                batch_output = test_output[start_idx:end_idx]
                eval_step(model, metrics, batch_input, batch_output)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
                metrics.reset()  # Reset the metrics for the next training epoch.

            clear_output(wait=True)
            # Plot loss (non-blocking)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.set_title('Training Loss')
            for dataset in ('train', 'test'):
                ax.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            
            # Use non-blocking display
            plt.ion()  # Turn on interactive mode
            plt.show()
            plt.pause(0.1)  # Brief pause to update display
            plt.close(fig)  # Close figure to prevent memory buildup

    return model, metrics_history


def main():
    print("Loading data...")
    resolution = 256
    path_to_data = f"data/burgers/datasets/burgers_n1200_res{resolution}.npz"
    data, _ = jax_fno.data_utils.load_dataset(path_to_data)

    # Get grid coordinates
    grid = data.get('grid', jnp.linspace(0, 1, resolution, endpoint=False))
    
    # Original data shapes: u_in, u_out are (n_samples, n_points)
    train_input = data['train_input']
    train_output = data['train_output']
    test_input = data['test_input']
    test_output = data['test_output']
    
    # Create input arrays with shape (n_samples, n_points, 2)
    # Stack initial condition + grid coordinates
    n_train_samples = train_input.shape[0]
    n_test_samples = test_input.shape[0]
    
    # Tile grid to match batch dimensions
    train_grid = jnp.tile(grid[None, :], (n_train_samples, 1))  # (n_train, n_points)
    test_grid = jnp.tile(grid[None, :], (n_test_samples, 1))    # (n_test, n_points)
    
    # Stack [input, coordinates] along last axis to get (n_samples, n_points, 2)
    data['train_input'] = jnp.stack([train_input, train_grid], axis=-1)
    data['test_input'] = jnp.stack([test_input, test_grid], axis=-1)
    
    # Create output arrays with shape (n_samples, n_points, 1)
    data['train_output'] = train_output[..., None]
    data['test_output'] = test_output[..., None]

    print("Creating model...")
    model = create_fno_burgers()
    
    print("Training model...")
    model, _ = train_fno(model, data, n_epochs=500)

    model.eval()  # Switch to evaluation mode

    save_dir = f"/Users/christian/Code/fourier_neural_operators/data/burgers/models/burgers_res{resolution}"
    plot_test_examples(model, data, save_path=os.path.join(save_dir, 'figures'))

    print("Saving model...")
    save_model(model, save_dir)

    print("Finished.")
    return None


if __name__ == "__main__":
    main()

