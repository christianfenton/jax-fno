# JAX-FNO: Fourier Neural Operators in JAX

This project provides JAX/Flax implementations of Fourier Neural Operators (FNOs) for solving partial differential equations (PDEs), along with a JAX-based initial-value problem (IVP) solver that can be used to generate new training and testing data.

The FNO work in this project is based on [Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).](https://arxiv.org/pdf/2010.08895)

## Installation

Clone this repository and install with Poetry
```bash
git clone <repository-url>
cd jax_fno
poetry install .
```

## Basic Usage

### Solving the heat equation with the IVP solver

The diffusion equation in 1D is
```math
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2},
```
where $u(t, x)$ is the density of some material and $D$ is the diffusivity of the material.

Starting from an initial condition
```math
u(t=t_0, x) = \frac{1}{\sqrt{4 \pi D t_0}} \exp^{-x^2 / 4 D t_0},
```
at time $t_0$, the diffusion equation has an analytical solution
```math
u(t, x) = \frac{1}{\sqrt{4 \pi D t}} \exp^{-x^2 / 4 D t}
```
at a later time $t$.

Let us solve the heat equation with the `jax_fno.solvers` module.

Import the required packages:
```python
import jax.numpy as jnp

from jax_fno import solver
```

Define the analytical solution:
```python
def heat_soln_dirichlet(
    t: float, 
    x: jnp.ndarray, 
    diffusivity: float,
    length: float
) -> jnp.ndarray:
    """
    Return the solution to a 1D heat equation with
        a Gaussian initial condition and Dirichlet boundary conditions.

    Initial condition at time t=1:
    u(t0, x) = exp(-(x-L/2)**2 / (4 * D * t0)) / j(4 * pi * D * t0)

    Boundary conditions:
    u(t, 0) = u(t, L) = 0

    Solution at a later time t:
    u(t, x) = exp(-(x-L/2)**2 / (4 * D * t)) / (4 * pi * D * t)

    Args:
        t: Time
        x: Grid points
        D: Diffusivity
        L: Grid size
    """
    k = 1 / jnp.sqrt(4 * jnp.pi * diffusivity * t)
    return k * jnp.exp(-((x - length / 2) ** 2) / (4 * diffusivity * t))
```

Define the governing equation:
```python
def laplacian_dirichlet_1d(
    u: jnp.ndarray, 
    bc_left: float, 
    bc_right: float, 
    dx: float
) -> jnp.ndarray:
    """
    Take the Laplacian in a finite difference approach.

    Assumes the boundary points are ghost points with Dirichlet conditions.

    The derivative

    Args:
        u: The array to be differenced
        bc_left: The value at the left boundary (first ghost point)
        bc_right: The value at the right boundary (last ghost point)
        dx: The grid spacing
    """
    dudx = jnp.diff(u, prepend=bc_left, append=bc_right)
    return jnp.diff(dudx) / dx**2

def heat_rhs_dirichlet(
    t: float,
    u: jnp.ndarray,
    diffusivity: float,
    bc_left: float,
    bc_right: float,
    dx: float,
) -> jnp.ndarray:
    """
    Return right-hand-side of heat equation du/dt = D d2u/dx2.
    """
    d2udx2 = laplacian_dirichlet_1d(u, bc_left, bc_right, dx)
    return diffusivity * d2udx2
```

For implicit time-stepping schemes, we can also pass a **matrix-free** Jacobian to improve performance (otherwise the solver defaults to
automatic differentiation).
```python
# This is optional and only used in implicit time-stepping schemes
def jvp_heat_rhs_dirichlet(
    t: float,
    u: jnp.ndarray,
    v: jnp.ndarray,
    diffusivity: float,
    bc_left: float,
    bc_right: float,
    dx: float,
) -> jnp.ndarray:
    """
    Return the matrix-free Jacobian df/du * v where f(u) = D d2u/dx2.
    """
    d2vdx2 = laplacian_dirichlet_1d(v, bc_left, bc_right, dx)
    return diffusivity * d2vdx2
```

Set system parameters:
```python
diffusivity = 2.0
length = 100.0  # domain length
nx = 128  # number of grid points
dx = length / (nx + 1)  # grid spacing (grid points are in the interior)
bc_values = (0.0, 0.0)  # boundary condition values
t_span = (1.0, 10.0)  # (start_time, end_time)
x = jnp.linspace(dx, length - dx, nx, endpoint=True)  # grid
```

Fix parameters in governing equations:
```python
f = lambda t, u: heat_rhs_dirichlet(
    t, u, 
    diffusivity, bc_values[0], bc_values[1], dx
)

jvp = lambda t, u, v: jvp_heat_rhs_dirichlet(
    t, u, v, 
    diffusivity, bc_values[0], bc_values[1], dx
)
```

Create an initial condition:
```python
u0 = heat_soln_dirichlet(t_span[0], x, diffusivity, length)
```

Solve the equation
```python
t_final, u_final = solver.integrate(
    f,
    u0,
    t_span,
    dt=1e-1,
    stepper=solver.BackwardEuler(),
    jvp=jvp
)
```

Plot the solutions:
```python
u0 = heat_soln_dirichlet(x, t_span[0], diffusivity, length)
uT = heat_soln_dirichlet(x, t_span[1], diffusivity, length)

fig, ax = plt.subplots(figsize=(4,3), layout='tight')
ax.plot(x, u0, ls=':', label="Initial condition $u_0$")
ax.plot(x, u_final, ls='-', marker='.', label=f"Numerical soln. at $t={t_final:.2f}$")
ax.plot(x, uT, ls=':', label=f"Analytical soln. at $t={t_span[1]:.2f}$")
ax.legend(fontsize=8)
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('u', fontsize=10)
plt.show()
```

### Training an FNO

In this example, we aim to learn the mapping $G \colon u \rightarrow v$ such that
```math
v(x) = \frac{du}{dx}
```
for
```math
u(x) = \sin(\alpha x),
```
where $x \in [0, 2 \pi]$ and $\alpha \in [0.5, 1]$.

Import the required dependencies:
```python
# Linear algebra tools
import jax
import jax.numpy as jnp

# FNO package
import jax_fno

# Machine learning framework
from flax import nnx

# Optimiser
import optax

# Read/write utilities
import orbax.checkpoint as ocp
```

Generate data:
```python
key = jax.random.key(0)

resolution = 32  # resolution

# test/train split
n_samples = 120 
n_train = 100
n_test = 20

L = 2 * jnp.pi  # domain length
h = L / resolution  # grid spacing
grid = jnp.linspace(0, L - h, resolution)

# generate u and v 
alpha = 0.5 * (1 + jax.random.uniform(key, (n_samples, )))
u_data = [jnp.sin(alpha[i] * grid) for i in range(n_samples)]
v_data = [-jnp.cos(alpha[i] * grid) / alpha[i] for i in range(n_samples)]

# stack data into tensors
x = jnp.tile(grid, (n_samples, 1))
u = jnp.stack(u_data)
v = jnp.stack(v_data)

# stack u and grid to get input tensor with 2 features
input = jnp.stack([u, x], axis=-1)
input = jnp.permute_dims(input, (0, 2, 1))  # shape: (batch, feature, position)

# reshape output to have a feature dimension with length 1
output = v[:, None, :]

# organise data into dictionaries for convenience
train_ds = {'input': input[:n_train, :, :], 'output': output[:n_train, :, :]}
test_ds = {'input': input[-n_test:, :, :], 'output': output[-n_test:, :, :]}
```

Define an iterator for the datasets:
```python
from typing import Dict

class DatasetIterator:
    def __init__(self, dataset: Dict[str, jnp.ndarray], batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = dataset['input'].shape[0]
        self.reset()
    
    def reset(self, key = jax.random.key(0)):
        """Reset the iterator to the beginning with optional reshuffling."""
        self.indices = jnp.arange(self.n_samples)
        if self.shuffle:
            self.indices = jnp.array(jax.random.permutation(key, self.indices))
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, jnp.ndarray]:
        if self.current_idx >= self.n_samples:
            raise StopIteration
        
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        batch = {
            'input': self.dataset['input'][batch_indices],
            'output': self.dataset['output'][batch_indices]
        }
        
        self.current_idx = end_idx
        return batch
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
```

Create an `FNO1D` model instance
```python
input_dim = 2
output_dim = 1   
model = jax_fno.FNO1D(
    key,
    input_dim, 
    output_dim, 
    width=64, 
    n_modes=12, 
    n_layers=4, 
    projection_hidden=128,
)
```

Display the model
```python
nnx.display(model)
```

Check the model works for a forward pass:
```python
x = train_ds['input'][:2, :, :]
y = model(x)
```

Create the optimiser with a scheduler and define metrics:
```python
# Create optimiser with exponentially decaying learning rate
learning_rate = 1e-3  # initial learning rate

# Calculate steps per epoch for scheduling
n_train = train_ds['input'].shape[0]
batch_size = 16
steps_per_epoch = (n_train + batch_size - 1) // batch_size  # ceil division

# Schedule that changes every 100 epochs
schedule = optax.schedules.exponential_decay(
    learning_rate, 
    transition_steps=steps_per_epoch * 100,  # 100 epochs worth of steps
    decay_rate=0.5, 
    staircase=True
)

optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'),)
```

Define training step functions:
```python
def l2_loss(pred, target, eps=1e-6):
    numerator = jnp.sum((pred - target)**2, axis=-1)
    denominator = jnp.sum(target**2, axis=-1) + eps
    return jnp.mean(numerator / denominator)


def loss_fn(model: jax_fno.FNO1D, inputs: jax.Array, targets: jax.Array):
    predictions = model(inputs)
    return l2_loss(predictions, targets)


@nnx.jit
def train_step(
        model: jax_fno.FNO1D,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        inputs: jax.Array,
        targets: jax.Array
    ):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs, targets)
    metrics.update(loss=loss)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: jax_fno.FNO1D, metrics: nnx.MultiMetric, inputs: jax.Array, targets: jax.Array):
    loss = loss_fn(model, inputs, targets)
    metrics.update(loss=loss)
```

Train and evaluate the model:
```python
from IPython.display import clear_output
import matplotlib.pyplot as plt

metrics_history = {'train_loss': [], 'test_loss': []}
n_epochs = 500

train_iter = DatasetIterator(train_ds, batch_size=batch_size, shuffle=True)
test_iter = DatasetIterator(test_ds, batch_size=batch_size, shuffle=False)

shuffle_key = jax.random.key(42)

for epoch in range(1, n_epochs+1):
	model.train()
	for batch in train_iter:
		train_step(model, optimizer, metrics, batch['input'], batch['output'])
	shuffle_key, subkey = jax.random.split(shuffle_key)
	train_iter.reset(subkey)
	
	metrics_history['train_loss'].append(metrics.compute()["loss"])

	# Evaluation
	model.eval()
	for batch in test_iter:
		eval_step(model, metrics, batch["input"], batch["output"])
	shuffle_key, subkey = jax.random.split(shuffle_key)
	test_iter.reset(subkey)
	metrics_history["test_loss"].append(metrics.compute()["loss"])
		
	# Reset the metrics before the next training epoch
	metrics.reset()

	# Plot loss
	clear_output(wait=True)
	fig, ax = plt.subplots(figsize=(4, 3))
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss')
	ax.set_yscale('log')
	ax.plot(metrics_history[f'train_loss'], label=f'train_loss')
	ax.plot(metrics_history[f'test_loss'], label=f'test_loss')
	ax.legend()
	plt.show()
```

Perform inference on the test set:
```python
model.eval()  # switch to evaluation mode

@nnx.jit
def pred_step(model: jax_fno.FNO1D, batch_input: jax.Array):
  return model(batch_input)
```

```python
import numpy as np

n_examples = 3
    
n_test = test_ds['input'].shape[0]
example_indices = np.random.choice(n_test, size=min(n_examples, n_test), replace=False)
    
# Get predictions for selected examples
selected_inputs = test_ds['input'][example_indices]
selected_outputs = test_ds['output'][example_indices]
predictions = model(selected_inputs)

# Create subplots
fig, axes = plt.subplots(1, n_examples, figsize=(3*4, 3), layout='tight')
if n_examples == 1:
    axes = [axes]

for i, (ax, idx) in enumerate(zip(axes, example_indices)):
    # Extract data for this example
    x = selected_inputs[i, 1, :]
    u0 = selected_inputs[i, 0, :]  # Initial condition
    u_true = selected_outputs[i, 0, :]  # Ground truth final state
    u_pred = predictions[i, 0, :]  # Model prediction
    
    # Plot
    ax.plot(x, u0, '-', label='Initial condition $u_0(x)$', linewidth=2, alpha=0.8)
    ax.plot(x, u_true, '-', label='Ground truth $u(x,t=1)$', linewidth=2, alpha=0.8)
    ax.plot(x, u_pred, ':', label='FNO prediction $u(x,t=1)$', linewidth=2, alpha=0.8)
    
    # Compute error metrics
    l2_error = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
    
    ax.set_title(f'Sample {idx+1} -- Relative L2 Error: {l2_error:.2e}', fontsize=10)
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$u(x)$', fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()

plt.show()
```