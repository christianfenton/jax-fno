# JAX-FNO: Fourier Neural Operators in JAX

This project provides JAX/Flax implementations of Fourier Neural Operators (FNOs) for solving partial differential equations (PDEs), along with JAX-based traditional PDE/IVP solvers for data generation and comparison.

The package provides two complementary approaches to solving PDEs:
1. **Traditional time-stepping solver** - JIT-compiled backward Euler method with Newton-Raphson iterations for generating training data and ground truth solutions
2. **Fourier Neural Operator (FNO)** - Modern neural network architecture that learns to map between function spaces

This enables a complete workflow: generate training data with the traditional solver, train FNO models, and compare performance on challenging PDEs like the Burgers equation. 

The work in this project is based on [Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).](https://arxiv.org/pdf/2010.08895)

## Features

This project is new and under heavy development. Currently implemented features are:
- **PDE Solver** (`fno.solve_ivp`): JAX-optimized backward Euler with matrix-free Newton-Raphson method
  - Built-in equations: Burgers, heat equation
  - Periodic and Dirichlet boundary conditions
- **FNO Architecture** (`fno`): Complete 1D Fourier Neural Operator
  - Spectral convolution layers (i.e. Fourier layers)
  - Lifting and projection networks
- **Data Generation**: Parallel data generation scripts
- **Profiling Tools**: JAX compilation analysis and performance debugging
- **Training Routines**: Example training script for Burgers' equation in 1D

**To Dos:**
- 2D/3D FNO extensions
- Additional PDE types (Navier-Stokes, wave equation)

## Installation

### Development Mode

1. Clone this repository:
```bash
git clone <repository-url>
cd jax_fno
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

## Quick Start

### Generate Training Data

Use the high-performance PDE solver to create datasets:

```python
import jax.numpy as jnp
import jax_fno

# Set up Burgers equation
L = 1.0  # Domain length
nx = 256  # Grid points
x = jnp.linspace(0, L, nx, endpoint=False)
u0 = jnp.sin(2 * jnp.pi * x)  # Initial condition

# Solve with Newton-Raphson + BiCGStab
params = {
    'nu': 0.01, 
    'bc_type': jax_fno.BCType.PERIODIC,
    'bc_left': 0.0, 'bc_right': 0.0
}

t, u = jax_fno.solve(
    initial_condition=u0,
    t_span=(0.0, 1.0),
    L=L,
    residual_fn=jax_fno.burgers_residual_1d,
    parameters=params,
    jvp_fn=jax_fno.burgers_jvp_1d,  # Analytical Jacobian for speed
    dt=1e-4
)
```

### Train a Fourier Neural Operator

```python
from flax import nnx
import jax_fno

# Create FNO model for Burgers equation
rngs = nnx.Rngs(42)
model = jax_fno.FNOBurgers1D(
    width=64,      # Hidden dimension
    modes=16,      # Fourier modes to keep
    n_layers=4,    # Number of Fourier layers
    rngs=rngs
)

# Prepare input: [initial_condition, x_coordinates]
x_input = jax_fno.create_burgers_input(u0, x)
prediction = model(x_input)  # Shape: (batch, n_points, 1)
```

### Generate Large Datasets

For training, use the parallel data generation script:

```bash
python data/burgers/generate.py \
    --n_train 1000 \
    --n_test 200 \
    --resolution 256 \
    --n_processes 8
```

For more details, see `src/fno/solve_ivp/README.md`.

## Project Structure

```
src/jax_fno/
├── __init__.py              # Main package exports  
├── operators/               # Neural operator implementations
│   ├── __init__.py          # Operator exports
│   ├── fno1d.py             # Complete 1D FNO (FNO1D, FNOBurgers1D)
│   ├── fourier_layer.py     # Spectral convolution layer
│   └── initializers.py      # Complex initialization utilities
├── solvers/                 # Traditional PDE solvers
│   ├── __init__.py          # Solver exports
│   ├── ivp.py               # Main solve() function with JIT optimization
│   ├── derivatives.py       # Finite difference operators
│   ├── grid.py              # Grid generation utilities
│   └── equations/           # Built-in PDE implementations
│       ├── __init__.py      # Equation exports
│       ├── burgers.py       # Burgers equation residual & Jacobian
│       └── heat.py          # Heat equation residual & Jacobian
└── utils/                   # Utilities and data helpers
    ├── __init__.py          # Utility exports
    └── readwrite.py         # Dataset I/O functions
data/burgers/
├── generate.py              # Parallel data generation script
└── datasets/                # Generated training/test data
tests/                       # Test suite
├── test_fourier_layer.py    # Fourier layer tests
└── test_fno1d.py            # Full FNO tests
profile/
└── profile_solve_ivp.py     # Performance profiling tools
```

## Key Components

- **`FourierLayer1D`**: Core spectral convolution layer with learnable Fourier modes
- **`FNO1D`**: Complete neural operator with lifting, Fourier layers, and projection
- **`jax_fno.solve()`**: JAX-optimized PDE/IVP solver with Newton-Raphson iterations
- **Data generation**: Parallel scripts for creating large training datasets

## License