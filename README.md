# JAX-FNO: Fourier Neural Operators in JAX

This project provides Fourier neural operators (FNOs) for solving partial 
differential equations (PDEs) and an ordinary differential equation (ODE) 
integrator, all written in JAX/Flax.

## Overview

**JAX-FNO** is organised into two main modules:

- **Learning** (`jax_fno.learn`): FNO architectures that learn PDE solution operators
- **Integration** (`jax_fno.integrate`): ODE integration methods

## Installation

[Poetry](https://python-poetry.org/docs/) is recommended for installation.

### Using Poetry

Create a Poetry environment and add the package:

With SSH:
```bash
poetry new my-project
cd my-project
poetry add git+ssh://git@github.com/christianfenton/jax-fno.git
```

With HTTPS:
```bash
poetry add git+https://github.com/christianfenton/jax-fno.git
```

### Using pip

Alternatively, you can install directly with pip:

With SSH:
```bash
pip install git+ssh://git@github.com/christianfenton/jax-fno.git
```

With HTTPS:
```bash
pip install git+https://github.com/christianfenton/jax-fno.git
```

## Getting started

To get started using the project, check out the tutorials:

- **[Solving the heat equation](https://christianfenton.github.io/jax-fno/tutorials/solving_heat_equation/)**
- **[Learning an antiderivative operator](https://christianfenton.github.io/jax-fno/tutorials/learning_antiderivative/)**

Check out the **[documentation page](https://christianfenton.github.io/jax-fno/)**
for further details.

## Citations

The work in this project is based on
> Li, Zongyi, et al. "Fourier neural operator for parametric partial 
> differential equations."
> *arXiv preprint arXiv:2010.08895* (2020).
> [https://arxiv.org/pdf/2010.08895](https://arxiv.org/pdf/2010.08895)

## Future Works

In the future, `jax_fno.integrate.solve_ivp` should be adapted to match 
[`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and added to [`jax.scipy.integrate`](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.integrate).