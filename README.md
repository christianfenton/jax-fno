# JAX-FNO: Fourier Neural Operators in JAX

This project provides JAX/Flax implementations of Fourier neural operators 
(FNOs) for solving partial differential equations (PDEs) along with time 
integration methods written in JAX for easy, efficient training data generation.

## Overview

**JAX-FNO** is organised into two main modules:

- **Fourier neural operators** (`jax_fno.operators`): FNO architectures 
    that learn PDE solution operators
- **Initial value problem (IVP) solver** (`jax_fno.solver`): Integration 
    methods for easy, efficient training data generation

## Installation

Clone this repository and install with 
[Poetry](https://python-poetry.org/docs/):

```bash
git clone https://github.com/christianfenton/jax-fno.git
cd jax_fno
poetry install
```

## Getting started

To get started using the project, check out the tutorials:

- **[Solving the heat equation with the IVP solver](https://christianfenton.github.io/jax-fno/tutorials/solving_heat_equation/)**
- **[Learning an antiderivative operator](https://christianfenton.github.io/jax-fno/tutorials/learning_antiderivative/)**

Check out the **[documentation page](https://christianfenton.github.io/jax-fno/)**
for further details.

## Citations

The work in this project is based on
> Li, Zongyi, et al. "Fourier neural operator for parametric partial 
> differential equations."
> *arXiv preprint arXiv:2010.08895* (2020).
> [https://arxiv.org/pdf/2010.08895](https://arxiv.org/pdf/2010.08895)

## Links

Check out the source code on 
[GitHub](https://github.com/christianfenton/jax-fno).

## Future Plans

In the future, `jax_fno.solver.solve_ivp` should be adapted to match 
[`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and added to [`jax.scipy.integrate`](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.integrate).