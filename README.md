# JAX-FNO: Fourier Neural Operators in JAX

This project provides JAX/Flax implementations of Fourier Neural Operators 
(FNOs) for solving partial differential equations (PDEs), along with a 
JAX-based initial-value problem (IVP) solver that can be used to generate 
new training and testing data.

The FNO work in this project is based on [Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).](https://arxiv.org/pdf/2010.08895)

## Installation

Clone this repository and install with Poetry
```bash
git clone <repository-url>
cd jax_fno
poetry install
```

# Overview

This project provides two Python modules:

- jax_fno.solver: Time-stepping solver for data generation
- jax_fno.operators: Fourier neural operators

Check out the [documentation page](https://christianfenton.github.io/jax-fno/) for further details on these modules.
