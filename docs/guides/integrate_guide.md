# Understanding the ODE integrator

The `jax_fno.integrate` module provides JAX-compatible time integration methods 
for solving initial value problems (IVPs). The solver performs temporal 
discretisation and integration, while users need to handle spatial 
discretisation during setup.

## Quick start

```python
import jax.numpy as jnp
from jax_fno.integrate import solve_ivp, RK4

# 1. Define your discretised PDE as an ODE
# NOTE: This must be written in a JAX-compatible (functionally pure) way
def my_pde_rhs(t, y, args):
    """Right-hand side: dy/dt = f(t, y, ...)

    Implement your spatial discretisation here.
    Handle boundary conditions within this function.
    """
    ...

# 2. Set initial condition
y0 = ...

# 3. Choose time-stepping method
method = RK4()

# 4. Integrate
t_final, y_final = integrate.solve_ivp(
    my_pde_rhs,
    t_span=(0.0, 1.0),
    y0=y0,
    method=method,
    step_size=0.001,
    args=(...,)
)
```

## Time-stepping methods

### Explicit methods

Explicit methods compute the next state directly from the current state. 
They're simple and fast but can require very small time-steps 
for stiff problems.

### Implicit methods

Implicit methods use a root-finding algorithm at each time step, and 
root-finding algorithms often use linear solvers at each iteration.
Implicit methods are usually more expensive than explicit methods per step but 
can allow much larger time steps for stiff problems.

### Extending the time-stepping methods

```python
from dataclasses import dataclass
from typing import Callable
from jax import Array
from jax_fno.integrate import AbstractStepper, solve_ivp

@dataclass(frozen=True)
class MyMethod(AbstractStepper):

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        h: Array,
        args: tuple = ()
    ) -> Array:
        """Advance one time step."""
        ...

t, y = solve_ivp(fun, t_span, y0, MyMethod(), step_size, args)
```

## Extending the root-finding algorithms

Currently only [NewtonRaphson][jax_fno.integrate.NewtonRaphson] is provided
as root-finding algorithm.

Users can extend the root finders by writing their own implementation by
inheriting from [AbstractRootFinder][jax_fno.integrate.AbstractRootFinder].

## Extending the linear solvers

The root-finders often use a linear solver as a subroutine.

The linear solvers currently available are 
[GMRES][jax_fno.integrate.GMRES], [CG][jax_fno.integrate.CG], 
[BiCGStab][jax_fno.integrate.BiCGStab] and 
[Direct][jax_fno.integrate.Direct].

Users can extend the linear solvers by writing their own implementation and 
inheriting from [AbstractLinearSolver][jax_fno.integrate.AbstractLinearSolver].

### JAX transformations

As long as `fun` and `method` are JAX-compatible, 
[solve_ivp][jax_fno.integrate.solve_ivp] should support most JAX transformations, 
however these features have not been properly tested yet.

JIT Compilation:

```python
import jax
from jax_fno.integrate import solve_ivp

# JIT-compile the entire integration
solve_jit = jax.jit(
    solve_ivp(fun, t_span, y0, method, h, args),
    static_argnames=['fun', 'method']
)

y_final = solve_jit(y0)
```

Vectorisation (batching):

```python
# Integrate multiple initial conditions in parallel
y0_batch = jnp.stack([y0_1, y0_2, y0_3])  # (batch, n)

solve_batch = jax.vmap(
    lambda y_: solve_ivp(fun, t_span, y_, method, dt, args)[1]
)

y_final_batch = solve_batch(y0_batch)  # (batch, n)
```
