# Understanding the solver

The `jax_fno.solver` module provides JAX-compatible time integration methods 
for solving initial value problems (IVPs). The solver performs temporal 
discretisation and integration, while users need to handle spatial 
discretisation in the themselves during setup.

## Quick start

```python
import jax.numpy as jnp
from jax_fno import solver

# 1. Define your discretized PDE as an ODE
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
method = solver.RK4()

# 4. Integrate
t_final, y_final = solver.integrate(
    my_pde_rhs,
    t_span=(0.0, 1.0),
    y0=y0,
    method=method,
    dt=0.001,
    args=(...,)
)
```

## Time-Stepping methods

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
from jax import Array
from typing import Callable
from jax_fno.solver import AbstractStepper

@dataclass(frozen=True)
class MyMethod(AbstractStepper):

    def step(
        self,
        fun: Callable,
        t: Array,
        y: Array,
        dt: Array,
        args: tuple = ()
    ) -> Array:
        """Advance one time step."""
        ...

t, y = solver.integrate(fun, t_span, y0, MyMethod(), dt, args)
```

## Extending the root-finding algorithms

Only one root-finding algorithm is 
currently provided ([NewtonRaphson][jax_fno.solver.NewtonRaphson]), but users 
can extend this by following the 
[RootFindingProtocol][jax_fno.solver.RootFindingProtocol] protocol.

## Extending the linear solvers

For the Newton-Raphson root-finding algorithm, a linear solver is used at each 
iteration. The linear solvers currently available are 
[GMRES][jax_fno.solver.GMRES], [CG][jax_fno.solver.CG], 
[BiCGStab][jax_fno.solver.BiCGStab] and 
[DirectSolve][jax_fno.solver.DirectSolve]. 
Users can extend this by following the 
[LinearSolverProtocol][jax_fno.solver.LinearSolverProtocol] protocol.

### JAX transformations

As long as `fun` and `method` are JAX-compatible, 
[integrate][jax_fno.solver.integrate] should support most JAX transformations, 
however these features have not been properly tested yet.

JIT Compilation:

```python
import jax
from jax_fno.solver import integrate

# JIT-compile the entire integration
integrate_jit = jax.jit(
    integrate(fun, t_span, y0, method, dt, args),
    static_argnames=['fun', 'method']
)

y_final = integrate_jit(y0)
```

Vectorisation (batching):

```python
# Integrate multiple initial conditions in parallel
y0_batch = jnp.stack([y0_1, y0_2, y0_3])  # (batch, n)

integrate_batch = jax.vmap(
    lambda y_: integrate(fun, t_span, y_, method, dt, args)[1]
)

y_final_batch = integrate_batch(y0_batch)  # (batch, n)
```
