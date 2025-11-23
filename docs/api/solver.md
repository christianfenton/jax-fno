# Solver

`jax_fno.solver` provides time integration methods for PDEs of the form
$$
\frac{\partial y}{\partial t} = f(t, y).
$$

::: jax_fno.solver.solve_ivp
    options:
        members: []

::: jax_fno.solver.integrate

## Time-stepping schemes

::: jax_fno.solver.ForwardEuler

::: jax_fno.solver.RK4

::: jax_fno.solver.BackwardEuler