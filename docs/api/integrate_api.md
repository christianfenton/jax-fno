# Integration (jax_fno.integrate)

`jax_fno.integrate` provides time integration methods for PDEs of the form
$$ \frac{\partial y}{\partial t} = f(t, y). $$

::: jax_fno.integrate.solve_ivp

::: jax_fno.integrate.solve_with_history
    options:
        members: []

## Time-stepping schemes

::: jax_fno.integrate.ForwardEuler

::: jax_fno.integrate.RK4

::: jax_fno.integrate.BackwardEuler

::: jax_fno.integrate.AbstractStepper

## Root-finding algorithms

::: jax_fno.integrate.NewtonRaphson

::: jax_fno.integrate.AbstractRootFinder

## Linear solvers

::: jax_fno.integrate.GMRES

::: jax_fno.integrate.CG

::: jax_fno.integrate.BiCGStab

::: jax_fno.integrate.Direct

::: jax_fno.integrate.AbstractLinearSolver