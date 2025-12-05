# Solver

`jax_fno.solver` provides time integration methods for PDEs of the form
$$
\frac{\partial y}{\partial t} = f(t, y).
$$

::: jax_fno.solver.integrate

::: jax_fno.solver.solve_ivp
    options:
        members: []

## Time-stepping schemes

::: jax_fno.solver.ForwardEuler

::: jax_fno.solver.RK4

::: jax_fno.solver.BackwardEuler

## Root-finding algorithms

::: jax_fno.solver.RootFindingProtocol

::: jax_fno.solver.NewtonRaphson

::: jax_fno.solver.GMRES

::: jax_fno.solver.CG

::: jax_fno.solver.BiCGStab

::: jax_fno.solver.DirectSolve

## Linear solvers

::: jax_fno.solver.LinearSolverProtocol