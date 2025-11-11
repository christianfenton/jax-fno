"""
Implicit time-stepping schemes.

This module implements implicit time integration methods that require solving
non-linear systems via Newton-Raphson iterations.

A linear solver is used inside each Newton-Raphson iteration.

Architecture:
    1. Configuration dataclasses (NewtonConfig, LinSolverConfig)
    2. Generic utilities (newton_raphson, dispatch_linear_solver)
    3. Scheme-specific classes (BackwardEuler, etc.)

Each implicit scheme provides:
    - make_residual: Creates residual function R(u_{n+1}) for the scheme
    - make_jvp: Creates matrix-free Jacobian-vector product function
    - step: Advance the solution by one time step

Note: Only matrix-free (JVP) Jacobians are supported. For small systems where
      dense Jacobians might be more efficient, explicit methods are recommended.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_sparse
from typing import Callable
from .stepper import Stepper


@dataclass(frozen=True)
class LinearSolverConfig:
    """
    Configuration for linear solver used within Newton-Raphson iterations.

    Attributes:
        method: Linear solver method (iterative only, for matrix-free Jacobians)
            - 'gmres': Generalised minimal residuals (default, robust)
            - 'cg': Conjugate gradients (for symmetric positive definite systems)
            - 'bicgstab': Stabilised biconjugate gradients (for non-symmetric systems)
        tol: Convergence tolerance for iterative solver
        maxiter: Maximum iterations for iterative solver
    """
    method: str = 'gmres'
    tol: float = 1e-6
    maxiter: int = 100

    def __post_init__(self):
        valid_methods = {'gmres', 'cg', 'bicgstab'}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid linear solver method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )


def newton_raphson(
        u_guess: jnp.ndarray,
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        jvp_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        linsolver: Callable[[Callable, jnp.ndarray], jnp.ndarray],
        tol: float,
        maxiter: int
        
    ) -> jnp.ndarray:
    """
    Solve the non-linear system R(u) = 0 using a Newton-Raphson method.

    Iterative update: u_{k+1} = u_k - J^{-1}(u_k) * R(u_k)

    Args:
        u_guess: Initial guess
        residual_fn: A function returning the residual R(u)
        jvp_fn: Jacobian-vector product function: (u, v) -> J(u) * v
        linsolver_fn: Linear solver: (J_op, rhs) -> solution
        tol: Convergence tolerance for Newton-Raphson method. Default: 1e-6.
        maxiter: Maximum number of Newton-Raphson iterations. Default: 50.

    Returns:
        Final solution u
    """
    
    u_k = u_guess
    r_k = residual_fn(u_k)
    state0 = (u_k, r_k, 0)

    def body_fun(state):
        """Update solution by solving linear system: J(u_k) * delta = -r_k"""
        u_k, r_k, k = state
        jac_op = lambda v: jvp_fn(u_k, v)
        delta = linsolver(jac_op, -r_k)
        u_kp1 = u_k + delta
        r_kp1 = residual_fn(u_kp1)
        return (u_kp1, r_kp1, k+1)

    def cond_fun(state):
        _, r_k, k = state
        return (jnp.linalg.norm(r_k) > tol) & (k < maxiter)
    
    u_final, _, niters = jax.lax.while_loop(cond_fun, body_fun, state0)

    callback = lambda k, kmax : print(
        f"Newton-Raphson method did not converge within {int(kmax)} iterations.") if k >= kmax else None
    jax.pure_callback(callback, None, niters, maxiter)

    return u_final


def dispatch_linear_solver(config: LinearSolverConfig) -> Callable[[Callable, jnp.ndarray], jnp.ndarray]:
    """
    Function factory for a matrix-free iterative linear solver.

    Args:
        config: Linear solver configuration

    Returns:
        Solver function with signature: (jvp_operator, rhs) -> solution
    """
    if config.method == 'gmres':
        def gmres_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.gmres(
                jvp_fn, rhs,
                tol=config.tol,
                maxiter=config.maxiter
            )
            return solution
        return gmres_solve

    elif config.method == 'cg':
        def cg_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.cg(
                jvp_fn, rhs,
                tol=config.tol,
                maxiter=config.maxiter
            )
            return solution
        return cg_solve

    elif config.method == 'bicgstab':
        def bicgstab_solve(jvp_fn, rhs):
            solution, _ = jax_sparse.bicgstab(
                jvp_fn, rhs,
                tol=config.tol,
                maxiter=config.maxiter
            )
            return solution
        return bicgstab_solve

    else:
        raise ValueError(f"Unknown linear solver method: {config.method}")
    

def default_linear_solver():
    """
    Default linear solver for use inside Newton-Raphson iterations.

    Default Settings:
        method: 'gmres'
        tol: 1e-6
        maxiter: 100
    """
    return dispatch_linear_solver(LinearSolverConfig())


class BackwardEuler(Stepper):
    """
    Backward Euler time-stepping scheme.

    Discretisation:
        du/dt = f(u, t) --> (u_{n+1} - u_n) / dt = f(u_{n+1}, t_{n+1})

    Residual:
        R(u_{n+1}) = u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1})

    Jacobian:
        J = ∂R/∂u_{n+1} = I - dt * ∂f/∂u(u_{n+1}, t_{n+1})
    """

    @staticmethod
    def make_residual(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u_prev: jnp.ndarray,
        t_prev: float,
        dt: float
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Create residual function for a backward Euler scheme.

        Args:
            f: RHS of ODE du/dt = f(u, t)
            u_prev: Solution at previous time step t_n
            t_prev: Time at previous step t_n
            dt: Time step size

        Returns:
            Residual function R(u_{n+1}) = u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1})
        """
        residual_fn = lambda u_new: u_new - u_prev - dt * f(u_new, t_prev + dt)
        return residual_fn

    @staticmethod
    def make_jvp(
        df: Callable[[jnp.ndarray, float], Callable[[jnp.ndarray], jnp.ndarray]],
        t_prev: float,
        dt: float
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Create matrix-free Jacobian-vector product function for a backward Euler.

        Jacobian of residual: J = I - dt * ∂f/∂u

        Args:
            df: Jacobian-vector product of f.
                df(u, t) returns a function v -> (∂f/∂u)*v
            t_prev: Time at previous step
            dt: Time step size

        Returns:
            Function (u, v) -> J(u) * v = v - dt * (∂f/∂u)*v
        """
        return lambda u, v: v - dt * df(u, t_prev + dt)(v)

    @staticmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t: float,
        dt: float,
        *,
        df: Callable[[jnp.ndarray, float], Callable[[jnp.ndarray], jnp.ndarray]],
        tol: float = 1e-6,
        maxiter: int = 50,
        linsolver: Callable[[Callable, jnp.ndarray], jnp.ndarray] = default_linear_solver()
    ) -> jnp.ndarray:
        """
        Advance solution by one time step.

        Solves: u_{n+1} - u_n - dt * f(u_{n+1}, t_{n+1}) = 0
        for u_{n+1} using Newton-Raphson method with a matrix-free Jacobian.

        Args:
            f: RHS function: du/dt = f(u, t)
            u: Current solution at time t
            t: Current time
            dt: Time step size
            df: Jacobian-vector product of f (REQUIRED).
                df(u, t) returns a function v -> (∂f/∂u)*v
            tol: Convergence tolerance for Newton-Raphson method
            maxiter: Maximum number of Newton-Raphson iterations
            linsolver: Iterative linear solver

        Returns:
            Solution at time t + dt
        """
        # Define residual function for this step
        residual_fn = BackwardEuler.make_residual(f, u, t, dt)

        # Define Jacobian-vector product
        jvp_fn = BackwardEuler.make_jvp(df, t, dt)

        # Initial guess (forward Euler step)
        u_guess = u + dt * f(u, t)

        # Solve system with Newton-Raphson method
        u_next = newton_raphson(u_guess, residual_fn, jvp_fn, linsolver, tol=tol, maxiter=maxiter)

        return u_next
