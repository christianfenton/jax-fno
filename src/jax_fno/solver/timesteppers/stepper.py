from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp


class Stepper(ABC):
    """Base class for time-stepping schemes."""
    
    @staticmethod
    @abstractmethod
    def step(
        f: Callable[[jnp.ndarray, float], jnp.ndarray],
        u: jnp.ndarray,
        t:float,
        dt: float,
        *args,
        **kwargs
    ) -> jnp.ndarray:
        """Advance the solution u from t to t+dt."""
        pass