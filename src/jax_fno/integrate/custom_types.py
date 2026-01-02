"""Type aliases to improve type hint readability."""

from typing import Callable
from jax import Array

type LinearMap = Callable[[Array], Array]
type JacobianConstructor = Callable[[Array], Array]
type JVPConstructor = Callable[[Array, Array], Array]
