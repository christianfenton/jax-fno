"""Abstract base class for time-stepping schemes."""

from abc import ABC


class AbstractStepper(ABC):
    """
    Base class for all time-stepping schemes.

    This provides a common type for both explicit and implicit steppers,
    enabling type-based dispatch in the integration routines.
    """
    pass
