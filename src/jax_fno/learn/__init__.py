"""
Fourier Neural Operators

Neural network architectures that learn mappings between function spaces.
"""

from .fno1d import FNO1D, FourierLayer1D, SpectralConv1D

__all__ = [
    "FNO1D",
    "FourierLayer1D",
    "SpectralConv1D"
]