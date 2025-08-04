"""
Built-in PDEs

Pre-implemented residual functions and Jacobian-vector products
for common problems like Burgers' and heat/diffusion equations.

All functions follow the standard interface:
- residual_fn(u_new, u_old, dt, dx, params) -> residual
- jvp_fn(u_new, dt, dx, params, v) -> jacobian @ v
"""

from .burgers import burgers_residual_1d, burgers_jvp_1d
from .heat import heat_residual_1d, heat_jvp_1d

__all__ = [
    "burgers_residual_1d",
    "burgers_jvp_1d", 
    "heat_residual_1d",
    "heat_jvp_1d",
]