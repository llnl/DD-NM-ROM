__all__ = [
  "GaussNewton",
  "Newton"
]

from .gauss_newton import GaussNewton
from .newton import Newton

# Data types
from typing import Union
SOLVER_TYPES = Union[GaussNewton, Newton]
