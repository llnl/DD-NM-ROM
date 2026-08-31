__all__ = [
  "GaussNewton",
  "Newton",
  "DistNewton"
]

from .gauss_newton import GaussNewton
from .newton import Newton
from .dist_newton import DistNewton

# Data types
from typing import Union
SOLVER_TYPES = Union[GaussNewton, Newton, DistNewton]
