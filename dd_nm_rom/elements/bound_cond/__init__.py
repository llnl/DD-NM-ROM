__all__ = [
  "DirichletBC",
  "NeumannBC",
  "PeriodicBC"
]

from .dirichlet import DirichletBC
from .neumann import NeumannBC
from .periodic import PeriodicBC
from .utils import *

# Data types
from typing import Union
BC_TYPES = Union[DirichletBC, NeumannBC, PeriodicBC]
