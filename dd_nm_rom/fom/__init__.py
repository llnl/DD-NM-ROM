__all__ = [
  "Burgers2D",
  "DDBurgers2D",
  "Poisson2D",
  "DDPoisson2D",
  "Elasticity2D",
  "DDElasticity2D"
]

from .domain_dec import DDBurgers2D, DDPoisson2D, DDElasticity2D
from .monolithic import Burgers2D, Poisson2D#, Elasticity2D

# Data types
from typing import Union
#FOM_TYPES = Union[Burgers2D, DDBurgers2D, Poisson2D, DDPoisson2D, Elasticity2D, DDElasticity2D]
FOM_TYPES = Union[Burgers2D, DDBurgers2D, Poisson2D, DDPoisson2D, DDElasticity2D]
