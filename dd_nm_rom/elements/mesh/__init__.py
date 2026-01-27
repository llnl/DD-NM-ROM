__all__ = [
  "MeshDD",
  "MeshMono"
]

from .domain_dec import MeshDD
from .monolithic import MeshMono

# Data types
from typing import Union
MESH_TYPES = Union[MeshDD, MeshMono]
