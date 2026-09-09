import numpy as np
import scipy.sparse as sp

from typing import Dict

from .neumann import NeumannBC
from .. import mesh as mesh_mod
from ... import backend as bkd


class PeriodicBC(NeumannBC):
  """
  Periodic boundary conditions class.

  This class handles the construction and application of periodic boundary 
  conditions for a given mesh and boundaries function values.

  :param nu: Viscosity or diffusion coefficient.
  :type nu: float
  :param mesh: Mesh object on which the boundary conditions are applied.
  :type mesh: MESH_TYPES
  :param funval: A dictionary defining the function values for 
                 the boundary conditions. The keys areand the 
                 values are dictionaries with axis ('x' or 'y')  
                 and method ('fwd' or 'bwd').
  :type funval: Dict[str, Dict[str, callable]]
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    mesh: mesh_mod.MESH_TYPES,
    funval: Dict[str,Dict[str,callable]],
    advection: bool = True
  ) -> None:
    super(PeriodicBC, self).__init__(nu, mesh, funval, advection)
    self.name = "periodic"

  # Building
  # ===================================
  def _build_src(
    self,
    funval: callable,
    axis: str = "x",
    method: str = "fwd"
  ) -> np.ndarray:
    """
    Build the zero-valued source term for the given boundary condition value.

    :param funval: Function value for the boundary condition.
    :type funval: callable
    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: Source term array.
    :rtype: np.ndarray
    """
    f = super(PeriodicBC, self)._build_src(funval, axis, method)
    return np.zeros_like(f)

  def _build_op(
    self,
    axis: str = "x",
    method: str = "fwd"
  ) -> sp.spmatrix:
    """
    Build the boundary condition operator.

    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: The operator matrix.
    :rtype: sp.spmatrix
    """
    n = self.mesh.n[axis]
    op = sp.lil_matrix((n, n))
    i, j = (0,-1) if (method == "fwd") else (-1,0)
    op[i,j] = 1.0
    return op.tocsr()
