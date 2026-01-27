import numpy as np
import scipy.sparse as sp

from typing import Dict

from .. import mesh as mesh_mod
from .dirichlet import DirichletBC
from .utils import get_axis_method


class NeumannBC(DirichletBC):
  """
  Neumann boundary conditions class.

  This class handles the construction and application of Neumann boundary 
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
    super(NeumannBC, self).__init__(nu, mesh, funval, advection)
    self.name = "neumann"
    self.update_built = False

  # Building
  # ===================================
  def build(self) -> None:
    """
    Build the boundary condition operators and source terms.
    """
    super(NeumannBC, self).build()
    self.op = {"x": {}, "y": {}}
    for side in self.funval.keys():
      axis, method = get_axis_method(side)
      self.op[axis][method] = self._build_op(
        axis=axis,
        method=method
      )
    self.built = True

  def _build_src(
    self,
    funval: callable,
    axis: str = "x",
    method: str = "fwd"
  ) -> np.ndarray:
    """
    Build the source term for the given boundary condition value.

    :param funval: Function value for the boundary condition.
    :type funval: callable
    :param axis: Axis for the boundary condition ('x' or 'y').
    :type axis: str
    :param method: Method for building the boundary condition ('fwd' or 'bwd').
    :type method: str

    :return: Source term array.
    :rtype: np.ndarray
    """
    f = super(NeumannBC, self)._build_src(funval, axis, method)
    sign = -1 if (method == "fwd") else 1
    f *= (sign * 2.0/3.0 * self.mesh.h[axis])
    return f

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
    # Remove inner points
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n)
    e0[index] = 1.0
    I0 = sp.diags(e0, 0)
    # Assemble
    e = np.ones(n)
    op = sp.spdiags([4.0/3.0*e, -1.0/3.0*e], [0, 1], n, n)
    if (method == "bwd"):
      op = op.T
    return I0 @ op
