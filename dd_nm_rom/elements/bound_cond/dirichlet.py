import numpy as np

from typing import Dict

from .. import mesh as mesh_mod
from .utils import get_axis_method, SIDES


class DirichletBC(object):
  """
  Dirichlet boundary conditions class.

  This class handles the construction and application of Dirichlet boundary 
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
    self.name = "dirichlet"
    self.nu = nu
    self.mesh = mesh
    self.funval = funval
    self.f = None
    self.op = None
    # Control variables
    self.built = False
    self.update_built = True
    self.advection = advection

  # Building
  # ===================================
  def is_built(self) -> None:
    """
    Check if the boundary conditions have been built.

    :raises ValueError: If the boundary conditions have not been built.
    """
    if (not self.built):
      raise ValueError("BC not built. Please, call 'build' method first.")

  def build(self) -> None:
    """
    Build the boundary condition operators and source terms.
    """
    self.f = {}
    # Loop over variables
    for k in ("u", "v"):
      f_k = {}
      # Loop over sides
      for (side, sfunval) in self.funval.items():
        axis, method = get_axis_method(side)
        f_k[side] = self._build_src(
          funval=sfunval[k],
          axis=axis,
          method=method
        )
      self.f[k] = {"D": self._compose_dif_src(f_k)}
      if self.advection:
        self.f[k]["A"] = self._compose_adv_src(f_k)
    if self.update_built:
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
    n, lim = self.mesh.n, self.mesh.lim
    index = 0 if (method == "fwd") else -1
    e0 = np.zeros(n[axis])
    e0[index] = 1.0
    if (axis == "x"):
      x = np.full(n["y"], lim["x"][index])
      y = self.mesh.nodes_1d["y"]
      return np.kron(funval(x, y), e0)
    else:
      x = self.mesh.nodes_1d["x"]
      y = np.full(n["x"], lim["y"][index])
      return np.kron(e0, funval(x, y))

  def _compose_adv_src(
    self,
    values: Dict[str, np.ndarray]
  ) -> Dict[str, np.ndarray]:
    """
    Compose the advection source term.

    :param values: Dictionary of source term values for each side.
    :type values: Dict[str, np.ndarray]

    :return: Advection source term for each axis.
    :rtype: Dict[str, np.ndarray]
    """
    f = {}
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] - values[si[1]]
      f[axis] = -(0.5/self.mesh.h[axis]) * fi
    return f

  def _compose_dif_src(
    self,
    values: Dict[str, np.ndarray]
  ) -> Dict[str, np.ndarray]:
    """
    Compose the diffusion source term.

    :param values: Dictionary of source term values for each side.
    :type values: Dict[str, np.ndarray]

    :return: Diffusion source term for each axis.
    :rtype: Dict[str, np.ndarray]
    """
    f = 0.0
    for axis in ("x", "y"):
      si = SIDES[axis]
      fi = values[si[0]] + values[si[1]]
      f = f + (self.nu/self.mesh.h[axis]**2) * fi
    return f
