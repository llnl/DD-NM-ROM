import numpy as np

from typing import Dict, List
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

from .basic import BasicField


class Burgers2DExact(BasicField):
  """
  Implementation of the exact solution for the 2D Burgers' equation.

  This class provides methods to compute the exact solution of the 2D Burgers' 
  equation under given boundary conditions. The exact solution is useful for 
  validating numerical methods and understanding the behavior of viscous 
  fluid flows.

  References:
    - https://onlinelibrary.wiley.com/doi/epdf/10.1002/fld.1650030302
    - https://doi.org/10.1016/j.cma.2021.113997

  :param mesh: The computational mesh.
  :type mesh: MESH_TYPES
  :param nu: The kinematic viscosity. Default is 1e-2.
  :type nu: float
  :param a_lim: Limits for the parameter 'a'. Default is [1e0, 1e4].
  :type a_lim: List[float]
  :param k_lim: Limits for the parameter 'k'. Default is [5.0, 25.0].
  :type k_lim: List[float]
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float = 1e-2,
    a_lim: List[float] = [1e0, 1e4],
    k_lim: List[float] = [5.0, 25.0],
    use_qmc: bool = True
  ) -> None:
    super(Burgers2DExact, self).__init__(mesh)
    self.x0 = 1.0
    self.a = np.zeros(4)
    self.nu = float(nu)
    self.Re = 1/self.nu
    self.a_lim = list(a_lim)
    self.k_lim = list(k_lim)
    self.use_qmc = use_qmc

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    self.design_space = np.array([self.a_lim, self.k_lim]).T

  def construct_design_mat(
    self,
    n_samples: int
  ) -> np.ndarray:
    if self.use_qmc:
      dmat, _ = super(Burgers2DExact, self).construct_design_mat_qmc(n_samples)
      return self._broadcast(dmat)
    return self._broadcast(super(Burgers2DExact, self).construct_design_mat(n_samples))

  def set_params(
    self,
    mu: np.ndarray
  ) -> None:
    """
    Abstract method to set parameters for the field.

    :param args: Positional arguments for setting parameters.
    :param kwargs: Keyword arguments for setting parameters.
    """
    mu = mu.reshape(-1)
    self.a[:2] = mu[0]
    self.k = mu[1]

  # Velocity fields
  # ===================================
  def u(
    self,
    x: np.ndarray,
    y: np.ndarray
  ) -> np.ndarray:
    """
    Compute the velocity field :math:`u`.

    :param x: x-coordinates of the points.
    :type x: np.ndarray
    :param y: y-coordinates of the points.
    :type y: np.ndarray

    :return: The computed velocity field :math:`u`.
    :rtype: np.ndarray
    """
    f = self.a[1] + self.a[3]*y + self.k*self._psi(x,-1)*np.cos(self.k*y)
    return f/self._phi(x, y)

  def v(
    self,
    x: np.ndarray,
    y: np.ndarray
  ) -> np.ndarray:
    """
    Compute the velocity field :math:`v`.

    :param x: x-coordinates of the points.
    :type x: np.ndarray
    :param y: y-coordinates of the points.
    :type y: np.ndarray

    :return: The computed velocity field :math:`v`.
    :rtype: np.ndarray
    """
    f = self.a[2] + self.a[3]*x - self.k*self._psi(x)*np.sin(self.k*y)
    return f/self._phi(x, y)

  def _phi(
    self,
    x: np.ndarray,
    y: np.ndarray
  ) -> np.ndarray:
    f = self.a[0] + self.a[1]*x + self.a[2]*y \
      + self.a[3]*x*y + self._psi(x)*np.cos(self.k*y)
    return -0.5*self.Re*f

  def _psi(
    self,
    x: np.ndarray,
    sign: int = 1
  ) -> np.ndarray:
    f = self.k*(x-self.x0)
    return np.exp(f) + sign*np.exp(-f)

  def get_init(self) -> None:
    raise NotImplementedError(
      f"The '{self.name}' field can be used only for steady-state problems."
    )

  # Boundary conditions
  # ===================================
  def get_bc_funval(self) -> Dict[str, Dict[str, callable]]:
    """
    Get the boundary condition function values for all sides.

    :return: Dictionary of boundary condition functions for each side.
    :rtype: Dict[str, Dict[str, callable]]
    """
    funval = {}
    for side in bc_mod.SIDES["all"]:
      funval[side] = {}
      for z in ("u", "v"):
        funval[side][z] = self.u if (z == "u") else self.v
    return funval
