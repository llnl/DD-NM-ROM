import numpy as np

from dd_nm_rom import ops
from typing import Any, Dict, List


class MeshMono(object):
  """
  Bidimensional structured mesh for a monolithic solver.

  :param nx: Number of nodes in the x-direction (excluding boundaries).
  :type nx: int
  :param ny: Number of nodes in the y-direction (excluding boundaries).
  :type ny: int
  :param x_lim: Physical limits in the x-direction.
  :type x_lim: List[float]
  :param y_lim: Physical limits in the y-direction.
  :type y_lim: List[float]
  :param with_bounds: Whether to include boundary nodes. Defaults to False.
  :type with_bounds: bool
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nx: int,
    ny: int,
    x_lim: List[float],
    y_lim: List[float],
    with_bounds: bool = False
  ) -> None:
    self.name = self.__class__.__name__
    # Include boundary nodes
    self.with_bounds = bool(with_bounds)
    self.shift = 0 if self.with_bounds else 1
    # Number of nodes
    self.n = {"x": int(nx), "y": int(ny)}
    if self.with_bounds:
      for axis in self.n.keys():
        self.n[axis] += 2
    # Physical limits
    self.phylim = {
      "x": np.sort(x_lim).astype(float),
      "y": np.sort(y_lim).astype(float)
    }
    # Control variables
    self.built = False
    self.update_built = True

  # Building
  # ===================================
  def is_built(self) -> None:
    """
    Check if the mesh has been built. Raises an error if not.

    :raises ValueError: If the mesh has not been built.
    """
    if (not self.built):
      raise ValueError("Mesh not built. Please, call 'build' method first.")

  def build(self) -> None:
    """
    Build the mesh by calculating node locations, spacings and indices.
    """
    # Total number of nodes
    self.nxy = self.n["x"] * self.n["y"]
    # Spacing and nodes location
    self.h, self.lim, self.nodes_1d = {}, {}, {}
    for (axis, lim) in self.phylim.items():
      n = self.n[axis]
      h = (lim[1]-lim[0])/(n+2*self.shift)
      self.h[axis] = h
      self.lim[axis] = [lim[0]+0.5*h, lim[1]-0.5*h]
      self.nodes_1d[axis] = lim[0] + (0.5+self.shift+np.arange(n))*h
    self.hxy = self.h["x"] * self.h["y"]
    self.grid = np.meshgrid(self.nodes_1d["x"], self.nodes_1d["y"])
    # Nodes indices
    self.nodes_val = [self.nodes_1d["y"], self.nodes_1d["x"]]
    self.nodes_val = ops.generate_combs(self.nodes_val)
    self.nodes_ind = np.arange(self.nxy).reshape(self.n["y"], self.n["x"])
    # > Inner/Bounds nodes indices
    if self.with_bounds:
      self.nodes_ind_inner = self.nodes_ind[1:-1,1:-1].flatten()
    else:
      self.nodes_ind_inner = self.nodes_ind.flatten()
    self.nodes_ind_bound = np.setdiff1d(
      self.nodes_ind.flatten(), self.nodes_ind_inner
    )
    # Update control variable
    if self.update_built:
      self.built = True

  # Configuration
  # ===================================
  def get_config(self) -> Dict[str, Any]:
    """
    Get the mesh configuration.

    :return: Configuration of the mesh.
    :rtype: Dict[str, Any]
    """
    return self.get_config_mono()

  def get_config_mono(self) -> Dict[str, Any]:
    """
    Get the mesh configuration for a monolithic solver.

    :return: Monolithic solver mesh configuration.
    :rtype: Dict[str, Any]
    """
    return {
      "nx": self.n["x"],
      "nx": self.n["y"],
      "x_lim": self.phylim["x"],
      "y_lim": self.phylim["y"],
      "with_bounds": self.with_bounds
    }

  def get_config_dd(
    self,
    n_sub_x: int = 2,
    n_sub_y: int = 2
  ) -> Dict[str, Any]:
    """
    Get the mesh configuration for a domain decomposition solver.

    :param n_sub_x: Number of subdomains in the x-direction. Defaults to 2.
    :type n_sub_x: int
    :param n_sub_y: Number of subdomains in the y-direction. Defaults to 2.
    :type n_sub_y: int

    :return: Domain decomposition solver mesh configuration.
    :rtype: Dict[str, Any]
    """
    return {
      "nx_intr": self._get_n_nodes_intr(self.n["x"], n_sub_x),
      "ny_intr": self._get_n_nodes_intr(self.n["y"], n_sub_y),
      "lx_sub": self._get_l_sub(self.phylim["x"], n_sub_x),
      "ly_sub": self._get_l_sub(self.phylim["y"], n_sub_y),
      "x0": self.phylim["x"][0],
      "y0": self.phylim["y"][0],
      "n_sub_x": n_sub_x,
      "n_sub_y": n_sub_y,
      "with_bounds": self.with_bounds
    }

  def _get_n_nodes_intr(
    self,
    n: int,
    n_sub: int
  ) -> int:
    """
    Compute number of interior nodes for each subdomain along a specific axis.

    :param n: Number of nodes along the axis.
    :type n: int
    :param n_sub: Number of subdomains along the axis.
    :type n_sub: int

    :return: Number of interior nodes for each subdomain.
    :rtype: int
    """
    if self.with_bounds:
      n -= 2
    return int((n+2)/n_sub-2)

  def _get_l_sub(
    self,
    phylim: List[float],
    n_sub: int
  ) -> float:
    """
    Compute the physical length for each subdomain along a specific axis.

    :param phylim: Physical limits along the axis.
    :type phylim: List[float]
    :param n_sub: Number of subdomains along the axis.
    :type n_sub: int

    :return: Physical length for each subdomain.
    :rtype: float
    """
    return (phylim[1]-phylim[0])/n_sub
