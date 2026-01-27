import numpy as np

from dd_nm_rom import ops
from typing import Any, Dict, List

from .monolithic import MeshMono


class MeshDD(MeshMono):
  """
  Bidimensional structured mesh for a domain decomposition solver.

  :param nx_intr: Number of interior nodes along the x-axis per subdomain.
  :type nx_intr: int
  :param ny_intr: Number of interior nodes along the y-axis per subdomain.
  :type ny_intr: int
  :param lx_sub: Physical length of each subdomain along the x-axis.
  :type lx_sub: float
  :param ly_sub: Physical length of each subdomain along the y-axis.
  :type ly_sub: float
  :param x0: Origin of the full domain along the x-axis. Defaults to 0.0.
  :type x0: float
  :param y0: Origin of the full domain along the y-axis. Defaults to 0.0.
  :type y0: float
  :param n_sub_x: Number of subdomains along the x-axis. Defaults to 2.
  :type n_sub_x: int
  :param n_sub_y: Number of subdomains along the y-axis. Defaults to 2.
  :type n_sub_y: int
  :param with_bounds: Whether to include boundary nodes. Defaults to False.
  :type with_bounds: bool
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nx_intr: int,
    ny_intr: int,
    lx_sub: float,
    ly_sub: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_sub_x: int = 2,
    n_sub_y: int = 2,
    with_bounds: bool = False
  ) -> None:
    # Number of interior nodes per subdomain
    self.n_nodes_intr = {"x": int(nx_intr), "y": int(ny_intr)}
    self.n_nodes_sub = {"x": int(nx_intr)+2, "y": int(ny_intr)+2}
    # Physical length per subdomain
    self.l_sub = {"x": float(lx_sub), "y": float(ly_sub)}
    # Full domain origin
    self.origin = {"x": float(x0), "y": float(y0)}
    # Number of subdomains
    self.n_subs = {"x": int(n_sub_x), "y": int(n_sub_y)}
    self.n_sub = self.n_subs["x"]*self.n_subs["y"]
    # Include boundary nodes
    self.with_bounds = bool(with_bounds)
    super(MeshDD, self).__init__(**self.get_config_mono())
    self.update_built = False

  # Building
  # ===================================
  def build(self) -> None:
    """
    Build the mesh by calculating node locations, spacings, indices,
    and subdomains residual nodes.
    """
    super(MeshDD, self).build()
    # Subdomains indices
    self.sub_ind = {axis: self._get_indices(axis) for axis in ("x", "y")}
    # x-y subdomains combinations
    self.sub_combs = [np.arange(self.n_subs["y"]), np.arange(self.n_subs["x"])]
    self.sub_combs = ops.generate_combs(self.sub_combs)
    self._set_res_nodes()
    # Update control variable
    self.built = True

  def _get_indices(
    self,
    axis: str
  ) -> List[np.ndarray]:
    """
    Get the indices for the subdomains along a specific axis.

    :param axis: Axis for which to get the indices ('x' or 'y').
    :type axis: str

    :return: List of arrays containing the indices for each subdomain.
    :rtype: List[np.ndarray]
    """
    sub_ind = np.arange(self.n[axis])
    sub_ind_split = []
    si, ei = 0, self.n_nodes_intr[axis]+1
    if self.with_bounds:
      ei += 1
    for _ in range(self.n_subs[axis]-1):
      sub_ind_split.append(sub_ind[si:ei])
      si, ei = ei, ei+self.n_nodes_intr[axis]+2
    sub_ind_split.append(sub_ind[si:])
    return sub_ind_split

  def _set_res_nodes(self) -> None:
    """
    Set the subdomains residual nodes for the mesh.
    """
    self.res_ind = []
    self.res_nodes = []
    for sub in self.sub_combs:
      ind_s = np.ix_(self.sub_ind["y"][sub[0]], self.sub_ind["x"][sub[1]])
      res_s = self.nodes_ind[ind_s].flatten()
      self.res_ind.append(ind_s)
      self.res_nodes.append(np.sort(res_s))

  # Configuration
  # ===================================
  def get_config(self) -> Dict[str, Any]:
    """
    Get the mesh configuration.

    :return: Configuration of the mesh.
    :rtype: Dict[str, Any]
    """
    return self.get_config_dd()

  def get_config_mono(self) -> Dict[str, Any]:
    """
    Get the mesh configuration for a monolithic solver.

    :return: Monolithic solver mesh configuration.
    :rtype: Dict[str, Any]
    """
    return {
      "nx": self._get_n("x"),
      "ny": self._get_n("y"),
      "x_lim": self._get_phylim("x"),
      "y_lim": self._get_phylim("y"),
      "with_bounds": self.with_bounds
    }

  def get_config_dd(self) -> Dict[str, Any]:
    """
    Get the mesh configuration for a domain decomposition solver.

    :return: Domain decomposition solver mesh configuration.
    :rtype: Dict[str, Any]
    """
    return {
      "nx_intr": self.n_nodes_intr["x"],
      "ny_intr": self.n_nodes_intr["y"],
      "lx_sub": self.l_sub["x"],
      "ly_sub": self.l_sub["y"],
      "x0": self.phylim["x"][0],
      "y0": self.phylim["y"][0],
      "n_sub_x": self.n_subs["x"],
      "n_sub_y": self.n_subs["y"],
      "with_bounds": self.with_bounds
    }

  def _get_n(
    self,
    axis: str
  ) -> int:
    """
    Compute number of total nodes along a specific axis.

    :param axis: Axis for which to compute the number of nodes ('x' or 'y').
    :type axis: str

    :return: Total number of nodes along the specified axis.
    :rtype: int
    """
    return int(self.n_subs[axis]*(self.n_nodes_intr[axis]+2)-2)

  def _get_phylim(
    self,
    axis: str
  ) -> List[float]:
    """
    Compute the physical limits along a specific axis.

    :param axis: Axis for which to compute the physical limits ('x' or 'y').
    :type axis: str

    :return: Physical limits along the specified axis.
    :rtype: List[float]
    """
    return [
      self.origin[axis],
      self.origin[axis] + float(self.l_sub[axis]*self.n_subs[axis])
    ]
