import numpy as np

from typing import List
from dd_nm_rom.elements import mesh as mesh_mod

from .sin_multi_peak import SinMultiPeak


class SinPeak(SinMultiPeak):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    mu_lim: List[float] = [0.9, 1.1],
    bc_type: str = "neumann"
  ) -> None:
    super(SinPeak, self).__init__(
      mesh=mesh,
      mu_lim=mu_lim,
      forced_config=None,
      bc_type=bc_type
    )

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    # Define possible combinations
    self.configs = np.array([1,0,0,0])
    # Define design space
    self.design_space = np.array(self.mu_lim).reshape(-1,1)

  def sample_design_space(self) -> np.ndarray:
    self.init_design_space()
    return self.configs * np.random.uniform(*self.mu_lim)

  def _convert_dmat_to_mu(
    self,
    dmat: np.ndarray
  ) -> np.ndarray:
    return self.configs.reshape(1,-1) * dmat
