import numpy as np

from typing import List, Union
from dd_nm_rom import ops
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

from .basic import BasicField


class PoissonForce(BasicField):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    mu_lim: List[float] = [0.9, 1.1],
    forced_config: Union[List[int], np.ndarray, None] = None,
    bc_type: str = "periodic",
    use_qmc: bool = True
  ) -> None:
    super(PoissonForce, self).__init__(mesh)
    bc_mod.check_bc_type(bc_type)
    self.bc_type = bc_type
    self.mu_lim = mu_lim
    self.configs = None
    self.forced_config = forced_config
    self.use_qmc = use_qmc
    if (self.forced_config is not None):
      self.forced_config = np.array(self.forced_config).reshape(-1)

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    if self.use_qmc:
      self.design_space = [self.mu_lim] * self.mesh.n_sub + [[2, 6]] * 2
    else:
      # Define possible combinations (legacy path).
      self.configs = ops.generate_combs([np.arange(2)]*self.mesh.n_sub)[1:]
      if (self.forced_config is not None):
        self.configs += self.forced_config.reshape(1,-1)
        self.configs = self.configs.astype(bool).astype(int)
        self.configs = np.unique(self.configs, axis=0)
      self.design_space = [[0,len(self.configs)]] + [self.mu_lim]*self.mesh.n_sub + [[2, 6]]*2
    self.design_space = np.array(self.design_space).T

  def sample_design_space(self) -> np.ndarray:
    force_magnitudes = np.random.uniform(*self.mu_lim, size=self.mesh.n_sub)
    force_magnitudes[3] = force_magnitudes[1] + force_magnitudes[2] - force_magnitudes[0]

    # Generate frequency parameters (integer and even values between 2 and 6)
    freq_x = 2 * np.random.randint(1, 4)  # Will give 2, 4, or 6
    freq_y = 2 * np.random.randint(1, 4)  # Will give 2, 4, or 6

    mu = np.concatenate([force_magnitudes, [freq_x, freq_y]])
    return mu

  def construct_design_mat(
    self,
    n_samples: int
  ) -> np.ndarray:
    if self.use_qmc:
      dmat, mask = super(PoissonForce, self).construct_design_mat_qmc(n_samples)
      if self.mesh.n_sub >= 4:
        dmat[:, self.mesh.n_sub - 1] = (
          dmat[:, 1] + dmat[:, 2] - dmat[:, 0]
        )
      return self._convert_dmat_to_mu(dmat, mask)

    # Legacy LHS path.
    dmat = super(PoissonForce, self).construct_design_mat(n_samples)
    if dmat.shape[1] == 7:  # Only if we have 4 subdomains + config index
        # Ensure column[0] + column[3] = column[1] + column[2]
        # So column[3] = column[1] + column[2] - column[0]
        dmat[:, 4] = dmat[:, 2] + dmat[:, 3] - dmat[:, 1]
        
        # Make sure values stay within valid range
        # Assumption: mu_lim is something like [0.9, 1.1]
#       min_val, max_val = self.mu_lim
#       dmat[:, 4] = np.clip(dmat[:, 4], min_val, max_val)
    return self._convert_dmat_to_mu(dmat)

  def _convert_dmat_to_mu(
    self,
    dmat: np.ndarray,
    mask: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    if self.use_qmc:
      force = dmat[:, :self.mesh.n_sub]
      if (self.forced_config is not None):
        forced = self.forced_config.reshape(1, -1)
        mask = np.maximum(mask[:, :self.mesh.n_sub], forced)
      else:
        mask = mask[:, :self.mesh.n_sub]
      force = mask * force
      frequencies = dmat[:, -2:].copy()
      frequencies = (2 * np.round(frequencies / 2)).astype(np.int32)
      return np.hstack([force, frequencies])

    cfg = np.floor(dmat[:,0]).astype(np.int32)
    # Ensure the last two columns are integer and even
    dmat[:, -2:] = (2 * np.round(dmat[:, -2:] / 2)).astype(np.int32)
#   return self.configs[cfg] * dmat[:,1:]
    return dmat[:,1:]

  def set_params(
    self,
    mu: np.ndarray
  ) -> None:
    #self.mu = mu.reshape(-1)
    mu = mu.reshape(-1)
    self.mu = mu[:self.mesh.n_sub]  # First 4 elements for force magnitudes
    self.freq_x = mu[-2]  # Second-to-last element for x frequency
    self.freq_y = mu[-1]  # Last element for y frequency

  # Velocity fields
  # ===================================
  def u(self) -> np.ndarray:
    return self.generate_field()

  def v(self) -> np.ndarray:
    return self.generate_field()

  def generate_field(self) -> np.ndarray:
    f = np.zeros(self.mesh.nxy)
    x, y = self.mesh.nodes_val.T
    for (i, mu_i) in enumerate(self.mu):
      ind = self.mesh.res_nodes[i]
      f[ind] = self._phi(x[ind], y[ind], mu_i)
    return f.reshape(self.mesh.n["y"], self.mesh.n["x"])

  def _phi(
    self,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray
  ) -> np.ndarray:
    # In order for the Poisson problem parameterized with forcing term to have solution
    # One has to becareful on creating the forcing term, it has to have zero mean.
    return mu*np.sin(self.freq_x*np.pi*x)*np.sin(self.freq_y*np.pi*y)

  def get_force(
    self,
    mu: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    if (mu is not None):
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])
