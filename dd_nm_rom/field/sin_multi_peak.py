import numpy as np

from typing import List, Union
from dd_nm_rom import ops
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

from .basic import BasicField


class SinMultiPeak(BasicField):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    mu_lim: List[float] = [0.9, 1.1],
    forced_config: Union[List[int], np.ndarray, None] = None,
    bc_type: str = "neumann"
  ) -> None:
    super(SinMultiPeak, self).__init__(mesh)
    bc_mod.check_bc_type(bc_type)
    self.bc_type = bc_type
    self.mu_lim = mu_lim
    self.configs = None
    self.forced_config = forced_config
    if (self.forced_config is not None):
      self.forced_config = np.array(self.forced_config).reshape(-1)

  # Design space
  # ===================================
  def _init_design_space(self) -> None:
    """
    2D array of shape (2, n_params) defining the lower and upper bounds for sampling input parameters.

    - The first parameter is a configuration index in [0, len(self.configs)), which selects a binary activation pattern for subdomains.
    - The remaining parameters are amplitude values mu_i for each subdomain, sampled uniformly from self.mu_lim = [mu_min, mu_max].
    - Only subdomains activated in the selected configuration will use their mu_i value; others are set to zero.

    The structure is:
        design_space[0, :] -> lower bounds
        design_space[1, :] -> upper bounds
    """
    # Define possible combinations
    self.configs = ops.generate_combs([np.arange(2)]*self.mesh.n_sub)[1:]
    if (self.forced_config is not None):
      self.configs += self.forced_config.reshape(1,-1)
      self.configs = self.configs.astype(bool).astype(int)
      self.configs = np.unique(self.configs, axis=0)
    # Define design space
    self.design_space = [[0,len(self.configs)]] + [self.mu_lim]*self.mesh.n_sub
    self.design_space = np.array(self.design_space).T

  def sample_design_space(self) -> np.ndarray:
    s = 0.0
    while (s == 0.0):
      config = np.random.binomial(1, p=0.5, size=self.mesh.n_sub)
      if (self.forced_config is not None):
        config += self.forced_config
        config = config.astype(bool).astype(int)
      s = np.sum(config)
    mu = config * np.random.uniform(*self.mu_lim, size=self.mesh.n_sub)
    return mu

  def construct_design_mat(
    self,
    n_samples: int
  ) -> np.ndarray:
    """
    Generate a design matrix of parameter vectors using Latin Hypercube Sampling (LHS),
    and apply subdomain activation masking based on predefined configurations.

    This method performs the following:
      1. Samples `n_samples` points from the continuous design space using LHS.
         - The first entry of each sample is a float in [0, len(self.configs)),
           interpreted as a configuration index.
         - The remaining entries are raw amplitude values μ_i ∈ [μ_min, μ_max]
           for each subdomain.
      2. Rounds down the configuration index to an integer to select a binary mask
         from `self.configs`.
      3. Applies the mask to zero out inactive μ_i entries.

    Args:
        n_samples (int): Number of parameter vectors to generate.

    Returns:
        np.ndarray: A (n_samples × n_sub) array of masked amplitude vectors, where
        each row represents a sample with μ_i values only in the active subdomains.
    """
    dmat = super(SinMultiPeak, self).construct_design_mat(n_samples)
    return self._convert_dmat_to_mu(dmat)

  def _convert_dmat_to_mu(
    self,
    dmat: np.ndarray
  ) -> np.ndarray:
    cfg = np.floor(dmat[:,0]).astype(np.int32)
    return self.configs[cfg] * dmat[:,1:]

  def set_params(
    self,
    mu: np.ndarray
  ) -> None:
    self.mu = mu.reshape(-1)

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
    return np.abs(mu*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

  def get_init(
    self,
    mu: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    if (mu is not None):
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])
