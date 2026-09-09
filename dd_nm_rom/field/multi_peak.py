import numpy as np

from typing import List, Union
from dd_nm_rom import ops
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

from .basic import BasicField


class MultiPeak(BasicField):

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MeshDD,
    mu_lim: List[float] = [0.9, 1.1],
    forced_config: Union[List[int], np.ndarray, None] = None,
    bc_type: str = "neumann",
    use_qmc: bool = True
  ) -> None:
    super(MultiPeak, self).__init__(mesh)
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
    """
    2D array of shape (2, n_params) defining the lower and upper bounds for sampling input parameters.

    - The first parameter is a configuration index in [0, len(self.configs)), which selects a binary activation pattern for subdomains.
    - The remaining parameters are amplitude values mu_i for each subdomain, sampled uniformly from self.mu_lim = [mu_min, mu_max].
    - Only subdomains activated in the selected configuration will use their mu_i value; others are set to zero.

    The structure is:
        design_space[0, :] -> lower bounds
        design_space[1, :] -> upper bounds
    """
    if self.use_qmc:
      # QMC samples each amplitude directly.  The activation mask is derived
      # from the unit-cube sample, so no configuration table is needed.
      self.design_space = [self.mu_lim] * (2*self.mesh.n_sub)
    else:
      # Define possible combinations (legacy path).
      self.configs = ops.generate_combs([np.arange(2)]*self.mesh.n_sub)[1:]
      if (self.forced_config is not None):
        self.configs += self.forced_config.reshape(1,-1)
        self.configs = self.configs.astype(bool).astype(int)
        self.configs = np.unique(self.configs, axis=0)
      self.design_space = [[0,len(self.configs)]] + [self.mu_lim]* (2*self.mesh.n_sub)
    self.design_space = np.array(self.design_space).T

  def sample_design_space(self) -> np.ndarray:
    s = 0.0
    while (s == 0.0):
      config = np.random.binomial(1, p=0.5, size=self.mesh.n_sub)
      if (self.forced_config is not None):
        config += self.forced_config
        config = config.astype(bool).astype(int)
      s = np.sum(config)
    mu_u = config * np.random.uniform(*self.mu_lim, size=self.mesh.n_sub)
    mu_v = config * np.random.uniform(*self.mu_lim, size=self.mesh.n_sub)

    return self._broadcast(np.concatenate([mu_u, mu_v]))

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
    if self.use_qmc:
      dmat, mask = super(MultiPeak, self).construct_design_mat_qmc(n_samples)
      return self._broadcast(self._convert_dmat_to_mu(dmat, mask))
    dmat = super(MultiPeak, self).construct_design_mat(n_samples)
    return self._broadcast(self._convert_dmat_to_mu(dmat))

  def _convert_dmat_to_mu(
    self,
    dmat: np.ndarray,
    mask: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    if self.use_qmc:
      if (self.forced_config is not None):
        forced = self.forced_config.reshape(1, -1)
        mask = np.maximum(mask, np.tile(forced, (mask.shape[0], 2)))
      return mask * dmat

    cfg = np.floor(dmat[:,0]).astype(np.int32)

    mu_u_raw = dmat[:, 1 : 1 + self.mesh.n_sub]
    mu_v_raw = dmat[:, 1 + self.mesh.n_sub : 1 + 2 * self.mesh.n_sub]

    mask = self.configs[cfg]  # shape: (n_samples, n_sub)

    mu_u = mask * mu_u_raw
    mu_v = mask * mu_v_raw

    return np.hstack([mu_u, mu_v])


  def set_params(self, mu: np.ndarray) -> None:
    mu = mu.reshape(-1)
    assert mu.shape[0] == 2 * self.mesh.n_sub
    self.mu_u = mu[:self.mesh.n_sub]
    self.mu_v = mu[self.mesh.n_sub:]

  # Velocity fields
  # ===================================
  def u(self) -> np.ndarray:
    return self.generate_field(self.mu_u, field="u")


  def v(self) -> np.ndarray:
    return self.generate_field(self.mu_v, field="v")

  def generate_field(self, mu_vec: np.ndarray, field: str = "u") -> np.ndarray:
    f = np.zeros(self.mesh.nxy)
    x, y = self.mesh.nodes_val.T
    for i, mu_i in enumerate(mu_vec):
        ind = self.mesh.res_nodes[i]
        f[ind] = self._phi(x[ind], y[ind], mu_i, field)
    return f.reshape(self.mesh.n["y"], self.mesh.n["x"])

  def _phi(self, x: np.ndarray, y: np.ndarray, mu: float, field: str = "u") -> np.ndarray:
    if field == "u":
#       return np.abs(mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
#       return (mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
        return (mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
    elif field == "v":
#       return np.abs(mu * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y))
#       return (mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
        return (mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
#       return np.abs(mu * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
    else:
        raise ValueError(f"Unknown field type: {field}")

  def get_init(
    self,
    mu: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    if (mu is not None):
      self.set_params(mu)
    return np.concatenate([self.u().reshape(-1), self.v().reshape(-1)])
