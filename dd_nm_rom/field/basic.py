import abc
import numpy as np

from pydoe import lhs
from scipy.stats import qmc
from typing import Dict, Union
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod
import dd_nm_rom.backend as bkd


class BasicField(object):
  """
  Base class for defining initial or steady-state fields for 2D
  Burgers' equation.

  This class serves as a template for creating fields associated with
  the 2D Burgers' equation, which may include initial conditions or
  steady-state solutions. Subclasses should define specific field
  behavior and properties.

  :param mesh: Mesh object containing grid information.
  :type mesh: MESH_TYPES
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    use_qmc: bool = True,
  ) -> None:
    self.name = self.__class__.__name__
    self.mu = None
    self.mesh = mesh
    self.design_space = None
    self.bc_type = "dirichlet"
    self.use_qmc = use_qmc

  # Design space
  # ===================================
  def init_design_space(self) -> None:
    """
    Initialize the design space if it is not already initialized.
    """
    if (self.design_space is None):
      self._init_design_space()

  def _broadcast(self, value):
    """Broadcast field sampling data consistently across distributed ranks."""
    return bkd.bcast(value)

  @abc.abstractmethod
  def _init_design_space(self) -> None:
    """
    Abstract method to be implemented by subclasses to initialize
    the design space.
    """
    pass

  def sample_design_space(self) -> np.ndarray:
    """
    Sample a point from the design space.

    :return: A sample point from the design space.
    :rtype: np.ndarray
    """
    self.init_design_space()
    return self._broadcast(self.construct_design_mat(n_samples=1).reshape(-1))

  def construct_design_mat(
    self,
    n_samples: int
  ) -> np.ndarray:
    """
    Construct a design matrix using Latin Hypercube Sampling (LHS).

    :param n_samples: Number of samples to generate.
    :type n_samples: int

    :return: Design matrix with sampled points.
    :rtype: np.ndarray
    """
    if self.use_qmc:
      dmat, _ = self.construct_design_mat_qmc(n_samples)
      return self._broadcast(dmat)

    self.init_design_space()
    # Construct
    ddim = self.design_space.shape[1]
    dmat = lhs(ddim, int(n_samples))
    # Rescale
    amin, amax = self.design_space
    return self._broadcast(dmat * (amax - amin) + amin)

  def construct_design_mat_qmc(
    self,
    n_samples: int,
    design_space: Union[np.ndarray, None] = None
  ) -> np.ndarray:
    """
    Construct a design matrix using Latin Hypercube Sampling (LHS).

    :param n_samples: Number of samples to generate.
    :type n_samples: int

    :return: Design matrix with sampled points.
    :rtype: np.ndarray
    """
    if (design_space is None):
      self.init_design_space()
      design_space = self.design_space
    else:
      design_space = np.asarray(design_space)
    # Construct
    ddim = design_space.shape[1]
    engine = qmc.LatinHypercube(d=ddim)
    dmat = engine.random(n_samples)
    # Get a binary mask based on the samples
    mask = (dmat >= 0.5).astype(int)
    # Rescale
    dmat = qmc.scale(dmat, design_space[0,:], design_space[1,:])

    return dmat, mask

  def construct_design_mat_test(
    self,
    n_samples: int,
    dmat_train: Union[list, np.ndarray] = [],
    tol: float = 1e-3
  ) -> np.ndarray:
    """
    Construct a test design matrix ensuring sufficient L2 distance
    between samples.

    :param n_samples: Number of test samples to generate.
    :type n_samples: int
    :param dmat_train: Training design matrix.
    :type dmat_train: Union[list, np.ndarray], optional
    :param tol: Minimum allowable L2 distance between samples.
    :type tol: float

    :return: Test design matrix with sufficient sample spacing.
    :rtype: np.ndarray
    """
    dmat_test = []
    for _ in range(n_samples):
      sample, dist = self._compute_sample_dist(dmat_test, dmat_train)
      if (len(dist) == 0):
        dist = 1.0 + tol
      while (np.any(dist < tol)):
        sample, dist = self._compute_sample_dist(dmat_test, dmat_train)
      if (len(dmat_test) == 0):
        dmat_test = sample
      else:
        dmat_test = np.vstack([dmat_test, sample])
    return dmat_test

  def _compute_sample_dist(
    self,
    dmat_test: Union[list, np.ndarray] = [],
    dmat_train: Union[list, np.ndarray] = []
  ) -> np.ndarray:
    """
    Compute the distance of a new sample from existing
    test and training samples.

    :param dmat_test: Existing test design matrix.
    :type dmat_test: Union[list, np.ndarray], optional
    :param dmat_train: Existing training design matrix.
    :type dmat_train: Union[list, np.ndarray], optional

    :return: The new sample and the computed distances.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    dist = np.array([])
    sample = self.sample_design_space().reshape(1,-1)
    if (len(dmat_test) > 0):
      dist = np.append(dist, np.linalg.norm(dmat_test - sample, axis=-1))
    if (len(dmat_train) > 0):
      dist = np.append(dist, np.linalg.norm(dmat_train - sample, axis=-1))
    return sample, dist

  @abc.abstractmethod
  def set_params(
    self,
    mu: np.ndarray
  ) -> None:
    """
    Abstract method to set parameters for the field.

    This method should be implemented in subclasses to configure the
    parameters specific to the field.

    :param mu: Array of parameter values to be set. The structure and
               meaning of these parameters depend on the specific field
               implementation.
    :type mu: np.ndarray
    """
    pass

  # Velocity fields
  # ===================================
  @abc.abstractmethod
  def u(self, *args, **kwargs) -> np.ndarray:
    """
    Abstract method to compute the velocity field :math:$u$.

    :param args: Positional arguments for initial condition retrieval.
    :param kwargs: Keyword arguments for initial condition retrieval.

    :return: Velocity field u.
    :rtype: np.ndarray
    """
    pass

  @abc.abstractmethod
  def v(self, *args, **kwargs) -> np.ndarray:
    """
    Abstract method to compute the velocity field :math:$v$.

    :param args: Positional arguments for initial condition retrieval.
    :param kwargs: Keyword arguments for initial condition retrieval.

    :return: Velocity field u.
    :rtype: np.ndarray
    """
    pass

  @abc.abstractmethod
  def get_init(self, *args, **kwargs) -> np.ndarray:
    """
    Abstract method to get the initial conditions for the field.

    :param args: Positional arguments for initial condition retrieval.
    :param kwargs: Keyword arguments for initial condition retrieval.

    :return: Initial conditions for the field.
    :rtype: np.ndarray
    """
    pass

  @abc.abstractmethod
  def get_force(self, *args, **kwargs) -> np.ndarray:
    """
    Abstract method to get the force for the field.

    :param args: Positional arguments for force retrieval.
    :param kwargs: Keyword arguments for force retrieval.

    :return: Force for the field.
    :rtype: np.ndarray
    """
    pass

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
        funval[side][z] = lambda x, y: np.zeros_like(x)
    return funval
