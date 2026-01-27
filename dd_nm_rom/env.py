import os

from typing import Union


def _set_cpu_threads(
  nb_threads: int
) -> None:
  """
  Set the number of CPU threads for various libraries.

  This function sets environment variables to specify the number of threads 
  used by different libraries such as OpenMP, OpenBLAS, MKL, and others.

  :param nb_threads: Number of threads to set for each environment variable.
  :type nb_threads: int

  :return: None
  :rtype: None
  """
  for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  ):
    os.environ[k] = str(nb_threads)

def set(
  backend: str = "numpy",
  device: str = "cpu",
  device_idx: int = 0,
  nb_threads: int = 8,
  epsilon: Union[float, None] = 1e-10,
  floatx: str = "float64",
  seed: Union[int, None] = None
) -> None:
  """
  Configure the settings for the computational environment.

  This function must be called at the beginning of your application to set up 
  the backend, device, and other configuration options.

  :param backend: Backend library to use (e.g., "numpy").
  :type backend: str
  :param device: Device to use (e.g., "cpu" or "gpu").
  :type device: str
  :param device_idx: Index of the device (relevant for multi-GPU setups).
  :type device_idx: int
  :param nb_threads: Number of CPU threads to use.
  :type nb_threads: int
  :param epsilon: Small value to prevent division by zero.
  :type epsilon: float or None
  :param floatx: Floating-point precision (e.g., "float64").
  :type floatx: str
  :param seed: Random seed for reproducibility.
  :type seed: int or None

  :return: None
  :rtype: None
  """
  nb_threads = int(nb_threads)
  _set_cpu_threads(nb_threads)
  from . import backend as bkd
  bkd.set(
    backend=backend,
    device=device,
    device_idx=device_idx,
    nb_threads=nb_threads,
    epsilon=epsilon,
    floatx=floatx,
    seed=seed
  )
