import os
import torch
import random
import numpy as np
import scipy as sp

from typing import Any, Union


# Global
# -------------------------------------
_SEED = None
_VALID_BKD = {"numpy", "torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}

# Setting
# -------------------------------------
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
  Configure the settings for the computational backend.

  This function sets up various parameters for the backend environment, 
  including the computational backend, device, number of threads, and 
  precision settings.

  :param backend: The computational backend to use (e.g., "numpy").
  :type backend: str
  :param device: The device to use (e.g., "cpu").
  :type device: str
  :param device_idx: The index of the device to use (e.g., 0 for the 
                     first device).
  :type device_idx: int
  :param nb_threads: The number of threads to use.
  :type nb_threads: int
  :param epsilon: A small value to avoid numerical instability. If None, 
                  a default value is used.
  :type epsilon: float or None
  :param floatx: The floating-point precision to use (e.g., "float64").
  :type floatx: str
  :param seed: The seed for random number generation. If None, the seed 
               is not set.
  :type seed: int or None

  :return: None
  :rtype: None
  """
  set_backend(backend)
  set_seed(seed)
  set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

def get_backend() -> str:
  """
  Returns the current backend identifier.

  :return: The backend identifier.
  :rtype: str
  """
  return _BKD

def set_backend(
  value: str = "numpy"
) -> None:
  """
  Set the backend for the library.

  :param value: The backend to be set.
  :type value: str

  :raises ValueError: If the provided backend is not in the list of valid 
                      backends.
  """
  global _BKD
  _BKD = value
  if (value not in _VALID_BKD):
    raise ValueError(
      f"Unknown backend: '{value}'. Valid options are: {_VALID_BKD}"
    )

# Conversion
# -------------------------------------
def to_numpy(x: Any) -> np.ndarray:
  """
  Convert the input to a NumPy array.

  If the input is already a NumPy array, it is returned as-is. If the input 
  is a PyTorch tensor, it is converted to a NumPy array. For other types 
  such as `int`, `float`, `list`, or `tuple`, the input is converted to a NumPy 
  array with a `float` data type. If the input does not match any of these 
  types, it is returned unchanged.

  :param x: The input to convert to a NumPy array. Can be a NumPy array, 
            PyTorch tensor, int, float, list, or tuple.
  :type x: Any

  :return: The converted NumPy array or the original input if it cannot be 
           converted.
  :rtype: np.ndarray or Any
  """
  if (x is not None):
    if isinstance(x, np.ndarray):
      return x
    elif (torch.is_tensor(x)):
      return x.numpy(force=True)
    elif isinstance(x, (int, float, list, tuple)):
      return np.array(x, dtype=floatx("numpy"))
    else:
      return x

def to_backend(x: Any) -> Union[np.ndarray, torch.Tensor]:
  """
  Convert input to a backend-specific format.

  If the backend is set to "torch" and the input `x` is not already a 
  PyTorch tensor, it converts `x` to a PyTorch tensor. If the backend is 
  not "torch", it converts `x` to a NumPy array.

  :param x: The input to be converted.
  :type x: Any

  :return: The input converted to the appropriate format based on the 
           backend setting.
  :rtype: Union[np.ndarray, torch.Tensor]
  """
  if (x is not None):
    if (_BKD == "torch"):
      if torch.is_tensor(x):
        return x
      else:
        return torch.as_tensor(to_numpy(x), dtype=floatx("torch"))
    else:
      return to_numpy(x)

def to_sparse(
  x: Union[np.ndarray, sp.sparse.spmatrix]
) -> sp.sparse.spmatrix:
  """
  Convert the input array or sparse matrix to a Compressed Sparse Row (CSR) 
  matrix.

  If the input `x` is already a sparse matrix, it will be converted to CSR 
  format. If `x` is a dense NumPy array, it will be converted to a CSR sparse 
  matrix.

  :param x: The input array or sparse matrix to convert.
  :type x: Union[np.ndarray, sp.sparse.spmatrix]

  :return: The input converted to a CSR sparse matrix.
  :rtype: sp.sparse.spmatrix
  """
  return x.tocsr() if sp.sparse.issparse(x) else sp.sparse.csr_matrix(x)

# Device
# -------------------------------------
def device() -> str:
  """
  Returns the current device identifier.

  :return: The device identifier as a string.
  :rtype: str
  """
  return _DEVICE

def set_device(
  value: str = None,
  index: int = 0,
  nb_threads: int = 8,
) -> None:
  """
  Set the device for computations.

  This function sets the global device for PyTorch operations and configures 
  the number of threads for operations.

  :param value: The device to set (e.g., "cpu", "cuda"). If None or "cuda", 
                the function will select "cuda" if available, otherwise "cpu".
  :type value: str, optional
  :param index: The device index, default is 0.
  :type index: int, optional
  :param nb_threads: Number of threads to use, default is 8.
  :type nb_threads: int, optional

  :return: None
  :rtype: None

  :raises ValueError: If the device specified in `value` is not valid.
  """
  if ((value is None) or (value == "cuda")):
    value = "cuda" if torch.cuda.is_available() else "cpu"
  if (value not in _VALID_DEVICE):
    raise ValueError(
      f"Unknown device: '{value}'. Valid options are: {_VALID_DEVICE}"
    )
  if (value == "cuda"):
    value += f":{index}"
  global _DEVICE
  _DEVICE = value
  # Set default device
  try:
    torch.set_default_device(torch.device(_DEVICE))
    torch.set_num_interop_threads(nb_threads)
    torch.set_num_threads(nb_threads)
  except:
    pass

# Epsilon
# -------------------------------------
def machine_eps() -> float:
  """
  Returns the machine epsilon for the floating-point precision defined 
  by `_FLOATX`.

  Machine epsilon is the smallest positive number :math:`\epsilon` such that 
  :math:`1.0 + \epsilon \neq 1.0`. This function returns the machine epsilon 
  for the data type specified by the global variable `_FLOATX`.

  :return: Machine epsilon for the specified floating-point precision.
  :rtype: float

  :raises KeyError: If `_FLOATX` is not one of 'float16', 'float32', or 'float64'.
  """
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon() -> float:
  """
  Returns the current small epsilon value used for numerical stability.

  :return: A small epsilon value.
  :rtype: float
  """
  return _EPSILON

def set_epsilon(
  value: Union[float, None] = None
) -> None:
  """
  Set the global epsilon value used for numerical precision.

  If no value is provided, the function sets epsilon to the machine epsilon.

  :param value: The epsilon value to set. If None, defaults to machine epsilon.
  :type value: float or None

  :return: None
  :rtype: None
  """
  if (value is None):
    value = machine_eps()
  global _EPSILON
  _EPSILON = value

# Float
# -------------------------------------
def floatx(
  bkd: str = "torch"
) -> Union[str, type, torch.dtype]:
  """
  Returns the floating point precision type based on the backend and global
  `_FLOATX` setting.

  :param bkd: The backend to use ("torch" or "numpy"). Default is "torch".
  :type bkd: str

  :return: The floating point precision type for the specified backend.
  :rtype: Union[str, type, torch.dtype]

  :raises ValueError: If the backend is not "torch" or "numpy".
  """
  if (bkd == "torch"):
    return {
      "float16": torch.float16,
      "float32": torch.float32,
      "float64": torch.float64
    }[_FLOATX]
  elif (bkd == "numpy"):
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  else:
    return _FLOATX

def set_floatx(
  value: str
) -> None:
  """
  Set the global floating-point precision type for the library.

  This function sets the global floating-point precision type (`_FLOATX`) to 
  the specified value. If the value is not in the list of valid data types, 
  it raises a `ValueError`. Additionally, it tries to set the default floating-
  point dtype in PyTorch.

  :param value: The desired floating-point precision type.
  :type value: str

  :raises ValueError: If the provided value is not in the list of valid dtypes.

  :return: None
  :rtype: None
  """
  global _FLOATX
  _FLOATX = value
  if (value not in _VALID_DTYPE):
    raise ValueError(
      f"Unknown dtype: '{value}'. Valid options are: {_VALID_DTYPE}"
    )
  try:
    torch.set_default_dtype(floatx())
  except:
    pass

# Seed
# -------------------------------------
def seed() -> Union[int, None]:
  """
  Retrieve the current seed value.

  :return: The current seed value if set, otherwise None.
  :rtype: Union[int, None]
  """
  return _SEED

def set_seed(
  value: Union[int, None] = None
) -> None:
  """
  Set random number generator seeds for reproducibility.

  This function sets the seed for Python"s built-in random module, NumPy, 
  and PyTorch, ensuring deterministic operations. It"s essential for 
  achieving reproducible results in data processing and machine learning 
  tasks. If `value` is provided, all random generators will use the same seed.

  :param value: An integer seed for random number generators.
  :type value: int or None

  :return: None
  :rtype: None
  """
  global _SEED
  _SEED = value
  if (value is not None):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(value)
