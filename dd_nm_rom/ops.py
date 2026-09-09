import collections
import numpy as np
import scipy as sp
import torch.sparse

import torch_sla

from typing import Any, Dict, List
from dd_nm_rom import backend as bkd

def sp_diag(
  x: np.ndarray,
  format : str = "csr"
) -> sp.sparse.spmatrix:
  """
  Create a sparse diagonal matrix from a 1D NumPy array.

  :param x: A 1D NumPy array to be used as the diagonal of the matrix.
  :type x: np.ndarray

  :return: A sparse diagonal matrix with `x` as the main diagonal.
  :rtype: sp.sparse.spmatrix

  :raises ValueError: If `x` is not a 1D array.
  """
  if (x.ndim != 1):
    raise ValueError("A 1D array is needed to build a sparse diagonal matrix.")

  if isinstance(x, np.ndarray):
    return sp.sparse.spdiags(x, 0, x.size, x.size)
  else:
    # (x.size[0], x.size[1])
    if format == "csr":
      return torch_sla.SparseTensor.diag(x, device=x.device).to_csr()
    else:
      return torch_sla.SparseTensor.diag(x, device=x.device).to_torch_sparse()

def sp_diag_sla(
  x: np.ndarray,
  format : str = "csr"
) -> sp.sparse.spmatrix:
  """
  Create a sparse diagonal matrix from a 1D NumPy array.

  :param x: A 1D NumPy array to be used as the diagonal of the matrix.
  :type x: np.ndarray

  :return: A sparse diagonal matrix with `x` as the main diagonal.
  :rtype: sp.sparse.spmatrix

  :raises ValueError: If `x` is not a 1D array.
  """
  if (x.ndim != 1):
    raise ValueError("A 1D array is needed to build a sparse diagonal matrix.")

  if isinstance(x, np.ndarray):
    return sp.sparse.spdiags(x, 0, x.size, x.size)
  else:
    return torch_sla.SparseTensor.diag(x, device=x.device)


def map_nested_dict(
  obj: Any,
  fun: callable,
  **kwargs
) -> Any:
  """
  Recursively apply a function to all values in a nested dictionary.

  This function traverses a nested dictionary and applies the given
  function to each value. It supports dictionaries, lists, and tuples.

  :param obj: The nested dictionary or other container to map.
  :type obj: dict or list or tuple or Any
  :param fun: The function to apply to each value.
  :type fun: Callable[[Any], Any]

  :return: A new nested structure with the function applied to all values.
  :rtype: Any
  """
  if isinstance(obj, collections.abc.Mapping):
    return {k: map_nested_dict(v, fun, **kwargs) for (k, v) in obj.items()}
  else:
    if isinstance(obj, (list, tuple)):
      return [fun(x, **kwargs) for x in obj]
    else:
      return fun(obj, **kwargs)

def face_splitting(
  A: np.ndarray,
  B: np.ndarray
) -> np.ndarray:
  """
  Compute the face-splitting product of matrices A and B.

  This operation is useful in the following identity:
    `hadamard(Ax, By) = face_splitting(A, B) @ kron(x, y)`

  It is related to the Khatri-Rao product by:
    `face_splitting(A, B).T = khatri_rao(A.T, B.T)`

  It is equivalent to the row-wise Kronecker product of A and B.

  :param A: (m, n) array-like matrix.
  :type A: np.ndarray
  :param B: (m, p) array-like matrix.
  :type B: np.ndarray

  :return: (m, np) array-like matrix representing the face-splitting product.
  :rtype: np.ndarray
  """
  return sp.linalg.khatri_rao(A.T, B.T).T

def generate_combs(
  arrays_1d: List[np.ndarray],
  **kwargs
) -> np.ndarray:
  """
  Generate all combinations of elements from multiple 1D arrays.

  This function creates a mesh grid from the provided 1D arrays and returns
  a 2D numpy array where each row represents a combination of elements
  from the input arrays.

  :param arrays_1d: List of 1D numpy arrays to generate combinations from.
  :type arrays_1d: List[np.ndarray]

  :return: 2D numpy array with each row representing a combination of
           elements from the input arrays.
  :rtype: np.ndarray
  """

  combs = np.meshgrid(*arrays_1d, indexing="ij", **kwargs)
  return np.array(combs).T.reshape(-1,len(arrays_1d))

def compute_stats(
  x: np.ndarray
) -> Dict[str, float]:
  """
  Compute basic statistics for a numpy array.

  :param x: Input array of numeric values.
  :type x: np.ndarray

  :return: A dictionary containing the mean and standard deviation of
           the input array.
  :rtype: Dict[str, float]

  The dictionary returned has the following keys:
    - "mean": The arithmetic mean of the array.
    - "std": The standard deviation of the array.
  """
  return {"mean": float(np.mean(x)), "std": float(np.std(x))}
