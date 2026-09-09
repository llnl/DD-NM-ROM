import os
import socket
from mpi4py import MPI
import torch
import torch.distributed as dist
import torch.cuda.comm as comm
from torch.distributed.tensor import DTensor, Shard, Replicate
import random
import numpy as np
import scipy as sp
import logging
import tracemalloc

import torch_sla

from typing import Any, List, Optional, Tuple, Union

from dd_nm_rom.utils import parallel_print
import dd_nm_rom.config as cfg

logger = logging.getLogger(__name__)


# Global
# -------------------------------------
_SEED = None
_VALID_BKD = {"numpy", "torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}
_COMM = None
_RANK = None
_NRANKS = None
_DMESH = None
_DEVICE_PER_RANK = 4

_USE_ZEROFILL = cfg.get_config_val("DDNMROM_MPI_BUFFER_ZEROFILL")
_DTENSOR_CHECKS = cfg.get_config_val("DDNMROM_DTENSOR_CHECKS")

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
  global _DEVICE_PER_RANK
  _DEVICE_PER_RANK = cfg.update_from_env("DDNMROM_DEVICE_PER_NODE", _DEVICE_PER_RANK)

  set_backend(backend)
  set_device(device, device_idx, nb_threads)

  if is_torch_backend():
    init_distributed(device)
    # TODO: some schedulers handle binding automatically, so device_count will always=1
    if (_NRANKS > 1 and torch.accelerator.device_count() > 1):
      # TODO: fix this to work for multiple nodes!
      #device_idx = _RANK % _NRANKS
      device_idx = 0
      logger.info("BACKEND: RANK %s reassigning device_idx to %s", _RANK, device_idx)

    #if _NRANKS > 1:
    #  device_idx = _RANK
    #torch.set_deterministic_debug_mode(1)
  # else:
  #   #global _COMM, _RANK, _NRANKS
  #   MPI.Init()
  #   _COMM = MPI.COMM_WORLD
  #   _RANK = 0
  #   _NRANKS = 1

  set_seed(seed)
  #set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

  if root():
    cfg.print_config_env()
    logger.debug("BACKEND ZEROFILL {} DTENSOR CHECK {}".format(_USE_ZEROFILL, _DTENSOR_CHECKS))
  barrier()

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
        return torch.as_tensor(to_numpy(x), dtype=floatx("torch"), device=device())
    else:
      return to_numpy(x)

def to_sparse(
  x: Union[np.ndarray, sp.sparse.spmatrix],
  format: str = "csr"
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
  if format == "csr":
    return x.tocsr() if sp.sparse.issparse(x) else sp.sparse.csr_matrix(x)
  else:
    return x.tocoo() if sp.sparse.issparse(x) else sp.sparse.coo_matrix(x)


def to_sp_backend(
  x: Optional[sp.sparse.spmatrix]
) -> Optional[Union[sp.sparse.spmatrix, torch.Tensor]]:
    """
    Convert a SciPy sparse matrix to the active backend's CSR representation.

    :param x: The sparse matrix to convert, or None.
    :type x: sp.sparse.spmatrix or None

    :return: A PyTorch CSR tensor for the torch backend, the original SciPy
             matrix for the NumPy backend, or None.
    :rtype: sp.sparse.spmatrix or torch.Tensor or None
    """
    if (x is not None):
        if (_BKD == "torch"):
            if torch.is_tensor(x):
                return x.to_sparse_csr().to(device())
            else:
                return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, x.shape, device=device())

        else:
            return x


def to_sp_coo_backend(x: sp.sparse.spmatrix) -> torch.Tensor:
    """
    Converts a scipy sparse matrix in COO to torch sparse COO
    This routine constructs the torch tensor using the direct data pointers from scipy,
    avoiding additional memory copies and object creation overhead

    If backend is not torch, then the original matrix is returned
    """
    if (x is not None):
        if (_BKD == "torch"):
            if torch.is_tensor(x):
                return x.to_sparse_coo().to(device())
            else:
                row = x.row
                col = x.col
                xcoo = torch.sparse_coo_tensor(torch.tensor(np.vstack((row, col))), x.data, size=x.shape, device=device())
                xcoo = xcoo.coalesce()
                return xcoo
        else:
            return x


def torch_csr_to_scipy(
  x: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[sp.sparse.spmatrix, List[sp.sparse.spmatrix]]:
    """
    Convert PyTorch CSR tensor(s) to SciPy CSR matrix/matrices.

    :param x: A PyTorch CSR tensor or a list of PyTorch CSR tensors.
    :type x: torch.Tensor or list[torch.Tensor]

    :return: The corresponding SciPy CSR matrix or list of matrices.
    :rtype: sp.sparse.spmatrix or list[sp.sparse.spmatrix]
    """
    if torch.is_tensor(x):
        assert x.layout == torch.sparse_csr
        return sp.sparse.csr_matrix((x.values().cpu(), x.col_indices().cpu(), x.crow_indices().cpu()), shape=(x.shape[0], x.shape[1]))
    elif isinstance(x, List):
        for i in range(len(x)):
            assert x[i].layout == torch.sparse_csr
            x[i] = sp.sparse.csr_matrix((x[i].values().cpu(), x[i].col_indices().cpu(), x[i].crow_indices().cpu()), shape=(x[i].shape[0], x[i].shape[1]))
        return x
    else:
        return x


def torch_coo_to_scipy(
  x: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[sp.sparse.spmatrix, List[sp.sparse.spmatrix]]:
    """
    Convert PyTorch COO tensor(s) to SciPy COO matrix/matrices.

    :param x: A PyTorch COO tensor or a list of PyTorch COO tensors.
    :type x: torch.Tensor or list[torch.Tensor]

    :return: The corresponding SciPy COO matrix or list of matrices.
    :rtype: sp.sparse.spmatrix or list[sp.sparse.spmatrix]
    """
    if torch.is_tensor(x):
        assert x.layout == torch.sparse_coo
        x = x.coalesce()
        inds = x._indices().cpu()
        return sp.sparse.coo_matrix((x._values().cpu(), (inds[0], inds[1])), shape=(x.shape[0], x.shape[1]))
    elif isinstance(x, List):
        for i in range(len(x)):
            assert x[i].layout == torch.sparse_coo
            x[i] = x[i].coalesce()
            inds = x[i]._indices().cpu()
            x[i] = sp.sparse.coo_matrix((x[i]._values().cpu(), (inds[0], inds[1])), shape=(x[i].shape[0], x[i].shape[1]))
        return x
    else:
        return x


def torch_hstack(
  x: Union[List[torch.Tensor], torch.Tensor],
  format: str = "csr"
) -> torch.Tensor:
    """
    Helper function around torch.hstack for CSR tensors
    this converts x to COO, since hstack does not work with CSR tensors
    after the hstack, returns x back in a CSR tensor

    :param x: A tensor or list of tensors to stack.
    :type x: torch.Tensor or list[torch.Tensor]
    :param format: Sparse format for the result; ``"csr"`` returns CSR.
    :type format: str

    :return: The horizontally stacked tensor.
    :rtype: torch.Tensor
    """
    if isinstance(x, List):
        for i in range(len(x)):
            x[i] = x[i].to_sparse_coo()
        x = torch.hstack(x)
        if format == "csr":
          return x.to_sparse_csr()
        else:
          return x
    else:
        if format == "csr":
          return torch.hstack(x.to_sparse_coo()).to_sparse_csr()
        else:
          return torch.hstack(x.to_sparse_coo())



def torch_bmat(x: List[List[Optional[torch.Tensor]]], format: str = "csr") -> torch.Tensor:
  """
  Assemble a block sparse matrix from PyTorch tensor blocks.

  :param x: Nested block rows containing tensors or None entries.
  :type x: list[list[torch.Tensor or None]]
  :param format: Sparse format for the result; ``"csr"`` returns CSR.
  :type format: str

  :return: The assembled sparse tensor.
  :rtype: torch.Tensor
  """
  if format == "csr":
    for i in range(len(x)):
      for j in range(len(x[i])):
        if x[i][j] is None:
          continue
        x[i][j] = torch_csr_to_scipy(x[i][j].to_sparse_csr())
    x = sp.sparse.bmat(x, format="csr")
    return to_sp_backend(x)
  else:
    for i in range(len(x)):
      for j in range(len(x[i])):
        if x[i][j] is None:
          continue
        x[i][j] = torch_coo_to_scipy(x[i][j].to_sparse_coo())
    x = sp.sparse.bmat(x, format="coo")
    return to_sp_coo_backend(x)


def torch_bmat_new(
  x: List[List[Optional[torch.Tensor]]],
  format: str = "csr"
) -> torch.Tensor:
  """
  Assemble a block sparse matrix directly from PyTorch tensor blocks.

  :param x: Nested block rows containing tensors or None entries.
  :type x: list[list[torch.Tensor or None]]
  :param format: Sparse format for the result; ``"csr"`` returns CSR.
  :type format: str

  :return: The assembled sparse tensor.
  :rtype: torch.Tensor
  """
  m, n = len(x), len(x[0])

  rows = [i for i in range(m) for j in range(len(x[i])) if x[i][j] is not None]
  cols = [j for i in range(m) for j in range(len(x[i])) if x[i][j] is not None]

  brow_lengths = torch.zeros(m, dtype=int)
  bcol_lengths = torch.zeros(n, dtype=int)
  nnz = 0
  for i in range(m):
    for j in range(n):
      if x[i][j] is None:
        continue

      # TODO: fix - this always converts to coo
      if x[i][j].is_sparse:
        x[i][j] = x[i][j].coalesce()
      else:
        x[i][j] = x[i][j].to_sparse_coo()

      brow_lengths[i] = x[i][j].shape[0]
      bcol_lengths[j] = x[i][j].shape[1]
      nnz += x[i][j].values().numel()

  row_offsets = torch.zeros(m + 1, dtype=int)
  col_offsets = torch.zeros(n + 1, dtype=int)
  row_offsets[1:] = torch.cumsum(brow_lengths, 0)
  col_offsets[1:] = torch.cumsum(bcol_lengths, 0)

  data = torch.empty(nnz, dtype=x[0][0].dtype)
  row = torch.empty(nnz, dtype=torch.int64)
  col = torch.empty(nnz, dtype=torch.int64)

  nnz = 0
  for i, j in zip(rows, cols):
    block_nnz = x[i][j].values().numel()
    idx = slice(nnz, nnz + block_nnz)
    data[idx] = x[i][j].values()
    inds = x[i][j].indices()
    row[idx] = row_offsets[i] + inds[0]
    col[idx] = col_offsets[j] + inds[1]
    nnz += block_nnz

  if format == "csr":
    return torch.sparse_coo_tensor(torch.vstack((row, col)), data, size=(row_offsets[-1], col_offsets[-1]), device=device()).to_sparse_csr()
  else:
    return torch.sparse_coo_tensor(torch.vstack((row, col)), data, size=(row_offsets[-1], col_offsets[-1]), device=device())


def speye(
  n: Union[int, Tuple[int, int]],
  format: str = "coo"
) -> torch.Tensor:
  """
  Create a sparse identity matrix using the active PyTorch dtype and device.

  :param n: Matrix dimension or two-dimensional output shape.
  :type n: int or tuple[int, int]
  :param format: Sparse format for the result; ``"coo"`` returns COO.
  :type format: str

  :return: A sparse identity tensor.
  :rtype: torch.Tensor
  """
  if isinstance(n, tuple):
    if len(n) == 1:
      n = n * 2
    shape = n
    diag = np.min(n)
  else:
    shape = (n, n)
    diag = n
  global _FLOATX
  data = torch.ones(diag, layout=torch.strided, dtype=_FLOATX, device=device())
  if format=="coo":
    indices = torch.arange(0, diag, dtype=int, device=device())
    return torch.sparse_coo_tensor(torch.vstack((indices, indices)), data, size=shape, device=device())
  else:
    col_indices = torch.arange(0, shape[0], dtype=int, device=device())
    crow_indices = torch.arange(0, shape[0]+1, dtype=int, device=device())
    return torch.sparse_csr_tensor(crow_indices, col_indices, data, size=shape, device=device())


def _start_mem_trace(reset: bool = False) -> Tuple[int, int]:
  """
  Starts a tracemalloc session if not already in one, returns current values
  Resets peak usage if reset=True
  """
  if reset:
    tracemalloc.reset_peak()
  if not tracemalloc.is_tracing():
    tracemalloc.start()
  return tracemalloc.get_traced_memory()


def _end_mem_trace(print_stats: bool = True) -> Tuple[int, int]:
  """
  Stop tracemalloc tracing and return the current and peak memory usage.

  :param print_stats: Whether to print the measured memory usage.
  :type print_stats: bool

  :return: Current and peak traced memory usage in bytes.
  :rtype: tuple[int, int]
  """
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  # convert to MB
  #current = current / 10**6
  #peak = peak / 10**6
  if print_stats:
    print(f"Current memory usage: {current / 10**6:.6f} MB")
    print(f"Peak memory usage:    {peak / 10**6:.6f} MB")
  return (current, peak)


def _get_mem_trace_stats(
  start: Tuple[int, int],
  end: Tuple[int, int],
  print_stats: bool = False
) -> Tuple[float, float]:
  """
  Computes and optionally reports difference in (current, peak) usage from start, end
  Returned values are converted to MB
  """
  diff_current, diff_peak = ((e - s) / 10**6 for e, s in zip(end, start))
  if print_stats:
    print(f"Change in memory usage over interval: {diff_current:.6f} MB   (start: {start[0]/10**6:.6f} MB, end: {end[0]/10**6:.6f} MB)")
    print(f"Change in peak   usage over interval: {diff_peak:.6f} MB      (start: {start[1]/10**6:.6f} MB, end: {end[1]/10**6:.6f} MB)")
  return (diff_current, diff_peak)


def same_ptr(x: torch.Tensor, y: torch.Tensor) -> bool:
  """
  Checks if x and y share the same underyling data pointer
  If x and y are sparse, then this compares the pointers for the values and indices also.
  Returns true if x and y share the same memory, False otherwise
  """
  if x.layout != y.layout:
    # automatically false if one is dense and other is sparse
    return False
  if (x.is_sparse and y.is_sparse) or (x.is_sparse_csr and y.is_sparse_csr):
    # todo - for COO tensors, this should probably check uncoalesced ._values() instead..
    return x.values().data_ptr() == y.values().data_ptr()
  return x.data_ptr() == y.data_ptr()


def get_tensor_ptr(
  x: torch.Tensor
) -> Union[int, Tuple[int, int], Tuple[int, int, int]]:
  """
  Helper to return underlying pointer for tensor
  If sparse tensor, returns tuple of values, col/crow indices
  """
  if x.layout == torch.strided:
    return x.data_ptr()
  elif x.layout == torch.sparse_csr:
    return (x.values().data_ptr(), x.col_indices().data_ptr(), x.crow_indices().data_ptr())
  elif x.layout == torch.sparse_coo:
    return (x.values().data_ptr(), x.indices().data_ptr())


def tensor_eq(
  x: torch.Tensor,
  y: torch.Tensor,
  check_indices: bool = False
) -> Union[bool, torch.Tensor]:
  """
  Helper function to compare equality on two tensors
  Supports regular tensors as well as sparse COO and sparse CSR tensors (unlike torch)
  :param check_indices: (sparse tensors only) If true, compares the sparse tensor row and col indices for equality,
    otherwise only compare the sparse element values.
  """
  if x.layout != y.layout:
    return False
  if x.numel() != y.numel():
    return False
  if x.layout == torch.strided:
    # non-sparse tensor comparison: use regular eq op:
    return torch.all(x == y)
  elif x.layout == torch.sparse_csr:
    is_equal = torch.all(x.values() == y.values())
    if check_indices and is_equal:
      # compare indices, skip if values were already non-equal
      is_equal &= torch.all(x.col_indices() == y.col_indices())
      is_equal &= torch.all(x.crow_indices() == y.crow_indices())
    return is_equal
  elif x.layout == torch.sparse_coo:
    is_equal = torch.all(x._values() == y._values())
    if check_indices and is_equal:
      # compare indices, skip if values were already non-equal
      is_equal &= torch.all(x.indices() == y.indices())
    return is_equal


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
  #_DEVICE = value
  _DEVICE = torch.device(value)
  # Set the device before configuring optional thread counts.  PyTorch only
  # permits changing the inter-op thread count before parallel work starts;
  # repeated backend fixture setup after MPI initialization must therefore
  # not turn an otherwise valid device selection into a failure.
  torch.set_default_device(_DEVICE)
  try:
    torch.set_num_interop_threads(nb_threads)
  except RuntimeError as exc:
    if "cannot set number of interop threads" not in str(exc):
      raise
    logger.debug("Inter-op thread count is already fixed: %s", exc)
  try:
    torch.set_num_threads(nb_threads)
  except RuntimeError as exc:
    if "cannot set number of intraop threads" not in str(exc):
      raise
    logger.debug("Intra-op thread count is already fixed: %s", exc)

# Epsilon
# -------------------------------------
def machine_eps() -> float:
  r"""
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
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(value)
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(value)


def is_torch_backend() -> bool:
  """
  Determine whether the active computational backend is PyTorch.

  :return: True when the configured backend is ``"torch"``.
  :rtype: bool
  """
  return _BKD == "torch"


def init_distributed(backend_type: str = "cuda") -> None:
  """
  Initialize MPI-backed PyTorch distributed state when multiple ranks exist.

  :param backend_type: Device backend used to select ``gloo`` or ``nccl``.
  :type backend_type: str

  :return: None
  :rtype: None
  """
  is_cuda = backend_type != "cpu"
  device_count = torch.cuda.device_count() if is_cuda else 0
  logger.debug("BEFORE INIT DIST: TORCH device counts = %s", device_count)

  global _COMM, _RANK, _NRANKS
  if dist.is_available() and dist.is_initialized():
    return

  _COMM = MPI.COMM_WORLD
  _RANK = _COMM.Get_rank()
  _NRANKS = _COMM.Get_size()

  logger.debug("INIT DISTRIBUTED: rank = %s, num ranks = %s", _RANK, _NRANKS)

  if _NRANKS > 1:
    # Broadcast root hostname to all other ranks
    root_addr = None
    if _RANK == 0:
      root_addr = os.environ.get("MASTER_ADDR", socket.gethostname())
    root_addr = _COMM.bcast(root_addr, root=0)

    os.environ["MASTER_ADDR"] = root_addr
    os.environ.setdefault("MASTER_PORT", "23457")

    #print(" -- FLUX_JOB_SIZE = {} FLUX_TASK_RANK = {}".format(int(os.environ.get('FLUX_JOB_SIZE')), int(os.environ.get('FLUX_TASK_RANK'))))

    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Use FLUX vars if available, else fallback to MPI
    world_size = int(os.environ.get('FLUX_JOB_SIZE', _NRANKS))
    rank_env = int(os.environ.get('FLUX_TASK_RANK', _RANK))
    assert rank_env == _RANK

    backend = "gloo" if backend_type == "cpu" else "nccl"

    logger.info(
      "Creating torch process group: world size = %s rank = %s, backend type = '%s'",
      world_size, rank_env, backend,
    )
    available_devices = torch.accelerator.device_count() if is_cuda else 1
    logger.info("RANK %s number of available devices = %s", _RANK, available_devices)
    #devid = _RANK % _DEVICE_PER_RANK
    # if torch.accelerator.device_count() > 1:
    #   devid = _DEVICE_PER_RANK // torch.accelerator.device_count()
    #   devid = devid % _NRANKS
    #     devid = _RANK * torch.accelerator.device_count()
    #   devid = devid % _DEVICE_PER_RANK

    # for 2 ranks:
    # devid = _RANK
    # print(" RANK {} DEVICE ID = {}".format(_RANK, devid))
    # global _DEVICE
    # _DEVICE = "cuda:{}".format(devid)
    # torch.set_default_device(torch.device("cuda:{}".format(devid)))

    # Set up the process group with an explicit local CUDA device.  Without
    # this, NCCL guesses from the global rank and emits a warning (and can
    # select the wrong device on multi-node jobs).
    process_group_args = {
      "backend": backend,
      "init_method": "env://",
      "rank": rank_env,
      "world_size": world_size,
    }
    if backend == "nccl":
      torch.cuda.set_device(_DEVICE)
      process_group_args["device_id"] = _DEVICE
    dist.init_process_group(**process_group_args)
    
    init_device_mesh(backend_type, use_2d=False)
  else:
    logger.info("Serial mode; no distributed")

  if is_cuda:
    device_name = torch.cuda.get_device_name()
    device_properties = torch.cuda.get_device_properties()
  else:
    device_name = str(_DEVICE)
    device_properties = "CPU"
  logger.info("RANK %s: Initialized on device '%s'", _RANK, device_name)
  logger.info("RANK %s: Device properties: %s", _RANK, device_properties)


def init_device_mesh(backend_type: str, use_2d: bool = True) -> None:
  """
  Initialize the device mesh used for distributed tensors.

  :param backend_type: Device backend passed to PyTorch's mesh initializer.
  :type backend_type: str
  :param use_2d: Whether to use a two-dimensional mesh when the rank layout
                 permits it.
  :type use_2d: bool

  :return: None
  :rtype: None
  """
  if not _BKD == "torch" or not distributed():
    return

  # fallback to 1D mesh if grid is not even
  if _NRANKS % _DEVICE_PER_RANK != 0:
    use_2d = False

  if use_2d:
    mesh = (_NRANKS // _DEVICE_PER_RANK, _DEVICE_PER_RANK)
    dims = ("GLOBAL", "LOCAL")
  else:
    mesh = (_NRANKS,)
    dims = ("GLOBAL",)

  logger.info("RANK %s: Initializing device mesh %s (%s)", _RANK, mesh, dims)
  global _DMESH
  _DMESH = dist.init_device_mesh(backend_type, mesh_shape=mesh)#, mesh_dim_names=dims)
  logger.info("RANK %s: Initialized device mesh: %s", _RANK, _DMESH)


def finalize_distributed() -> None:
  """
  Destroy the PyTorch process group and clear distributed global state.

  :return: None
  :rtype: None
  """
  global _COMM, _RANK, _NRANKS, _DMESH
  if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
  _COMM = None
  _RANK = None
  _NRANKS = None
  _DMESH = None


def distributed() -> bool:
  """
  Determine whether distributed execution is active.

  :return: True when more than one rank has initialized distributed state.
  :rtype: bool
  """
  if _NRANKS is None or _NRANKS <= 1:
    return False
  if _BKD == "torch":
    return dist.is_available() and dist.is_initialized()
  return True

def get_rank() -> int:
  """
  Return the current rank, defaulting to zero in serial execution.

  :return: The current process rank.
  :rtype: int
  """
  if _RANK is not None:
    return _RANK
  else:
    return 0

def get_nranks() -> int:
  """
  Return the number of ranks, defaulting to one in serial execution.

  :return: The number of participating processes.
  :rtype: int
  """
  if _NRANKS is not None:
    return _NRANKS
  else:
    return 1

def barrier() -> None:
  """
  Synchronize all distributed ranks when distributed execution is active.

  :return: None
  :rtype: None
  """
  if not distributed():
    return

  if _DEVICE.type == "cuda":
    torch.accelerator.synchronize()
  _COMM.Barrier()
  dist.barrier()


def bcast(value: Any, root: int = 0) -> Any:
  """
  Broadcast a value from the given rank to all other ranks.

  In serial execution, the value is returned unchanged.

  :param value: Value to broadcast.
  :type value: Any
  :param root: Rank that provides the value.
  :type root: int
  :return: The broadcast value.
  :rtype: Any
  """
  if not distributed():
    return value
  return _COMM.bcast(value, root=root)


def root() -> bool:
  """
  Determine whether the current process is the root rank.

  :return: True in serial execution or on rank zero.
  :rtype: bool
  """
  if not distributed():
    return True
  return _RANK == 0


# Parallel helpers:
def get_local_sizes(
  data: Union[np.ndarray, torch.Tensor],
  dim: int = 0,
  root: int = 0
) -> Optional[List[int]]:
  """
  Returns a list with the size of data (at the given dim) for each rank
  Returned list is only defined on rank root, other ranks are undefined

  :param data: Local array or tensor whose dimension size is collected.
  :type data: np.ndarray or torch.Tensor
  :param dim: Dimension whose local size is collected.
  :type dim: int
  :param root: Rank that receives the gathered sizes.
  :type root: int

  :return: Local sizes on the root rank, or None on other ranks.
  :rtype: list[int] or None
  """
  lsize = data.shape[dim]
  if not distributed():
    return [lsize]

  rank_sizes = _COMM.gather(lsize, root=root)
  barrier()

  return rank_sizes


def get_local_sizes_all(
  data: Union[np.ndarray, torch.Tensor],
  dim: int = 0
) -> List[int]:
  """
  All-node version of get_local_sizes (size of data returned for all ranks)

  :param data: Local array or tensor whose dimension size is collected.
  :type data: np.ndarray or torch.Tensor
  :param dim: Dimension whose local size is collected.
  :type dim: int

  :return: Local sizes from every rank.
  :rtype: list[int]
  """
  lsize = data.shape[dim]
  if not distributed():
    return [lsize]

  rank_sizes = _COMM.allgather(lsize)
  barrier()

  return rank_sizes

def get_local_shape(
  data: Union[np.ndarray, torch.Tensor],
  root: int = 0
) -> Optional[List[Tuple[int, ...]]]:
  """
  Returns a list with the size of data (at the given dim) for each rank
  Returned list is only defined on rank root, other ranks are undefined

  :param data: Local array or tensor whose shape is collected.
  :type data: np.ndarray or torch.Tensor
  :param root: Rank that receives the gathered shapes.
  :type root: int

  :return: Local shapes on the root rank, or None on other ranks.
  :rtype: list[tuple[int, ...]] or None
  """
  lsize = data.shape
  if not distributed():
    return [lsize]

  rank_sizes = _COMM.gather(lsize, root=root)
  barrier()

  return rank_sizes


def _ensure_contiguous(
  x: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """Return tensor inputs with contiguous storage when required by MPI."""
  if isinstance(x, list):
    return [value if value.is_contiguous() else value.contiguous() for value in x]
  return x if x.is_contiguous() else x.contiguous()


def gather_tensor(
  x: torch.Tensor,
  sizes: Optional[List[int]] = None,
  dim: int = 0,
  root: int = 0,
  as_list: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Gathers the local tensor x from each rank onto root rank.

  If sizes is not provided, shape of each input x is determined. Otherwise,
  sizes is a list of sizes over all ranks.

  Rank local tensors x are assumed to be split over specified dim, so all other
  dimensions are assumed the same size on all ranks.

  :param x: Local tensor to gather.
  :type x: torch.Tensor
  :param sizes: Optional local sizes for every rank.
  :type sizes: list[int] or None
  :param dim: Dimension along which to concatenate gathered tensors.
  :type dim: int
  :param root: Rank that receives gathered data.
  :type root: int
  :param as_list: Return individual gathered tensors instead of concatenating.
  :type as_list: bool

  :return: Gathered data on root, or the local tensor on other ranks.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if not distributed():
    return x

  x_sparse = False
  if x.is_sparse or x.is_sparse_csr:
    x_sparse = True
    assert x.layout == torch.sparse_csr
    x = x.to_dense()
  x = _ensure_contiguous(x)

  data = None

  rank_sizes = sizes
  calc_sizes = False
  if rank_sizes is None and _RANK == root:
    # if rank sizes is undefined on root rank, then we need to determine the sizes
    calc_sizes = True
  calc_sizes = _COMM.bcast(calc_sizes, root=root)
  barrier()

  if calc_sizes:
    rank_sizes = get_local_sizes(x, dim, root)

  if _RANK == root:
    data = []
    for rank in range(_NRANKS):
      # check each rank has same size tensor (torch gather requires this)
      assert rank_sizes[rank] == rank_sizes[0]
      # NOTE: assumes data is 2D tensor, where the second dim is always constant across ranks (e.g domain size)
      data.append(torch.zeros_like(x, device=device()))

  dist.gather(x, data, root)

  barrier()

  if _RANK == root and not as_list:
    # Root rank stacks all gathered tensors
    data = torch.cat(data, dim=dim)

  barrier()
  
  if x_sparse:
    # convert dense back to sparse
    if _RANK == root:
      if as_list:
        for rank in range(_NRANKS):
          data[rank] = data[rank].to_sparse_csr()
      else:
        data = data.to_sparse_csr()
    else:
      x = x.to_sparse_csr()

  # Root rank returns combined tensor, all others just return input tensor
  if _RANK == root:
    return data
  else:
    return x
  

def gatherv_tensor(
  x: Union[torch.Tensor, List[torch.Tensor]],
  sizes: Optional[List[int]] = None,
  offsets: Optional[List[int]] = None,
  dim: int = 0,
  root: int = 0,
  as_list: bool = False,
  coalesce: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Gathers the local tensor x from each rank onto root rank.

  If sizes is not provided, shape of each input x is determined. Otherwise,
  sizes is a list of sizes over all ranks.

  Rank local tensors x are assumed to be split over specified dim, so all other
  dimensions are assumed the same size on all ranks.

  :param x: Local tensor or list of tensors to gather.
  :type x: torch.Tensor or list[torch.Tensor]
  :param sizes: Optional local sizes for every rank.
  :type sizes: list[int] or None
  :param offsets: Optional receive-buffer offsets for every rank.
  :type offsets: list[int] or None
  :param dim: Dimension along which to concatenate gathered tensors.
  :type dim: int
  :param root: Rank that receives gathered data.
  :type root: int
  :param as_list: Return individual gathered tensors instead of concatenating.
  :type as_list: bool
  :param coalesce: Coalesce sparse COO input before gathering.
  :type coalesce: bool

  :return: Gathered data on root, or the local tensor on other ranks.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if not distributed():
    return x

  x_sparse = False
  if isinstance(x, List):
    for i in range(len(x)):
      if x[i].is_sparse_csr:
        x_sparse = True
        x[i] = x[i].to_sparse_coo()
        

    sub_sizes = [s.shape[dim] for s in x]
    tmp = torch.cat(x, dim=dim)#.contiguous()
    res = gatherv_tensor(tmp, sizes, offsets, dim, root, as_list)
    if as_list:
      sub_sizes = _COMM.gather(sub_sizes, root=root)
    if _RANK == root:
      if as_list:
        for r in range(len(res)):
          res[r] = list(res[r].split(sub_sizes[r], dim))
    if x_sparse:
      # convert dense back to sparse
      if _RANK == root:
        if as_list:
          for rank in range(_NRANKS):
            res[rank] = res[rank].to_sparse_csr()
        else:
          res = res.to_sparse_csr()
      else:
        res = res.to_sparse_csr()
    return res

  if x.is_sparse:
    return gatherv_spcoo(x, sizes, offsets, dim, root, as_list, coalesce=coalesce)

  if x.is_sparse_csr:
    return gatherv_spcsr(x, sizes, offsets, dim, root, as_list)

  data = None
  data_ptr = None

  rank_sizes = sizes
  rank_offsets = offsets
  calc_sizes = False
  #if (rank_sizes is None or rank_offsets is None) and _RANK == root:
  if rank_sizes is None and _RANK == root:
    # if rank sizes is undefined on root rank, then we need to determine the sizes
    calc_sizes = True
  calc_sizes = _COMM.bcast(calc_sizes, root=root)
  barrier()

  if calc_sizes:
    rank_sizes = get_local_sizes(x, dim, root)

  if _RANK == root:
    total_size = list(x.shape)
    total_size[dim] = np.sum(rank_sizes, dtype=int)
    flat_total_size = np.prod(total_size, dtype=int)

    if as_list:
      rank_sizes_dim = rank_sizes

    # scale rank_sizes by size of all other dimensions
    other_dims = x.shape[dim+1:] + x.shape[:dim]

    rank_sizes = [r * np.prod(other_dims, dtype=int) for r in rank_sizes]

    if rank_offsets is None:
      rank_offsets  = [int(0)]
      for rank in range(1, _NRANKS):
        rank_offsets.append(rank_offsets[rank-1] + rank_sizes[rank-1])

    alloc_fn = torch.zeros if _USE_ZEROFILL else torch.empty
    data = alloc_fn(total_size, dtype=x.dtype, device="cpu")


    rank_sizes = tuple(rank_sizes)
    rank_offsets = tuple(rank_offsets)


  dtype = MPI.Datatype.Match_size(MPI.TYPECLASS_REAL if x.is_floating_point() else MPI.TYPECLASS_INTEGER, x.element_size())

  #assert x.storage_offset() == 0

  # mpi4py requires a contiguous host buffer.  State vectors assembled by the
  # DD-FOM are frequently tensor views, and passing those views directly can
  # trigger DLPack ``buffer is not contiguous`` errors.
  send_buffer = _ensure_contiguous(x).detach().cpu().numpy()
  recv_buffer = data.numpy() if _RANK == root else None
  _COMM.Gatherv(send_buffer, [recv_buffer, rank_sizes, rank_offsets, dtype], root)

  #data = comm.gather(x, dim=dim, out=data)
  #tmp = comm.gather(x, dim=dim)
  #print(" GATHERV OUT: ", tmp)

  barrier()

  if _RANK == root and as_list:
    data = data.to(device())
    # Root rank stacks all gathered tensors
    data = list(torch.split(data, rank_sizes_dim, dim=dim))
  if _RANK == root and not as_list:
    data = data.to(device())

  if x_sparse:
    # convert dense back to sparse
    if _RANK == root:
      if as_list:
        for rank in range(_NRANKS):
          data[rank] = data[rank].to_sparse_csr()
      else:
        data = data.to_sparse_csr()
    else:
      x = x.to_sparse_csr()

  # Root rank returns combined tensor, all others just return input tensor
  if _RANK == root:
    return data
  else:
    return x


def gatherv_spcoo(
  x: torch.Tensor,
  sizes: Optional[List[int]] = None,
  offsets: Optional[List[int]] = None,
  dim: int = 0,
  root: int = 0,
  as_list: bool = False,
  coalesce: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Wrapper around gatherv for sparse coo tensors. This is usually called by gatherv
  Indices and values of sp tensor x are gathered and reassembled

  :param x: Local sparse COO tensor to gather.
  :type x: torch.Tensor
  :param sizes: Optional local sizes for every rank.
  :type sizes: list[int] or None
  :param offsets: Optional receive-buffer offsets for every rank.
  :type offsets: list[int] or None
  :param dim: Dimension along which to concatenate gathered tensors.
  :type dim: int
  :param root: Rank that receives gathered data.
  :type root: int
  :param as_list: Return individual gathered tensors instead of concatenating.
  :type as_list: bool
  :param coalesce: Coalesce the COO tensor before gathering.
  :type coalesce: bool

  :return: Gathered sparse data on root, or the local tensor on other ranks.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if coalesce:
    x = x.coalesce()

  if not distributed():
    return x

  assert x.layout == torch.sparse_coo

  indices = gatherv_tensor(x._indices().ravel(), sizes, offsets, dim=0, root=root, as_list=True)


  values = gatherv_tensor(x._values(), sizes, offsets, dim=0, root=root, as_list=True)

  shapes = _COMM.gather(x.shape)

  if _RANK == root:


    sp_x = []
    for rank in range(_NRANKS):
      sp_x.append(torch.sparse_coo_tensor(indices[rank].reshape(x.ndim,-1), values[rank], size=shapes[rank], device=device()))


    if not as_list:
      sp_x = torch.cat(sp_x, dim=dim)
    return sp_x
  else:
    return x


def gatherv_spcsr(
  x: torch.Tensor,
  sizes: Optional[List[int]] = None,
  offsets: Optional[List[int]] = None,
  dim: int = 0,
  root: int = 0,
  as_list: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Wrapper around gatherv for sparse csr tensors. This is usually called by gatherv
  Indices and values of sp tensor x are gathered and reassembled

  :param x: Local sparse CSR tensor to gather.
  :type x: torch.Tensor
  :param sizes: Optional local sizes for every rank.
  :type sizes: list[int] or None
  :param offsets: Optional receive-buffer offsets for every rank.
  :type offsets: list[int] or None
  :param dim: Dimension along which to concatenate gathered tensors.
  :type dim: int
  :param root: Rank that receives gathered data.
  :type root: int
  :param as_list: Return individual gathered tensors instead of concatenating.
  :type as_list: bool

  :return: Gathered sparse data on root, or the local tensor on other ranks.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if not distributed():
    return x

  assert x.layout == torch.sparse_csr
  crow_indices = gatherv_tensor(x.crow_indices(), sizes, offsets, dim=0, root=root, as_list=True)
  col_indices = gatherv_tensor(x.col_indices(), sizes, offsets, dim=0, root=root, as_list=True)
  values = gatherv_tensor(x.values(), sizes, offsets, dim=0, root=root, as_list=True)

  shapes = _COMM.gather(x.shape)

  if _RANK == root:


    sp_x = []
    for rank in range(_NRANKS):
      sp_x.append(torch.sparse_csr_tensor(crow_indices[rank], col_indices[rank], values[rank], size=shapes[rank], device=device()))


    if not as_list:
      sp_x = torch.cat([s.to_sparse_coo() for s in sp_x], dim=dim).to_sparse_csr()
    return sp_x
  else:
    return x


def scatter_tensor(
  x: Union[torch.Tensor, List[torch.Tensor]],
  x_out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
  dim: int = 0,
  root: int = 0,
  as_list: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Scatters a distributed tensor x across all ranks (from source rank root) and returns result
  Each rank gets 1/nranks portion of x tensor (along dim)
  NOTE: assumes x is distributed across all ranks, and each rank must have same size

  If x_out=None, then result tensor will be allocated with size according to x.shape[dim] // nranks
  NOTE: x_out will be overwritten by this operation

  :param x: Source tensor or list of tensors on the root rank.
  :type x: torch.Tensor or list[torch.Tensor]
  :param x_out: Optional output buffer or buffers for local data.
  :type x_out: torch.Tensor or list[torch.Tensor] or None
  :param dim: Dimension along which to split the source tensor.
  :type dim: int
  :param root: Rank that provides source data.
  :type root: int
  :param as_list: Treat ``x`` and ``x_out`` as per-rank tensor lists.
  :type as_list: bool

  :return: The local output tensor or tensor list.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if not distributed():
    return x

  full_size = 0
  data = None
  if _RANK == root:
    if not as_list:
      x = _ensure_contiguous(x)
      data = _ensure_contiguous(list(torch.tensor_split(x, _NRANKS, dim=dim)))
      full_size = x.shape[dim]
      assert len(data) == _NRANKS
      if x_out is None:
        full_size = list(x.size())
        full_size[dim] = x.shape[dim] // _NRANKS
    else:
      assert len(x) == _NRANKS
      x = _ensure_contiguous(x)
      full_size = x[0].shape[dim]

  if x_out is None:
    full_size = _COMM.bcast(full_size, root=root)
    if not as_list:
      x_out = torch.empty(full_size, device=device())
    else:
      x_out = []
      for rank in range(_NRANKS):
        x_out.append(torch.empty(full_size, device=device()))

  barrier()
  x_out = _ensure_contiguous(x_out)
  if not as_list:
    dist.scatter(x_out, data, root)
  else:
    for rank in range(_NRANKS):
      dist.scatter(x_out[rank], x[rank] if _RANK == root else None, root)
  barrier()

  return x_out


def broadcast_tensor(
  x: Union[torch.Tensor, List[torch.Tensor]],
  x_out: Optional[torch.Tensor] = None,
  root: int = 0
) -> Union[torch.Tensor, List[torch.Tensor]]:
  """
  Broadcast a dense or sparse tensor, or a list of tensors, from the root rank.

  :param x: Tensor or tensor list supplied by the root rank.
  :type x: torch.Tensor or list[torch.Tensor]
  :param x_out: Optional output buffer for a dense tensor broadcast.
  :type x_out: torch.Tensor or None
  :param root: Rank that supplies the source tensor or tensors.
  :type root: int

  :return: The broadcast tensor or tensor list.
  :rtype: torch.Tensor or list[torch.Tensor]
  """
  if not distributed():
    return x
  
  x_list = False
  x_size = 0
  x_spcoo = False
  x_spcsr = False
  if _RANK == root:
    x_list = isinstance(x, List)
    if x_list:
      x_size = len(x)
  
  x_list, x_size = _COMM.bcast((x_list, x_size), root=root)

  if x_list:
    if _RANK != root:
      x = [None] * x_size
    return [broadcast_tensor(t, x_out, root=root) for t in x]

  if _RANK == root:
    x_spcoo = x.is_sparse
    x_spcsr = x.is_sparse_csr
  x_spcoo, x_spcsr = _COMM.bcast((x_spcoo, x_spcsr), root=root)

  if x_spcoo:
    return broadcast_spcoo(x, x_out, root)

  if x_spcsr:
    return broadcast_spcsr(x, x_out, root)

  out_size = None
  dtype = None
  if _RANK == root:
    out_size = x.size()
    dtype = x.dtype

  out_size = _COMM.bcast(out_size, root=root)
  dtype = _COMM.bcast(dtype, root=root)

  if _RANK != root:
    if x_out is None:
      # create output space on all other ranks
      x = torch.zeros(out_size, dtype=dtype, device=device())
    else:
      x = x_out

  x = _ensure_contiguous(x)

  dist.broadcast(x, src=root)

  return x


def broadcast_spcoo(
  x: torch.Tensor,
  x_out: Optional[torch.Tensor] = None,
  root: int = 0
) -> torch.Tensor:
  """
  Broadcast a sparse COO tensor from the root rank.

  :param x: Sparse COO tensor supplied by the root rank.
  :type x: torch.Tensor
  :param x_out: Optional output buffer used for component broadcasts.
  :type x_out: torch.Tensor or None
  :param root: Rank that supplies the source tensor.
  :type root: int

  :return: The broadcast sparse COO tensor.
  :rtype: torch.Tensor
  """
  if not distributed(): return x

  shape = None
  indices = None
  values = None
  if _RANK == root:
    indices = x._indices()
    values = x._values()
    shape = x.shape
  shape = _COMM.bcast(shape, root=root)

  indices = broadcast_tensor(indices, x_out, root)
  values = broadcast_tensor(values, x_out, root)

  return torch.sparse_coo_tensor(indices, values, size=shape, device=device())


def broadcast_spcsr(
  x: torch.Tensor,
  x_out: Optional[torch.Tensor] = None,
  root: int = 0
) -> torch.Tensor:
  """
  Broadcast a sparse CSR tensor from the root rank.

  :param x: Sparse CSR tensor supplied by the root rank.
  :type x: torch.Tensor
  :param x_out: Optional output buffer used for component broadcasts.
  :type x_out: torch.Tensor or None
  :param root: Rank that supplies the source tensor.
  :type root: int

  :return: The broadcast sparse CSR tensor.
  :rtype: torch.Tensor
  """
  if not distributed(): return x

  shape = None
  crow_indices = None
  col_indices = None
  values = None
  if _RANK == root:
    crow_indices = x.crow_indices()
    col_indices = x.col_indices()
    values = x.values()
    shape = x.shape
  
  shape = _COMM.bcast(shape, root=root)

  crow_indices = broadcast_tensor(crow_indices, x_out, root)
  col_indices = broadcast_tensor(col_indices, x_out, root)
  values = broadcast_tensor(values, x_out, root)

  return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=shape, device=device())


def get_mesh() -> Any:
  """
  Return the current distributed device mesh.

  :return: The initialized device mesh, or None when no mesh is configured.
  :rtype: Any
  """
  return _DMESH


# Methods for DTensor creation
def to_sharded_dtensor(x: torch.Tensor,
                       shape: List[int] = None,
                       stride: List[int] = None) -> DTensor:
  """
  Creates a distributed tensor from each rank's local tensor x.
  The returned distributed tensor is sharded across the current device mesh on dim 1,
  so the DTensor represents a full tensor of x combined across all ranks.
  """
  if not distributed() or _DMESH is None:
    return x

  dtensor = DTensor.from_local(x,
                               device_mesh=_DMESH,
                               placements=[Shard(0)],
                               shape=shape,
                               stride=stride,
                               run_check=_DTENSOR_CHECKS)
  return dtensor


def to_replica_dtensor(x: torch.Tensor,
                       shape: List[int] = None,
                       stride: List[int] = None) -> DTensor:
  """
  Creates a distributed tensor from each rank's local tensor x.
  The returned distributed tensor is replicated across all ranks in the current device mesh on dim 1,
  so the DTensor represents a full tensor of x combined across all ranks.
  """
  if not distributed() or _DMESH is None:
    return x

  dtensor = DTensor.from_local(x,
                               device_mesh=_DMESH,
                               placements=[Replicate()],
                               shape=shape,
                               stride=stride,
                               run_check=_DTENSOR_CHECKS)
  return dtensor
