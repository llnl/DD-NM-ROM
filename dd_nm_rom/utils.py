import os
import sys
import types
import inspect
import joblib as jl
import dill as pickle
import numpy as np

from tqdm import tqdm
from typing import Any, List, Union
from itertools import groupby

from . import backend as bkd

# Classes
# =====================================
def get_class(
  modules: Union[types.ModuleType, List[types.ModuleType]],
  name: Union[str, None] = None,
  kwargs: Union[dict, None] = None
) -> callable:
  """
  Return a class object given its name and the module it belongs to.

  This function searches for a class with the specified name within the given 
  module(s). If found, it can return an instance of the class with optional 
  keyword arguments provided in `kwargs`. If no class is found, an error is 
  raised.

  :param modules: A module or a list of modules to search for the class.
  :type modules: list or module
  :param name: The name of the class to retrieve.
  :type name: str, optional
  :param kwargs: Optional keyword arguments to pass when initializing the 
                 class (it can contain the name of the class if 'name' 
                 is not provided).
  :type kwargs: dict, optional

  :return: An instance of the class if found, or the class itself.
  :rtype: object or class
  """
  # Check class name
  if ((name is None) and (kwargs is not None)):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      raise ValueError("Class name not provided.")
  # Loop over modules to find class
  if (not isinstance(modules, (list, tuple))):
    modules = [modules]
  for module in modules:
    members = inspect.getmembers(module, inspect.isclass)
    for (name_i, cls_i) in members:
      if (name_i == name):
        if (kwargs is not None):
          return cls_i(**kwargs)
        else:
          return cls_i
  # Raise error if class not found
  names = [module.__name__ for module in modules]
  raise ValueError(f"Class `{name}` not found in modules: {names}.")

def check_path(path: str) -> None:
  """
  Check if the specified path exists.

  :param path: The path to check.
  :type path: str

  :return: None
  :rtype: None

  :raises IOError: If the path does not exist.
  """
  if (not os.path.exists(path)):
    raise IOError(f"Path '{path}' does not exist.")

# Data
# =====================================
def save_case(
  path: str,
  index: int,
  data: Any
) -> None:
  """
  Save simluated case to a file with specific format.

  The file is saved with a name formatted as `case_{index}.p`, where `index` 
  is zero-padded to 5 digits.

  :param path: Directory path where the file will be saved.
  :type path: str
  :param index: Index used to generate the filename.
  :type index: int
  :param data: Data to be saved in the file.
  :type data: Any

  :return: None
  :rtype: None
  """
  filename = path + f"/case_{str(index+1).zfill(5)}.p"
  pickle.dump(data, open(filename, "wb"))

def load_case(
  path: str,
  index: int,
  key: Union[str, None] = None
) -> Any:
  """
  Load simluated case from a file and optionally retrieve a specific item.

  Constructs the filename from the provided path and index, then loads 
  the data from this file. If a key is specified, return the value associated 
  with that key. Otherwise, return the entire data.

  :param path: Directory path where the case file is located.
  :type path: str
  :param index: Index used to generate the filename.
  :type index: int
  :param key: Optional key to retrieve a specific item from the data.
  :type key: Union[str, None]

  :return: The data from the file, or the specific item if a key is provided.
  :rtype: Any
  """
  filename = path + f"/case_{str(index+1).zfill(5)}.p"
  if os.path.exists(filename):
    data = pickle.load(open(filename, "rb"))
    if (key is None):
      return data
    else:
      return data[key]

def load_case_parallel(
  path: str,
  ranges: List[int],
  key: Union[str, None] = None,
  n_workers: int = 1
) -> List[Any]:
  """
  Load simluated cases in parallel or sequentially based on the number 
  of workers.

  This function uses `joblib` to parallelize the loading of cases if 
  `n_workers` is greater than 1. Otherwise, it loads the cases sequentially.

  :param path: Path to the data source.
  :type path: str
  :param ranges: Range of indices for the cases to be loaded.
  :type ranges: List[int]
  :param key: Optional key to pass to the `load_case` function.
  :type key: Union[str, None]
  :param n_workers: Number of parallel workers to use. Default is 1.
  :type n_workers: int

  :return: A list of loaded cases.
  :rtype: List[Any]
  """
  #"""
  single_case = False
  if bkd.distributed():
    ranges = np.arange(*ranges).tolist()
    range_per_rank = len(ranges) // bkd.get_nranks()
    if range_per_rank == 0:
      single_case = True
    else:
      print(" TOTAL RANGES {} split into {} per rank!".format(len(ranges), range_per_rank))
      rank = bkd.get_rank()
      start = rank * range_per_rank
      end = (rank+1) * range_per_rank
      extra = len(ranges) % bkd.get_nranks()
      if extra != 0 and rank == bkd.get_nranks() - 1:
        end += extra
      ranges = ranges[start:end]
      print(" RANK {} loading {} ranges: {} ({})".format(rank, len(ranges), ranges, ranges))
      bkd._COMM.Barrier()
  else:
    ranges = range(*ranges)
    print(" RANK 0 loading {} ranges: {} ({})".format(len(ranges), ranges, ranges))
  #"""

  iterable = tqdm(
    iterable=ranges,
    ncols=80,
    desc="> Cases",
    file=sys.stdout
  )
  if (n_workers > 1):
    return jl.Parallel(n_workers)(
      jl.delayed(load_case)(path=path, index=i, key=key) for i in iterable
    )
  else:
    cases = None
    if not bkd.distributed() or not single_case or (single_case and bkd.root()):
        cases = [load_case(path=path, index=i, key=key) for i in iterable]
    if single_case:
        cases = bkd._COMM.bcast(cases, 0)

    if bkd.distributed():
        bkd._COMM.Barrier()
    return cases


def generate_case_parallel(
  sol_fun: callable,
  n_samples: int,
  n_workers: int = 1,
  desc: str = "> Cases",
  verbose: bool = True,
  indices: Union[List[int], None] = None,
) -> None:

  """
  Generate cases in parallel and check solver convergence.

  The `sol_fun` callable function should return 0 or 1 to indicate whether 
  the solver has converged or not.

  :param sol_fun: A callable that performs the solver operation and returns 
                  convergence status as 0 or 1.
  :type sol_fun: callable
  :param n_samples: Number of samples or cases to generate.
  :type n_samples: int
  :param n_workers: Number of parallel workers to use. Defaults to 1.
  :type n_workers: int
  :param desc: Description to display in the progress bar. Defaults to
               "> Cases".
  :type desc: str
  :param verbose: If True, prints the total number of converged cases. 
                  Defaults to True.
  :type verbose: bool

  :return: None
  :rtype: None

  This function uses `joblib` for parallel processing and `tqdm` for showing 
  a progress bar. It applies the `sol_fun` function to a range of sample 
  indices and collects convergence results. If `verbose` is True, it prints 
  the total number of converged cases.
  """
  iterable_indices = list(range(n_samples)) if indices is None else list(indices)
  iterable = tqdm(iterable=iterable_indices, ncols=80, desc=desc, file=sys.stdout)
  if (n_workers > 1):
    converged = jl.Parallel(n_workers)(
      jl.delayed(sol_fun)(i) for i in iterable
    )
  else:
    converged = [sol_fun(i) for i in iterable]
  if verbose:
    print(f"> Total converged cases: {sum(converged)}/{len(iterable_indices)}")


def parallel_print(str):
  """
  Prints str in order on each rank
  NOTE: assumes all ranks will call this
  """
  if not bkd.distributed():
    print(str)
    return

  for rank in range(bkd.get_nranks()):
    if (rank == bkd.get_rank()):
      print(str)
    bkd.barrier()
  bkd.barrier()


def compress_ranges(numbers: Union[List[int], np.array]):
    """
    Helper function to take a range of numbers [0, 1, 2, 3, ..., 10] and return
    a shortened text representation: "0-10"
    """
    parts = []
    for _, group in groupby(enumerate(numbers), key=lambda t: t[1] - t[0]):
        group = list(group)
        start = group[0][1]
        end = group[-1][1]
        parts.append(f"{start}-{end}" if start != end else str(start))

    return ",".join(parts)


def is_in_range(ranges: str, n: int):
    """
    Helper function to check if a given number is in a compressed representation of a range.
    Ex: ranges: [0-10, 20-30], check if n=5 is in the range
    """
    for part in ranges.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            if start <= n <= end:
                return True
        elif int(part) == n:
            return True
    return False
