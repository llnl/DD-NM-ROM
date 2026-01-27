import os
import sys
import types
import inspect
import joblib as jl
import dill as pickle

from tqdm import tqdm
from typing import Any, List, Union


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
  iterable = tqdm(
    iterable=range(*ranges),
    ncols=80,
    desc="> Cases",
    file=sys.stdout
  )
  if (n_workers > 1):
    return jl.Parallel(n_workers)(
      jl.delayed(load_case)(path=path, index=i, key=key) for i in iterable
    )
  else:
    return [load_case(path=path, index=i, key=key) for i in iterable]

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
# iterable = tqdm(
#   iterable=range(n_samples),
#   ncols=80,
#   desc=desc,
#   file=sys.stdout
# )
  if (n_workers > 1):
    converged = jl.Parallel(n_workers)(
      jl.delayed(sol_fun)(i) for i in iterable
    )
  else:
    converged = [sol_fun(i) for i in iterable]
  if verbose:
#   print(f"> Total converged cases: {sum(converged)}/{n_samples}")
    print(f"> Total converged cases: {sum(converged)}/{len(iterable_indices)}")
