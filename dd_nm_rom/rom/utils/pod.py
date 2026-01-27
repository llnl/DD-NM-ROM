import numpy as np
import scipy as sp
import dill as pickle
import dask.array as da

from dd_nm_rom import backend as bkd
from typing import Dict, List, Tuple, Union

SVD_TYPE = Dict[str, List[Dict[str, np.ndarray]]]
BASES_TYPE = Dict[str, List[np.ndarray]]


def compute_svd(
  data: Dict[str, List[np.ndarray]],
  energy_min: float = 1e-5,
  n_bases: int = -1,
  nb_samples: int = -1,
  get_bases: bool = False,
  save_dir: Union[str, None] = None,
  verbose: bool = True
) -> Union[Tuple[SVD_TYPE, BASES_TYPE], Tuple[SVD_TYPE, None]]:
  """
  Compute the Singular Value Decomposition (SVD) for the given datasets.

  :param data: Dictionary containing the datasets.
  :type data: Dict[str, List[np.ndarray]]
  :param energy_min: Minimum energy threshold for basis selection.
                     Default is 1e-8.
  :type energy_min: float
  :param n_bases: Number of bases to retain.
                  Default is -1 (determined by energy_min).
  :type n_bases: int
  :param nb_samples: Number of samples to use from each dataset.
                     Default is -1 (use all samples).
  :type nb_samples: int
  :param get_bases: Flag indicating whether to compute and return POD bases.
                    Default is False.
  :type get_bases: bool
  :param save_dir: Directory to save the SVD and bases results.
                   Default is None (do not save).
  :type save_dir: Union[str, None]
  :param verbose: Flag indicating whether to print progress messages.
                  Default is True.
  :type verbose: bool

  :return: Tuple containing the SVD results and the POD bases (if requested).
  :rtype: Union[Tuple[SVD_TYPE, BASES_TYPE], Tuple[SVD_TYPE, None]]
  """
  svd = {}
  for (k, data_k) in data.items():
    svd[k] = []
    for (i, data_ki) in enumerate(data_k):
      if (nb_samples > 0):
        nb_samples_max = len(data_ki)
        nb_samples_ki = min(nb_samples_max, nb_samples)
        indices = np.arange(nb_samples_ki)
        np.random.seed(bkd.seed())
        indices = np.random.choice(indices, size=nb_samples_ki, replace=False)
        data_ki = data_ki[indices]
      if verbose:
        print(f"> Performing SVD for dataset '{k}-{i+1}' ...")
      svd[k].append(perform_svd(data_ki.T))
  if (save_dir is not None):
    if verbose:
      print("> Saving SVD data ...")
    pickle.dump(svd, open(save_dir+"/svd.p", "wb"))
  if get_bases:
    bases = get_bases_from_svd(
      svd,
      energy_min=energy_min,
      n_bases=n_bases
    )
    if (save_dir is not None):
      if verbose:
        print("> Saving POD bases ...")
      pickle.dump(bases, open(save_dir+"/bases.p", "wb"))
    return svd, bases
  else:
    return svd, None

def get_bases_from_svd(
  svd: SVD_TYPE,
  energy_min: float = 1e-5,
  n_bases: int = -1
) -> BASES_TYPE:
  """
  Extract POD bases from the given SVD results.

  :param svd: Dictionary containing the SVD results.
  :type svd: SVD_TYPE
  :param energy_min: Minimum energy threshold for basis selection.
                     Default is 1e-8.
  :type energy_min: float
  :param n_bases: Number of bases to retain.
                  Default is -1 (determined by 'energy_min').
  :type n_bases: int

  :return: Dictionary containing the POD bases.
  :rtype: BASES_TYPE
  """
  bases = {}
  for (k, svd_k) in svd.items():
    bases[k] = []
    n_bases_k = n_bases[k] if isinstance(n_bases, dict) else n_bases
    energy_min_k = energy_min[k] if isinstance(energy_min, dict) else energy_min
    for svd_ki in svd_k:
      bases[k].append(
        get_pod_bases(
          svd=svd_ki,
          energy_min=energy_min_k,
          n_bases=n_bases_k
        )
      )
  return bases

def get_pod_bases(
  svd: Union[Dict[str, np.ndarray], None] = None,
  data: Union[np.ndarray, None] = None,
  energy_min: float = 1e-5,
  n_bases: int = -1
) -> np.ndarray:
  """
  Compute POD bases from SVD results or directly from data.

  :param svd: Dictionary containing the SVD results. If None, SVD will be
              computed from data.
  :type svd: Union[Dict[str, np.ndarray], None], optional
  :param data: Data matrix to compute SVD from, if svd is None.
               Default is None.
  :type data: Union[np.ndarray, None], optional
  :param energy_min: Minimum energy threshold for basis selection.
                     Default is 1e-5.
  :type energy_min: float
  :param n_bases: Number of bases to retain.
                  Default is -1 (determined by 'energy_min').
  :type n_bases: int

  :return: Matrix containing the selected POD bases.
  :rtype: np.ndarray
  """
  svd = perform_svd(data) if (svd is None) else svd
  if (n_bases <= 0):
    s_sq = svd["s"]**2
    energy = np.cumsum(s_sq) / np.sum(s_sq)
    n_bases = np.where(energy > 1-energy_min)[0].min()+1
  n_bases = min(svd["u"].shape[0], n_bases)
  return svd["u"][:,:n_bases]

def perform_svd(data: np.ndarray) -> Dict[str, np.ndarray]:
  """
  Perform Singular Value Decomposition (SVD) on the given data matrix.

  :param data: Data matrix to perform SVD on.
  :type data: np.ndarray

  :return: Dictionary containing the left singular vectors ('u') and
           singular values ('s').
  :rtype: Dict[str, np.ndarray]
  """
# u, s, _ = sp.linalg.svd(data, full_matrices=False)
  dask_data = da.from_array(data, chunks=(500, 'auto'))
  k = 500
# k = min(data.shape)
  u, s, _ = da.linalg.svd_compressed(dask_data, k=k)
  u = u.compute()
  s = s.compute()
  return {"u": u, "s": s}