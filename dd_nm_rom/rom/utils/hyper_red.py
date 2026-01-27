import numpy as np
import scipy as sp

from typing import Dict, List, Tuple, Union


def get_col_indices(
  row_ind: np.ndarray,
  matrices: List[sp.sparse.spmatrix]
) -> np.ndarray:
  """
  Get column indices for given row indices from a list of matrices.

  :param row_ind: Array of row indices.
  :type row_ind: np.ndarray
  :param matrices: List of sparse matrices.
  :type matrices: List[sp.sparse.spmatrix]

  :return: Array of column indices.
  :rtype: np.ndarray
  """
  matrices = [matrices] if (not isinstance(matrices, list)) else matrices
  col_ind = set()
  for m in matrices:
    if (m is not None):
      m = sp.sparse.coo_matrix(m)
      for row in row_ind:
        col_ind = col_ind.union(set(m.col[m.row==row]))
  return np.sort(np.array(list(col_ind)))

def select_sample_nodes(
  bases: np.ndarray,
  n_samples: int,
  n_bases: int = -1,
  samples: Union[np.ndarray, None] = None
) -> np.ndarray:
  """
  Greedy algorithm to select sample nodes for hyper-reduction.

  :param bases: Array of residual basis vectors.
  :type bases: np.ndarray
  :param n_samples: Number of desired sample nodes.
  :type n_samples: int
  :param n_bases: Number of working columns of bases.
  :type n_bases: int, optional
  :param samples: Existing sample nodes.
  :type samples: Union[np.ndarray, None], optional

  :return: Array of sample nodes.
  :rtype: np.ndarray
  """
  n_bases = bases.shape[1] if (n_bases <= 0) else n_bases
  if (samples is None):
    samples = np.array([], dtype=np.int32)
  # Initialize greedy algorithm
  dn_samples = n_samples - len(samples)                     # number of additional nodes to sample
  n_bases_it = 0                                            # intializes counter for the number of working basis vectors used
  n_it = min(n_bases, dn_samples)                           # number of greedy iterations to perform
  n_rhs = int(np.ceil(n_bases/dn_samples))                  # max number of RHS in least squares problem
  n_bases_it_min = int(np.floor(n_bases/n_it))              # minimum number of working basis vectors per iteration
  dn_samples_min = int(np.floor(dn_samples*n_rhs/n_bases))  # minimum number of sample nodes to add per iteration
  full_indices = np.arange(bases.shape[0])                  # array of all node inidices
  # Greedy algorithm iterations
  for it in range(n_it):
    # Set number of working basis vectors for current iteration
    nb = n_bases_it_min
    if (it <= (n_bases % n_it - 1)):
      nb += 1
    # Set number of sample nodes to add during current iteration
    dn = dn_samples_min
    if ((n_rhs == 1) and (it <= (dn_samples % n_bases - 1))):
      dn += 1
    if (it == 0):
      r = bases[:,:nb]
    else:
      # Compute least-squares solutions
      a = bases[:,:n_bases_it]
      # for q in range(nb):
      #   b = bases[:,n_bases_it+q-1]
      #   x = np.linalg.lstsq(a[samples], b[samples], rcond=None)[0]
      #   r[:,q] = b - a@x
      b = bases[:,n_bases_it:n_bases_it+nb]
      x = sp.linalg.lstsq(a[samples], b[samples], cond=None)[0]
      r = b - a@x
    for _ in range(dn):
      # choose node with largest average error
      i = np.setdiff1d(full_indices, samples)
      n = np.where(np.isin(full_indices, i))[0]
      n = n[np.argmax(np.sum(r[i]*r[i], axis=1))]
      samples = np.append(samples, n)
    n_bases_it += nb
  # Return samples
  return np.sort(samples)
