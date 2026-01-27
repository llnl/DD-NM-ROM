import numpy as np
import scipy.sparse as sp

from typing import List, Tuple, Union


SOL_TYPE = Tuple[Union[List[np.ndarray], np.ndarray], ...]
EVAL_TYPE = Tuple[np.ndarray, Union[np.ndarray, sp.spmatrix], float]
PRINT_FMT = "| {0:11d} | {1:11.4e} | {2:11.4e} |"
