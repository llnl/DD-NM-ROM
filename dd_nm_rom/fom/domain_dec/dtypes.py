import numpy as np
import scipy.sparse as sp

from ..monolithic import Burgers2D
from typing import Dict, List, Tuple, Union

FOM_TYPE = Burgers2D
CMAT_TYPE = Dict[str, Union[List[sp.spmatrix], sp.spmatrix]]
KKT_TYPE = Tuple[np.ndarray, np.ndarray, sp.spmatrix, sp.spmatrix]
UV_TYPE = Dict[str, Dict[str, Union[List[np.ndarray], np.ndarray]]]
RES_JAC_TYPE = Tuple[np.ndarray, Union[Dict[str, sp.spmatrix], sp.spmatrix]]
SOL_TYPE = Tuple[UV_TYPE, np.ndarray, Union[np.ndarray, List[np.ndarray]], bool]
