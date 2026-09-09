import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops
from dd_nm_rom import backend as bkd

from . import dtypes


class SubdomainElementState(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    name: str,
    monolithic: dtypes.FOM_TYPE,
    nodes_res: np.ndarray,
    nodes_state: np.ndarray
  ) -> None:
    self.name = name
    self.monolithic = monolithic
    self.nodes_res = nodes_res
    self.nodes_state = nodes_state
    self.n_nodes_res = self.nodes_res.size
    self.n_nodes_state = self.nodes_state.size
    self.submat = np.ix_(self.nodes_res, self.nodes_state)
    # Intersection between state and residual nodes
    self.intersect = {
      "state_res": np.nonzero(np.isin(self.nodes_state, self.nodes_res))[0],
      "res_state": np.nonzero(np.isin(self.nodes_res, self.nodes_state))[0]
    }
    # Operators
    self.set_ops_bc()

  def __repr__(self):
    return "SubdomainElementState(name={}, monolithic={}, n_nodes_res={}, n_nodes_state={})".format(self.name, self.monolithic, self.n_nodes_res, self.n_nodes_state)

  # Operators
  # -----------------------------------
  def set_ops_bc(self) -> None:
    # Inclusion operators
    if bkd.is_torch_backend():
      self.iden = bkd.torch_csr_to_scipy(self.monolithic.iden.to_sparse_csr())[self.submat]
    else:
      self.iden = self.monolithic.iden[self.submat]
    self.iden_uv = sp.block_diag([self.iden, self.iden])
    # Differential operators
    self.ops = {}
    for (op_k, op_v) in self.monolithic.ops.items():
      op_v = bkd.torch_csr_to_scipy(op_v)
      self.ops[op_k] = op_v[self.submat]
      if bkd.is_torch_backend():
        self.ops[op_k] = bkd.to_sp_coo_backend(self.ops[op_k].tocoo()) #.to_dense()
    # Boundary conditions
    self.bc_f = ops.map_nested_dict(
      self.monolithic.bc_f, lambda x: x[self.nodes_res]
    )


    if bkd.is_torch_backend():
      self.iden = bkd.to_sp_coo_backend(self.iden.tocoo())
      self.iden_uv = bkd.to_sp_coo_backend(self.iden_uv)
