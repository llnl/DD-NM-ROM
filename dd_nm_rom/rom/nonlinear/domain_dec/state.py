import copy
import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops
from dd_nm_rom.rom.utils import hyper_red as hr_mod
from dd_nm_rom import backend as bkd


class SubdomainElementStateROM(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    state_fom,
    nn_model=None
  ):
    self.state_fom = state_fom
    self.name = self.state_fom.name
    self.nn_model = nn_model
    self.hr_init = False
    self.hr_active = False
    self.set_ops_bc()
    self.set_hr_mode(active=False)

  # Autoencoders
  # ===================================
  def encode(self, x, with_jac=True):
    return self.nn_model.encoder(x, with_jac)

  def decode(self, x, with_jac=True):
    return self.nn_model.decoder(x, with_jac)

  def encdec(self, x):
    return self.nn_model(x)

  # Operators
  # ===================================
  def set_ops_bc(self):
    self._iden = copy.copy(self.state_fom.iden)
    self._iden_uv = copy.copy(self.state_fom.iden_uv)
    if bkd.is_torch_backend():
      self._iden = bkd.to_sp_coo_backend(self._iden)
      self._iden_uv = bkd.to_sp_coo_backend(self._iden_uv)
    self._ops = ops.map_nested_dict(self.state_fom.ops, copy.copy)
    self._bc_f = ops.map_nested_dict(self.state_fom.bc_f, copy.copy)


  # Hyper-reduction (HR)
  # ===================================
  def set_hr_mode(self, active=False):
    self.hr_active = active
    if self.hr_active:
      if (not self.hr_init):
        raise ValueError(
          f"HR mode not initialized for element '{self.name}'."
        )
      self.size = {x_k: self.hr_nodes_state[x_k].size for x_k in ("u", "v")}
      self.iden = self._iden_hr
      self.iden_uv = self._iden_uv_hr
      self.ops = self._ops_hr
      self.bc_f = self._bc_f_hr
    else:
      self.size = {x_k: self.state_fom.n_nodes_state for x_k in ("u", "v")}
      self.iden = self._iden
      self.iden_uv = self._iden_uv
      self.ops = self._ops
      self.bc_f = self._bc_f
    self.set_decoder_hr(active=self.hr_active)

  def init_hr_mode(self, hr_nodes_res):
    self.hr_nodes_res = hr_nodes_res
    self.set_indices_hr()
    self.set_ops_bc_hr()
    self.set_decoder_hr(active=True, row_ind=self.hr_nodes_state_all)
    self.hr_init = True

  # Indices
  # -----------------------------------
  def set_indices_hr(self):
    # Vector indices
    # -------------
    # Get state nodes
    self.hr_nodes_state = {}
    self.hr_nodes_state_all = []
    # Collect matrices
    matrices = list(self._ops.values()) + [self._iden]
    # Loop over variables
    for x_k in ("u", "v"):
      # Get column indices
      row_ind = self.hr_nodes_res[x_k]
      col_ind = hr_mod.get_col_indices(row_ind, matrices)
      # Store
      self.hr_nodes_state[x_k] = col_ind
      self.hr_nodes_state_all.append(copy.deepcopy(col_ind))
      if (x_k == "v"):
        self.hr_nodes_state_all[-1] += self.state_fom.n_nodes_state
    self.hr_nodes_state_all = np.concatenate(
      self.hr_nodes_state_all, dtype=np.int32
    )
    # Sub-matrix indices
    # -------------
    self.submat_hr = {}
    for x_i in ("u", "v"):
      for x_j in ("u", "v"):
        nodes_ij = [self.hr_nodes_res[x_i], self.hr_nodes_state[x_j]]
        self.submat_hr[x_i+"_"+x_j] = np.ix_(*nodes_ij)

  # Operators
  # -----------------------------------
  def set_ops_bc_hr(self):
    # Inclusion operators
    self._iden_hr = {}
    for (k, submat_k) in self.submat_hr.items():
      self._iden_hr[k] = self._iden[submat_k]
    self._iden_uv_hr = sp.block_diag(
      [self._iden_hr["u_u"], self._iden_hr["v_v"]]
    )
    # Differential operators
    self._ops_hr = {}
    for (op_k, op_v) in self._ops.items():
      self._ops_hr[op_k] = {}
      for x_k in ("u", "v"):
        submat_k = self.submat_hr[x_k+"_"+x_k]
        self._ops_hr[op_k][x_k] = op_v[submat_k]
    # Boundary conditions
    self._bc_f_hr = {}
    for x_k in ("u", "v"):
      self._bc_f_hr[x_k] = ops.map_nested_dict(
        self._bc_f[x_k], lambda x: x[self.hr_nodes_res[x_k]]
      )

  # Decoder
  # -----------------------------------
  def set_decoder_hr(
    self,
    active=True,
    row_ind=None
  ):
    if (self.nn_model is not None):
      self.nn_model.set_hr_mode(active, row_ind)
