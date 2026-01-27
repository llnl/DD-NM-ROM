import copy
import torch
import numpy as np
import scipy.sparse as sp

from time import time
from typing import Tuple
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom.rom.utils import hyper_red as hr_mod

from .autoencoder import AutoencoderNP


class NM_ROM(object):
  """
  Class for generating a non-hyper-reduced subdomain of
  the DD NM-ROM for the 2D steady-state Burgers' equation.
  """

  def __init__(
    self,
    fom,
    nn_configfile,
    res_bases=None,
    hr_active=False,
    hr_n_samples=1
  ):
    # FOM
    # -------------
    self.fom = fom
    for k in ("runtime", "mesh"):
      setattr(self, k, getattr(self.fom, k))
    # Autoencoder
    # -------------
    self.nn_config = torch.load(nn_configfile)
    self.nn_model = AutoencoderNP(self.nn_config)
    # HR
    # -------------
    self.hr_init = False
    self.hr_active = False
    self.res_bases = res_bases
    self.hr_n_samples = int(hr_n_samples)
    self.set_hr_mode(active=bool(hr_active))
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0

  def get_ndof(self):
    return self.nn_config["decoder"]["latent_dim"]

  # Autoencoder
  # ===================================
  def encode(self, x, with_jac=True):
    return self.nn_model.encoder(x, with_jac)

  def decode(self, x, with_jac=True):
    return self.nn_model.decoder(x, with_jac)

  def encdec(self, x):
    return self.nn_model(x)

  def reconstruct_static(self, x):
    if (x.ndim == 2):
      return np.vstack([self.encdec(xi) for xi in x.T]).T
    else:
      return self.encdec(x)

  # Hyper-reduction (HR)
  # ===================================
  def set_hr_mode(self, active=False):
    self.hr_active = active
    if self.hr_active:
      if (not self.hr_init):
        self.init_hr_mode()
      self.compute_res_jac = self._compute_res_jac_hr
      self.size = {x_k: self.hr_nodes_rows[x_k].size for x_k in ("u", "v")}
    else:
      self.compute_res_jac = self._compute_res_jac
      self.size = {x_k: self.fom.mesh.nxy for x_k in ("u", "v")}
    self.set_decoder_hr(active=self.hr_active)

  def init_hr_mode(self):
    self.set_indices_hr()
    self.set_ops_bc_hr()
    self.set_decoder_hr(active=True, row_ind=self.hr_nodes_cols_all)
    self.hr_init = True

  # Indices
  # -----------------------------------
  def set_indices_hr(self):
    # Sample HR nodes
    # -------------
    self.hr_n_samples = np.clip(self.hr_n_samples, *self.res_bases.T.shape)
    self.hr_nodes_rows_all = hr_mod.select_sample_nodes(
      bases=self.res_bases, n_samples=self.hr_n_samples
    )
    # Vector indices
    # -------------
    # Split HR residual nodes for u and v
    self.hr_nodes_rows = {
      "u": self.hr_nodes_rows_all[self.hr_nodes_rows_all  < self.fom.mesh.nxy],
      "v": self.hr_nodes_rows_all[self.hr_nodes_rows_all >= self.fom.mesh.nxy] \
           - self.fom.mesh.nxy
    }
    self.hr_nodes_cols = {}
    self.hr_nodes_cols_all = []
    for x_k in ("u", "v"):
      col_ind = hr_mod.get_col_indices(
        row_ind=self.hr_nodes_rows[x_k], matrices=list(self.fom.ops.values())
      )
      # Store
      self.hr_nodes_cols[x_k] = col_ind
      self.hr_nodes_cols_all.append(copy.deepcopy(col_ind))
      if (x_k == "v"):
        self.hr_nodes_cols_all[-1] += self.fom.mesh.nxy
    self.hr_nodes_cols_all = np.concatenate(self.hr_nodes_cols_all)
    self.hr_nodes_cols_all = self.hr_nodes_cols_all.astype(np.int32)
    # Sub-matrix indices
    # -------------
    self.submat_hr = {}
    for x_i in ("u", "v"):
      for x_j in ("u", "v"):
        nodes_ij = [self.hr_nodes_rows[x_i], self.hr_nodes_cols[x_j]]
        self.submat_hr[x_i+"_"+x_j] = np.ix_(*nodes_ij)

  # Operators
  # -----------------------------------
  def set_ops_bc_hr(self):
    # Inclusion operators
    self.iden_hr = {}
    for (k, submat_k) in self.submat_hr.items():
      self.iden_hr[k] = self.fom.iden[submat_k]
    self.iden_uv_hr = sp.block_diag([self.iden_hr["u_u"], self.iden_hr["v_v"]])
    # Differential operators
    self.ops_hr = {}
    for (op_k, op_v) in self.fom.ops.items():
      self.ops_hr[op_k] = {}
      for x_k in ("u", "v"):
        submat_k = self.submat_hr[x_k+"_"+x_k]
        self.ops_hr[op_k][x_k] = op_v[submat_k]
    # Boundary conditions
    self.bc_f_hr = {}
    for x_k in ("u", "v"):
      self.bc_f_hr[x_k] = ops.map_nested_dict(
        self.fom.bc_f[x_k], lambda x: x[self.hr_nodes_rows[x_k]]
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

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> Tuple[np.ndarray, sp.spmatrix]:
    start = time()
    res, jac = self.compute_res_jac(x)
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def _compute_res_jac(
    self,
    x: np.ndarray
  ) -> Tuple[np.ndarray, sp.spmatrix]:
    x, de_jac = self.decode(x, with_jac=True)
    res, jac = self.fom.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      x_old = self.decode(self.x_old, with_jac=False)
      res = x - x_old - self.dt*res
      jac = self.fom.iden_uv - self.dt*jac
    return res, jac @ de_jac

  def _compute_res_jac_hr(
    self,
    x: np.ndarray
  ) -> Tuple[np.ndarray, sp.spmatrix]:
    # Reconstruct u and v
    uv, dec_jac = self.reconstruct_uv(x, with_jac=True)
    uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
    # Precompute actions of operators
    ops_uv = self.action_ops(uv)
    # Compute residual
    res = self.compute_res(uv, uv_diag, ops_uv)
    # Compute Jacobian
    jac = self.compute_jac(uv_diag, ops_uv)
    # Return
    return res, jac @ dec_jac

  def reconstruct_uv(
    self,
    x,
    with_jac=True
  ):
    y = self.decode(x, with_jac=with_jac)
    uv = {
      "u": y[0][:self.size["u"]],
      "v": y[0][self.size["u"]:]
    }
    for x_i in ("u", "v"):
      for x_j in ("u", "v"):
        k = x_i+"_"+x_j
        uv[k] = self.iden_hr[k] @ uv[x_j]
    return (uv, y[1]) if with_jac else uv

  def action_ops(
    self,
    uv
  ):
    ops_uv = {}
    for x_k in ("u", "v"):
      ops_uv[x_k] = {}
      for (op_k, op_v) in self.ops_hr.items():
        ops_uv[x_k][op_k] = op_v[x_k] @ uv[x_k]
      for axis in ("x", "y"):
        ops_uv[x_k][f"A{axis}"] -= self.bc_f_hr[x_k]["A"][axis]
    return ops_uv

  def compute_res(
    self,
    uv,
    uv_diag,
    ops_uv
  ):
    if (not self.steady):
      uv_old = self.reconstruct_uv(self.x_old, with_jac=False)
    res = {}
    for x_k in ("u", "v"):
      res[x_k] = uv_diag[f"{x_k}_u"] @ ops_uv[x_k]["Ax"] \
               + uv_diag[f"{x_k}_v"] @ ops_uv[x_k]["Ay"] \
               + ops_uv[x_k]["D"] + self.bc_f_hr[x_k]["D"]
      # Backward Euler for integration
      if (not self.steady):
        x_kk = x_k+"_"+x_k
        res[x_k] = uv[x_kk] - uv_old[x_kk] - self.dt * res[x_k]
    return np.concatenate([res[x_k] for x_k in ("u", "v")])

  def compute_jac(
    self,
    uv_diag,
    ops_uv
  ):
    jac_uu = ops.sp_diag(ops_uv["u"]["Ax"])
    jac_uv = ops.sp_diag(ops_uv["u"]["Ay"])
    jac_vu = ops.sp_diag(ops_uv["v"]["Ax"])
    jac_vv = ops.sp_diag(ops_uv["v"]["Ay"])
    jac_xx = {}
    for x_k in ("u", "v"):
      jac_xx[x_k] = uv_diag[x_k+"_u"] @ self.ops_hr["Ax"][x_k] \
                  + uv_diag[x_k+"_v"] @ self.ops_hr["Ay"][x_k] \
                  + self.ops_hr["D"][x_k]
    jac_uu = jac_uu @ self.iden_hr["u_u"] + jac_xx["u"]
    jac_uv = jac_uv @ self.iden_hr["u_v"]
    jac_vu = jac_vu @ self.iden_hr["v_u"]
    jac_vv = jac_vv @ self.iden_hr["v_v"] + jac_xx["v"]
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    # Backward Euler for integration
    if (not self.steady):
      jac = self.iden_uv_hr - self.dt * jac
    return jac

  # Solution
  # ===================================
  def solve(
    self,
    x0,
    dt=0.0,
    nt=1,
    steady=True,
    guess=None,
    use_guess=False,
    runtime=0.0,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    verbose=False
  ):
    """
    Solves for the u and v states of the FOM using Newton"s method.
    """
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    self.runtime["total"] += runtime
    # Initialize solver
    solver = solvers.GaussNewton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    z, res, *_, flag = solver(x0, dt, nt, guess, use_guess)
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv = self.assemble_sol(z)
    return uv, z, res, converged

  def get_init_sol(
    self,
    x
  ):
    return self.encode(x)

  def assemble_sol(
    self,
    z
  ):
    if (z.ndim == 2):
      x = [self.decode(zi, with_jac=False) for zi in z.T]
      x = np.vstack(x).T
    else:
      x = self.decode(z, with_jac=False)
    return self.fom.extract_uv(x, diag=False)
