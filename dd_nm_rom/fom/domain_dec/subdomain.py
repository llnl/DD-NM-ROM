import numpy as np
import scipy.sparse as sp
import torch
import torch_sla

from dd_nm_rom import ops
from dd_nm_rom import backend as bkd
from typing import Dict, Tuple, Union

from . import dtypes
from .state import SubdomainElementState


class Subdomain(object):
  """
  Class for generating a subdomain of the DD-FOM for
  the 2D steady-state Burgers' equation.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    identifier: Tuple[int, int],
    monolithic: dtypes.FOM_TYPE,
    nodes_ind: Dict[str, np.ndarray],
    cmat: Dict[str, sp.spmatrix],
    ports: np.ndarray,
    port_to_nodes: Dict[int, np.ndarray],
    scaling: float = 1.0
  ) -> None:
    self.identifier = identifier
    self.monolithic = monolithic
    for k in ("ops_names", "mesh", "compact"):
      setattr(self, k, getattr(self.monolithic, k))
    self.nodes_ind = nodes_ind
    self.cmat = cmat
    self.ports = ports
    self.port_to_nodes = port_to_nodes
    self.scaling = scaling
    # Set states
    self.elem_states = {e_k: SubdomainElementState(
      name=e_k,
      monolithic=self.monolithic,
      nodes_res=self.nodes_ind["res"],
      nodes_state=self.nodes_ind[e_k]
    ) for e_k in ("res", "interior", "interface")}

  def __repr__(self):
    return "Subdomain(identifier={}, monolithic={}, nodes_ind={}, cmat={}, ports={}, port_to_nodes={}, scaling={})".format(self.identifier, self.monolithic, self.nodes_ind, self.cmat, self.ports, self.port_to_nodes, self.scaling)

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    uv: dtypes.UV_TYPE,
    lambdas: np.ndarray,
    steady: bool = True,
    dt: float = 0.0,
    uv_old: Union[dtypes.UV_TYPE, None] = None,
    force: dtypes.UV_TYPE = None,
    class_name: str = None
  ) -> dtypes.KKT_TYPE:
    """
    Compute residual and its jacobians with respect
    to interior and interface states.
    """
    # Assemble u and v on residual region
    uv = self.map_on_res(uv)
    if force is not None:
      force = self.map_on_res(force)
    if (not steady):
      uv_old = self.map_on_res(uv_old)
    # Residual and Jacobian
    res, jac = self.compute_res_jac(
      uv=uv,
      elem_states=self.elem_states,
      steady=steady,
      dt=dt,
      uv_old=uv_old,
      force=force,
      class_name=class_name
    )
    cres, cjac = self.compute_cres_cjac(uv=uv)
    # Return KKT system
    return self.assemble_kkt(
      res=res,
      cres=cres,
      lambdas=lambdas,
      jac=jac,
      cjac=cjac,
      scaling=self.scaling
    )

  def map_on_res(
    self,
    uv: dtypes.UV_TYPE
  ) -> dtypes.UV_TYPE:
    uv["res"] = {}
    for x_k in ("u", "v"):
      x_v = 0.0
      for e_k in ("interior", "interface"):
        x_v = x_v + self.elem_states[e_k].iden @ uv[e_k][x_k]
      uv["res"][x_k] = x_v
    return uv

  # Residual/Jacobian - PDE
  # -----------------------------------
  def compute_res_jac(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable],
    steady: bool = True,
    dt: float = 0.0,
    uv_old: Union[dtypes.UV_TYPE, None] = None,
    jac_fun: Union[callable, None] = None,
    force: dtypes.UV_TYPE = None,
    class_name: str = None
  ) -> dtypes.RES_JAC_TYPE:

    if self.compact:
      ops_uv = self.action_ops_gen(uv, elem_states)
    else:
      # Precompute actions of operators
      ops_uv = self.action_ops(uv, elem_states)
    # Residual and Jacobian
    res = self.compute_res(uv, elem_states, ops_uv, steady, dt, uv_old, force, class_name)
    jac = self.compute_jac(uv, elem_states, ops_uv, steady, dt, jac_fun, class_name)
    return res, jac

  def action_ops(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable]
  ) -> dtypes.UV_TYPE:
    ops_uv = {}
    for x_k in ("u", "v"):
      ops_uv[x_k] = {}
      for op_k in self.ops_names:
        op_v = 0.0
        for e_k in ("interior", "interface"):
          op_i = elem_states[e_k].ops[op_k]
          if isinstance(op_i, dict):
            op_i = op_i[x_k]
          op_v = op_v + op_i @ uv[e_k][x_k]
        ops_uv[x_k][op_k] = op_v
    return ops_uv

  def action_ops_gen(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable]
) -> dtypes.UV_TYPE:
    ops_uv = {}
    for x_k in ("u", "v"):
        ops_uv[x_k] = {}
        for op_k in self.ops_names:
            op_v = 0.0
            for e_k in ("interior", "interface"):
                # Retrieve the operator from the element state
                op_i = elem_states[e_k].ops[op_k]

                if op_k in ("Ax", "Ay") and (
                    f"{op_k}_pos" in elem_states[e_k].ops and f"{op_k}_neg" in elem_states[e_k].ops
                ):
                    # Choose velocity field for this axis
                    vel_field = "u" if op_k == "Ax" else "v"
                    vel = uv["res"][vel_field]

                    # Diagonal selection matrices
                    if bkd.is_torch_backend():
                        pos_mask = vel.ge(0.0).to(dtype=vel.dtype)
                        neg_mask = vel.lt(0.0).to(dtype=vel.dtype)
                    else:
                        pos_mask = (vel >= 0).astype(float)
                        neg_mask = 1-pos_mask #(vel < 0)

                    A_pos = elem_states[e_k].ops[f"{op_k}_pos"]
                    A_neg = elem_states[e_k].ops[f"{op_k}_neg"]

                    if bkd.is_torch_backend():
                        op_v += (
                          pos_mask * (A_pos @ uv[e_k][x_k])
                          + neg_mask * (A_neg @ uv[e_k][x_k])
                        )
                    else:
                        A_blend = sp.diags(pos_mask) @ A_pos + sp.diags(neg_mask) @ A_neg
                        op_v += A_blend @ uv[e_k][x_k]

                elif op_k not in ("Ax_pos", "Ax_neg", "Ay_pos", "Ay_neg"):
                    # Standard case
                    if isinstance(op_i, dict):
                        op_i = op_i[x_k]
                    if bkd.is_torch_backend():
                        op_i = bkd.to_sp_backend(op_i)
                    op_v += op_i @ uv[e_k][x_k]

            ops_uv[x_k][op_k] = op_v

    return ops_uv


  def compute_res(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable],
    ops_uv: dtypes.UV_TYPE,
    steady: bool = True,
    dt: float = 0.0,
    uv_old: Union[dtypes.UV_TYPE, None] = None,
    force: dtypes.UV_TYPE = None,
    class_name: str = None
  ) -> np.ndarray:
    # Compute
    res = self._compute_res(uv, elem_states, ops_uv, force, class_name)
    # Backward Euler for integration
    if (not steady):
      for x_k in ("u", "v"):
        x_kk = x_k if (x_k in uv["res"].keys()) else x_k+"_"+x_k
        res[x_k] = uv["res"][x_kk] - uv_old["res"][x_kk] - dt * res[x_k]
    # Return
    if bkd.is_torch_backend():
      return torch.cat([res[x_k] for x_k in ("u", "v")])
    else:
      return np.concatenate([res[x_k] for x_k in ("u", "v")])

  def _compute_res(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable],
    ops_uv: dtypes.UV_TYPE,
    force: dtypes.UV_TYPE = None,
    class_name: str = None
  ) -> Dict[str, np.ndarray]:
    bc_f = elem_states["res"].bc_f
    dx = {}
    for x_k in ("u", "v"):
      if (x_k not in uv["res"].keys()):
        u, v = [uv["res"][x_k+"_"+x_i] for x_i in ("u", "v")]
      else:
        u, v = [uv["res"][x_i] for x_i in ("u", "v")]
      if class_name == 'DDPoisson2D':
        dx[x_k] = ops_uv[x_k]["D"] + bc_f[x_k]["D"] - force["res"][x_k]
      else:
        bc_f[x_k]["A"]["x"] = bkd.to_backend(bc_f[x_k]["A"]["x"])
        bc_f[x_k]["A"]["y"] = bkd.to_backend(bc_f[x_k]["A"]["y"])
        bc_f[x_k]["D"] = bkd.to_backend(bc_f[x_k]["D"])

        adv_act_x = ops_uv[x_k]["Ax"] - bc_f[x_k]["A"]["x"]
        adv_act_y = ops_uv[x_k]["Ay"] - bc_f[x_k]["A"]["y"]
        dx[x_k] = ops.sp_diag(u) @ adv_act_x \
                + ops.sp_diag(v) @ adv_act_y \
                + ops_uv[x_k]["D"] + bc_f[x_k]["D"]
    return dx

  def compute_jac(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable],
    ops_uv: dtypes.UV_TYPE,
    steady: bool = True,
    dt: float = 0.0,
    jac_fun: Union[callable, None] = None,
    class_name: str = None
  ) -> Dict[str, sp.spmatrix]:
    # Compute
    jac_fun = self._compute_jac if (jac_fun is None) else jac_fun
    jac = jac_fun(uv, elem_states, ops_uv, class_name)
    if bkd.is_torch_backend():
      dt = torch.tensor(dt, dtype=float, device=bkd.device())
    # Backward Euler for integration
    if (not steady):
      for e_k in ("interior", "interface"):
        if bkd.is_torch_backend():
          jac[e_k] = elem_states[e_k].iden_uv.to_dense() - dt * jac[e_k].to_sparse_coo()
        else:
          jac[e_k] = elem_states[e_k].iden_uv - dt * jac[e_k]
    # Return
    return jac

  def _compute_jac(
    self,
    uv: dtypes.UV_TYPE,
    elem_states: Dict[str, callable],
    ops_uv: dtypes.UV_TYPE,
    class_name: str = None
  ) -> Dict[str, sp.spmatrix]:
    jac = {}
    bc_f = elem_states["res"].bc_f

    if class_name == 'DDPoisson2D':
      for e_k in ("interior", "interface"):
        state_k = elem_states[e_k]
        m = state_k.iden.shape[0]
        n = state_k.iden.shape[1]
        jac_xx_k = state_k.ops["D"]
        jac_uu_k = jac_xx_k
        jac_uv_k = sp.csr_matrix((m, n))
        jac_vu_k = sp.csr_matrix((m, n))
        jac_vv_k = jac_xx_k
        jac[e_k] = sp.bmat(
          [[jac_uu_k, jac_uv_k],
          [jac_vu_k, jac_vv_k]],
          format="csr"
        )
    else:

      jac_uu = ops.sp_diag(ops_uv["u"]["Ax"] - bc_f["u"]["A"]["x"], format="coo")
      jac_uv = ops.sp_diag(ops_uv["u"]["Ay"] - bc_f["u"]["A"]["y"], format="coo")
      jac_vu = ops.sp_diag(ops_uv["v"]["Ax"] - bc_f["v"]["A"]["x"], format="coo")
      jac_vv = ops.sp_diag(ops_uv["v"]["Ay"] - bc_f["v"]["A"]["y"], format="coo")
      uv_diag = ops.map_nested_dict(uv["res"], ops.sp_diag, format="coo")


      for e_k in ("interior", "interface"):
        state_k = elem_states[e_k]
        # Should be separate out as an individual function
        if self.compact:
          # --- Blend compact upwind operators based on local velocity ---
          for op_k, vel_field in zip(("Ax", "Ay"), ("u", "v")):
              vel = uv["res"][vel_field]
              if bkd.is_torch_backend():
                pos_mask = vel.ge(0.0).to(dtype=vel.dtype)
                neg_mask = vel.lt(0.0).to(dtype=vel.dtype)
              else:
                pos_mask = (vel >= 0).astype(float)
                neg_mask = 1.0 - pos_mask

              A_pos = state_k.ops[f"{op_k}_pos"]
              A_neg = state_k.ops[f"{op_k}_neg"]
              if bkd.is_torch_backend():
                A_pos = A_pos.to_dense() if A_pos.layout != torch.strided else A_pos
                A_neg = A_neg.to_dense() if A_neg.layout != torch.strided else A_neg
                A_blend = (
                  pos_mask.unsqueeze(1) * A_pos
                  + neg_mask.unsqueeze(1) * A_neg
                )
              else:
                A_blend = sp.diags(pos_mask) @ A_pos + sp.diags(neg_mask) @ A_neg
              state_k.ops[op_k] = A_blend  # overwrite the value in keys Ax and Ay

        if bkd.is_torch_backend():
          jac_xx_k = uv_diag["u"] @ state_k.ops["Ax"].to_dense() \
                  + uv_diag["v"] @ state_k.ops["Ay"].to_dense() \
                  + state_k.ops["D"]


        else:
          jac_xx_k = uv_diag["u"] @ state_k.ops["Ax"] \
                  + uv_diag["v"] @ state_k.ops["Ay"] \
                  + state_k.ops["D"]
        # TODO: fix -- coo -> csr conversion for matmul/addmm
        iden = state_k.iden.to_dense() if bkd.is_torch_backend() else state_k.iden
        jac_uu_k = jac_uu @ iden + jac_xx_k
        jac_uv_k = jac_uv @ iden
        jac_vu_k = jac_vu @ iden
        jac_vv_k = jac_vv @ iden + jac_xx_k
        jac[e_k] = [[jac_uu_k, jac_uv_k],
                    [jac_vu_k, jac_vv_k]]
        if bkd.is_torch_backend():
          jac[e_k] = bkd.torch_bmat_new(jac[e_k], format="coo")
        else:
          jac[e_k] = sp.bmat(jac[e_k], format="csr")
    return jac

  # Residual/Jacobian - Constraints
  # -----------------------------------
  def compute_cres_cjac(
    self,
    uv: dtypes.UV_TYPE
  ) -> dtypes.RES_JAC_TYPE:
    cat_func = torch.cat if bkd.is_torch_backend() else np.concatenate
    cx = cat_func([uv["interface"][x_k] for x_k in ("u", "v")])
    cres = self.cmat["interface"] @ cx
    return cres, self.cmat

  # KKT system
  # -----------------------------------
  def assemble_kkt(
    self,
    res: np.ndarray,
    cres: np.ndarray,
    lambdas: np.ndarray,
    jac: Dict[str, Union[np.ndarray, sp.spmatrix]],
    cjac: Dict[str, Union[np.ndarray, sp.spmatrix]],
    scaling: float
  ) -> dtypes.KKT_TYPE:
    # To sparse
    if bkd.is_torch_backend():
      jac = ops.map_nested_dict(jac, bkd.to_sp_coo_backend)
      cjac = ops.map_nested_dict(cjac, bkd.to_sp_coo_backend)
    else:
      jac = ops.map_nested_dict(jac, bkd.to_sparse)
      cjac = ops.map_nested_dict(cjac, bkd.to_sparse)
    cat_func = torch.cat if bkd.is_torch_backend() else np.concatenate
    hstack_f = torch.hstack if bkd.is_torch_backend() else sp.hstack

    # Residual
    if bkd.is_torch_backend():
      # .T operator does not work for sparse tensors
      interior_t = torch.t(jac["interior"]).to_dense()
      interface_t = torch.t(jac["interface"]).to_dense()
      cjac_interface_t = torch.t(cjac["interface"]).to_dense()
      res = cat_func([
        scaling*(interior_t@res),
        scaling*(interface_t@res) + cjac_interface_t@lambdas
      ])
    else:
      res = cat_func([
        scaling*(jac["interior"].T@res),
        scaling*(jac["interface"].T@res) + cjac["interface"].T@lambdas
      ])
    if bkd.is_torch_backend():
      # Constraints
      cjac = hstack_f([cjac["interior"].to_sparse_coo(), cjac["interface"].to_sparse_coo()])
      # Hessian
      jac = hstack_f([jac["interior"].to_sparse_coo(), jac["interface"].to_sparse_coo()])
      hess = scaling*(torch.t(jac)@jac)

    else:
      # Constraints
      cjac = hstack_f([cjac["interior"], cjac["interface"]])
      # Hessian
      jac = hstack_f([jac["interior"], jac["interface"]])
      hess = scaling*(jac.T@jac)
    return res, cres, hess, cjac
