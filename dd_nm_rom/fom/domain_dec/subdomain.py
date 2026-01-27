import numpy as np
import scipy.sparse as sp

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

                    pos_mask = (vel >= 0).astype(float)
                    neg_mask = 1-pos_mask #(vel < 0)
                    P = sp.diags(pos_mask)  # shape (n, n)
                    N = sp.diags(neg_mask)  # shape (n, n)

                    A_pos = elem_states[e_k].ops[f"{op_k}_pos"]
                    A_neg = elem_states[e_k].ops[f"{op_k}_neg"]

                    A_blend = P @ A_pos + N @ A_neg

                    op_v += A_blend @ uv[e_k][x_k]

                elif op_k not in ("Ax_pos", "Ax_neg", "Ay_pos", "Ay_neg"):
                    # Standard case
                    if isinstance(op_i, dict):
                        op_i = op_i[x_k]
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
    # Backward Euler for integration
    if (not steady):
      for e_k in ("interior", "interface"):
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
      jac_uu = ops.sp_diag(ops_uv["u"]["Ax"] - bc_f["u"]["A"]["x"])
      jac_uv = ops.sp_diag(ops_uv["u"]["Ay"] - bc_f["u"]["A"]["y"])
      jac_vu = ops.sp_diag(ops_uv["v"]["Ax"] - bc_f["v"]["A"]["x"])
      jac_vv = ops.sp_diag(ops_uv["v"]["Ay"] - bc_f["v"]["A"]["y"])
      uv_diag = ops.map_nested_dict(uv["res"], ops.sp_diag)

      for e_k in ("interior", "interface"):
        state_k = elem_states[e_k]
        # Should be separate out as an individual function
        if self.compact:
          # --- Blend compact upwind operators based on local velocity ---
          for op_k, vel_field in zip(("Ax", "Ay"), ("u", "v")):
              vel = uv["res"][vel_field]
              pos_mask = (vel >= 0).astype(float)
              neg_mask = 1.0 - pos_mask
              P = sp.diags(pos_mask)
              N = sp.diags(neg_mask)

              A_pos = state_k.ops[f"{op_k}_pos"]
              A_neg = state_k.ops[f"{op_k}_neg"]
              A_blend = P @ A_pos + N @ A_neg
              state_k.ops[op_k] = A_blend  # overwrite the value in keys Ax and Ay

        jac_xx_k = uv_diag["u"] @ state_k.ops["Ax"] \
                + uv_diag["v"] @ state_k.ops["Ay"] \
                + state_k.ops["D"]
        jac_uu_k = jac_uu @ state_k.iden + jac_xx_k
        jac_uv_k = jac_uv @ state_k.iden
        jac_vu_k = jac_vu @ state_k.iden
        jac_vv_k = jac_vv @ state_k.iden + jac_xx_k
        jac[e_k] = sp.bmat(
          [[jac_uu_k, jac_uv_k],
          [jac_vu_k, jac_vv_k]],
          format="csr"
        )
    return jac

  # Residual/Jacobian - Constraints
  # -----------------------------------
  def compute_cres_cjac(
    self,
    uv: dtypes.UV_TYPE
  ) -> dtypes.RES_JAC_TYPE:
    cx = np.concatenate([uv["interface"][x_k] for x_k in ("u", "v")])
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
    jac = ops.map_nested_dict(jac, bkd.to_sparse)
    cjac = ops.map_nested_dict(cjac, bkd.to_sparse)
    # Residual
    res = np.concatenate([
      scaling*(jac["interior"].T@res),
      scaling*(jac["interface"].T@res) + cjac["interface"].T@lambdas
    ])
    # Constraints
    cjac = sp.hstack([cjac["interior"], cjac["interface"]])
    # Hessian
    jac = sp.hstack([jac["interior"], jac["interface"]])
    hess = scaling*(jac.T@jac)
    return res, cres, hess, cjac
