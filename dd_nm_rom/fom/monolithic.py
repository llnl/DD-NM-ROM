import numpy as np
import scipy.sparse as sp


from time import time
from dd_nm_rom import ops, solvers
from dd_nm_rom import field as field_mod
from typing import Dict, List, Tuple, Union

from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import DiffOperators
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

# Data types
RES_JAC_TYPE = Tuple[np.ndarray, sp.spmatrix]
UV_TYPE = Dict[str, Union[np.ndarray, sp.spmatrix]]
SOL_TYPE = Tuple[UV_TYPE, Union[np.ndarray, List[np.ndarray]], bool]


class Burgers2D(object):
  """
  Generate FOM for 2D Burgers equation solutions.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float,
    upwind: bool = False,
    upwind_order: int =1,
    compact: bool = False
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity
    self.nu = nu
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Scheme
    self.upwind = upwind
    self.upwind_order = upwind_order
    self.compact = compact

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=self.upwind,
      upwind_order=self.upwind_order
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval()
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    if self.compact:
      res, jac = self.compute_res_jac_compact(x)
    else:
      res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)
    # Action of advection operator on vectors
    adv_act = {}
    for axis in ("x", "y"):
      adv_act[axis] = {}
      for k in ("u", "v"):
        adv_act[axis][k] = self.ops[f"A{axis}"] @ uv[k] \
                         - self.bc_f[k]["A"][axis]
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = uv_diag["u"] @ adv_act["x"][k] \
           + uv_diag["v"] @ adv_act["y"][k] \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"]
      dx.append(dx_k)
    res = np.concatenate(dx)
    # Compute Jacobian
    jac_xx = uv_diag["u"] @ self.ops["Ax"] \
           + uv_diag["v"] @ self.ops["Ay"] \
           + self.ops["D"]
    jac_uu = ops.sp_diag(adv_act["x"]["u"]) + jac_xx
    jac_uv = ops.sp_diag(adv_act["y"]["u"])
    jac_vu = ops.sp_diag(adv_act["x"]["v"])
    jac_vv = ops.sp_diag(adv_act["y"]["v"]) + jac_xx
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
  # ===================================
  def solve(
    self,
    x0: Union[np.ndarray, None] = None,
    dt: float = 0.0,
    nt: int = 1,
    steady: bool = True,
    tol: float = 1e-8,
    maxit: int = 50,
    stepsize_min: float = 1e-10,
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton"s method.
    """
    self.is_built()
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime["total"] += time()-start
    # Initialize solver
    solver = solvers.Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged

  def compute_res_jac_compact(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    """Compute residual and jacobian for the compact upwind scheme
    """
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)
    # Action of advection operator on vectors
    adv_act = {}
    for axis in ("x", "y"):
      # Determine which velocity component to check
      vel_component = "u" if axis == "x" else "v"
      vel = uv[vel_component]

      # Create masks for positive and negative velocities
      pos_mask = (vel >= 0)
      neg_mask = (vel < 0)

      adv_act[axis] = {}
      for k in ("u", "v"):
        # Apply both operators to the solution vector
        pos_result = self.ops[f"A{axis}_pos"] @ uv[k]
        neg_result = self.ops[f"A{axis}_neg"] @ uv[k]

        if np.all(pos_mask):
            result = pos_result
        elif np.all(neg_mask):
            result = neg_result
        else:
            result = np.zeros_like(pos_result)
            result[pos_mask] = pos_result[pos_mask]
            result[neg_mask] = neg_result[neg_mask]

        # Apply boundary condition adjustments
        adv_act[axis][k] = result - self.bc_f[k]["A"][axis]

    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = uv_diag["u"] @ adv_act["x"][k] \
           + uv_diag["v"] @ adv_act["y"][k] \
           + self.ops["D"] @ uv[k] + self.bc_f[k]["D"]
      dx.append(dx_k)
    res = np.concatenate(dx)

    # Compute Jacobian with direction-dependent operators
    jac_operators = {}
    for axis in ("x", "y"):
      vel_component = "u" if axis == "x" else "v"
      vel = uv[vel_component]
      pos_mask = (vel >= 0).astype(float)
      neg_mask = 1-pos_mask #(vel < 0)

      # Diagonal selection matrices
      P = sp.diags(pos_mask)  # shape (n, n)
      N = sp.diags(neg_mask)  # shape (n, n)

      A_pos = self.ops[f"A{axis}_pos"]
      A_neg = self.ops[f"A{axis}_neg"]

      # Efficient blending
      A_blend = P @ A_pos + N @ A_neg

      jac_operators[f"A{axis}"] = A_blend

    # Compute Jacobian with blended operators
    jac_xx = uv_diag["u"] @ jac_operators["Ax"] \
          + uv_diag["v"] @ jac_operators["Ay"] \
          + self.ops["D"]

    jac_uu = ops.sp_diag(adv_act["x"]["u"]) + jac_xx
    jac_uv = ops.sp_diag(adv_act["y"]["u"])
    jac_vu = ops.sp_diag(adv_act["x"]["v"])
    jac_vv = ops.sp_diag(adv_act["y"]["v"]) + jac_xx
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    return res, jac

class Poisson2D(object):
  """
  Generate FOM for 2D Poisson equation solutions.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mesh: mesh_mod.MESH_TYPES,
    nu: float
  ) -> None:
    # Mesh
    self.mesh = mesh
    self.mesh.is_built()
    # Viscosity
    self.nu = nu
    # Integration
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Runtime
    self.runtime = {k: 0.0 for k in ("total", "lin_solve", "res_jac")}
    self.built = False
    # Force term in the Poisson
    self.f = None

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "FOM model not built. Please, call 'build' method first."
      )

  def build(
    self,
    field: field_mod.FIELD_TYPES,
    force: np.ndarray = None
  ) -> None:
    # BC
    self.bc = self.build_bc(field)
    self.bc_f = self.bc.f
    # Operators
    self.diff_ops = DiffOperators(
      nu=self.nu,
      bc=self.bc,
      mesh=self.mesh,
      upwind=False,
      upwind_order=1
    )
    self.diff_ops.build()
    self.ops = self.diff_ops.ops
    self.ops_names = list(self.ops.keys())
    # Identities
    self.iden = sp.eye(self.mesh.nxy).tocsr()
    self.iden_uv = sp.eye(self.get_ndof()).tocsr()

    # Set forcing term
    if force is None:
      # Default to zero force if not provided
      self.f = np.zeros(self.get_ndof())
    else:
      self.f = force
    self.built = True

  def build_bc(
    self,
    field: field_mod.FIELD_TYPES
  ) -> bc_mod.BC_TYPES:
    if (field.bc_type == "dirichlet"):
      bc_cls = bc_mod.DirichletBC
    elif (field.bc_type == "neumann"):
      bc_cls = bc_mod.NeumannBC
    else:
      bc_cls = bc_mod.PeriodicBC
    bc = bc_cls(
      nu=self.nu,
      mesh=self.mesh,
      funval=field.get_bc_funval(),
      advection=False
    )
    bc.build()
    return bc

  def get_ndof(self) -> int:
    return 2*self.mesh.nxy

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    start = time()
    res, jac = self.compute_res_jac(x)
    # Backward Euler for integration
    if (not self.steady):
      res = x - self.x_old - self.dt*res
      jac = self.iden_uv - self.dt*jac
    delta = time()-start
    self.runtime["total"] += delta
    self.runtime["res_jac"] += delta
    return res, jac

  def compute_res_jac(
    self,
    x: np.ndarray
  ) -> RES_JAC_TYPE:
    # Extract u and v
    uv, uv_diag = self.extract_uv(x, diag=True)

    # Split force into u and v components
    f_uv = {"u": self.f[:self.mesh.nxy], "v": self.f[self.mesh.nxy:]}
#   # Action of advection operator on vectors
#   adv_act = {}
#   for axis in ("x", "y"):
#     adv_act[axis] = {}
#     for k in ("u", "v"):
#       adv_act[axis][k] = self.ops[f"A{axis}"] @ uv[k] \
#                        - self.bc_f[k]["A"][axis]
    # Compute residual
    dx = []
    for k in ("u", "v"):
      dx_k = self.ops["D"] @ uv[k] + self.bc_f[k]["D"] - f_uv[k]
      dx.append(dx_k)
    res = np.concatenate(dx)
    # Compute Jacobian
    jac_xx = self.ops["D"]
    jac_uu = jac_xx
    jac_uv = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac_vu = sp.csr_matrix((self.mesh.nxy, self.mesh.nxy))
    jac_vv = jac_xx
    jac = sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )
    return res, jac

  def extract_uv(
    self,
    x: np.ndarray,
    diag: bool = True
  ) -> Union[Tuple[UV_TYPE, UV_TYPE], UV_TYPE]:
    uv = {"u": x[:self.mesh.nxy], "v": x[self.mesh.nxy:]}
    if diag:
      uv_diag = ops.map_nested_dict(uv, ops.sp_diag)
      return uv, uv_diag
    else:
      return uv

  # Solving
  # ===================================
  def solve(
    self,
    x0: Union[np.ndarray, None] = None,
    dt: float = 0.0,
    nt: int = 1,
    steady: bool = True,
    tol: float = 1e-8,
    maxit: int = 50,
    stepsize_min: float = 1e-10,
    iostep: int = 1,
    verbose: bool = False
  ) -> SOL_TYPE:
    """
    Solves for the u and v states of the FOM using Newton"s method.
    """
    self.is_built()
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime["total"] += time()-start
    # Initialize solver
    solver = solvers.Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      iostep=iostep,
      verbose=verbose
    )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    # Return solution
    uv = self.extract_uv(x, diag=False)
    converged = True if (flag[-1] == 0) else False
    return uv, res, converged
