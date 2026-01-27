import numpy as np
import scipy.sparse as sp

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd
from typing import Dict, List, Tuple, Union

from . import dtypes
from .indices import DDIndices
from .subdomain import Subdomain
from .state import SubdomainElementState


class DDBurgers2D(object):
  """
  Class to compute domain decomposition model from steady-state 2D Burgers FOM.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    monolithic: dtypes.FOM_TYPE,
    n_constraints_weak: int = 1,
    constraint_type: str = "strong",
    scaling: float = 1.0
  ) -> None:
    # FOM monolithic
    # -------------
    self.monolithic = monolithic
    for k in ("runtime", "mesh", "compact"):
      setattr(self, k, getattr(self.monolithic, k))
    if (self.mesh.name != "MeshDD"):
      raise ValueError(
        "The mesh needs to be of type 'MeshDD' using the DD-FOM model."
      )
    # > Scaling factor for residual
    self.scaling = self.mesh.hxy if (scaling <= 0) else scaling
    # Constraints
    # -------------
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = int(n_constraints_weak)
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Force term in the Poisson
    self.f = None
    # Control variables
    # -------------
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    self.monolithic.is_built()
    if (not self.built):
      raise ValueError(
        "DD-FOM model not built. Please, call 'build' method first."
      )

  def build(self) -> None:
    self.monolithic.is_built()
    # DD-FOM subdomains indices
    self.dd_indices = DDIndices(self.monolithic)
    self.dd_indices.build()
    # Constraints
    self.cmat = self.assemble_cmat()
    # DD-FOM subdomains
    self.subdomains = []
    for s in range(self.mesh.n_sub):
      cmat_s, indices_s = {}, {}
      for e_k in ("res", "interior", "interface"):
        indices_s[e_k] = getattr(self.dd_indices, e_k)[s]
        if (e_k != "res"):
          cmat_s[e_k] = self.cmat[e_k][s]
      self.subdomains.append(
        Subdomain(
          identifier=tuple(self.mesh.sub_combs[s]),
          monolithic=self.monolithic,
          nodes_ind=indices_s,
          cmat=cmat_s,
          ports=self.dd_indices.sub_to_ports[s],
          port_to_nodes=self.dd_indices.port_to_nodes,
          scaling=self.scaling
        )
      )
    # Update control variables
    self.built = True

  def get_ndof(self) -> int:
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.elem_states[e_k].n_nodes_state
    ndof *= 2
    ndof += self.n_constraints
    return ndof

  # Constraint matrices
  # -----------------------------------
  def assemble_cmat(self) -> dtypes.CMAT_TYPE:
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      n_ports = len(self.dd_indices.port_to_nodes[p])
      self.n_constraints += (len(subs_p)-1) * n_ports
    # Assemble constraints matrices
    cmat = {
      "interior": self.init_cmat(element="interior"),
      "interface": self.assemble_cmat_intf()
    }
    # > Make constraint matrices block diagonal for u and v components
    for (e_k, cmat_k) in cmat.items():
      cmat[e_k] = [sp.block_diag([m, m]) for m in cmat_k]
    self.n_constraints *= 2
    # Convert to weak constraints
    if (self.constraint_type == "weak"):
      cmat, self.n_constraints = self.assemble_cmat_weak(
        cmat=cmat,
        n_constraints_weak=self.n_constraints_weak,
        n_constraints=self.n_constraints
      )
    return cmat

  def init_cmat(
    self,
    element: str
  ) -> List[sp.spmatrix]:
    cmat = []
    for nodes_ind in getattr(self.dd_indices, element):
      cmat.append(sp.coo_matrix((self.n_constraints, len(nodes_ind))))
    return cmat

  def assemble_cmat_intf(self) -> List[sp.spmatrix]:
    # Initialize matrices
    cmat = self.init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      port_nodes = self.dd_indices.port_to_nodes[p]
      port_size = port_nodes.size
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          s = subs_p[l]
          intf_nodes = self.dd_indices.interface[s]
          col = np.where(np.isin(intf_nodes, port_nodes))[0]
          row = np.arange(port_size) + shift
          dat = (-1)**j * np.ones(port_size)
          cmat[s].col = np.concatenate((cmat[s].col, col))
          cmat[s].row = np.concatenate((cmat[s].row, row))
          cmat[s].data = np.concatenate((cmat[s].data, dat))
        shift += port_size
    return cmat

  def assemble_cmat_weak(
    self,
    cmat: dtypes.CMAT_TYPE,
    n_constraints_weak: int,
    n_constraints: int
  ) -> Tuple[dtypes.CMAT_TYPE, int]:
    n_constraints_weak = max(n_constraints_weak, 1)
    n_constraints_weak = min(n_constraints_weak, n_constraints)
    rgen = np.random.default_rng(bkd.seed())
    rmat = rgen.standard_normal((n_constraints_weak, n_constraints))
    for e_k in ("interior", "interface"):
      cmat[e_k] = [rmat @ m for m in cmat[e_k]]
    cmat = ops.map_nested_dict(cmat, bkd.to_sparse)
    return cmat, n_constraints_weak

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> dtypes.RES_JAC_TYPE:
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res, hess, cjac = [], [], []
    cres = np.zeros(self.n_constraints)
    runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    uv, lambdas = self.assemble_sol(x, map_on_res=False)
    if (not self.steady):
      uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False)
    runtime += (time()-start) / self.mesh.n_sub
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    uv_s_old = None
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      uv_s = self.extract_uv_sub_from_dict(uv, s)
      if (not self.steady):
        uv_s_old = self.extract_uv_sub_from_dict(uv_old, s)
      # > Compute quantities needed for KKT system
      res_s, cres_s, hess_s, cjac_s = sub.res_jac(
        uv=uv_s,
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        uv_old=uv_s_old
      )
      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      res.append(res_s)
      cres += cres_s
      cjac.append(cjac_s)
      hess.append(hess_s)
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    res, jac = self.assemble_kkt(res, cres, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, jac

  def assemble_sol(
    self,
    x: np.ndarray,
    map_on_res: bool = False
  ) -> Tuple[dtypes.UV_TYPE, np.ndarray]:
    shape = [self.mesh.nxy]
    if (x.ndim == 2):
      shape.append(x.shape[1])
    uv = self.init_uv(shape=shape, map_on_res=map_on_res)
    # Loop over subdomains
    si = 0
    for sub in self.subdomains:
      # Loop over elements
      for e_k in ("interior", "interface"):
        state_k = sub.elem_states[e_k]
        ei = si + 2*state_k.n_nodes_state
        uv = self.extract_uv_sub_from_vec(
          uv=uv,
          uv_i=x[si:ei],
          elem_state=state_k,
          map_on_res=map_on_res
        )
        si = ei
    lambdas = x[-self.n_constraints:]
    return uv, lambdas

  def init_uv(
    self,
    shape: List[int],
    map_on_res: bool = False
  ) -> dtypes.UV_TYPE:
    uv = {}
    for e_k in ("interior", "interface"):
      uv[e_k] = {}
      for x_k in ("u", "v"):
        uv[e_k][x_k] = []
    if map_on_res:
      uv["res"] = {}
      for x_k in ("u", "v"):
        uv["res"][x_k] = np.zeros(shape)
    return uv

  def extract_uv_sub_from_vec(
    self,
    uv: dtypes.UV_TYPE,
    uv_i: np.ndarray,
    elem_state: SubdomainElementState,
    map_on_res: bool = False
  ) -> dtypes.UV_TYPE:
    size = elem_state.n_nodes_state
    indices = elem_state.nodes_state
    # u velocity
    u = uv_i[:size]
    uv[elem_state.name]["u"].append(u)
    if map_on_res:
      uv["res"]["u"][indices] = u
    # v velocity
    v = uv_i[size:]
    uv[elem_state.name]["v"].append(v)
    if map_on_res:
      uv["res"]["v"][indices] = v
    return uv

  def extract_uv_sub_from_dict(
    self,
    uv: dtypes.UV_TYPE,
    index: int
  ) -> dtypes.UV_TYPE:
    uv_s = {}
    for e_k in ("interior", "interface"):
      uv_s[e_k] = {}
      for x_k in ("u", "v"):
        uv_s[e_k][x_k] = uv[e_k][x_k][index]
    return uv_s

  def assemble_kkt(
    self,
    res: np.ndarray,
    cres: np.ndarray,
    cjac: sp.spmatrix,
    hess: sp.spmatrix
  ) -> dtypes.RES_JAC_TYPE:
    # > Residual
    res.append(cres)
    res = np.concatenate(res)
    # > Constraints Jacobian
    cjac = sp.hstack(cjac)
    # > Hessians
    hess = sp.block_diag(hess)
    # > Full Jacobian
    jac = sp.bmat(
      [[hess, cjac.T],
       [cjac,   None]],
      format="csr"
    )
    return res, jac

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
  ) -> dtypes.SOL_TYPE:
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
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, lambdas, res, converged

  def get_init_sol(
    self,
    x: np.ndarray
  ) -> np.ndarray:
    self.is_built()
    # Map init sol on elements
    x = self.map_sol_on_elements(x.reshape(1,-1))
    # Initial guess
    x_dd = []
    # Loop over subdomains
    for s in range(self.mesh.n_sub):
      # Loop over elements
      for e_k in ("interior", "interface"):
        x_dd.append(x[e_k][s].reshape(-1))
    x_dd.append(np.zeros(self.n_constraints))
    return np.concatenate(x_dd)

  def map_sol_on_elements(
    self,
    solutions: np.ndarray,
    map_on_ports: bool = False
  ) -> Dict[str, List[np.ndarray]]:
    """
    Map a monolithic solution to domain decomposition elements
    """
    self.is_built()
    if (solutions.shape[1] != self.monolithic.get_ndof()):
      solutions = solutions.T
    data = {}
    for e_k in ("res", "interior", "interface"):
      data[e_k] = []
      for sub in self.subdomains:
        indices = sub.elem_states[e_k].nodes_state
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data[e_k].append(solutions[:,indices])
    if map_on_ports:
      data["port"] = []
      for p in self.dd_indices.ports:
        indices = self.dd_indices.port_to_nodes[p]
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data["port"].append(solutions[:,indices])
    return data

class DDPoisson2D(object):
  """
  Class to compute domain decomposition model from 2D Poisson FOM.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    monolithic: dtypes.FOM_TYPE,
    n_constraints_weak: int = 1,
    constraint_type: str = "strong",
    scaling: float = 1.0
  ) -> None:
    # FOM monolithic
    # -------------
    self.monolithic = monolithic
    for k in ("runtime", "mesh"):
      setattr(self, k, getattr(self.monolithic, k))
    if (self.mesh.name != "MeshDD"):
      raise ValueError(
        "The mesh needs to be of type 'MeshDD' using the DD-FOM model."
      )
    # > Scaling factor for residual
    self.scaling = self.mesh.hxy if (scaling <= 0) else scaling
    # Constraints
    # -------------
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = int(n_constraints_weak)
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0
    # Force term in the Poisson
    self.f = None
    # Control variables
    # -------------
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    self.monolithic.is_built()
    if (not self.built):
      raise ValueError(
        "DD-FOM model not built. Please, call 'build' method first."
      )

  def build(self) -> None:
    self.monolithic.is_built()
    # DD-FOM subdomains indices
    self.dd_indices = DDIndices(self.monolithic)
    self.dd_indices.build()
    # Constraints
    self.cmat = self.assemble_cmat()
    # DD-FOM subdomains
    self.subdomains = []
    for s in range(self.mesh.n_sub):
      cmat_s, indices_s = {}, {}
      for e_k in ("res", "interior", "interface"):
        indices_s[e_k] = getattr(self.dd_indices, e_k)[s]
        if (e_k != "res"):
          cmat_s[e_k] = self.cmat[e_k][s]
      self.subdomains.append(
        Subdomain(
          identifier=tuple(self.mesh.sub_combs[s]),
          monolithic=self.monolithic,
          nodes_ind=indices_s,
          cmat=cmat_s,
          ports=self.dd_indices.sub_to_ports[s],
          port_to_nodes=self.dd_indices.port_to_nodes,
          scaling=self.scaling
        )
      )
    # Update control variables
    self.built = True

  def get_ndof(self) -> int:
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.elem_states[e_k].n_nodes_state
    ndof *= 2
    ndof += self.n_constraints
    return ndof

  # Constraint matrices
  # -----------------------------------
  def assemble_cmat(self) -> dtypes.CMAT_TYPE:
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      n_ports = len(self.dd_indices.port_to_nodes[p])
      self.n_constraints += (len(subs_p)-1) * n_ports
    # Assemble constraints matrices
    cmat = {
      "interior": self.init_cmat(element="interior"),
      "interface": self.assemble_cmat_intf()
    }
    # > Make constraint matrices block diagonal for u and v components
    for (e_k, cmat_k) in cmat.items():
      cmat[e_k] = [sp.block_diag([m, m]) for m in cmat_k]
    self.n_constraints *= 2
    # Convert to weak constraints
    if (self.constraint_type == "weak"):
      cmat, self.n_constraints = self.assemble_cmat_weak(
        cmat=cmat,
        n_constraints_weak=self.n_constraints_weak,
        n_constraints=self.n_constraints
      )
    return cmat

  def init_cmat(
    self,
    element: str
  ) -> List[sp.spmatrix]:
    cmat = []
    for nodes_ind in getattr(self.dd_indices, element):
      cmat.append(sp.coo_matrix((self.n_constraints, len(nodes_ind))))
    return cmat

  def assemble_cmat_intf(self) -> List[sp.spmatrix]:
    # Initialize matrices
    cmat = self.init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_indices.port_to_subs.items():
      port_nodes = self.dd_indices.port_to_nodes[p]
      port_size = port_nodes.size
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          s = subs_p[l]
          intf_nodes = self.dd_indices.interface[s]
          col = np.where(np.isin(intf_nodes, port_nodes))[0]
          row = np.arange(port_size) + shift
          dat = (-1)**j * np.ones(port_size)
          cmat[s].col = np.concatenate((cmat[s].col, col))
          cmat[s].row = np.concatenate((cmat[s].row, row))
          cmat[s].data = np.concatenate((cmat[s].data, dat))
        shift += port_size
    return cmat

  def assemble_cmat_weak(
    self,
    cmat: dtypes.CMAT_TYPE,
    n_constraints_weak: int,
    n_constraints: int
  ) -> Tuple[dtypes.CMAT_TYPE, int]:
    n_constraints_weak = max(n_constraints_weak, 1)
    n_constraints_weak = min(n_constraints_weak, n_constraints)
    rgen = np.random.default_rng(bkd.seed())
    rmat = rgen.standard_normal((n_constraints_weak, n_constraints))
    for e_k in ("interior", "interface"):
      cmat[e_k] = [rmat @ m for m in cmat[e_k]]
    cmat = ops.map_nested_dict(cmat, bkd.to_sparse)
    return cmat, n_constraints_weak

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray
  ) -> dtypes.RES_JAC_TYPE:
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res, hess, cjac = [], [], []
    cres = np.zeros(self.n_constraints)
    runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    uv, lambdas = self.assemble_sol(x, map_on_res=False)
    force, *_ = self.assemble_sol(self.f, map_on_res=False)
    if (not self.steady):
      uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False)
    runtime += (time()-start) / self.mesh.n_sub
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    uv_s_old = None
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      uv_s = self.extract_uv_sub_from_dict(uv, s)
      force_s = self.extract_uv_sub_from_dict(force, s)
      if (not self.steady):
        uv_s_old = self.extract_uv_sub_from_dict(uv_old, s)
      # > Compute quantities needed for KKT system
      res_s, cres_s, hess_s, cjac_s = sub.res_jac(
        uv=uv_s,
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        uv_old=uv_s_old,
        force=force_s,
        class_name = self.__class__.__name__
      )
      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      res.append(res_s)
      cres += cres_s
      cjac.append(cjac_s)
      hess.append(hess_s)
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    res, jac = self.assemble_kkt(res, cres, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, jac

  def assemble_sol(
    self,
    x: np.ndarray,
    map_on_res: bool = False
  ) -> Tuple[dtypes.UV_TYPE, np.ndarray]:
    shape = [self.mesh.nxy]
    if (x.ndim == 2):
      shape.append(x.shape[1])
    uv = self.init_uv(shape=shape, map_on_res=map_on_res)
    # Loop over subdomains
    si = 0
    for sub in self.subdomains:
      # Loop over elements
      for e_k in ("interior", "interface"):
        state_k = sub.elem_states[e_k]
        ei = si + 2*state_k.n_nodes_state
        uv = self.extract_uv_sub_from_vec(
          uv=uv,
          uv_i=x[si:ei],
          elem_state=state_k,
          map_on_res=map_on_res
        )
        si = ei
    lambdas = x[-self.n_constraints:]
    return uv, lambdas

  def init_uv(
    self,
    shape: List[int],
    map_on_res: bool = False
  ) -> dtypes.UV_TYPE:
    uv = {}
    for e_k in ("interior", "interface"):
      uv[e_k] = {}
      for x_k in ("u", "v"):
        uv[e_k][x_k] = []
    if map_on_res:
      uv["res"] = {}
      for x_k in ("u", "v"):
        uv["res"][x_k] = np.zeros(shape)
    return uv

  def extract_uv_sub_from_vec(
    self,
    uv: dtypes.UV_TYPE,
    uv_i: np.ndarray,
    elem_state: SubdomainElementState,
    map_on_res: bool = False
  ) -> dtypes.UV_TYPE:
    size = elem_state.n_nodes_state
    indices = elem_state.nodes_state
    # u velocity
    u = uv_i[:size]
    uv[elem_state.name]["u"].append(u)
    if map_on_res:
      uv["res"]["u"][indices] = u
    # v velocity
    v = uv_i[size:]
    uv[elem_state.name]["v"].append(v)
    if map_on_res:
      uv["res"]["v"][indices] = v
    return uv

  def extract_uv_sub_from_dict(
    self,
    uv: dtypes.UV_TYPE,
    index: int
  ) -> dtypes.UV_TYPE:
    uv_s = {}
    for e_k in ("interior", "interface"):
      uv_s[e_k] = {}
      for x_k in ("u", "v"):
        uv_s[e_k][x_k] = uv[e_k][x_k][index]
    return uv_s

  def assemble_kkt(
    self,
    res: np.ndarray,
    cres: np.ndarray,
    cjac: sp.spmatrix,
    hess: sp.spmatrix
  ) -> dtypes.RES_JAC_TYPE:
    # > Residual
    res.append(cres)
    res = np.concatenate(res)
    # > Constraints Jacobian
    cjac = sp.hstack(cjac)
    # > Hessians
    hess = sp.block_diag(hess)
    # > Full Jacobian
    jac = sp.bmat(
      [[hess, cjac.T],
       [cjac,   None]],
      format="csr"
    )
    return res, jac

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
  ) -> dtypes.SOL_TYPE:
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
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, lambdas, res, converged

  def get_force(
    self,
    x: np.ndarray
  ) -> np.ndarray:
    self.is_built()
    # Map init sol on elements
    x = self.map_sol_on_elements(x.reshape(1,-1))
    # Initial guess
    x_dd = []
    # Loop over subdomains
    for s in range(self.mesh.n_sub):
      # Loop over elements
      for e_k in ("interior", "interface"):
        x_dd.append(x[e_k][s].reshape(-1))
    x_dd.append(np.zeros(self.n_constraints))
    self.f = np.concatenate(x_dd)
    return self.f

  def map_sol_on_elements(
    self,
    solutions: np.ndarray,
    map_on_ports: bool = False
  ) -> Dict[str, List[np.ndarray]]:
    """ 
    Map a monolithic solution to domain decomposition elements
    """
    self.is_built()
    print(self.monolithic.get_ndof())
    if (solutions.shape[1] != self.monolithic.get_ndof()):
      solutions = solutions.T
    print(solutions.shape)
    data = {}
    for e_k in ("res", "interior", "interface"):
      data[e_k] = []
      for sub in self.subdomains:
        indices = sub.elem_states[e_k].nodes_state
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data[e_k].append(solutions[:,indices])
    if map_on_ports:
      data["port"] = []
      for p in self.dd_indices.ports:
        indices = self.dd_indices.port_to_nodes[p]
        indices = np.concatenate([indices, indices+self.mesh.nxy])
        data["port"].append(solutions[:,indices])
    return data

class DDElasticity2D(object):
    """
    Class to compute domain decomposition model from steady-state 2D Linear
    Elasticity FOM. This model solves for the x-displacement (u) and
    y-displacement (v) fields, constrained by the Lame constants (lambda, mu).
    """

    # Initialization
    # ===================================
    def __init__(
        self,
        monolithic: dtypes.FOM_TYPE,
        lame_lambda: float = 1.0,
        lame_mu: float = 0.8,
        n_constraints_weak: int = 1,
        constraint_type: str = "strong",
        scaling: float = 1.0
    ) -> None:
        # FOM monolithic
        # -------------
        self.monolithic = monolithic
        for k in ("runtime", "mesh"):
            setattr(self, k, getattr(self.monolithic, k))
        if (self.mesh.name != "MeshDD"):
            raise ValueError(
                "The mesh needs to be of type 'MeshDD' using the DD-FOM model."
            )

        # > Physical Lame Constants for Linear Elasticity
        # These are typically defined in the monolithic FOM, but are stored here
        # to ensure the correct context for the model.
        self.lame_lambda = lame_lambda
        self.lame_mu = lame_mu

        # > Scaling factor for residual
        self.scaling = self.mesh.hxy if (scaling <= 0) else scaling
        # Constraints
        # -------------
        self.constraint_type = constraint_type
        if (self.constraint_type not in ("weak", "strong")):
            raise ValueError(
                f"Could not interpret constraint type: '{self.constraint_type}'. " \
                "Valid options are: ['weak', 'strong']."
            )
        self.n_constraints_weak = int(n_constraints_weak)
        # Integration (Elasticity is usually static/steady-state)
        # -------------
        self.steady = True
        self.x_old = None
        self.dt = 0.0
        # Force term (Applied external loads)
        self.f = None
        # Control variables
        # -------------
        self.built = False

    # Building
    # ===================================
    def is_built(self) -> None:
        """Checks if the monolithic FOM and the DD-FOM framework are built."""
        self.monolithic.is_built()
        if (not self.built):
            raise ValueError(
                "DD-FOM model not built. Please, call 'build' method first."
            )

    def build(self) -> None:
        """Builds the DD indices, constraint matrices, and subdomains."""
        self.monolithic.is_built()
        # DD-FOM subdomains indices
        self.dd_indices = DDIndices(self.monolithic)
        self.dd_indices.build()
        # Constraints
        self.cmat = self.assemble_cmat()
        # DD-FOM subdomains
        self.subdomains = []
        for s in range(self.mesh.n_sub):
            cmat_s, indices_s = {}, {}
            for e_k in ("res", "interior", "interface"):
                indices_s[e_k] = getattr(self.dd_indices, e_k)[s]
                if (e_k != "res"):
                    cmat_s[e_k] = self.cmat[e_k][s]
            self.subdomains.append(
                Subdomain(
                    identifier=tuple(self.mesh.sub_combs[s]),
                    monolithic=self.monolithic,
                    nodes_ind=indices_s,
                    cmat=cmat_s,
                    ports=self.dd_indices.sub_to_ports[s],
                    port_to_nodes=self.dd_indices.port_to_nodes,
                    scaling=self.scaling
                )
            )
        # Update control variables
        self.built = True

    def get_ndof(self) -> int:
        """
        Computes the total number of degrees of freedom in the DD system.
        (2 components (u, v) per node + Lagrange multipliers).
        """
        ndof = 0
        for sub in self.subdomains:
            for e_k in ("interior", "interface"):
                # Total nodes * 2 (for u and v displacement components)
                ndof += sub.elem_states[e_k].n_nodes_state
        ndof *= 2
        ndof += self.n_constraints
        return ndof

    # Constraint matrices
    # -----------------------------------
    def assemble_cmat(self) -> dtypes.CMAT_TYPE:
        """Assembles the constraint matrices (C) that enforce continuity 
        of u and v displacements across subdomain interfaces."""
        # Compute total number of constraints (initially for one component)
        self.n_constraints = 0
        for (p, subs_p) in self.dd_indices.port_to_subs.items():
            n_ports = len(self.dd_indices.port_to_nodes[p])
            self.n_constraints += (len(subs_p)-1) * n_ports
        # Assemble constraints matrices
        cmat = {
            "interior": self.init_cmat(element="interior"),
            "interface": self.assemble_cmat_intf()
        }
        # > Make constraint matrices block diagonal for u and v components
        for (e_k, cmat_k) in cmat.items():
            cmat[e_k] = [sp.block_diag([m, m]) for m in cmat_k]
        # Total number of constraints is doubled for two displacement components
        self.n_constraints *= 2
        # Convert to weak constraints if requested
        if (self.constraint_type == "weak"):
            cmat, self.n_constraints = self.assemble_cmat_weak(
                cmat=cmat,
                n_constraints_weak=self.n_constraints_weak,
                n_constraints=self.n_constraints
            )
        return cmat

    def init_cmat(
        self,
        element: str
    ) -> List[sp.spmatrix]:
        """Initializes sparse constraint matrices (C) with correct size."""
        cmat = []
        for nodes_ind in getattr(self.dd_indices, element):
            cmat.append(sp.coo_matrix((self.n_constraints, len(nodes_ind))))
        return cmat

    def assemble_cmat_intf(self) -> List[sp.spmatrix]:
        """Fills the constraint matrices for the interface nodes to enforce
        strong coupling (u_i - u_j = 0, v_i - v_j = 0)."""
        # Initialize matrices
        cmat = self.init_cmat(element="interface")
        # Fill matrices
        shift = 0
        for (p, subs_p) in self.dd_indices.port_to_subs.items():
            port_nodes = self.dd_indices.port_to_nodes[p]
            port_size = port_nodes.size
            for i in range(len(subs_p)-1):
                for (j, l) in enumerate((i,i+1)):
                    s = subs_p[l]
                    intf_nodes = self.dd_indices.interface[s]
                    col = np.where(np.isin(intf_nodes, port_nodes))[0]
                    row = np.arange(port_size) + shift
                    dat = (-1)**j * np.ones(port_size)
                    cmat[s].col = np.concatenate((cmat[s].col, col))
                    cmat[s].row = np.concatenate((cmat[s].row, row))
                    cmat[s].data = np.concatenate((cmat[s].data, dat))
                shift += port_size
        return cmat

    def assemble_cmat_weak(
        self,
        cmat: dtypes.CMAT_TYPE,
        n_constraints_weak: int,
        n_constraints: int
    ) -> Tuple[dtypes.CMAT_TYPE, int]:
        """Applies randomized projection for weak constraints."""
        n_constraints_weak = max(n_constraints_weak, 1)
        n_constraints_weak = min(n_constraints_weak, n_constraints)
        rgen = np.random.default_rng(bkd.seed())
        rmat = rgen.standard_normal((n_constraints_weak, n_constraints))
        for e_k in ("interior", "interface"):
            cmat[e_k] = [rmat @ m for m in cmat[e_k]]
        cmat = ops.map_nested_dict(cmat, bkd.to_sparse)
        return cmat, n_constraints_weak

    # Residual/Jacobian
    # ===================================
    def res_jac(
        self,
        x: np.ndarray
    ) -> dtypes.RES_JAC_TYPE:
        """
        Computes the global residual (R) and Jacobian (J, KKT matrix)
        for the linear elasticity problem.
        """
        runtime = 0.0
        # Initialize
        # -------------
        start = time()
        res, hess, cjac = [], [], []
        cres = np.zeros(self.n_constraints)
        runtime += time()-start
        # Assemble solution vector (displacement u and v)
        # -------------
        start = time()
        uv, lambdas = self.assemble_sol(x, map_on_res=False)
        # Get external force terms (f)
        force, *_ = self.assemble_sol(self.f, map_on_res=False)
        if (not self.steady):
            uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False)
        runtime += (time()-start) / self.mesh.n_sub
        # Loop over subdomains
        # -------------
        runtime_s = 0.0
        uv_s_old = None
        for (s, sub) in enumerate(self.subdomains):
            start_s = time()
            # > Get u and v (x- and y-displacements) for subdomain 's'
            uv_s = self.extract_uv_sub_from_dict(uv, s)
            force_s = self.extract_uv_sub_from_dict(force, s)
            if (not self.steady):
                uv_s_old = self.extract_uv_sub_from_dict(uv_old, s)
            # > Compute quantities needed for KKT system (delegating to Subdomain)
            res_s, cres_s, hess_s, cjac_s = sub.res_jac(
                uv=uv_s,
                lambdas=lambdas,
                steady=self.steady,
                dt=self.dt,
                uv_old=uv_s_old,
                force=force_s,
                class_name = self.__class__.__name__ # Tells subdomain to use Elasticity physics
            )
            runtime_s = max(time()-start_s, runtime_s)
            # > Store subdomain-related quantities
            start = time()
            res.append(res_s)
            cres += cres_s
            cjac.append(cjac_s)
            hess.append(hess_s)
            runtime += time()-start
        runtime += runtime_s
        # Assemble
        # -------------
        start = time()
        res, jac = self.assemble_kkt(res, cres, cjac, hess)
        runtime += time()-start
        self.runtime["total"] += runtime
        self.runtime["res_jac"] += runtime
        return res, jac

    def assemble_sol(
        self,
        x: np.ndarray,
        map_on_res: bool = False
    ) -> Tuple[dtypes.UV_TYPE, np.ndarray]:
        """
        Maps the global solution vector (x) back to the internal uv structure,
        separating u, v, and lambda (Lagrange multipliers).
        """
        shape = [self.mesh.nxy]
        if (x.ndim == 2):
            shape.append(x.shape[1])
        uv = self.init_uv(shape=shape, map_on_res=map_on_res)
        # Loop over subdomains
        si = 0
        for sub in self.subdomains:
            # Loop over elements
            for e_k in ("interior", "interface"):
                state_k = sub.elem_states[e_k]
                ei = si + 2*state_k.n_nodes_state
                uv = self.extract_uv_sub_from_vec(
                    uv=uv,
                    uv_i=x[si:ei],
                    elem_state=state_k,
                    map_on_res=map_on_res
                )
                si = ei
        lambdas = x[-self.n_constraints:]
        return uv, lambdas

    def init_uv(
        self,
        shape: List[int],
        map_on_res: bool = False
    ) -> dtypes.UV_TYPE:
        """Initializes the dictionary structure for displacements (u, v)."""
        uv = {}
        for e_k in ("interior", "interface"):
            uv[e_k] = {}
            for x_k in ("u", "v"):
                uv[e_k][x_k] = []
        if map_on_res:
            uv["res"] = {}
            for x_k in ("u", "v"):
                uv["res"][x_k] = np.zeros(shape)
        return uv

    def extract_uv_sub_from_vec(
        self,
        uv: dtypes.UV_TYPE,
        uv_i: np.ndarray,
        elem_state: SubdomainElementState,
        map_on_res: bool = False
    ) -> dtypes.UV_TYPE:
        """Extracts u and v components for a subdomain from the flat solution vector."""
        size = elem_state.n_nodes_state
        indices = elem_state.nodes_state
        # u displacement (x-component)
        u = uv_i[:size]
        uv[elem_state.name]["u"].append(u)
        if map_on_res:
            uv["res"]["u"][indices] = u
        # v displacement (y-component)
        v = uv_i[size:]
        uv[elem_state.name]["v"].append(v)
        if map_on_res:
            uv["res"]["v"][indices] = v
        return uv

    def extract_uv_sub_from_dict(
        self,
        uv: dtypes.UV_TYPE,
        index: int
    ) -> dtypes.UV_TYPE:
        """Extracts u and v components for a specific subdomain index."""
        uv_s = {}
        for e_k in ("interior", "interface"):
            uv_s[e_k] = {}
            for x_k in ("u", "v"):
                uv_s[e_k][x_k] = uv[e_k][x_k][index]
        return uv_s

    def assemble_kkt(
        self,
        res: np.ndarray,
        cres: np.ndarray,
        cjac: sp.spmatrix,
        hess: sp.spmatrix
    ) -> dtypes.RES_JAC_TYPE:
        """Assembles the final KKT system (Residual vector and Jacobian matrix)."""
        # > Residual
        res.append(cres)
        res = np.concatenate(res)
        # > Constraints Jacobian
        cjac = sp.hstack(cjac)
        # > Hessians (Stiffness matrix for elasticity)
        hess = sp.block_diag(hess)
        # > Full Jacobian (KKT matrix)
        jac = sp.bmat(
            [[hess, cjac.T],
             [cjac,    None]],
            format="csr"
        )
        return res, jac

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
    ) -> dtypes.SOL_TYPE:
        """
        Solves for the u and v displacements using Newton's method.
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
        # Solving (Elasticity is usually steady-state)
        self.steady = bool(steady)
        if self.steady:
            dt, nt = 0.0, 1
        x, res, *_, flag = solver(x0, dt, nt)
        converged = True if (flag[-1] == 0) else False
        # Assemble solution
        uv, lambdas = self.assemble_sol(x, map_on_res=True)
        return uv, lambdas, res, converged

    def get_force(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Maps a monolithic force vector (external loads) to the DD element
        structure, ready for use in the res_jac routine.
        """
        self.is_built()
        # Map force vector on elements
        x = self.map_sol_on_elements(x.reshape(1,-1))
        # Assemble DD force vector
        x_dd = []
        # Loop over subdomains
        for s in range(self.mesh.n_sub):
            # Loop over elements
            for e_k in ("interior", "interface"):
                x_dd.append(x[e_k][s].reshape(-1))
        x_dd.append(np.zeros(self.n_constraints))
        self.f = np.concatenate(x_dd)
        return self.f

    def map_sol_on_elements(
        self,
        solutions: np.ndarray,
        map_on_ports: bool = False
    ) -> Dict[str, List[np.ndarray]]:
        """ 
        Map a monolithic solution (displacement or force) to domain 
        decomposition elements (interior/interface).
        """
        self.is_built()
        if (solutions.shape[1] != self.monolithic.get_ndof()):
            solutions = solutions.T
        data = {}
        for e_k in ("res", "interior", "interface"):
            data[e_k] = []
            for sub in self.subdomains:
                indices = sub.elem_states[e_k].nodes_state
                # Concatenate indices for both u and v components
                indices = np.concatenate([indices, indices+self.mesh.nxy])
                data[e_k].append(solutions[:,indices])
        if map_on_ports:
            data["port"] = []
            for p in self.dd_indices.ports:
                indices = self.dd_indices.port_to_nodes[p]
                indices = np.concatenate([indices, indices+self.mesh.nxy])
                data["port"].append(solutions[:,indices])
        return data

    def get_init_sol(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """Maps an initial solution from the monolithic form to the DD vector form."""
        self.is_built()
        # Map init sol on elements
        x = self.map_sol_on_elements(x.reshape(1,-1))
        # Initial guess
        x_dd = []
        # Loop over subdomains
        for s in range(self.mesh.n_sub):
            # Loop over elements
            for e_k in ("interior", "interface"):
                x_dd.append(x[e_k][s].reshape(-1))
        x_dd.append(np.zeros(self.n_constraints))
        return np.concatenate(x_dd)
