import numpy as np
import scipy.sparse as sp
import torch
import json
from torch import distributed as dist
from mpi4py import MPI
import torch_sla

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print, compress_ranges
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
    scaling: float = 1.0,
    subs_per_rank: int = 1
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
  
    self.subs_per_rank = subs_per_rank

    # Cache for residual shapes across ranks
    self.local_res_shapes = None
  

  # Building
  # ===================================
  def is_built(self) -> None:
    self.monolithic.is_built()
    if (not self.built):
      raise ValueError(
        "DD-FOM model not built. Please, call 'build' method first."
      )


  def get_rank_subdomains(self, subs_per_rank) -> List[int]:
    # Returns a list of subdomain IDs assigned to this rank

    if not bkd.distributed():
       return np.arange(0, self.mesh.n_sub, dtype=int).tolist()

    # check that the mapping fits in the mesh
    assert subs_per_rank * bkd.get_nranks() == self.mesh.n_sub

    # simple mapping: rank 0 gets 0..subs_per_rank-1,
    #                 rank 1 gets subs_per_rank..2*subs_per_rank-1, etc
    # TODO: support more complex mapping and locality between ranks
    id = bkd.get_rank()
    subs = np.arange(id * subs_per_rank, (id+1) * subs_per_rank, 1, dtype=int).tolist() # row wise split
    parallel_print(" DDFOM GET RANK SUBDOMAINS: RANK {} assigned to subdomains {}".format(bkd.get_rank(), compress_ranges(subs)))
    return subs


  def build(self) -> None:
    self.monolithic.is_built()
    # DD-FOM subdomains indices
    self.dd_indices = DDIndices(self.monolithic)
    self.dd_indices.build()
    # Constraints
    self.cmat = self.assemble_cmat()
    # DD-FOM subdomains
    self.subdomains = []
    self.local_subdomains = []
    self.global_subdomains = []
    subs = self.get_rank_subdomains(self.subs_per_rank)
    ls = 0 # local subdomain index
    for s in range(self.mesh.n_sub):
      if not np.isin(s, subs):
        continue
      cmat_s, indices_s = {}, {}
      for e_k in ("res", "interior", "interface"):
        indices_s[e_k] = getattr(self.dd_indices, e_k)[s]

      cmat_s["interior"] = self.cmat["interior"][s]  # use rank-subdomain index
      cmat_s["interface"] = self.cmat["interface"][s] # interface matrices indexed over global subs

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

      self.global_subdomains.append(s)
      self.local_subdomains.append(ls)
      ls+=1
    # Update control variables
    self.built = True

    parallel_print("RANK {} FOM global subdomains = {} (local subdomains = {})".format(bkd.get_rank(), compress_ranges(self.global_subdomains), compress_ranges(self.local_subdomains)))
    parallel_print("RANK {} ndof = {}".format(bkd.get_rank(), self.get_ndof()))

    self.global_submap = bkd._COMM.allgather(self.global_subdomains) if bkd.distributed() else [self.global_subdomains]
    parallel_print("RANK {} GLOBAL SUBDOMAIN MAPPING: {}".format(bkd.get_rank(), self.global_submap))

    self.offsets = self.get_global_offsets()
    parallel_print("RANK {} GLOBAL SUBDOMAIN OFFSETS: {}".format(bkd.get_rank(), self.offsets))

    self.owned = self.get_owned_nodes()
    parallel_print("RANK {} OWNED NODES: {}".format(bkd.get_rank(), [f"{e_k}: {x.size}" for (e_k, x) in self.owned.items()]))

    self.overlaps = self.build_distributed_map()
    for e_k in self.overlaps.keys():
      parallel_print("RANK {} OVERLAPPING {} NODES: {}".format(bkd.get_rank(), e_k, [f"{rank}: {nodes.size}" for (rank, nodes) in self.overlaps[e_k].items()])) 

    if bkd.distributed():
      assert len(self.subdomains) == self.subs_per_rank


  def get_ndof(self) -> int:
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.elem_states[e_k].n_nodes_state
    if bkd.distributed():
      ndof = bkd._COMM.allreduce(ndof, op=MPI.SUM)
    ndof *= 2 # 
    ndof += self.n_constraints
    return ndof
  
  def find_overlap_nodes_from_rank(self, rank, element="interface"):
    overlap_nodes = []
    subs_other = self.global_submap[rank]
    indices_elem = getattr(self.dd_indices, element)
    for sub in self.global_subdomains:
      for other_sub in subs_other:
        overlap_nodes.append(np.intersect1d(indices_elem[sub], indices_elem[other_sub]))
    overlap_nodes = np.concatenate(overlap_nodes)
    overlap_nodes = np.unique(overlap_nodes)
    return overlap_nodes
  
  def build_distributed_map(self):
    overlap_nodes = {}
    for e_k in ("res", "interface"):
      overlap_nodes[e_k] = {}
      for rank in range(len(self.global_submap)):
        if rank == bkd.get_rank(): continue
        overlap_nodes[e_k][rank] = self.find_overlap_nodes_from_rank(rank, element=e_k)
  
    return overlap_nodes

  def get_owned_nodes(self):
    owned = {}
    for e_k in ("res", "interior", "interface"):
      owned[e_k] = []
      for sub in self.subdomains:
        owned[e_k].append(sub.nodes_ind[e_k])
      owned[e_k] = np.concatenate(owned[e_k])
    return owned


  def flatten_across_domain(self, x):
    tmp = [None] * self.mesh.n_sub
    for s in range(self.subs_per_rank):
      for r in range(bkd.get_nranks()):
        global_s = self.global_submap[r][s]
        tmp[global_s] = x[s][r]
    x = tmp
    return x
  
  
  def flatten_across_dist(self, x):
    tmp = [None] * self.mesh.n_sub
    for s in range(self.subs_per_rank):
      for r in range(bkd.get_nranks()):
        global_s = self.global_submap[r][s]
        tmp[global_s] = x[r][s]
    x = tmp
    return x

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
    cmat = ops.map_nested_dict(cmat, bkd.to_sp_coo_backend)
    return cmat

  def init_cmat(
    self,
    element: str
  ) -> List[sp.spmatrix]:
    cmat = []
    for nodes_ind in getattr(self.dd_indices, element):
      n_c = len(nodes_ind) if element != "interior" else 1
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
    cmat = ops.map_nested_dict(cmat, bkd.torch_coo_to_scipy)
    for e_k in ("interior", "interface"):
      cmat[e_k] = [rmat @ m for m in cmat[e_k]]
    cmat = ops.map_nested_dict(cmat, bkd.to_sparse, format="coo")
    cmat = ops.map_nested_dict(cmat, bkd.to_sp_coo_backend)
    return cmat, n_constraints_weak

  def residual(self, x, use_global: bool = True):
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res = []
    cres = torch.zeros(self.n_constraints, device=bkd.device()) if bkd.is_torch_backend() else np.zeros(self.n_constraints)

    runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    #uv, lambdas = self.assemble_sol(x, map_on_res=False, use_global=use_global)
    uv, lambdas = self.assemble_sol(x, map_on_res=False, use_global=False)
    if (not self.steady):
      #uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False, use_global=use_global)
      uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False, use_global=False)
    runtime += (time()-start) / len(self.subdomains)
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    uv_s_old = None
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      global_s = s
      uv_s = self.extract_uv_sub_from_dict(uv, global_s)
      if (not self.steady):
        uv_s_old = self.extract_uv_sub_from_dict(uv_old, global_s)

      uv_s = sub.map_on_res(uv_s)
      if (not self.steady):
        uv_s_old = sub.map_on_res(uv_s_old)

      if sub.compact:
        ops_uv = sub.action_ops_gen(uv_s, sub.elem_states)
      else:
        ops_uv = sub.action_ops(uv_s, sub.elem_states)
      res_s = sub.compute_res(uv_s, sub.elem_states, ops_uv, self.steady, self.dt, uv_s_old)
      cres_s, cjac_s = sub.compute_cres_cjac(uv_s)

      jac_s = sub.compute_jac(uv_s, sub.elem_states, ops_uv, self.steady, self.dt)

      if bkd.is_torch_backend():
        interior_t = torch.t(jac_s["interior"])
        interface_t = torch.t(jac_s["interface"])
        cjac_interface_t = torch.t(cjac_s["interface"])
        res_s = torch.cat([
          sub.scaling*(interior_t@res_s),
          sub.scaling*(interface_t@res_s) + cjac_interface_t@lambdas
        ])
      else:
        res_s = np.concatenate([
          sub.scaling*(jac_s["interior"].T@res_s),
          sub.scaling*(jac_s["interface"].T@res_s) \
            + cjac_s["interface"].T@lambdas
        ])


      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      res.append(res_s)
      cres += cres_s
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    bkd.barrier()
    if bkd.distributed():
      if use_global:
        global_res = [bkd.gatherv_tensor(r, as_list=True) for r in res]
        dist.reduce(cres, dst=0, op=dist.ReduceOp.SUM)
        if bkd.root():
          res = self.flatten_across_domain(global_res)
        res = bkd.broadcast_tensor(res, root=0)
        res = torch.cat(res) if bkd.is_torch_backend() else np.concatenate(res)
      else:
        res = torch.cat(res)

        # gather res sizes
        if self.local_res_shapes is None:
          res_sizes = bkd.get_local_sizes_all(res, dim=0)
          self.local_res_shapes = res_sizes
        else:
          res_sizes = self.local_res_shapes
        res_sizes = np.sum(res_sizes)
        res = bkd.to_sharded_dtensor(res, shape=(res_sizes,), stride=(1,))
    else:
      res = torch.cat(res) if bkd.is_torch_backend() else np.concatenate(res)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, cres

  def local_trial_vector(self, x):
    """Return the replicated FOM KKT vector needed by local residual work.

    Unlike the ROM, FOM subdomains index their state using global offsets in
    :meth:`assemble_sol`. Thus a rank-local state slice cannot be passed to
    ``residual(..., use_global=False)``; every rank retains the complete
    regular KKT vector and only residual/Jacobian assembly is local.
    """
    return x


  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x: np.ndarray,
    use_global: bool = True
  ) -> dtypes.RES_JAC_TYPE:
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res, hess, cjac = [], [], []
    cres = torch.zeros(self.n_constraints, device=bkd.device()) if bkd.is_torch_backend() else np.zeros(self.n_constraints)

    runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    #uv, lambdas = self.assemble_sol(x, map_on_res=False, use_global=use_global)
    uv, lambdas = self.assemble_sol(x, map_on_res=False, use_global=False)
    if (not self.steady):
      #uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False, use_global=use_global)
      uv_old, _ = self.assemble_sol(self.x_old, map_on_res=False, use_global=False)
    runtime += (time()-start) / len(self.subdomains)
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    uv_s_old = None
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      global_s = s
      uv_s = self.extract_uv_sub_from_dict(uv, global_s)
      if (not self.steady):
        uv_s_old = self.extract_uv_sub_from_dict(uv_old, global_s)
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
      if bkd.is_torch_backend():
        cjac.append(cjac_s.to_sparse_coo())
        hess.append(hess_s.to_sparse_coo())
      else:
        cjac.append(cjac_s)
        hess.append(hess_s)
      runtime += time()-start
    runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    bkd.barrier()


    if bkd.distributed():
      if use_global:
        # TODO: [0] below indicates # of subdomains per rank, fix this to generalize
        global_res = [bkd.gatherv_tensor(r, as_list=True) for r in res]
        global_cjac = [bkd.gatherv_tensor(r, dim=1, as_list=True, coalesce=True) for r in cjac]
        global_hess = [bkd.gatherv_tensor(r, as_list=True) for r in hess]

        
        dist.reduce(cres, dst=0, op=dist.ReduceOp.SUM)


        jac = None
        if bkd.root():

            global_res = self.flatten_across_domain(global_res)
            global_cjac = self.flatten_across_domain(global_cjac)
            global_hess = self.flatten_across_domain(global_hess)


            res, jac = self.assemble_kkt(global_res, cres, global_cjac, global_hess)

        res = bkd.broadcast_tensor(res, root=0)
        jac = bkd.broadcast_tensor(jac, root=0)
        bkd.barrier()
      else:
        #res_f, jac = self.assemble_kkt(res, cres, cjac, hess)

        res, jac = self.assemble_kkt(res, cres, cjac, hess)
    else: 
      res, jac = self.assemble_kkt(res, cres, cjac, hess)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, jac

  def get_global_offsets(self):
    offsets = []
    si = 0
    for s in range(self.mesh.n_sub):
      nnodes = self.dd_indices.interface[s].size + self.dd_indices.interior[s].size
      offsets.append(si)
      si = si + 2*nnodes
    return offsets

  def assemble_sol(
    self,
    x: np.ndarray,
    map_on_res: bool = False,
    use_global: bool = True
  ) -> Tuple[dtypes.UV_TYPE, np.ndarray]:
    shape = [self.mesh.nxy]
    if (x.ndim == 2):
      shape.append(x.shape[1])
    uv = self.init_uv(shape=shape, map_on_res=map_on_res)
    # Loop over subdomains
    offset = (self.get_ndof() - self.n_constraints) // self.mesh.n_sub
    offset = 48


    # assemble local subdomains
    si = 0
    for (s, sub) in enumerate(self.subdomains):
      si = self.offsets[self.global_subdomains[s]]
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
    
    # gather all subdomains and broadcast back - every rank should have full assembled soln
    if bkd.distributed() and use_global:
      uv_global = {}
      for e_k in ("interior", "interface"):
        uv_global[e_k] = {}
        for x_k in ("u", "v"):
          uv_global[e_k][x_k] = [bkd.gatherv_tensor(s, as_list=True) for s in uv[e_k][x_k]]
      if map_on_res:
        uv_global["res"] = {}
        uv_global["res"]["u"] = None
        uv_global["res"]["v"] = None

      if map_on_res and bkd.root():
        global_res = {}
        global_res["u"] = torch.zeros_like(uv["res"]["u"])
        global_res["v"] = torch.zeros_like(uv["res"]["v"])
        for rank in range(bkd.get_nranks()):
          subs = self.global_submap[rank]
          si = 0
          for s in range(len(subs)):
            # note: s is local subdomain on rank
            global_s = subs[s] # global subdomain on rank
            si = offset * global_s
            for e_k in ("interior", "interface"):
              state_k = self.dd_indices.__dict__[e_k][global_s]
              global_res["u"][state_k] = uv_global[e_k]["u"][s][rank]
              global_res["v"][state_k] = uv_global[e_k]["v"][s][rank]
        uv_global["res"] = {}
        uv_global["res"]["u"] = global_res["u"]
        uv_global["res"]["v"] = global_res["v"]

      for e_k in ("interior", "interface"):
        for x_k in ("u", "v"):
          uv[e_k][x_k] = bkd.broadcast_tensor(uv_global[e_k][x_k])
      if map_on_res:
        uv["res"]["u"] = bkd.broadcast_tensor(uv_global["res"]["u"])
        uv["res"]["v"] = bkd.broadcast_tensor(uv_global["res"]["v"])

      for e_k in ("interior", "interface"):
        for x_k in ("u", "v"):
          #uv[e_k][x_k][:] = uv[e_k][x_k][self.global_subdomains]
          uv[e_k][x_k] = self.flatten_across_domain(uv[e_k][x_k])

      bkd.barrier()

      
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
        uv["res"][x_k] = torch.zeros(shape, device=bkd.device()) if bkd.is_torch_backend() else np.zeros(shape)
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
  
  def get_rank_portion(self, vec, elem):
    # returns a list of rank-local subdomain portions from vec corresponding to the current rank
    local_vec = []
    for (s, sub) in enumerate(self.subdomains):
      global_s = self.global_subdomains[s]
      ind = self.dd_indices[elem][global_s]
      local_vec.append(vec[ind])
    return local_vec

  def assemble_kkt(
    self,
    res: np.ndarray,
    cres: np.ndarray,
    cjac: sp.spmatrix,
    hess: sp.spmatrix
  ) -> dtypes.RES_JAC_TYPE:
    # > Residual
    res.append(cres)
    if bkd.is_torch_backend():
        res = torch.cat(res)
        # > Constraints Jacobian
        cjac = bkd.torch_hstack(cjac, format="coo")

        # > Hessians

        
        hess = torch_sla.SparseTensorList.from_torch_sparse_list(hess)
        hess = hess.to_block_diagonal().to_torch_sparse()

        # > Full Jacobian
        jac = bkd.torch_bmat_new([[hess, torch.t(cjac)],
                                  [cjac,   None]], format="coo")
    else:
        res = np.concatenate(res)
        # > Constraints Jacobian
        cjac = sp.hstack(cjac)
        # > Hessians
        hess = sp.block_diag(hess)
        # > Full Jacobian
        jac = sp.bmat([[hess, cjac.T],
                       [cjac,   None]],
                       format="csr")
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
    nt = 1
    self.is_built()
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = torch.zeros(self.get_ndof(), device=bkd.device()) if bkd.is_torch_backend() else np.zeros(self.get_ndof())
    self.runtime["total"] += time()-start
    if bkd.distributed():
      solver = solvers.DistNewton(
        model=self,
        tol=tol,
        maxit=maxit,
        stepsize_min=stepsize_min,
        iostep=iostep,
        verbose=verbose,
        distributed=bkd.distributed()
      )
    else:
      # Initialize solver
      solver = solvers.Newton(
        model=self,
        tol=tol,
        maxit=maxit,
        stepsize_min=stepsize_min,
        iostep=iostep,
        verbose=verbose,
        distributed=bkd.distributed()
      )
    # Solving
    self.steady = bool(steady)
    if self.steady:
      dt, nt = 0.0, 1
    x, res, *_, flag = solver(x0, dt, nt)
    if not isinstance(flag, List):
      flag = [flag]
    converged = True if (flag[-1] == 0) else False
    bkd.barrier()
    # Assemble solution
    uv, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, lambdas, res, converged

  def get_init_sol(
    self,
    x: np.ndarray
  ) -> np.ndarray:
    self.is_built()
    x = bkd.to_backend(x)
    # Map init sol on elements
    x = self.map_sol_on_elements(x.reshape(1,-1))

    x_g = {}
    for e_k in ("interior", "interface"):
      # TODO: [0] below indicates # of subdomains per rank, fix this to generalize
      # todo: change to cat? bcast? 4/24
      x_g[e_k] = [bkd.gatherv_tensor(s, dim=1, as_list=True) for s in x[e_k]]

    # broadcast collected subdomain states

    # Initial guess
    x_dd = None
    if bkd.distributed() and bkd.root():
      # Convert gathered [local subdomain][rank] data to global subdomain
      # order before assembling the vector. This matches the serial layout:
      # [subdomain 0 interior, subdomain 0 interface, ...].
      x = {
        e_k: self.flatten_across_domain(x_g[e_k])
        for e_k in ("interior", "interface")
      }
      x_dd = []
      for s in range(self.mesh.n_sub):
        for e_k in ("interior", "interface"):
          x_dd.append(x[e_k][s].reshape(-1))
      x_dd.append(torch.zeros(self.n_constraints, device=bkd.device()) if bkd.is_torch_backend() else np.zeros(self.n_constraints))
      if bkd.is_torch_backend():
        x_dd = torch.cat(x_dd)
      else:
        x_dd = np.concatenate(x_dd)
    elif not bkd.distributed():
      x = x_g
      x_dd = []
      for s in range(self.mesh.n_sub):
        for e_k in ("interior", "interface"):
          x_dd.append(x[e_k][s].reshape(-1))

      x_dd.append(torch.zeros(self.n_constraints, device=bkd.device()) if bkd.is_torch_backend() else np.zeros(self.n_constraints))
      if bkd.is_torch_backend():
        x_dd = torch.cat(x_dd)
      else:
        x_dd = np.concatenate(x_dd)

    x_dd = bkd.broadcast_tensor(x_dd, root=0)
      
    return x_dd

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
      for (s, sub) in enumerate(self.subdomains):
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
