import copy
import torch
import numpy as np
import scipy.sparse as sp
import numpy_indexed as npi
from typing import List
from torch import distributed as dist
from mpi4py import MPI

import json
import logging

from time import time
from dd_nm_rom import ops
from dd_nm_rom import solvers
from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print

from .rbf_model import RBFModel
from .subdomain import SubdomainROM
from ..autoencoder import AutoencoderNP, MultiAutoencoderNP

logger = logging.getLogger(__name__)


class DD_NM_ROM(object):
  """
  Compute DD NM-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.
  """

  def __init__(
    self,
    dd_fom,
    nn_configfiles,
    res_bases=None,
    hr_active=False,
    hr_n_samples=1,
    hr_n_edge_samples_ratio=0.75,
    hr_sample_small_ports=False,
    hr_small_ports_dim=5,
    constraint_type='weak',
    n_constraints_weak=1,
    scaling=1.0,
    subs_per_rank: int = 1,
    check_unique_models=False
  ):
    # DD-FOM
    # -------------
    self.dd_fom = dd_fom
    for k in ("runtime", "mesh"):
      setattr(self, k, getattr(self.dd_fom, k))
    # Scaling factor for residual
    self.scaling = self.mesh.hxy if (scaling <= 0) else scaling

    global_submap = self.dd_fom.global_submap
    parallel_print("RANK {}: DDROM: FOM global subdomain mapping = {}".format(bkd.get_rank(), global_submap))

    self.subs_per_rank = subs_per_rank
    subs = self.get_rank_subdomains(self.subs_per_rank)
    self._subs = subs

    self.debug = logger.isEnabledFor(logging.DEBUG)

    # Autoencoders
    # -------------
    load_opts = {"weights_only":False, "map_location":{"cuda:0":"cpu"}}
    self.nn_configs = self._load_nn_configs(nn_configfiles, **load_opts)
    self.nn_models = self.init_nn_models(self.nn_configs, check_unique_models) # NOTE: FIX! this assumes nn_models loaded by rank # == subdomain #! 
    if self.debug: logger.debug(" DD_ROM: NN MODELS: {}".format(["{}: {}".format(key, len(models)) for (key, models) in self.nn_models.items()]))
    # Forcing term
    self.force = self.dd_fom.f is not None
    # DD-ROM Constraints
    # -------------
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = n_constraints_weak
    self.compute_rom_dim()
    if self.debug: logger.debug("Rom dim = {}".format(self.rom_dim))
    if (self.constraint_type == "strong"):
      self.set_port_indices()
      self.init_nn_model_intf()
    self.assemble_cmat()
    bkd.barrier()
    # HR
    # -------------
    self.res_bases = res_bases
    self.hr_active = hr_active
    self.hr_n_samples = int(hr_n_samples)
    self.hr_n_edge_samples_ratio = np.clip(hr_n_edge_samples_ratio, 0, 1)
    self.hr_sample_small_ports = float(hr_sample_small_ports)
    self.hr_small_ports_dim = int(hr_small_ports_dim)
    # DD-ROM subdomains
    # -------------
    self.subdomains = []
    self.local_subdomains = []
    self.global_subdomains = []
    ls = 0 # local subdomain index
    for (s_lf, sub) in enumerate(self.dd_fom.subdomains):
      s = self.dd_fom.global_subdomains[s_lf]
      if bkd.distributed():
        if np.isin(s, subs, invert=True):
          continue

      logger.info(" DD-NM-ROM INIT: RANK {} : assigned to global subdomain {}".format(bkd.get_rank(), s))
      inputs_s = {}
      # Retrieves the corresponding attribute from the current object
      if self.debug: logger.debug(" DD-ROM INIT: FOM SUBDOMAIN {} (local = {}) global = {}".format(s_lf, ls, s))

      attr_k = getattr(self, "rom_dim")
      inputs_s["rom_dim"] = {}
      inputs_s["rom_dim"]["interior"] = attr_k["interior"][ls]
      # Strong constraints compose interface states from globally indexed
      # ports; weak constraints retain an interface model per local subdomain.
      interface_idx = s if self.constraint_type == "strong" else ls
      inputs_s["rom_dim"]["interface"] = attr_k["interface"][interface_idx]


      attr_k = getattr(self, "nn_models")
      inputs_s["nn_models"] = {}
      inputs_s["nn_models"]["interior"] = attr_k["interior"][ls]
      inputs_s["nn_models"]["interface"] = attr_k["interface"][ls]

      attr_k = getattr(self, "cmat")
      inputs_s["cmat"] = {}
      inputs_s["cmat"]["interior"] = attr_k["interior"][ls]  # use rank-subdomain index
      inputs_s["cmat"]["interface"] = attr_k["interface"][s] # interface matrices indexed over global subs
      self.subdomains.append(
        SubdomainROM(
          sub_fom=sub,
          scaling=self.scaling,
          constraint_type=self.constraint_type,
          res_bases=self.get_res_bases(s),
          hr_n_samples=self.hr_n_samples,
          hr_n_edge_samples_ratio=self.hr_n_edge_samples_ratio,
          hr_sample_small_ports=self.hr_sample_small_ports,
          hr_small_ports_dim=self.hr_small_ports_dim,
          **inputs_s
        )
      )
      self.global_subdomains.append(s)
      self.local_subdomains.append(ls)
      if self.hr_active:
        self.subdomains[-1].set_hr_mode(active=True)
      ls+=1

    # Trigger torch.compile on each rank's local activation functions before
    # any rank can enter the solver's distributed collectives.  Compiled
    # callables are process-local, so every rank must warm its own models.
    compile_exception = None
    try:
      self.compile_activations()
    except Exception as exc:
      compile_exception = exc

    if bkd.distributed():
      compile_errors = bkd._COMM.allgather(
        None if compile_exception is None else repr(compile_exception)
      )
      if any(error is not None for error in compile_errors):
        details = "; ".join(
          "rank {}: {}".format(rank, error)
          for rank, error in enumerate(compile_errors)
          if error is not None
        )
        raise RuntimeError(
          "DDNMROM activation compilation failed before synchronization: "
          + details
        ) from compile_exception
    elif compile_exception is not None:
      raise compile_exception

    bkd.barrier()
    self.resjac_it = 0

    self.local_res_shapes = None

    self.local_offsets, self.local_sizes = self.get_local_offsets()
    parallel_print("RANK {} ROM LOCAL SUBDOMAIN OFFSETS: {} - LOCAL SUBDOMAIN SIZES: {}".format(bkd.get_rank(), self.local_offsets, self.local_sizes))

    self.global_offsets, self.global_sizes = self.get_global_offsets()
    parallel_print("RANK {} ROM GLOBAL SUBDOMAIN OFFSETS: {} - GLOBAL SUBDOMAIN SIZES: {}".format(bkd.get_rank(), self.global_offsets, self.global_sizes))

    # cache the ndof for reuse
    self._ndof = self.get_ndof()
    self._ndof_sol = self._ndof - self.n_constraints
    if self.debug: logger.debug(" NDOF = {} NDOF (SOL) = {}".format(self._ndof, self._ndof_sol))
    assert self._ndof is not None
    assert self._ndof_sol is not None

    # Interpolator
    # -------------
    self.rbf_model = RBFModel(self.subdomains, self.n_constraints, self.dd_fom)
    # Integration
    # -------------
    self.steady = True
    self.x_old = None
    self.dt = 0.0


  def get_rank_subdomains(self, subs_per_rank) -> List[int]:
    # Returns a list of subdomain IDs assigned to this rank

    # # check that the mapping fits in the mesh

    # # simple mapping: rank 0 gets 0..subs_per_rank-1,
    # #                 rank 1 gets subs_per_rank..2*subs_per_rank-1, etc
    # # TODO: support more complex mapping and locality between ranks
    subs = self.dd_fom.global_subdomains
    parallel_print(" DDROM GET RANK SUBDOMAINS: RANK {} assigned to subdomains {}".format(bkd.get_rank(), subs))
    return subs

  def compile_activations(self):
    compiled = set()
    for sub in self.subdomains:
      for state in sub.elem_states.values():
        model = state.nn_model
        if model is None or id(model) in compiled:
          continue
        model.compile_activations()
        compiled.add(id(model))
  
  def get_res_bases(self, index=0):
    if (self.res_bases is not None):
      if isinstance(self.res_bases, (list, tuple)):
        if (len(self.res_bases) != self.mesh.n_sub):
          raise ValueError(
            "The number of residual bases matrices provided " \
            "does not match the number of subdomains."
          )
        else:
          return self.res_bases[index]
      elif isinstance(self.res_bases, np.ndarray):
        return copy.deepcopy(self.res_bases)
      else:
        raise ValueError(
          "The 'res_bases' input must be either a list or a NumPy array."
        )

  def get_ndof(self):
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.rom_dim[e_k]
    if bkd.distributed():
      ndof = bkd._COMM.allreduce(ndof, op=MPI.SUM)
    if self.debug: logger.debug("GET_NDOF: RANK = {}: ndof = {} nconstraints = {} (total ndof = {})".format(bkd.get_rank(), ndof, self.n_constraints, ndof+self.n_constraints))
    ndof += self.n_constraints
    return ndof

  # ROM dimensions
  # ===================================
  def compute_rom_dim(self):
    # Read latent dimension of each autoencoder
    self.rom_dim = {}
    for (e_k, cfg_k) in self.nn_configs.items():
      if (e_k not in self.rom_dim):
        self.rom_dim[e_k] = []
      for cfg_ki in cfg_k:
        dim = cfg_ki["decoder"]["latent_dim"]
        self.rom_dim[e_k].append(dim)
      if self.debug: logger.debug("-- {} - rom dim = ({}) {}".format(e_k, len(self.rom_dim[e_k]), self.rom_dim[e_k]))

    if self.debug: logger.debug(" NUM NN CONFIGS FOR PORT = {} DDFOM PORTS = {}".format(len(self.nn_configs["port"]),len(self.dd_fom.dd_indices.ports) ))

    # Compute interface latent dimension form ports ones
    if (self.constraint_type == 'strong'):
      if (len(self.nn_configs["port"]) != len(self.dd_fom.dd_indices.ports)):
        logger.warning("The number of port autoencoders doesn't " \
                       "match the number of ports available.\n" \
                       "(got {}, expected {})".format(len(self.nn_configs["port"]), len(self.dd_fom.dd_indices.ports)))
      self.rom_dim["interface"] = []
      for (s, sub) in enumerate(self.dd_fom.subdomains): # FOR GLOBAL INTERFACE SIZE (ALL SUBDOMAINS)
        # -- LOCAL INTERFACE SIZE -- compute interface from ports that only touch this subdomain!
        dim = 0
        for p in sub.ports:
          dim += self.rom_dim["port"][p]
        self.rom_dim["interface"].append(dim)
      if bkd.distributed():
        global_intf_sizes = bkd._COMM.allgather(self.rom_dim["interface"])
        assert (len(global_intf_sizes) == bkd.get_nranks())
        for r in range(len(global_intf_sizes)):
          if r == bkd.get_rank():
            continue
    
        bkd.barrier()
        
        # interface sizes are gathered by rank (ordered), reorder them back to the appropriate physical location [0,...,nsubs]
        if self.debug: logger.debug(" IN global intf sizes = {}".format(global_intf_sizes))
        global_intf_sizes = self.dd_fom.flatten_across_dist(global_intf_sizes)
        if self.debug: logger.debug(" OUT global intf sizes = {}".format(global_intf_sizes))

        self.rom_dim["interface"] = np.array(global_intf_sizes).flatten()
      else:
        self.rom_dim["interface"] = np.array(self.rom_dim["interface"]).flatten()


  # Port to nodes indices
  # ===================================
  def set_port_indices(self):
    # Check number of port autoencoders
    if (len(self.nn_configs["port"]) != len(self.dd_fom.dd_indices.ports)):
      raise ValueError(
        "The number of port autoencoders doesn't " \
          "match the number of ports available."
      )
    # Set port nodes indices
    self.port_to_nodes = []
    ls = 0
    for s in range(self.mesh.n_sub):
      if s not in self.dd_fom.global_subdomains:
        # skip over other subdomains for now
        self.port_to_nodes.append({})
        continue
      sub = self.dd_fom.subdomains[ls]
      ls+=1

      port_to_nodes_s = {}
      shift = 0
      for p in sub.ports:
        # FOM
        # > Port/interface indices
        port_ind = self.dd_fom.dd_indices.port_to_nodes[p]
        intf_ind = sub.elem_states["interface"].nodes_state
        # > Duplicate for u and v
        port_ind = np.concatenate([port_ind, port_ind+self.mesh.nxy])
        intf_ind = np.concatenate([intf_ind, intf_ind+self.mesh.nxy])
        fom_ind = npi.indices(intf_ind, port_ind, missing="mask").compressed()
        # ROM
        port_dim = self.rom_dim["port"][p]
        rom_ind = np.arange(port_dim)+shift
        # Update
        port_to_nodes_s[p] = {"fom": fom_ind, "rom": rom_ind}
        shift += port_dim
      self.port_to_nodes.append(port_to_nodes_s)
    
    
    bkd.barrier()
    # gather indices from other subdomains
    tmp = []
    for ls in range(self.mesh.n_sub):
      if len(self.port_to_nodes[ls]) > 0:
        tmp.append((ls, self.port_to_nodes[ls]))

    if bkd.distributed():
      tmp = bkd._COMM.allgather(tmp)
    else:
      tmp = [tmp]

    for ls in range(self.mesh.n_sub):
      if len(self.port_to_nodes[ls]) == 0:
        for r in range(bkd.get_nranks()):
          if r != bkd.get_rank() and len(tmp[r]) > 0:
            for (r_ls, v) in tmp[r]:
              if r_ls == ls:
                self.port_to_nodes[ls] = v


  # Autoencoders
  # ===================================
  @classmethod
  def _load_nn_configs(cls, nn_configfiles, **kwargs):
    """
    Helper function to load a saved torch nn checkpoint file
    Only unique filepaths are loaded, otherwise a shallow copy of the data is made
    :@param nn_configfiles: dictionary of NN files to load for each element (interior, port, etc)
    :@param kwargs: options passed to torch.load
    """
    if not isinstance(nn_configfiles, dict):
      raise ValueError("Expected a dictionary of NN config files for each element")
    if len(nn_configfiles) == 0 or len(nn_configfiles.values()) == 0:
      raise ValueError("No nn_configfiles provided, need files for at least one element")

    # mapping of unique file to loaded data
  
    # loaded nn data for each element

    nn_configs = dict()
    unique_files = dict()


    # -- debugging - memory stats
    _mem_start = bkd._start_mem_trace()

    for element in nn_configfiles.keys():
      nn_configs[element] = []
      unique_files[element] = {}

      files = nn_configfiles[element]
      for file in files:
        if file in unique_files[element]:
          # this file has already been loaded, shallow copy the nn config
          nn_configs[element].append(copy.copy(unique_files[element][file]))
        else:
          unique_files[element][file] = torch.load(file, **kwargs)
          nn_configs[element].append(unique_files[element][file])
          if logger.isEnabledFor(logging.DEBUG): logger.debug(" -- Loaded file '{}' ({}: {} total configs)".format(file, element, len(nn_configs[element])))


    _mem_end = bkd._end_mem_trace(print_stats=False)
    _mem_usage = bkd._get_mem_trace_stats(_mem_start, _mem_end, print_stats=False)

    for (element, configs) in nn_configs.items():
      print(" DD-ROM RANK {}: loaded {} configs for '{}' ({} unique files, {} MB)".format(bkd.get_rank(), len(configs), element, len(unique_files[element]), _mem_usage[0]))

    _mem_usage = bkd._get_mem_trace_stats(_mem_start, _mem_end, print_stats=True)

    for (element, configs) in nn_configs.items():
      # make sure we have a model for every input file
      if len(configs) != len(nn_configfiles[element]):
        print(" --- Error: expected {} models for '{}' but created {}".format(len(nn_configfiles[element]), element, len(configs)))
      assert len(configs) == len(nn_configfiles[element])

    bkd.barrier()

    return nn_configs


  def init_nn_models(
    self,
    configs,
    check_unique=False
  ):
    # Loop over elements: interior and interface/ports
    nn_models = {}

    
    for (key_i, cfg_i) in configs.items():
      # Loop over instances in each element
      if (not isinstance(cfg_i, (list, tuple))):
        cfg_i = [cfg_i]

      if key_i not in nn_models:
        nn_models[key_i] = []

      if not check_unique or key_i != "interior":
        if self.debug: logger.debug(" Creating autoencoder {}".format(len(nn_models[key_i])))
        nn_models[key_i] = [AutoencoderNP(cfg_ij) for cfg_ij in cfg_i]
      else:
        # Check for autoencoder reuse
        # TODO: redo more efficiently using metadata from checkpoint file
        for cfg_ij in cfg_i:
          if key_i == "interior":
            # check for existing Autoencoder
            new_autoencoder = AutoencoderNP(cfg_ij)
            found = False
            for (id, model) in enumerate(nn_models[key_i]):
              if self.debug: logger.debug(" DDROM: INIT NN MODELS: Checking for unique {} autoencoder.. id {}".format(key_i, id))
              if model == new_autoencoder:
                if self.debug: logger.debug("    -- Found existing autoencoder.. reusing autoencoder id {}".format(id))

                # found matching autoencoder
                found = True
                nn_models[key_i].append(AutoencoderNP.makeShared(model))
                del new_autoencoder
                break

            # unique autoencoder, create a new instance
            if not found:
              nn_models[key_i].append(AutoencoderNP(cfg_ij))
          else:
            nn_models[key_i].append(AutoencoderNP(cfg_ij))
    return nn_models

  # Interface from ports
  # -----------------------------------
  def init_nn_model_intf(self):
    models = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      g_s = self.dd_fom.global_subdomains[s]

      n_nodes_intf = sub.elem_states["interface"].n_nodes_state
      models.append(
        MultiAutoencoderNP(
          indices=self.port_to_nodes[g_s],
          input_dim=2*n_nodes_intf,
          autoencoders={p: self.nn_models["port"][p] for p in sub.ports} # NOTE: this assumes port models are assembled/ordered over global domain
        )
      )
    self.nn_models["interface"] = models

  # Constraint matrices
  # ===================================
  def assemble_cmat(self):
    if (self.constraint_type == "weak"):
      self.cmat = copy.deepcopy(self.dd_fom.cmat)
      if (self.dd_fom.constraint_type != "weak"):
        self.cmat, self.n_constraints = self.dd_fom.assemble_cmat_weak(
          cmat=self.cmat,
          n_constraints_weak=self.n_constraints_weak,
          n_constraints=self.dd_fom.n_constraints
        )
      else:
        self.n_constraints = self.dd_fom.n_constraints
    else:
      self.cmat = self._assemble_cmat_strong()
    self.cmat = ops.map_nested_dict(
      self.cmat,
      lambda x: x.to_dense() if bkd.is_torch_backend() else x.toarray()
    )


  def _assemble_cmat_strong(self):
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_fom.dd_indices.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]
      self.n_constraints += (len(subs_p)-1) * port_dim
    # Assemble constraints matrices

    cmat_interior = self._init_cmat(element="interior")
    cmat_intf = self._assemble_cmat_intf()

    cmat = {
      "interior": cmat_interior,
      "interface": cmat_intf,
    }

    cmat = ops.map_nested_dict(cmat, bkd.to_sp_coo_backend)

    return cmat

  def _init_cmat(
    self,
    element
  ):
    cmat = []
    for dim in self.rom_dim[element]:
      cmat.append(sp.coo_matrix((self.n_constraints, dim)))
    return cmat

  def _assemble_cmat_intf(self):
    # Initialize matrices
    cmat = self._init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_fom.dd_indices.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]

      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          # TODO: gather self.port_to_nodes[subs_p[l]] from rank directly?
          col = self.port_to_nodes[subs_p[l]][p]["rom"]
          row = np.arange(port_dim) + shift
          data = (-1)**j * np.ones(port_dim)
          cmat[subs_p[l]].col  = np.concatenate((cmat[subs_p[l]].col,  col))
          cmat[subs_p[l]].row  = np.concatenate((cmat[subs_p[l]].row,  row))
          cmat[subs_p[l]].data = np.concatenate((cmat[subs_p[l]].data, data))
        shift += port_dim
    return cmat

  def residual(
    self,
    x,
    use_global: bool = True
  ):
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res = []
    cres = torch.zeros(self.n_constraints) if bkd.is_torch_backend() else np.zeros(self.n_constraints)
    # > Set Lagrangian multipliers
    lambdas = x[-self.n_constraints:]
    runtime += time()-start
    # Loop over subdomains
    # -------------
    if not use_global and bkd.distributed():
      rank_size = np.sum(self.local_sizes)
      if x.shape[0] != (rank_size + self.n_constraints):
        print(" **** Inconsistent x size in residual! Got input {} expected {} ({} + {}) ****".format(x.shape[0], rank_size + self.n_constraints, rank_size, self.n_constraints))
        rank_offset = self.global_offsets[bkd.get_rank()]
        x = x[rank_offset:rank_offset+rank_size]
    z = self.extract_z_sub_from_vec(x, use_global)

    z_old = None
    if (not self.steady):
      z_old = self.extract_z_sub_from_vec(self.x_old, use_global)
    if self.force:
      force, *_ = self.dd_fom.assemble_sol(self.dd_fom.f, map_on_res=False, use_global=False)
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      global_s = self.global_subdomains[s]

      start_s = time()
      force_s = None
      if self.force:
        force_s = self.dd_fom.extract_uv_sub_from_dict(force, global_s)
      # > Compute quantities needed for KKT system
      uv_s, dec_jac = sub.reconstruct_uv(z[s], with_jac=True, map_on_res=True)
      if self.force:
        force_s = sub.map_on_res(force_s)
      uv_s_old = None
      if not self.steady:
        uv_s_old = sub.reconstruct_uv(z_old[s], with_jac=False, map_on_res=True)

      # FOM subdomain.compute_res_jac
      if sub.sub_fom.compact:
        ops_uv_s = sub.sub_fom.action_ops_gen(uv_s, sub.elem_states)
      else:
        ops_uv_s = sub.sub_fom.action_ops(uv_s, sub.elem_states)
      
      res_s = sub.sub_fom.compute_res(uv_s, sub.elem_states, ops_uv_s, self.steady, self.dt, uv_s_old, force_s, self.dd_fom.__class__.__name__)
      jac_s = sub.sub_fom.compute_jac(uv_s, sub.elem_states, ops_uv_s, self.steady, self.dt, sub.compute_jac, self.dd_fom.__class__.__name__)

      for e_k in ("interior", "interface"):
        jac_s[e_k] = jac_s[e_k] @ (dec_jac[e_k].to_dense() if bkd.is_torch_backend() else dec_jac[e_k])

      # compute cres
      celem = "interface"
      if sub.constraint_type == "weak":  
        if sub.hr_active:
          cstate = sub.elem_states[celem]
          cstate.set_decoder_hr(active=False)
          cx, cdec_jac = cstate.decode(z[s][celem], with_jac=True)
          cstate.set_decoder_hr(active=True)
        else:
          cx = torch.cat([uv_s[celem][x_k] for x_k in ("u", "v")])
          cdec_jac = dec_jac[celem]
        cres_s = sub.cmat[celem] @ cx
        cjac_s = torch.t(sub.cmat[celem] @ cdec_jac) if bkd.is_torch_backend() else (sub.cmat[celem] @ cdec_jac).transpose()
      else:
        cres_s = sub.cmat[celem] @ z[s][celem]
        cjac_s = sub.cmat_interface_t

      # fom.sub.assemble_kkt
      interior_t = torch.t(jac_s["interior"]) if bkd.is_torch_backend() else jac_s["interior"].transpose()
      interface_t = torch.t(jac_s["interface"]) if bkd.is_torch_backend() else jac_s["interface"].transpose()
      if bkd.is_torch_backend():
        res_s = torch.cat([
          torch.atleast_1d(sub.scaling*(interior_t.to_dense() @ res_s)),
          torch.atleast_1d(sub.scaling*(interface_t.to_dense() @ res_s) + cjac_s.to_dense() @ lambdas)
        ])
      else:
        res_s = np.concatenate([
          sub.scaling*(interior_t@res_s),
          sub.scaling*(interface_t@res_s) + cjac_s@lambdas
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
    if bkd.distributed():
      if use_global:
        # assemble global residual
        global_res = [bkd.gatherv_tensor(r, as_list=True) for r in res]
        dist.reduce(cres, dst=0, op=dist.ReduceOp.SUM)
        if bkd.root():
          res = self.dd_fom.flatten_across_domain(global_res)
        res = bkd.broadcast_tensor(res, root=0)
        res = torch.cat(res) if bkd.is_torch_backend() else np.concatenate(res)
      else:
        # return rank-portion of global residual vector as DTensor
        res = torch.cat(res)
        if self.local_res_shapes is None:
          res_sizes = bkd.get_local_sizes_all(res, dim=0)
          self.local_res_shapes = res_sizes
        else:
          res_sizes = self.local_res_shapes
        res_sizes = np.sum(res_sizes)
        res = bkd.to_sharded_dtensor(res, shape=(res_sizes,), stride=(1,))
    else:
      res = np.concatenate(res)
    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime
    return res, cres

  def split_lambdas(self, x, use_global: bool = True):
    """Split a global or rank-local ROM KKT vector without DTensor reshards.

    A global vector has ``[all rank states, multipliers]`` layout.  A local
    vector has ``[this rank's state, multipliers]`` layout.  The latter is a
    regular tensor deliberately: converting an uneven state partition back
    through :meth:`DTensor.full_tensor` silently corrupts its ordering.
    """
    rank_size = int(np.sum(self.local_sizes))
    n_constraints = int(self.n_constraints)
    state_size = int(np.sum(self.global_sizes))
    if x.ndim != 1:
      raise ValueError("ROM KKT vector must be one-dimensional; got {}.".format(tuple(x.shape)))

    if use_global:
      expected_size = state_size + n_constraints
      if x.shape[0] != expected_size:
        raise ValueError(
          "Global ROM KKT vector has size {}; expected {} (state {} + constraints {}).".format(
            x.shape[0], expected_size, state_size, n_constraints,
          )
        )
      rank_offset = int(self.global_offsets[bkd.get_rank()]) if bkd.distributed() else 0
      rank_x = x[rank_offset:rank_offset + rank_size]
    else:
      expected_size = rank_size + n_constraints
      if x.shape[0] != expected_size:
        raise ValueError(
          "Local ROM KKT vector has size {}; expected {} (state {} + constraints {}).".format(
            x.shape[0], expected_size, rank_size, n_constraints,
          )
        )
      rank_x = x[:rank_size]

    lambdas = x[-n_constraints:] if n_constraints else x[:0]
    if self.debug:
      logger.debug(
        "SPLIT LAMBDAS: rank %s input=%s state=%s lambdas=%s",
        bkd.get_rank(), x.shape, rank_x.shape, lambdas.shape,
      )
    return rank_x, lambdas

  def local_trial_vector(self, x):
    """Return this rank's ROM state followed by the global multipliers."""
    if not bkd.distributed():
      return x
    rank_x, lambdas = self.split_lambdas(x, use_global=True)
    return torch.cat((rank_x, lambdas)) if bkd.is_torch_backend() else np.concatenate((rank_x, lambdas))

  # Residual/Jacobian
  # ===================================
  def res_jac(
    self,
    x,
    use_global: bool = True
  ):
    runtime = 0.0
    # Initialize
    # -------------
    start = time()
    res, hess, cjac = [], [], []
    cres = torch.zeros(self.n_constraints) if bkd.is_torch_backend() else np.zeros(self.n_constraints)
    # > Set Lagrangian multipliers
    lambdas = x[-self.n_constraints:]
    runtime += time()-start

    # Loop over subdomains
    # -------------
    if not use_global and bkd.distributed():
      rank_size = np.sum(self.local_sizes)
      if x.shape[0] != (rank_size + self.n_constraints):
        #print(" **** Inconsistent x size in res_jac! Got input {} expected {} ({} + {}) ****".format(x.shape[0], rank_size + self.n_constraints, rank_size, self.n_constraints))
        rank_offset = self.global_offsets[bkd.get_rank()]
        x = x[rank_offset:rank_offset+rank_size]

    z = self.extract_z_sub_from_vec(x, use_global)

    z_old = None
    if (not self.steady):
      z_old = self.extract_z_sub_from_vec(self.x_old, use_global)
    if self.force:
      force, *_ = self.dd_fom.assemble_sol(self.dd_fom.f, map_on_res=False, use_global=False)
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      global_s = self.global_subdomains[s]
      start_s = time()
      force_s = None
      if self.force:
        force_s = self.dd_fom.extract_uv_sub_from_dict(force, global_s)
      # > Compute quantities needed for KKT system
      res_s, cres_s, hess_s, cjac_s = sub.res_jac(
        z=z[s],
        lambdas=lambdas,
        steady=self.steady,
        dt=self.dt,
        z_old=z_old[s] if (z_old is not None) else None,
        force=force_s,
        class_name = self.dd_fom.__class__.__name__
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

    if bkd.distributed():
      if use_global:
        dist.reduce(cres, dst=0, op=dist.ReduceOp.SUM)
        global_res = [bkd.gatherv_tensor(r, as_list=True) for r in res]
        global_cjac = [bkd.gatherv_tensor(r, dim=1, as_list=True, coalesce=True) for r in cjac]
        global_hess = [bkd.gatherv_tensor(r, as_list=True) for r in hess]

        bkd.barrier()

        jac = None
        if bkd.root():
          global_res = self.dd_fom.flatten_across_domain(global_res)
          global_cjac = self.dd_fom.flatten_across_domain(global_cjac)
          global_hess = self.dd_fom.flatten_across_domain(global_hess)

          res, jac = self.dd_fom.assemble_kkt(global_res, cres, global_cjac, global_hess)

        res = bkd.broadcast_tensor(res, root=0)
        jac = bkd.broadcast_tensor(jac, root=0)
        bkd.barrier()
        jac = jac.to_sparse_csr()
      else:
        res, jac = self.dd_fom.assemble_kkt(res, cres, cjac, hess)
    else:
      res, jac = self.dd_fom.assemble_kkt(res, cres, cjac, hess)

    runtime += time()-start
    self.runtime["total"] += runtime
    self.runtime["res_jac"] += runtime

    self.resjac_it += 1
  
    return res, jac

  def get_local_offsets(self):
    """
    Returns a list containing the solution offsets for each LOCAL subdomain on this rank
    """
    # local offsets
    offsets = []
    sizes = []
    si = 0
    for (s, sub) in enumerate(self.subdomains):
      # local subdomains are packed, but they may correspond to different locations on the global grid depending
      # on the layout (e.g, if subdomains are split across ranks in columns)
      # interior rom dims are defined locally, but interface dimensions are assembled over the global domain
      # therefore, interface dimensions must be indexed according to our rank-local->global subdomain map
      # NOTE: here we can't yet use self.global_subdomains as it hasn't been assembled at the time of this call
      # instead, use the cached self._subs which is grabbed directly from the FOM (and currently we assume the 
      # domain split is the same between FOM and ROM)
      global_s = self._subs[s]
      nnodes = self.rom_dim["interior"][s] + self.rom_dim["interface"][global_s]
      sizes.append(nnodes)
      offsets.append(si)
      si = si + nnodes
    return offsets, sizes

  def get_global_offsets(self):
    # local offsets
    # get rank-local offsets
    local_offsets, local_sizes = self.get_local_offsets()

    # gather and assemble to global
    if bkd.distributed():
      global_sizes = bkd._COMM.allgather(np.sum(local_sizes))
    else:
      global_sizes = np.sum(local_sizes)
    global_sizes_t = np.cumsum(np.array(global_sizes).flatten())

    global_offsets = np.zeros_like(global_sizes_t)
    global_offsets[1:] = global_sizes_t[:-1]
    return global_offsets, global_sizes


  def extract_z_sub_from_vec(
    self,
    x,
    use_global: bool = True
  ):
    # NOTE: this always returns a list of LOCAL subdomain solution state from x, regardless of whether full solution is input (use_global=True)
    z = []
    si = 0
    runtime_s = 0.0

    # rank offset determines the shift in the global vector,
    # then the input is indexed according to local subdomain size (local subdomains are packed in order)
    rank_offset = 0
    if use_global and bkd.distributed():
      rank_offset = self.global_offsets[bkd.get_rank()]
    if self.debug: logger.debug("EXTRACT Z SUB: RANK {} offset = {} x shape = {} ({})".format(bkd.get_rank(), rank_offset, x.shape, type(x)))

    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      z_s = {}
      si = rank_offset + self.local_offsets[s]
      for e_k in ("interior", "interface"):
        ei = si + sub.rom_dim[e_k]
        #if self.debug: logger.debug(" ---- EXTRACT Z SUB: RANK {} - si = {} subdomain s = {} (global {}), ei = {} e_k = {} (x[{}:{}]) (x size = {})".format(bkd.get_rank(), si, s, self.global_subdomains[s], ei, e_k, si, ei, x.shape))
        z_s[e_k] = x[si:ei]
        # sanity checks for extracted solution state
        assert len(z_s[e_k]) > 0
        si = ei
      z.append(z_s)
      runtime_s = max(time()-start_s, runtime_s)
    self.runtime["total"] += runtime_s
    self.runtime["res_jac"] += runtime_s
    return z

  # Encode/Decode
  # ===================================
  def encode(
    self,
    x
  ):
    is_2d = (x.ndim == 2)
    if (is_2d and (x.shape[0] == 2*self.mesh.nxy)):
      x = x.T
    if is_2d:
      return np.vstack([self._encode(xi) for xi in x]).T
    else:
      return self._encode(x)

  def _encode(
    self,
    x
  ):
    if self.debug: logger.debug("ENCODE INPUT: RANK = {}: x = (shape = {})".format(bkd.get_rank(), x.shape))
    # Map on elements
    uv = self.dd_fom.map_sol_on_elements(x.reshape(1,-1))

    # Loop over subdomains/elements
    z = []
    for (s, sub) in enumerate(self.subdomains):
      for e_k in ("interior", "interface"):
        xi = uv[e_k][s].reshape(-1)
        zi = sub.elem_states[e_k].encode(xi, with_jac=False)
        z.append(zi)
    if bkd.is_torch_backend():
      z.append(torch.zeros(self.n_constraints))
      return torch.cat(z)
    else:
      z.append(np.zeros(self.n_constraints))
      return np.concatenate(z)

  def decode(
    self,
    x,
    map_on_res=False
  ):
    is_2d = (x.ndim == 2)
    shape = [self.mesh.nxy, x.shape[1]] if is_2d else [self.mesh.nxy]
    # Initialize solution containers
    uv = self.dd_fom.init_uv(shape=shape, map_on_res=map_on_res)
    z = {k: [] for k in ("interior", "interface")}
    # Loop over subdomains/elements
    si = 0

    # rank offset determines the shift in the global vector,
    # then the input is indexed according to local subdomain size (local subdomains are packed in order)
    rank_offset = 0
    if bkd.distributed():
      rank_offset = self.global_offsets[bkd.get_rank()]
      if self.debug: logger.debug("DECODE SUB: RANK {} offset = {} uv shape = {} X shape = {}".format(bkd.get_rank(), rank_offset, shape, x.shape))


    for (s, sub) in enumerate(self.subdomains):

      si = rank_offset + self.local_offsets[s]

      for e_k in ("interior", "interface"):
        # Extract latent space subvector
        ei = si + sub.rom_dim[e_k]
        #if self.debug: logger.debug("   DECODE SUB: RANK {} - si = {} subdomain s = {} (global {}), ei = {} e_k = {} (x[{}:{}])".format(bkd.get_rank(), si, s, self.global_subdomains[s], ei, e_k, si, ei))
        xi = x[si:ei]
        si = ei
        # Set element state
        state_k = sub.elem_states[e_k]
        state_k.set_decoder_hr(active=False)
        # Store latent space
        z[e_k].append(xi)
        # Reconstruct/store physical space
        if is_2d:
          uv_i = [state_k.decode(xj, with_jac=False) for xj in xi.T]
          uv_i = torch.vstack(uv_i).T if bkd.is_torch_backend() else np.vstack(uv_i).T
        else:
          uv_i = state_k.decode(xi, with_jac=False)
        uv = self.dd_fom.extract_uv_sub_from_vec(
          uv=uv,
          uv_i=uv_i,
          elem_state=sub.sub_fom.elem_states[e_k],
          map_on_res=map_on_res
        )
    lambdas = x[-self.n_constraints:]

    return uv, z, lambdas

  def reconstruct_static(
    self,
    x
  ):
    return self.decode(self.encode(x), map_on_res=True)[0]

  # Solution
  # ===================================
  def solve(
    self,
    x0=None,
    mu=None,
    dt=0.0,
    nt=1,
    steady=True,
    guess=None,
    use_guess=False,
    runtime=0.0,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    iostep: int = 1,
    verbose=False
  ):
    """
    Solves for the u and v states of the FOM using Newton's method.
    """
    nt = 1
    self.runtime = ops.map_nested_dict(self.runtime, lambda _: 0.0)
    self.runtime["total"] += runtime
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    # Initialize solution
    if (mu is not None):
      x0, runtime = self.rbf_model(mu)
      self.runtime["total"] += runtime
    # Initialize solver
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
    if bkd.is_torch_backend():
      x0 = bkd.to_backend(x0)
    if self.debug: logger.debug(" DDROM SOLVE: RANK {}: INITIAL X0 size = {}".format(bkd.get_rank(), x0.shape))
    x, res, *_, flag = solver(x0, dt, nt, guess, use_guess)
    if not isinstance(flag, List):
      flag = [flag]
    converged = True if (flag[-1] == 0) else False
    # Assemble solution
    uv, z, lambdas = self.assemble_sol(x, map_on_res=True)

    if bkd.is_torch_backend():
      uv = bkd.to_numpy(uv)
    return uv, z, lambdas, res, converged

  def get_init_sol(
    self,
    x
  ):
    if bkd.is_torch_backend():
      x = bkd.to_backend(x)
    if bkd.distributed():
      enc = self.encode(x)
      if self.debug: logger.debug("GET INIT SOL: RANK {} x.shape = {} ENC(X) = shape {}".format(bkd.get_rank(), x.shape, enc.shape))

      # split off constraints:
      offset = enc.shape[0] - self.n_constraints

      x_s = enc[:offset]
      x_c = enc[offset:]

      global_x_s = bkd.gatherv_tensor(x_s, as_list=True)
      if bkd.root():
        x_s = torch.cat(global_x_s)
      dist.reduce(x_c, dst=0, op=dist.ReduceOp.SUM)

      x_s = bkd.broadcast_tensor(x_s)
      x_c = bkd.broadcast_tensor(x_c)

      enc = torch.hstack([x_s, x_c])

      if self.debug: logger.debug("GET INIT SOL: RANK {} RETURN enc ({})= [{}, {}]".format(bkd.get_rank(), enc.shape, x_s.shape, x_c.shape))

      return enc
    else:
      enc = self.encode(x)

      return enc

  def assemble_sol(
    self,
    x,
    map_on_res=False
  ):
    return self.decode(x, map_on_res)

  def compute_error(
    self,
    uv_fom,
    uv_rom,
    scaling=False,
    relative=True,
    axis=None
  ):
    """
    Compute error between DD-ROM and DD-FOM DD solutions.
    """
    err = 0.0
    for s in range(self.mesh.n_sub):
      num_s, den_s = 0.0, 0.0
      for e_k in ("interior", "interface"):
        for x_k in ("u", "v"):
          x_fom = uv_fom[e_k][x_k][s]
          x_rom = uv_rom[e_k][x_k][s]
          num_s += np.sum(np.square(x_rom - x_fom), axis=axis)
          if relative:
            den_s += np.sum(np.square(x_fom), axis=axis)
      err += num_s/den_s if relative else num_s
    if scaling:
      err *= self.mesh.hxy
    return np.amax(np.sqrt(err/self.mesh.n_sub))

  def compute_error_new(
    self,
    uv_fom,
    uv_rom,
    scaling=False,
    relative=True,
    axis=0,
    return_dict=True,
    rel_eps=1e-6,
  ):
    """
    Compute error between DD-ROM and DD-FOM DD solutions.

    Default return (return_dict=False):
      - float: L2_timemax  (same as your original function)

    If return_dict=True:
      - dict with:
          "L2_timemax"   : max-in-time L2 error (matches your original)
          "Linf_timeavg" : time-averaged spatial L-infinity error
    """

    err = 0.0

    linf_err_t = None
    linf_fom_t = None
    fom_global_sup = 0.0

    for s in range(len(self.local_subdomains)):
        num_s, den_s = 0.0, 0.0  # per-subdomain accumulators (vectors over time)
        global_sub = self.global_subdomains[s]
        for e_k in ("interior", "interface"):
            for x_k in ("u", "v"):

                x_fom = uv_fom[e_k][x_k][global_sub]
                x_rom = uv_rom[e_k][x_k][s]
                x_rom_shape = x_rom.shape[-1]
                # trim fom solution in case rom isn't over all timesteps (mainly for debugging)
                if x_fom.shape[-1] != x_rom.shape[-1]:
                  x_fom = x_fom[:, 0:x_rom.shape[-1]]
                if bkd.is_torch_backend():
                  x_fom = bkd.to_numpy(x_fom)
                  x_rom = bkd.to_numpy(x_rom)

                # ---- L2 contributions (sum over space, keep time) ----
                num_s += np.sum((x_rom - x_fom) ** 2, axis=axis)
                if relative:
                    den_s += np.sum((x_fom) ** 2, axis=axis)

                # ---- Linf contribution for this component (max over space, keep time) ----
                e_sup = np.max(np.abs(x_rom - x_fom), axis=axis)
                f_sup = np.max(np.abs(x_fom),     axis=axis)

                if linf_err_t is None:
                    linf_err_t = e_sup.astype(float, copy=True)
                    linf_fom_t = f_sup.astype(float, copy=True)
                else:
                    linf_err_t = np.maximum(linf_err_t, e_sup)
                    linf_fom_t = np.maximum(linf_fom_t, f_sup)

                fom_global_sup = max(fom_global_sup, float(np.max(f_sup)))

        # accumulate L2 across subdomains
        err += (num_s / np.maximum(den_s, 1e-30)) if relative else num_s

    if bkd.distributed():
      bkd.barrier()
      req = []
      req.append(bkd._COMM.Iallreduce(MPI.IN_PLACE, err, op=MPI.SUM))
      req.append(bkd._COMM.Iallreduce(MPI.IN_PLACE, linf_err_t, op=MPI.MAX))
      req.append(bkd._COMM.Iallreduce(MPI.IN_PLACE, linf_fom_t, op=MPI.MAX))
      fom_global_sup = bkd._COMM.allreduce(fom_global_sup, op=MPI.MAX)
      MPI.Request.Waitall(req)
      bkd.barrier()
      parallel_print(" COMPUTE ERR (GLOBAL) RANK {}: err = {} linf_err_t = {} linf_fom_t = {} fom_global_sup = {}".format(bkd.get_rank(), err, linf_err_t, linf_fom_t, fom_global_sup))

    # Apply area scaling to L2
    if scaling:
        err *= self.mesh.hxy

    L2_t = np.sqrt(err / self.mesh.n_sub)

    # Linf-L2 metric
    L2_timemax = float(np.amax(L2_t))
    if not return_dict:
        return L2_timemax
    # L2-L2 metric
    L2_timeavg   = float(np.mean(L2_t))

    if relative:
        eps = max(1e-12, rel_eps * max(1.0, fom_global_sup))
        denom = np.maximum(linf_fom_t, eps)
        linf_t = linf_err_t / denom
    else:
        linf_t = linf_err_t

    # L2-Linf metric
    Linf_timeavg = float(np.mean(linf_t))
    # Linf-Linf
    Linf_timemax  = float(np.max(linf_t))


    return {
        "L2_timemax": L2_timemax,
        "L2_timeavg": L2_timeavg,
        "Linf_timeavg": Linf_timeavg,
        "Linf_timemax": Linf_timemax,
    }
