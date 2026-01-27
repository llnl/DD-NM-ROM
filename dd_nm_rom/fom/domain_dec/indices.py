import numpy as np

from . import dtypes


class DDIndices(object):
  """
  Class for generating residual, interior, and interface subdomain 
  indices for a steady-state 2D Burgers FOM.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    monolithic: dtypes.FOM_TYPE
  ) -> None:
    self.monolithic = monolithic
    for k in ("mesh", "ops"):
      setattr(self, k, getattr(self.monolithic, k))
    self.ops = list(self.ops.values())
    # Active residual nodes for each subdomain
    self.res = self.mesh.res_nodes
    # Set ports
    self.n_ports = 0
    self.ports = None
    self.port_to_subs = None
    self.port_to_nodes = None
    self.built = False

  # Building
  # ===================================
  def is_built(self) -> None:
    if (not self.built):
      raise ValueError(
        "Indices for DD are not built. Please, call 'build' method first."
      )

  def build(self):
    # Interior and interface
    self.set_allactive_nodes()
    self.set_interior_interface_nodes()
    # Ports
    self.set_ports_nodes(self.skeleton_inner)
    self.set_ports_nodes(self.skeleton_bound)
    self.set_ports_maps()
    self.reorder_port_nodes()
    self.built = True

  # All-active (interior+interface) nodes
  # -----------------------------------
  def set_allactive_nodes(self) -> None:
    self.allactive = []
    for res_s in self.res:
      # Initialize subdomain "all" indices
      all_s = set(res_s)
      # For each row (i.e., node in the grid),
      # finds nonzero columns (due to FD stencil)
      for op in self.ops:
        cols = op[res_s].nonzero()[1]
        cols = np.unique(cols).tolist()
        all_s = all_s.union(set(cols))
      # Store indices for current subdomain
      all_s = np.sort(np.array(list(all_s)))
      self.allactive.append(all_s)

  # Interior and interface nodes
  # -----------------------------------
  def set_interior_interface_nodes(self) -> None:
    self.skeleton = set()     # All interface nodes
    self.interior = []        # Interior nodes for each subdomain
    self.interface = []       # Interface nodes for each subdomain
    # Loop over subs
    subs = np.arange(self.mesh.n_sub)
    for i in subs:
      # Set i-th subdomain
      sub_i = set(self.allactive[i])
      # Set remaining subs
      subs_left = np.delete(subs, i)
      # Define interface nodes for i-th subdomain
      intf_i = set()
      for j in subs_left:
        # > Take intersection between subdomain i and j
        sub_j = set(self.allactive[j])
        intf_ij = sub_i.intersection(sub_j)
        intf_i = intf_i.union(intf_ij)
      # Define interior nodes for i-th subdomain
      intr_i = sub_i.difference(intf_i)
      # Store i-th subdomain indices
      self.skeleton = self.skeleton.union(intf_i)
      self.interior.append(np.sort(np.array(list(intr_i))))
      self.interface.append(np.sort(np.array(list(intf_i))))
    self.skeleton = np.array(list(self.skeleton))
    self.skeleton_inner = np.intersect1d(self.skeleton, self.mesh.nodes_ind_inner)
    self.skeleton_bound = np.intersect1d(self.skeleton, self.mesh.nodes_ind_bound)

  # Ports nodes
  # -----------------------------------
  def set_ports_nodes(
    self,
    skeleton: np.ndarray
  ) -> None:
    if (len(skeleton) == 0):
      return None
    # Assign each interface node to subdomains
    # -------------
    node_intf_to_subs = np.zeros((len(skeleton), self.mesh.n_sub), dtype=bool)
    for (i, node_i) in enumerate(skeleton):
      for (j, intf_j) in enumerate(self.interface):
        node_intf_to_subs[i,j] = node_i in intf_j
    # Assign subdomains to each port
    # -------------
    subs = np.arange(self.mesh.n_sub)
    _port_to_subs = set([])
    for mask in node_intf_to_subs:
      _port_to_subs.add(frozenset(subs[mask]))
    _port_to_subs = list(_port_to_subs)
    for (p, subs_p) in enumerate(_port_to_subs):
      _port_to_subs[p] = np.array(list(subs_p))
    # Assign nodes to each port
    # -------------
    _port_to_nodes = []
    for subs_p in _port_to_subs:
      indices = np.zeros(self.mesh.n_sub, dtype=bool)
      indices[subs_p] = True
      mask = (node_intf_to_subs == indices).all(axis=1)
      _port_to_nodes.append(np.sort(skeleton[mask]))
    # Create containers for ports information
    # -------------
    port_to_subs, port_to_nodes = {}, {}
    for (p, subs_p) in enumerate(_port_to_subs):
      if (len(subs_p) > 2):
        # > Split single-node ports
        for node in _port_to_nodes[p]:
          port_to_subs[self.n_ports] = subs_p
          port_to_nodes[self.n_ports] = np.array(node).reshape(1)
          self.n_ports += 1
      else:
        port_to_subs[self.n_ports] = subs_p
        port_to_nodes[self.n_ports] = _port_to_nodes[p]
        self.n_ports += 1
    # > List all the ports
    ports = np.array(list(port_to_subs.keys()))
    # Update class attributes
    # -------------
    if (self.ports is None):
      self.ports = ports
      self.port_to_subs = port_to_subs
      self.port_to_nodes = port_to_nodes
    else:
      self.ports = np.sort(np.append(self.ports, ports))
      self.port_to_subs.update(port_to_subs)
      self.port_to_nodes.update(port_to_nodes)

  # Ports nodes - Reordering
  # -----------------------------------
  def reorder_port_nodes(self) -> None:
    for p in self.ports:
      self.port_to_nodes[p] = self._reorder_port_nodes(
        orient=self.port_to_orientsize[p][0],
        nodes=self.port_to_nodes[p]
      )

  def _reorder_port_nodes(
    self,
    orient: str,
    nodes: np.ndarray
  ) -> np.ndarray:
    nodes = np.sort(nodes)
    if ((len(nodes) > 1) and np.in1d(nodes, self.skeleton_bound).all()):
      if (orient == "vert"):
        shape, axis = (-1,2), 1
      else:
        shape, axis = (2,-1), 0
      nodes = np.flip(nodes.reshape(*shape), axis=axis).flatten()
    return nodes

  # Ports maps
  # -----------------------------------
  def set_ports_maps(self):
    self._set_map_sub_to_ports()
    self._set_map_size_to_ports()
    self._set_map_orient_to_ports()
    self._set_map_orientsize_to_ports()

  def _set_map_sub_to_ports(self) -> None:
    """
    Assigns each port to multiple subdomains
    """
    self.sub_to_ports = {}
    for s in range(self.mesh.n_sub):
      sub = set([])
      for (p, subs_p) in self.port_to_subs.items():
        if (s in subs_p):
          sub.add(p)
      self.sub_to_ports[s] = np.sort(list(sub))

  def _set_map_size_to_ports(self) -> None:
    """
    Assigns each port to its size.
    """
    self.size_to_ports = {}
    for p in self.ports:
      size = self.port_to_nodes[p].size
      if (size not in self.size_to_ports):
        self.size_to_ports[size] = []
      self.size_to_ports[size].append(p)

  def _set_map_orient_to_ports(self) -> None:
    """
    Assigns each port to its orientation.
    """
    self.orient_to_ports = {"vert": [], "horiz": [], "inner": []}
    for (p, subs_p) in self.port_to_subs.items():
      if (subs_p.size > 2):
        self.orient_to_ports["inner"].append(p)
      else:
        sub_yx_1 = self.mesh.sub_combs[subs_p[0]]
        sub_yx_2 = self.mesh.sub_combs[subs_p[1]]
        if (sub_yx_1[1]-sub_yx_2[1] == 0):
          self.orient_to_ports["horiz"].append(p)
        else:
          self.orient_to_ports["vert"].append(p)

  def _set_map_orientsize_to_ports(self) -> None:
    """
    Assigns each port to its orientation and size.
    """
    # From orientation-size to ports
    self.orientsize_to_ports = {}
    for (orient, ports) in self.orient_to_ports.items():
      self.orientsize_to_ports[orient] = {}
      for p in ports:
        size = self.port_to_nodes[self.ports[p]].size
        if (size not in self.orientsize_to_ports[orient]):
          self.orientsize_to_ports[orient][size] = []
        self.orientsize_to_ports[orient][size].append(p)
    # From port to orientation-size
    self.port_to_orientsize = {}
    for p in self.ports:
      for (orient, size_to_ports) in self.orientsize_to_ports.items():
        for (size, ports) in size_to_ports.items():
          if (p in ports):
            self.port_to_orientsize[p] = (orient, size)
