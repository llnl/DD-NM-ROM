import abc
import copy
import numpy as np
import scipy.sparse as sp
import torch

from dd_nm_rom.ops import sp_diag
from dd_nm_rom.rom.utils import hyper_red as hr
from dd_nm_rom.utils import parallel_print
import dd_nm_rom.backend as bkd

from . import activation as act_mod


class Block(object):

  def __init__(
    self,
    config
  ):
    self.config = config
    for k in ("input_dim", "latent_dim"):
      setattr(self, k, self.config[k])
    if isinstance(config["activation"], dict):
      self._activation = act_mod.get(**config["activation"])
    else:
      self._activation = act_mod.get(config["activation"])
    # Model weights
    self.set_weights()
    # Set weights/activation
    self.w = self._w
    self.activation = self._activation


  def __eq__(self, other):
    comp = (self.input_dim == other.input_dim)
    comp &= (self.latent_dim == other.latent_dim)
    comp &= (len(self.w) == len(other.w))
    for (k,v) in self.w.items():
      comp &= ((k in other.w) and (bkd.same_ptr(v, other.w[k]) or bkd.tensor_eq(v, other.w[k], check_indices=True)))
    return comp

  @classmethod
  def makeShared(cls, other):
    """
    Makes a shallow copy of this autoencoder such that all tensors are created using views of those owned by other
    """
    view = cls.__new__(cls)
    view.config = copy.copy(other.config)
    view.input_dim = other.input_dim
    view.latent_dim = other.latent_dim
    view.name = getattr(other, "name", cls.__name__.lower())
    view._activation = copy.copy(other._activation)
    view.activation = view._activation
    view.w = {k: copy.copy(v) for (k, v) in other.w.items()}
    view._w = view.w
    return view

  def set_weights(self):
    self._w = self.config["weights"]
    self._w["ref"] = self.config["ref"]
    self._w["scale"] = self.config["scale"]
    self._w["ov_scale"] = 1.0/self.config["scale"]

    if bkd.is_torch_backend():
      w_torch = {}
      for (k, b) in self._w.items():
        if isinstance(b, np.ndarray):
          b = bkd.to_backend(b)
          w_torch[k] = b
        elif isinstance(b, sp.spmatrix):
          if sp.isspmatrix_csr(b):
            w_torch[k] = bkd.to_sp_backend(b)
          else:
            w_torch[k] = bkd.to_sp_coo_backend(b.tocoo())
        elif isinstance(b, torch.Tensor):
          w_torch[k] = b
        else:
          raise RuntimeError("Unexpected type for key {}: type = {}, {}".format(k, type(b), b))
      self._w = w_torch

    for k in ("scale", "ov_scale"):
      self._w[k+"_diag"] = sp_diag(self._w[k])

  def __call__(self, x, with_jac=True):
    return self.fun_jac(x) if with_jac else self.fun(x)

  def _warmup_activation(self, activation, size, reference):
    if not bkd.is_torch_backend():
      return
    act_mod.warmup(
      activation,
      int(size),
      device=bkd.device(),
      dtype=reference.dtype
    )

  @abc.abstractmethod
  def fun(self, x):
    pass

  @abc.abstractmethod
  def fun_jac(self, x):
    pass


class Encoder(Block):

  def __init__(
    self,
    config
  ):
    self.name = "encoder"
    super(Encoder, self).__init__(config)

  def set_weights(self):
    super(Encoder, self).set_weights()
    self._w["W1_scale"] = self._w["W1"] @ self._w["ov_scale_diag"]
    self._w["b1_ref"] = self._w["b1"] - self._w["W1_scale"] @ self._w["ref"]


  def __eq__(self, other):
    comp = (self.name == other.name)
    comp &= super().__eq__(other)
    return comp

  def compile_activations(self):
    self._warmup_activation(
      self.activation,
      self.w["b1_ref"].shape[0],
      self.w["b1_ref"]
    )

  def fun(self, x):
    # Apply encoder
    z = self.w["W1_scale"] @ x + self.w["b1_ref"]
    z = self.activation(z, with_jac=False)
    z = self.w["W2"] @ z
    return z

  def fun_jac(self, x):
    # Apply encoder
    z = self.w["W1_scale"] @ x + self.w["b1_ref"]
    z, dz = self.activation(z, with_jac=True)
    z = self.w["W2"] @ z
    if bkd.is_torch_backend():
      jac = (self.w["W2"] * dz.unsqueeze(0)) @ self.w["W1_scale"]
    else:
      jac = self.w["W2"] @ dz @ self.w["W1_scale"]
    # Return output and Jacobian
    return z, jac


class Decoder(Block):

  def __init__(
    self,
    config
  ):
    self.name = "decoder"
    super(Decoder, self).__init__(config)
    # Hyper-reduction (HR) weights
    self._w_hr = None
    self.hr_active = False

  def __eq__(self, other):
    comp = (self.name == other.name)
    comp &= super().__eq__(other)
    return comp

  def compile_activations(self):
    self._warmup_activation(
      self.activation,
      self.w["b1"].shape[0],
      self.w["b1"]
    )


  @classmethod
  def makeShared(cls, other):
    view = cls.__new__(cls)
    view.config = copy.copy(other.config)
    view.input_dim = other.input_dim
    view.latent_dim = other.latent_dim
    view._activation = copy.copy(other._activation)
    view.activation = view._activation
    view._w = {k: copy.copy(v) for (k, v) in other._w.items()}
    view.w = view._w
    view._w_hr = other._w_hr
    view.hr_active = other.hr_active
    view.name = other.name
    return view

  
  def set_weights(self):
    super(Decoder, self).set_weights()
    self._w["scale_W2"] = self._w["scale_diag"] @ self._w["W2"]

  def set_hr_mode(
    self,
    active=False,
    row_ind=None
  ):
    self.hr_active = active
    if self.hr_active:
      if (self._w_hr is None):
        self.set_weights_act_hr(row_ind)
      self.w = self._w_hr
      self.activation = self._activation_hr
    else:
      self.w = self._w
      self.activation = self._activation

    if bkd.is_torch_backend():
      w_torch = {}
      for (k, b) in self.w.items():
        if isinstance(b, np.ndarray):
          b = bkd.to_backend(b)
          w_torch[k] = b
        elif isinstance(b, sp.spmatrix):
          if sp.isspmatrix_csr(b):
            b = bkd.to_sp_backend(b)
            w_torch[k] = b
          else:
            b = bkd.to_sp_backend(bkd.to_sparse(b))
            w_torch[k] = b
        elif isinstance(b, torch.Tensor):
          w_torch[k] = b
        else:
          raise RuntimeError("Unexpected type for key {}: type = {}, {}".format(k, type(b), b))
      self.w = w_torch

  def set_weights_act_hr(
    self,
    row_ind
  ):
    col_ind = hr.get_col_indices(row_ind, self._w["W2"])
    # Weights
    # -------------
    # Initialize weights
    self._w_hr = copy.deepcopy(self._w)
    # Input layer
    for k in ("b1", "W1"):
      self._w_hr[k] = self._w_hr[k][col_ind]
    # Hidden layer
    submat = np.ix_(row_ind, col_ind)
    for k in ("W2", "scale_W2"):
      self._w_hr[k] = self._w_hr[k][submat]
    # Normalization layer
    for k in ("ref", "scale", "ov_scale"):
      self._w_hr[k] = self._w_hr[k][row_ind]
      if ("scale" in k):
        self._w_hr[k+"_diag"] = sp_diag(self._w_hr[k])
    # Masked activation function
    # -------------
    if isinstance(self._activation, act_mod.Mixed):
      # > Map masked activations to a unique 1d array
      acts = np.full(len(self._w["b1"]), "linear", dtype=object)
      for (act_id, (act_obj, mask)) in self._activation.masks.items():
        acts[mask] = act_id
      # > Extract new masks for HR
      masks_hr = {}
      acts_hr = acts[col_ind]
      for act_id in np.unique(acts_hr):
        masks_hr[act_id] = np.where(acts_hr == act_id)[0]
      self._activation_hr = act_mod.Mixed(masks_hr)
    else:
      self._activation_hr = self._activation

  def fun(self, z):
    # Apply decoder
    x = self.w["W1"] @ z + self.w["b1"]
    x = self.activation(x, with_jac=False)
    x = self.w["scale_W2"] @ x + self.w["ref"]
    return x

  def fun_jac(self, z):
    # Apply decoder
    x = self.w["W1"] @ z + self.w["b1"]
    x, dx = self.activation(x, with_jac=True)
    x = self.w["scale_W2"] @ x + self.w["ref"]
    if bkd.is_torch_backend():
      jac = self.w["scale_W2"] @ sp_diag(dx) @ self.w["W1"]
    else:
      jac = self.w["scale_W2"] @ dx @ self.w["W1"]
    # Return output and Jacobian
    return x, jac


class Autoencoder(object):

  def __init__(
    self,
    config
  ):
    self.name = "autoencoder"
    self.config = config
    for k in ("input_dim", "latent_dim", "activation"):
      setattr(self, k, self.config["decoder"][k])
    if isinstance(self.activation, str):
      self.activation = self.activation.lower()
    # Layers
    self.decoder = Decoder(self.config["decoder"])
    self.encoder = Encoder(self.config["encoder"])


  @classmethod
  def makeShared(cls, other):
    """
    Makes a shallow copy of this autoencoder such that all tensors are created using views of those owned by other
    """
    import tracemalloc
    if not tracemalloc.is_tracing():
      mem_start = bkd._start_mem_trace(True)
    else:
      mem_start = tracemalloc.get_traced_memory()

    view = cls.__new__(cls)
    view.name = other.name
    view.config = copy.copy(other.config)
    view.input_dim = other.input_dim
    view.latent_dim = other.latent_dim
    view.activation = other.activation
    view.decoder = Decoder.makeShared(other.decoder)
    view.encoder = Encoder.makeShared(other.encoder)
    mem_end = tracemalloc.get_traced_memory()
    bkd._get_mem_trace_stats(mem_start, mem_end, print_stats=True)

    return view

  def __copy__(self):
    return Autoencoder(copy.copy(self.config))


  def __call__(self, x):
    return self.decoder(self.encoder(x, with_jac=False), with_jac=False)

  def compile_activations(self):
    self.encoder.compile_activations()
    self.decoder.compile_activations()

  def __eq__(self, other):
    comp = (self.name == other.name)
    #comp &= (self.config == other.config)
    comp &= (self.input_dim == other.input_dim)
    comp &= (self.latent_dim == other.latent_dim)
    comp &= (self.decoder == other.decoder)
    comp &= (self.encoder == other.encoder)
    return comp
  
  def set_hr_mode(
    self,
    active=False,
    row_ind=None
  ):
    self.decoder.set_hr_mode(active=active, row_ind=row_ind)


class MultiAutoencoder(Autoencoder):

  def __init__(
    self,
    indices,
    input_dim,
    autoencoders
  ):
    self.indices = indices
    self.input_dim = input_dim
    self.autoencoders = autoencoders

    super(MultiAutoencoder, self).__init__(self.get_config())

  # ROM dimensions
  # ===================================
  def get_config(self):
    config = {}
    config_init = self._init_config()
    # Loop over layers
    for l in ("encoder", "decoder"):
      cfg = copy.deepcopy(config_init)
      # Update configuration by looping over ports
      if self.mixed_act:
        shift = 0
      for (k, autoencoder) in self.autoencoders.items():
        layer = getattr(autoencoder, l)
        cfg = self._update_config(
          config=cfg,
          layer=layer,
          indices=self.indices[k]
        )
        # Update mixed activation function
        if self.mixed_act:
          dim = layer.config["hidden_dim"]
          ind = np.arange(dim) + shift
          act = autoencoder.activation
          cfg["activation"]["masks"][act].append(ind)
          shift += dim
      if self.mixed_act:
        for (act, indices) in cfg["activation"]["masks"].items():
          cfg["activation"]["masks"][act] = np.sort(np.concatenate(indices))
      # Assemble weights
      if bkd.is_torch_backend():
        stack = cfg["weights"]["W1"][0].to_dense()
        for spw in range(1, len(cfg["weights"]["W1"])):
          stack = torch.vstack((stack, cfg["weights"]["W1"][spw].to_dense()))
        cfg["weights"]["W1"] = bkd.to_sp_backend(stack)
        stack = cfg["weights"]["W2"][0].to_dense()
        for spw in range(1, len(cfg["weights"]["W2"])):
          stack = torch.hstack((stack, cfg["weights"]["W2"][spw].to_dense()))
        cfg["weights"]["W2"] = bkd.to_sp_backend(stack)
        cfg["weights"]["b1"] = torch.cat(cfg["weights"]["b1"])
      else:
        cfg["weights"]["W1"] = sp.vstack(cfg["weights"]["W1"])
        cfg["weights"]["W2"] = sp.hstack(cfg["weights"]["W2"])
        cfg["weights"]["b1"] = np.concatenate(cfg["weights"]["b1"])
      # Store configuration
      config[l] = cfg
    return config

  def _init_config(self):
    latent_dim = 0
    activation = set()
    for autoencoder in self.autoencoders.values():
      latent_dim += autoencoder.latent_dim
      activation.add(autoencoder.activation)
    if (len(activation) != 1):
      self.mixed_act = True
      activation = {
        "identifier": "mixed",
        "masks": {act: [] for act in activation}
      }
    else:
      self.mixed_act = False
      activation = list(activation)[0]
    return {
      "input_dim": self.input_dim,
      "latent_dim": latent_dim,
      "activation": activation,
      "ref": torch.zeros(self.input_dim) if bkd.is_torch_backend() else np.zeros(self.input_dim),
      "scale": torch.ones(self.input_dim) if bkd.is_torch_backend() else np.ones(self.input_dim),
      "mask_shape": None,
      "mask_indices": None,
      "weights": {w: [] for w in ("W1", "b1", "W2")}
    }

  def _update_config(
    self,
    config,
    layer,
    indices
  ):
    # Set dimensions
    dims = {k: config[k+"_dim"] for k in ("input", "latent")}
    dims["hidden"] = layer.config["hidden_dim"]
    # Get weights
    if (layer.name == "encoder"):
      get_weights = self._get_weights_single_encoder
    else:
      get_weights = self._get_weights_single_decoder
    weights = get_weights(dims, layer, indices)
    # Store weights
    for w in ("W1", "b1", "W2"):
      config["weights"][w].append(weights[w])
    # Update reference value
    if (layer.name == "decoder"):
      config["ref"][indices["fom"]] += layer._w["ref"]
    return config

  def _get_weights_single_encoder(
    self,
    dims,
    layer,
    indices
  ):
    # Input layer
    if bkd.is_torch_backend():
      W1 = torch.zeros((dims["hidden"], dims["input"]))
      W1[:,indices["fom"]] += layer._w["W1_scale"]
      W1 = bkd.to_sp_backend(W1)
      # Hidden layer
      W2 = torch.zeros((dims["latent"], dims["hidden"]))
      W2[indices["rom"],:] += layer._w["W2"]
      W2 = bkd.to_sp_backend(W2)
    else:
      W1 = np.zeros((dims["hidden"], dims["input"]))
      W1[:,indices["fom"]] += layer._w["W1_scale"]
      W1 = sp.csr_matrix(W1)
      # Hidden layer
      W2 = np.zeros((dims["latent"], dims["hidden"]))
      W2[indices["rom"],:] += layer._w["W2"]
      W2 = sp.csr_matrix(W2)

    b1 = layer._w["b1_ref"]
    # Return weights
    return {"W1": W1, "b1": b1, "W2": W2}

  def _get_weights_single_decoder(
    self,
    dims,
    layer,
    indices
  ):
    # Input layer
    b1 = layer._w["b1"]
    if bkd.is_torch_backend():
      W1 = torch.zeros((dims["hidden"], dims["latent"]))
      W1[:,indices["rom"]] += layer._w["W1"]
      W1 = bkd.to_sp_backend(W1)
      # Hidden layer
      W2 = torch.zeros((dims["input"], dims["hidden"]))
      W2[indices["fom"],:] += layer._w["scale_W2"]
      W2 = bkd.to_sp_backend(W2)
    else:
      W1 = np.zeros((dims["hidden"], dims["latent"]))
      W1[:,indices["rom"]] += layer._w["W1"]
      W1 = sp.csr_matrix(W1)
      # Hidden layer
      W2 = np.zeros((dims["input"], dims["hidden"]))
      W2[indices["fom"],:] += layer._w["scale_W2"]
      W2 = sp.csr_matrix(W2)
    # Return weights
    return {"W1": W1, "b1": b1, "W2": W2}
