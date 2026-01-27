import abc
import copy
import numpy as np
import scipy.sparse as sp

from dd_nm_rom.ops import sp_diag
from dd_nm_rom.rom.utils import hyper_red as hr

from . import activation as act_mod


class Block(object):

  def __init__(
    self,
    config
  ):
    self.config = config
    for k in ("input_dim", "latent_dim"):
      setattr(self, k, self.config[k])
    self._activation = act_mod.get('linear')
    # Model weights
    self.set_weights()
    # Set weights/activation
    self.w = self._w
    self.activation = self._activation

  def set_weights(self):
    self._w = self.config["weights"]

  def __call__(self, x, with_jac=True):
    return self.fun_jac(x) if with_jac else self.fun(x)

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

  def fun(self, x):
    # Apply encoder
    z = self.w['W1'] @ x
    z = self.activation(z, with_jac=False)
    return z

  def fun_jac(self, x):
    # Apply encoder
    z = self.w['W1'] @ x
    jac = self.w['W1']
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

  def set_weights(self):
    super(Decoder, self).set_weights()

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
    x = self.w['W1'] @ z
    return x

  def fun_jac(self, z):
    # Apply decoder
    x = self.w['W1'] @ z 
    jac = self.w['W1']
    # Return output and Jacobian
    return x, jac


class Autoencoder(object):

  def __init__(
    self,
    config
  ):
    self.name = "autoencoder"
    self.config = config
    for k in ("input_dim", "latent_dim"):
      setattr(self, k, self.config["decoder"][k])
    self.activation = 'linear'

    # Layers
    self.decoder = Decoder(self.config["decoder"])
    self.encoder = Encoder(self.config["encoder"])

  def __call__(self, x):
    return self.decoder(self.encoder(x, with_jac=False), with_jac=False)

  def set_hr_mode(
    self,
    active=False,
    row_ind=None
  ):
    self.decoder.set_hr_mode(active=active, row_ind=row_ind)


class MultiAutoencoder(Autoencoder):
  # Combine multiple autoencoders into a single unified structure.
  # Integrate multiply autoencoders, each with its own encoder and decoder layers, into a single model.
  # This single model contains an encoder and a decoder with one layer in each.

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
    cfg = copy.deepcopy(config_init)
    # Loop over layers
    for l in ("encoder", "decoder"):
      cfg = copy.deepcopy(config_init)
      # Update configuration by looping over ports
      for (k, autoencoder) in self.autoencoders.items():
        layer = getattr(autoencoder, l)
        cfg = self._update_config(
          config=cfg,
          layer=layer,
          indices=self.indices[k]
        )
      # Assemble weights
      if l == "encoder":
        if cfg["weights"]["W1"]:  # Check if list is not empty
          combined_matrix = cfg["weights"]["W1"][0]
          for matrix in cfg["weights"]["W1"][1:]:
            combined_matrix = combined_matrix + matrix
          cfg["weights"]["W1"] = combined_matrix
      elif l == "decoder":
        if cfg["weights"]["W1"]:  # Check if list is not empty
          combined_matrix = cfg["weights"]["W1"][0]
          for matrix in cfg["weights"]["W1"][1:]:
            combined_matrix = combined_matrix + matrix
          cfg["weights"]["W1"] = combined_matrix
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
      "ref": np.zeros(self.input_dim),
      "scale": np.ones(self.input_dim),
      "mask_shape": None,
      "mask_indices": None,
      "weights": {"W1": []}
    }

  def _update_config(
    self,
    config,
    layer,
    indices
  ):
    # Set dimensions
    dims = {k: config[k+"_dim"] for k in ("input", "latent")}
    # Get weights
    if (layer.name == "encoder"):
      get_weights = self._get_weights_single_encoder
    else:
      get_weights = self._get_weights_single_decoder
    weights = get_weights(dims, layer, indices)
    # Store weights
    for w in ("W1", ):
      config["weights"][w].append(weights[w])
    return config

  def _get_weights_single_encoder(
    self,
    dims,
    layer,
    indices
  ):
    # Input layer
    W1 = np.zeros((dims["latent"], dims["input"]))
    mesh_indices = np.ix_(indices["rom"], indices["fom"])
    W1[mesh_indices] += layer._w["W1"]
    W1 = sp.csr_matrix(W1)
    return {"W1": W1}

  def _get_weights_single_decoder(
    self,
    dims,
    layer,
    indices
  ):
    # Input layer
    W1 = np.zeros((dims["input"], dims["latent"]))
    mesh_indices = np.ix_(indices["fom"], indices["rom"])
    W1[mesh_indices] += layer._w["W1"]
    W1 = sp.csr_matrix(W1)
    # Return weights
    return {"W1": W1}
