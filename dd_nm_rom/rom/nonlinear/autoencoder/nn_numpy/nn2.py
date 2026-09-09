"""
Supports Encoder with multiple dense layers
"""
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
    if isinstance(config["activation"], dict):
      self._activation = act_mod.get(**config["activation"])
    else:
      self._activation = act_mod.get(config["activation"])
    # Model weights
    self.set_weights()
    # Set weights/activation
    self.w = self._w
    self.activation = self._activation

  def set_weights(self):
    self._w = self.config["weights"]
    self._w["ref"] = self.config["ref"]
    self._w["scale"] = self.config["scale"]
    self._w["ov_scale"] = 1.0/self.config["scale"]
    for k in ("scale", "ov_scale"):
      self._w[k+"_diag"] = sp_diag(self._w[k])

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
    self.weight_keys = sorted([k for k in self._w.keys() if k.startswith("W") and k[1:].isdigit()], key=lambda x: int(x[1:]))
    self.hidden_keys = self.weight_keys[:-1]
    self.final_key = self.weight_keys[-1]
    self.activations = []
    first_hidden_dim = self._w["W1"].shape[0]
    for k in self.hidden_keys:
      if k == "W1":
        self.activations.append(self._activation)
      else:
        cur_dim = self._w[k].shape[0]
        if isinstance(self._activation, act_mod.Mixed) and cur_dim != first_hidden_dim:
          self.activations.append(act_mod.get("linear"))
        else:
          self.activations.append(self._activation)

  def set_weights(self):
    super(Encoder, self).set_weights()
    self._w["W1_scale"] = self._w["W1"] @ self._w["ov_scale_diag"]
    self._w["b1_ref"] = self._w["b1"] - self._w["W1_scale"] @ self._w["ref"]

  def fun(self, x):
    h = self.w["W1_scale"] @ x + self.w["b1_ref"]
    a = self.activations[0](h, with_jac=False)
    for i, k in enumerate(self.hidden_keys[1:]):
      h = self.w[k] @ a
      a = self.activations[i+1](h, with_jac=False)
    z = self.w[self.final_key] @ a
    return z

  def fun_jac(self, x):
    h = self.w["W1_scale"] @ x + self.w["b1_ref"]
    a, da = self.activations[0](h, with_jac=True)
    if bkd.is_torch_backend():
      jac_chain = da.unsqueeze(1) * self.w["W1_scale"].to_dense()
    else:
      jac_chain = da @ self.w["W1_scale"]
    for i, k in enumerate(self.hidden_keys[1:]):
      h = self.w[k] @ a
      a_next, da_next = self.activations[i+1](h, with_jac=True)
      if bkd.is_torch_backend():
        jac_chain = da_next.unsqueeze(1) * (self.w[k].to_dense() @ jac_chain)
      else:
        jac_chain = da_next @ self.w[k] @ jac_chain
      a = a_next
    z = self.w[self.final_key] @ a
    if bkd.is_torch_backend():
      jac = self.w[self.final_key].to_dense() @ jac_chain
    else:
      jac = self.w[self.final_key] @ jac_chain
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
      jac = (self.w["scale_W2"].to_dense() * dx.unsqueeze(0)) @ self.w["W1"].to_dense()
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

  def __call__(self, x):
    return self.decoder(self.encoder(x, with_jac=False), with_jac=False)

  def set_hr_mode(
    self,
    active=False,
    row_ind=None
  ):
    self.decoder.set_hr_mode(active=active, row_ind=row_ind)


class MixedEncoder(object):
  """Encoder wrapper for mixed architecture ports (some 2-layer, some 3-layer).

  Delegates encoding to each port's own encoder and assembles the global latent vector.
  Provides fun and fun_jac interfaces matching Encoder.
  """

  def __init__(
    self,
    autoencoders,
    indices,
    input_dim,
    latent_dim
  ):
    self.name = "mixed_encoder"
    self.autoencoders = autoencoders
    self.indices = indices
    self.input_dim = input_dim
    self.latent_dim = latent_dim

  def __call__(self, x, with_jac=True):
    return self.fun_jac(x) if with_jac else self.fun(x)

  def fun(self, x):
    z = np.zeros(self.latent_dim)
    for (k, ae) in self.autoencoders.items():
      ind_fom = self.indices[k]["fom"]
      ind_rom = self.indices[k]["rom"]
      x_port = x[ind_fom]
      z_port = ae.encoder.fun(x_port)
      z[ind_rom] = z_port
    return z

  def fun_jac(self, x):
    if bkd.is_torch_backend():
      z = torch.zeros(self.latent_dim, device=bkd.device())
      jac = torch.zeros((self.latent_dim, self.input_dim), device=bkd.device())
    else:
      z = np.zeros(self.latent_dim)
      jac = sp.csr_matrix((self.latent_dim, self.input_dim))
    for (k, ae) in self.autoencoders.items():
      ind_fom = self.indices[k]["fom"]
      ind_rom = self.indices[k]["rom"]
      x_port = x[ind_fom]
      z_port, jac_port = ae.encoder.fun_jac(x_port)
      z[ind_rom] = z_port
      if bkd.is_torch_backend():
        rows = torch.as_tensor(ind_rom, device=jac.device, dtype=torch.long)
        cols = torch.as_tensor(ind_fom, device=jac.device, dtype=torch.long)
        jac[rows[:, None], cols[None, :]] = jac_port
      else:
        jac_block = sp.csr_matrix(jac_port)
        if jac_block.nnz > 0:
          rows, cols = jac_block.nonzero()
          data = jac_block.data
          jac_add = sp.csr_matrix((data, (ind_rom[rows], ind_fom[cols])), shape=(self.latent_dim, self.input_dim))
          jac = jac + jac_add
    return z, jac


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
    config = self.get_config()
    super(MultiAutoencoder, self).__init__(config)
    if config["encoder"].get("mixed_depth", False):
      self.encoder = MixedEncoder(
        autoencoders=self.autoencoders,
        indices=self.indices,
        input_dim=self.input_dim,
        latent_dim=config["encoder"]["latent_dim"]
      )

  # ROM dimensions
  # ===================================
  def get_config(self):
    depth_list = [len([k for k in ae.encoder._w.keys() if k.startswith("W") and k[1:].isdigit()]) for ae in self.autoencoders.values()]
    hidden_counts = [d-1 for d in depth_list]
    all_same = all(h == hidden_counts[0] for h in hidden_counts)
    mixed = not all_same
    config = {}
    config_init = self._init_config()

    # Decoder aggregation
    cfg_dec = copy.deepcopy(config_init)
    if self.mixed_act:
      shift = 0
    for (k, ae) in self.autoencoders.items():
      cfg_dec = self._update_config(
        config=cfg_dec,
        layer=ae.decoder,
        indices=self.indices[k]
      )
      if self.mixed_act:
        dim = ae.decoder.config["hidden_dim"]
        ind = np.arange(dim) + shift
        act = ae.activation
        cfg_dec["activation"]["masks"][act].append(ind)
        shift += dim
    if self.mixed_act:
      for (act, indices_arr) in cfg_dec["activation"]["masks"].items():
        cfg_dec["activation"]["masks"][act] = np.sort(np.concatenate(indices_arr))
    cfg_dec["weights"]["W1"] = sp.vstack(cfg_dec["weights"]["W1"])
    cfg_dec["weights"]["W2"] = sp.hstack(cfg_dec["weights"]["W2"])
    cfg_dec["weights"]["b1"] = np.concatenate(cfg_dec["weights"]["b1"])
    config["decoder"] = cfg_dec

    # Encoder aggregation: branch by depth structure
    if mixed:
      cfg_enc = {
        "input_dim": config_init["input_dim"],
        "latent_dim": config_init["latent_dim"],
        "activation": "linear",
        "ref": np.zeros(config_init["input_dim"]),
        "scale": np.ones(config_init["input_dim"]),
        "mask_shape": None,
        "mask_indices": None,
        "hidden_dim": 1,
        "mixed_depth": True,
        "weights": {
          "W1": sp.csr_matrix((1, config_init["input_dim"])),
          "W2": sp.csr_matrix((config_init["latent_dim"], 1)),
          "b1": np.zeros(1)
        }
      }
      config["encoder"] = cfg_enc
    else:
      cfg_enc = copy.deepcopy(config_init)
      if self.mixed_act:
        shift = 0
      for (k, ae) in self.autoencoders.items():
        cfg_enc = self._update_config(
          config=cfg_enc,
          layer=ae.encoder,
          indices=self.indices[k]
        )
        if self.mixed_act:
          dim = ae.encoder.config["hidden_dim"]
          ind = np.arange(dim) + shift
          act = ae.activation
          cfg_enc["activation"]["masks"][act].append(ind)
          shift += dim
      if self.mixed_act:
        for (act, indices_arr) in cfg_enc["activation"]["masks"].items():
          cfg_enc["activation"]["masks"][act] = np.sort(np.concatenate(indices_arr))
      hidden_layers = hidden_counts[0]
      cfg_enc["weights"]["W1"] = sp.vstack(cfg_enc["weights"]["W1"])
      for h in range(2, hidden_layers+1):
        key = f"W{h}"
        cfg_enc["weights"][key] = sp.block_diag(cfg_enc["weights"][key])
      final_key = f"W{hidden_layers+1}"
      cfg_enc["weights"][final_key] = sp.hstack(cfg_enc["weights"][final_key])
      cfg_enc["weights"]["b1"] = np.concatenate(cfg_enc["weights"]["b1"])
      present_keys = set(cfg_enc["weights"].keys())
      for k in list(present_keys):
        if k.startswith("W") and isinstance(cfg_enc["weights"][k], list):
          cfg_enc["weights"].pop(k)
      config["encoder"] = cfg_enc

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
      "weights": {"W1": [], "b1": []}
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
    for w, mat in weights.items():
      if w not in config["weights"]:
        config["weights"][w] = []
      config["weights"][w].append(mat)
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
    b1 = layer._w["b1_ref"]
    W1 = np.zeros((dims["hidden"], dims["input"]))
    W1[:,indices["fom"]] += layer._w["W1_scale"]
    W1 = sp.csr_matrix(W1)
    # Hidden / latent mapping layers
    w_enc_keys = sorted([k for k in layer._w.keys() if k.startswith("W") and k[1:].isdigit()], key=lambda x: int(x[1:]))
    result = {"W1": W1, "b1": b1}
    for wk in w_enc_keys[1:-1]:
      mat_local = layer._w[wk]
      Wk = np.zeros(mat_local.shape)
      Wk[:, :] += mat_local
      result[wk] = sp.csr_matrix(Wk)
    w_last = w_enc_keys[-1]
    mat_last = layer._w[w_last]
    if mat_last.shape[0] != len(indices["rom"]):
      raise ValueError(f"Unexpected {w_last} shape {mat_last.shape}; expected rows == len(indices['rom'])={len(indices['rom'])}")
    W_last = np.zeros((dims["latent"], self._get_hidden_out_dim(layer)))
    W_last[indices["rom"],:] += mat_last
    result[w_last] = sp.csr_matrix(W_last)
    return result

  def _get_hidden_out_dim(self, layer):
    w_keys = sorted([k for k in layer._w.keys() if k.startswith("W") and k[1:].isdigit()], key=lambda x: int(x[1:]))
    last_hidden_key = w_keys[-2]
    return layer._w[last_hidden_key].shape[0]

  def _get_weights_single_decoder(
    self,
    dims,
    layer,
    indices
  ):
    # Input layer
    b1 = layer._w["b1"]
    W1 = np.zeros((dims["hidden"], dims["latent"]))
    W1[:,indices["rom"]] += layer._w["W1"]
    W1 = sp.csr_matrix(W1)
    # Hidden layer
    W2 = np.zeros((dims["input"], dims["hidden"]))
    W2[indices["fom"],:] += layer._w["scale_W2"]
    W2 = sp.csr_matrix(W2)
    # Return weights
    return {"W1": W1, "b1": b1, "W2": W2}
