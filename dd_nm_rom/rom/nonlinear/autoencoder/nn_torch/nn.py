import os
import copy
import torch
import torchinfo
import numpy as np
import scipy.sparse as sp

from dd_nm_rom import ops
from dd_nm_rom import backend as bkd
from sparselinear import SparseLinear
from .elements import generate_mask, get_activation


class GaussianNoise(torch.nn.Module):
  """
  Gaussian noise regularizer.
  """

  def __init__(
    self,
    stddev=0.1
  ):
    super(GaussianNoise, self).__init__()
    self.stddev = stddev
    self.register_buffer("noise", torch.tensor(0))

  def forward(self, x):
    if (self.training and (self.stddev > 0)):
      noise = self.noise.expand(*x.shape).to(bkd.floatx())
      noise = noise.normal_(mean=1.0, std=self.stddev)
      return x * noise
    else:
      return x


class Identity(torch.nn.Module):

  def __init__(
    self,
    dim,
    ref,
    scale,
    mask,
    activation="linear"
  ):
    super(Identity, self).__init__()
    self.dim = int(dim)
    self.input_dim = self.dim
    self.hidden_dim = self.dim
    self.output_dim = self.dim
    self.ref = ref
    self.scale = scale
    self.activation = activation
    connect = bkd.to_backend(np.vstack((mask.row, mask.col))).to(torch.int64)
    self.net = torch.nn.Sequential(
      SparseLinear(self.dim, self.dim, connectivity=connect, bias=True),
      get_activation(self.activation),
      SparseLinear(self.dim, self.dim, connectivity=connect, bias=False)
    )
    self.net.apply(self.init_weights)

  def init_weights(self, module):
    if isinstance(module, SparseLinear):
      torch.nn.init.ones_(module.weights)
      if (module.bias is not None):
        torch.nn.init.zeros_(module.bias)

  def forward(self, x):
    return self.net(x)


class Encoder(torch.nn.Module):

  def __init__(
    self,
    input_dim,
    hidden_dim,
    output_dim,
    ref,
    scale,
    mask,
    activation,
    dense=False
  ):
    super(Encoder, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.ref = ref
    self.scale = scale
    self.activation = activation
    if dense:
      input_lay = torch.nn.Linear(input_dim, hidden_dim)
    else:
      connect = bkd.to_backend(np.vstack((mask.row, mask.col))).to(torch.int64)
      input_lay = SparseLinear(input_dim, hidden_dim, connectivity=connect)
    self.net = torch.nn.Sequential(
      input_lay,
      get_activation(self.activation),
      torch.nn.Linear(hidden_dim, output_dim, bias=False)
    )

  def forward(self, x):
    return self.net((x-self.ref)/self.scale)


class Decoder(torch.nn.Module):

  def __init__(
    self,
    input_dim,
    hidden_dim,
    output_dim,
    ref,
    scale,
    mask,
    activation
  ):
    super(Decoder, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.ref = ref
    self.scale = scale
    self.activation = activation
    connect = bkd.to_backend(np.vstack((mask.row, mask.col))).to(torch.int64)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      get_activation(self.activation),
      SparseLinear(hidden_dim, output_dim, connectivity=connect, bias=False)
    )

  def forward(self, x):
    return self.scale*self.net(x)+self.ref


class Autoencoder(torch.nn.Module):
  """
  Class for implementing and training an autoencoder
  with sparse-masked decoder for model reduction.
  """

  def __init__(
    self,
    ref,
    scale,
    input_dim,
    latent_dim,
    row_shift,
    row_nonzero,
    min_input_dim=5,
    encoder_dense=False,
    encoder_hidden=-1,
    activation="sigmoid",
    add_noise=False,
    loading=False,
    saved_model=None
  ):
    super(Autoencoder, self).__init__()
    self.trainable = True
    # compute scaling vectors
    self.ref = ref.reshape(-1)
    self.scale = scale.reshape(-1)
    # sizes of layers in encoder and decoder
    self.input_dim = int(input_dim)
    self.min_input_dim = int(min_input_dim)
    if (self.input_dim <= self.min_input_dim):
      self.trainable = False
      self.latent_dim = self.input_dim
      self.decoder_hidden_dim = self.input_dim
      self.encoder_hidden_dim = self.input_dim
      self.activation = "linear"
      self.mask = sp.eye(self.input_dim).tocoo()
      self.encoder = Identity(
        dim=self.input_dim,
        ref=self.ref,
        scale=self.scale,
        mask=self.mask.T
      )
      self.decoder = Identity(
        dim=self.input_dim,
        ref=self.ref,
        scale=self.scale,
        mask=self.mask
      )
    else:
      self.latent_dim = int(latent_dim)
      # compute sparsity mask
      self.row_shift = int(row_shift)
      self.row_nonzero = int(row_nonzero)
      self.mask, self.decoder_hidden_dim = generate_mask(
        output_dim=self.input_dim,
        row_shift=self.row_shift,
        row_nonzero=self.row_nonzero
      )
      # activation function
      self.activation = str(activation)
      # initialize encoder and decoder
      self.encoder_dense = bool(encoder_dense)
      if encoder_dense:
        if (encoder_hidden < 0):
          encoder_hidden_dim = 2*self.input_dim
        else:
          encoder_hidden_dim = encoder_hidden
      else:
        encoder_hidden_dim = self.decoder_hidden_dim
      self.encoder_hidden_dim = int(encoder_hidden_dim)
      self.encoder = Encoder(
        input_dim=self.input_dim,
        hidden_dim=self.encoder_hidden_dim,
        output_dim=self.latent_dim,
        ref=self.ref,
        scale=self.scale,
        mask=self.mask.T,
        activation=self.activation,
        dense=encoder_dense
      )
      self.decoder = Decoder(
        input_dim=self.latent_dim,
        hidden_dim=self.decoder_hidden_dim,
        output_dim=self.input_dim,
        ref=self.ref,
        scale=self.scale,
        mask=self.mask,
        activation=self.activation
      )
    # Collect layers
    self.layers = {"encoder": self.encoder, "decoder": self.decoder}
    self.add_noise = add_noise
    if self.add_noise:
      self.layers["noise"] = GaussianNoise(stddev=0.02)
    self.layers = torch.nn.ModuleDict(self.layers)
    # Load
    if loading:
      saved_model = os.path.abspath(saved_model)
      print(f"Restoring network from file '{saved_model}'")
      model_checkpoint = torch.load(saved_model, weights_only=False)
      torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(model_checkpoint, "module.")
      self.load_state_dict(model_checkpoint)
    # To float/device
    self.to(dtype=bkd.floatx("torch"), device=bkd.device())

  def forward(self, x):
    if self.add_noise:
      x = self.layers["noise"](x)
    x = self.layers["encoder"](x)
    x = self.layers["decoder"](x)
    if self.add_noise:
      x = self.layers["noise"](x)
    return x

  def summary(self, filename=None, verbose=2):
    if bkd.distributed() and bkd.get_rank() != 0:
      # only print summary on root rank
      return
    stats = torchinfo.summary(
      model=self,
      input_size=(self.input_dim,),
      col_names=(
        "input_size",
        "output_size",
        "kernel_size",
        "num_params"
      ),
      depth=4,
      verbose=verbose
    )
    if (filename is not None):
      with open(filename, "w") as file:
        file.write(str(stats))

  def state_dict_np(self):
    config_common = {
      "ref": bkd.to_numpy(self.ref).reshape(-1),
      "scale": bkd.to_numpy(self.scale).reshape(-1),
      "input_dim": self.input_dim,
      "latent_dim": self.latent_dim,
      "activation": self.activation
    }
    config = {}
    for l in ("encoder", "decoder"):
      mask = self.mask if (l == "decoder") else self.mask.T
      mask_shape = tuple(mask.shape)
      config[l] = {
        "weights": self.get_weights_np(self.layers[l], mask_shape),
        "mask_shape": mask_shape,
        "mask_indices": np.vstack(mask.nonzero()),
        "hidden_dim": self.layers[l].hidden_dim
      }
      config[l].update(config_common)
    return config

  def get_weights_np(self, layer, mask_shape):
    weights = copy.deepcopy(dict(layer.state_dict()))
    weights = ops.map_nested_dict(weights, bkd.to_numpy)
    weights_np = {}
    for (i, k) in enumerate((0,2)):
      if (f"net.{k}.indices" in weights):
        mask = tuple([weights[f"net.{k}.indices"][j] for j in range(2)])
        wi = sp.csr_matrix(
          (weights[f"net.{k}.weights"], mask), shape=mask_shape
        )
      else:
        wi = weights[f"net.{k}.weight"]
      weights_np[f"W{i+1}"] = wi
    weights_np["b1"] = weights["net.0.bias"]
    return weights_np
