import torch
import numpy as np

from dd_nm_rom import backend as bkd


class Data(object):

  def __init__(
    self,
    snapshots,
    validation_split=0.2,
    batch_size=32,
    eps=1e-5
  ):
    self.snapshots = torch.from_numpy(snapshots)
    self.snapshots = self.snapshots.to(device="cpu", dtype=bkd.floatx())

    # Data
    self.train = None
    self.valid = None
    self.batches = None
    self.batches_valid = None
    # Split train and validation
    self.validation_split = validation_split
    self.split_train_valid(self.snapshots)
    self.batch_size = batch_size

  def normalize(self, data, eps=1e-5):
    amin = np.amin(data, axis=0)
    amax = np.amax(data, axis=0)
    ref, scale = 0.5*(amax+amin), 0.5*(amax-amin)
    indices = np.isclose(scale, 0.0, rtol=0.0, atol=eps)
    scale[indices] = 1.0
    self.ref = bkd.to_backend(ref)
    self.scale = bkd.to_backend(scale)

  def split_train_valid(self, data):
    data = self.shuffle(data, seed=bkd.seed())
    if (self.validation_split > 0.0):
      nb_samples = data.shape[0]
      valid_size = int(self.validation_split*nb_samples)
      self.valid = data[:valid_size]
      self.train = data[valid_size:]
    else:
      self.train = data

  def batch(self, data):
    data = self.shuffle(data)
    nb_samples = data.shape[0]
    nb_batches = int(np.ceil(nb_samples/self.batch_size))
    return torch.tensor_split(data, nb_batches, dim=0)

  def shuffle(self, data, seed=None):
    np.random.seed(seed)
    i = np.random.permutation(data.shape[0])
    return data[i]

  def on_epoch_begin(self):
    self.batches = self.batch(self.train)
    if (self.valid is not None):
      self.batches_valid = self.batch(self.valid)
