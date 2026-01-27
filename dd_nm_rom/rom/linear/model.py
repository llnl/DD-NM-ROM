import os
import numpy as np
import scipy.linalg as la
import dill as pickle

from dd_nm_rom import backend as bkd


class Model(object):

  def __init__(
    self,
    data=None,
    path="./"
  ):
    self.data = data

    # Define paths
    self.path = path
    self.dirs = {
      "save": self.path + "/saving/",
      "train": self.path + "/training/",
      "ckpt": self.path + "/training/ckpt/"
    }
    for d in self.dirs.values():
      os.makedirs(d, exist_ok=True)

  # Compiling
  # ---------------------------------
  def compile(
    self,
    optimizer="adam",
    lr=1e-3,
    lr_decay=None,
    weight_decay=0.0,
    loss="mse",
    monitor="loss",
    reduction="sum",
    callbacks=None
  ):
    print("Compiling the model ...")
    # Write nn summary
    self.net.summary(filename=self.dirs["save"] + "/summary.txt")
    # Initializing loss function
    self.loss = optim.losses.get(loss, reduction=reduction)
    # Monitor metrics
    self.monitor = monitor
    options = list(self.train_state.logs.keys())
    if (self.monitor not in options):
      raise ValueError(f"Monitor metrics not valid. Please choose: {options}")
    # Initializing the optimizer
    self.optimizer, self.lr_scheduler = optim.optimizers.get(
      self.net.parameters(),
      optimizer,
      lr=lr,
      lr_decay=lr_decay,
      weight_decay=weight_decay
    )
    # Callbacks
    self.callbacks = optim.callbacks.get_callbacks(callbacks)
    self.callbacks.set_model(self)
    self.is_compiled = True

  # Training
  # ---------------------------------
  def train(
    self,
    saving=True
  ):
    print("Computing SVD ...")

    u_, s_, vh_i = la.svd(self.data.train.T, full_matrices=False, check_finite=False)

    # puts left singular vectors and singular values into dictionaries
    self.svd = {'left_vecs': u_, 
                'sing_vals': s_}

    self.basis = self.compute_bases_from_svd(self.svd, ec=1e-2)

    # Saving
    if saving:
      print("Saving the SVD ...")
      self.save()

  # Testing
  # ---------------------------------
  def evaluate(self):
    if (self.data.valid is not None):
      self.net.train(mode=False)
      for batch in self.data.batches_valid:
        loss = self.evaluate_step(batch.to(bkd.device()))
        # Update logs
        self.train_state.on_batch_end({"val_loss": loss})

  def evaluate_step(self, data):
    return self.loss(self.net(data), data)

  # compute POD bases given SVD data
  def compute_bases_from_svd(self, data_dict,
                           ec=1e-8, 
                           nbasis=-1):
    '''
    Computes POD bases given saved SVD data.
    
    inputs:
    data_dict: dictionary with fields
                'left_vecs' : Matrix of left singular vectors of snapshot data 
                'sing_vals' : Vector of singular values of snapshot data
    ec: [optional] energy criterior for choosing size of basis. Default is 1e-8
    nbasis: [optional] size of basis. Setting to -1 uses energy criterion. Default is -1.
    
    output:
    bases: Matrix containing POD basis
    '''
    U = data_dict['left_vecs']
    if nbasis <= 0:
      s = data_dict['sing_vals']
      ss = s*s
      dim = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
    else:
      dim = nbasis
    return U[:, :dim]

  # Saving
  # ---------------------------------
  def save(self, filename=None):
    if (filename is None):
      filename = self.dirs["save"] + "/svd"
#   torch.save(self.net.state_dict(), filename+"_torch.p")
#   torch.save(self.net.state_dict_np(), filename+"_numpy.p")
    pickle.dump(self.svd, open(filename+".p",'wb'))