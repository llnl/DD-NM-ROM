import numpy as np
import torch
import torch.distributed as dist

from dd_nm_rom import backend as bkd
from .callback import Callback


class ModelCheckpoint(Callback):

  def __init__(
    self,
    frequency=1,
    monitor="val_loss",
    save_best_only=True,
    overwrite=True
  ):
    super(ModelCheckpoint, self).__init__()
    self.frequency = frequency
    self.save_best_only = save_best_only
    self.overwrite = overwrite
    self.best = np.inf
    self.monitor = monitor
    self.monitor_op = np.less if not bkd.distributed() else torch.less
    self.epochs_since_last_save = 0
    if bkd.distributed():
      self.best = bkd.to_backend(self.best)

  def on_epoch_end(self):
    # Update epochs count
    self.epochs_since_last_save += 1
    if (self.epochs_since_last_save < self.frequency):
      return
    self.epochs_since_last_save = 0
    # Save model
    prefix = self.model.dirs['ckpt']
    filename = prefix + f"/model_epoch_{self.model.train_state.epoch}"
    if self.save_best_only:
      if self.overwrite:
        filename = prefix + f"/model_best"
      current = self.get_monitor_value()
      if bkd.distributed():
        current = bkd.to_backend(current)
        dist.all_reduce(current, op=dist.ReduceOp.AVG)
      if self.monitor_op(current, self.best):
        self.model.save(filename)
        self.best = current
    else:
      if self.overwrite:
        filename = prefix + f"/model_last"
      self.model.save(filename)

  def get_monitor_value(self):
    if (self.monitor in self.model.train_state.logs):
      return self.model.train_state.logs[self.monitor]
    else:
      keys = list(self.model.train_state.logs.keys())
      raise ValueError(
        self.header + "The specified monitor quantity is incorrect. " \
          f"The available ones are: {keys}."
      )
