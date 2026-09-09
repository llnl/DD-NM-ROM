import json
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

from dd_nm_rom import backend as bkd

from .callback import Callback


class EarlyStopping(Callback):

  def __init__(
    self,
    min_delta=0,
    patience=0,
    baseline=None,
    monitor="loss"
  ):
    super(EarlyStopping, self).__init__()
    self.baseline = baseline
    self.monitor = monitor
    self.patience = patience
    self.min_delta = min_delta
    self.wait = 0
    self.stopped_epoch = 0
    self.monitor_op = np.less
    self.min_delta *= -1

  def on_train_begin(self):
    self.wait = 0
    if (self.baseline is not None):
      self.best = self.baseline
    else:
      self.best = np.inf if (self.monitor_op == np.less) else -np.inf

  def on_epoch_end(self):
    current = self.get_monitor_value()
    if bkd.distributed():
      current = bkd._COMM.allreduce(current, op=MPI.SUM)
      current = current / bkd.get_nranks()
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
    else:
      self.wait += 1
      if (self.wait >= self.patience):
        self.stopped_epoch = self.model.train_state.epoch
        self.model.stop_training = True

  def on_train_end(self):
    if (self.stopped_epoch > 0):
      print(self.header + f"Early stopping at epoch {self.stopped_epoch}.")

  def get_monitor_value(self):
    if (self.monitor in self.model.train_state.logs):
      return self.model.train_state.logs[self.monitor]
    else:
      keys = list(self.model.train_state.logs.keys())
      raise ValueError(
        self.header + "The specified monitor value is incorrect. " \
          f"The ones available are: {keys}."
      )


class ValueEarlyStopping(EarlyStopping):

  def __init__(
    self,
    epsilon=1e-8,
    monitor="loss"
  ):
    super(ValueEarlyStopping, self).__init__()
    self.epsilon = epsilon
    self.monitor = monitor
    self.monitor_op = np.less

  def on_train_begin(self):
    self.stopped_epoch = 0

  def on_epoch_end(self):
    current = self.get_monitor_value()
    if bkd.distributed():
      current = bkd._COMM.allreduce(current, op=MPI.SUM)
      current = current / bkd.get_nranks()
    if self.monitor_op(current, self.epsilon):
      self.stopped_epoch = self.model.train_state.epoch
      self.model.stop_training = True


class Terminator(Callback):

  def __init__(
    self,
    frequency=1
  ):
    super(Terminator, self).__init__()
    self.frequency = frequency
    self.stopped_epoch = 0

  def on_train_begin(self):
    self.filename = self.model.dirs["train"] + "/train_ctrl.json"
    if bkd.distributed() and bkd.get_rank() != 0:
      return
    with open(self.filename, "w") as file:
      json.dump({"stop_training": False}, file, indent=4)

  def on_epoch_end(self):
    epoch = self.model.train_state.epoch
    if (epoch % self.frequency == 0):
      if not bkd.distributed() or (bkd.distributed() and bkd.get_rank() == 0):
        with open(self.filename) as file:
          control = json.load(file)

        if control["stop_training"]:
          self.stopped_epoch = epoch
          self.model.stop_training = True

      if bkd.distributed():
        self.stopped_epoch = bkd._COMM.bcast(self.stopped_epoch, 0)
        self.model.stop_training = bkd._COMM.bcast(self.model.stop_training, 0)

  def on_train_end(self):
    if (self.stopped_epoch > 0):
      print(self.header + f"Early stopping at epoch {self.stopped_epoch}.")
