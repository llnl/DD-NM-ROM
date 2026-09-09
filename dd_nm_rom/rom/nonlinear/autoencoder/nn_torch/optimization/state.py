import time
import numpy as np
from mpi4py import MPI

from dd_nm_rom import backend as bkd

class TrainState(object):

  def __init__(self, start_epoch=0):
    self.epoch = start_epoch
    self._logs = {}
    self.logs = {}
    self.epochs = 1
    self.display_freq = 1
    self.epoch_start = 0.0
    self.rank = bkd.get_rank() if bkd.distributed() else -1

  def init_logs(self, logs_ids):
    for k in logs_ids:
      self._logs[k] = []
      self.logs[k] = 0.0

  def on_batch_end(self, logs):
    for (k, v) in logs.items():
      self._logs[k].append(float(v))

  def on_epoch_begin(self):
    for k in self._logs.keys():
        self._logs[k] = []
    self.epoch_start = time.time()
    if (self.epoch % self.display_freq == 0):
      text = "Epoch {:4d}/{:d}".format(self.epoch+1, self.epochs)
      #if self.rank != -1:
      #  text = "rank {:d}: {}".format(self.rank, text)
      if self.rank == -1 or self.rank == 0:
        print(' '*2 + text)

  def on_epoch_end(self):
    for k in self.logs.keys():
      self.logs[k] = float(np.mean(self._logs[k]))
    self.epoch_exec = time.time() - self.epoch_start
    if bkd.distributed():
      epoch_time = bkd._COMM.allreduce(self.epoch_exec, op=MPI.SUM)
      self.epoch_exec = epoch_time / bkd.get_nranks()
    if (self.epoch % self.display_freq == 0):
      text = "> Logs: | "
      for (k, v) in self.logs.items():
        text += k + ": {:.5e} | ".format(v)
      #if self.rank != -1:
      #  text = "rank {:d}: {}".format(self.rank, text)
      if self.rank == -1 or self.rank == 0:
        print(' '*4 + text)

      #if self.rank != -1:
      #  print(' '*4 + "rank {:d}: > Epoch execution time: {:.5e} s".format(self.rank, self.epoch_exec))
      #else:
      if self.rank == -1 or self.rank == 0:
        print(' '*4 + "> Epoch execution time: {:.5e} s".format(self.epoch_exec))
