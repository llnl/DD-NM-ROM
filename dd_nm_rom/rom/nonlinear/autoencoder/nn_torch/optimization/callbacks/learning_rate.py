import pandas as pd

from dd_nm_rom import postproc
from dd_nm_rom import backend as bkd

from .callback import Callback


class LearningRateTracker(Callback):

  def __init__(
    self,
    plotting=True,
    verbose=True
  ):
    super(LearningRateTracker, self).__init__()
    self.verbose = bool(verbose)
    self.plotting = bool(plotting)

  def on_train_begin(self):
    self.history = {'epoch': [], 'lr': []}
    self.filename = self.model.dirs['train'] + '/lr.csv'
    if bkd.distributed() and bkd.get_rank() != 0:
      return
    # Create file
    df = pd.DataFrame.from_dict(self.history)
    df.to_csv(self.filename, index=False)

  def on_epoch_begin(self):
    if (self.model.lr_scheduler is not None):
      epoch = self.model.train_state.epoch
      self.history['epoch'].append(epoch)
      lr = self.model.optimizer.param_groups[0]['lr']
      self.history['lr'].append(lr)
      if bkd.distributed() and bkd.get_rank() != 0:
        return
      self._write()
      if (self.verbose and (epoch % self.display_freq == 0)):
        print(' '*4 + self.header + "learning rate = %.5e" % (lr,))

  def on_train_end(self):
    if bkd.distributed() and bkd.get_rank() != 0:
      return
    self._write()
    if self.plotting:
      hist_df = pd.read_csv(self.filename)
      postproc.plot_lr(hist_df, self.model.dirs['train'])

  def _write(self):
    hist_df = pd.DataFrame.from_dict(self.history)
    hist_df.to_csv(self.filename, mode='a', index=False, header=False)
    self.history = {k: [] for k in self.history.keys()}
