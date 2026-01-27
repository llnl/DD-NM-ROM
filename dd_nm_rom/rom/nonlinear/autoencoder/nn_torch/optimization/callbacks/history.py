import pandas as pd

from dd_nm_rom import postproc

from .callback import Callback


class History(Callback):

  def __init__(
    self,
    plotting=1,
    scale='log',
    vars=['loss','val_loss'],
    labels=['Training', 'Validation'],
    frequency=1
  ):
    super(History, self).__init__()
    self.plotting = plotting
    self.scale = scale
    self.vars = vars
    self.labels = labels
    self.frequency = frequency

  def on_train_begin(self):
    self.filename = self.model.dirs['train'] + '/history.csv'
    self.history = {'epoch': []}
    self.history.update({
      k: [] for k in self.model.train_state.logs.keys()
    })
    # Create file
    df = pd.DataFrame.from_dict(self.history)
    df.to_csv(self.filename, index=False)

  def on_epoch_end(self):
    epoch = self.model.train_state.epoch
    self.history['epoch'].append(epoch)
    for (k, v) in self.model.train_state.logs.items():
      self.history[k].append(v)
    if (epoch % self.frequency == 0):
      self._write()

  def on_train_end(self):
    self._write()
    if self.plotting:
      postproc.plot_loss(
        hist_df=pd.read_csv(self.filename),
        path=self.model.dirs['train'],
        scale=self.scale,
        title='Loss',
        vars=self.vars,
        fig_name='loss',
        label=self.labels,
        window=1
      )

  def _write(self):
    hist_df = pd.DataFrame.from_dict(self.history)
    hist_df.to_csv(self.filename, mode='a', index=False, header=False)
    self.history = {k: [] for k in self.history.keys()}
