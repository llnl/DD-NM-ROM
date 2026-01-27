class Callback(object):

  def __init__(self):
    self.name = self.__class__.__name__
    self.model = None
    self.display_freq = 1
    self.header = "> " + self.name + ": "

  def set_model(self, model):
    if (model is not self.model):
      self.model = model
      self.init()

  def set_display_freq(self, display_freq):
    self.display_freq = display_freq

  def init(self):
    """Init after setting a model."""

  def on_epoch_begin(self):
    """Called at the beginning of every epoch."""

  def on_epoch_end(self):
    """Called at the end of every epoch."""

  def on_batch_begin(self):
    """Called at the beginning of every batch."""

  def on_batch_end(self):
    """Called at the end of every batch."""

  def on_train_begin(self):
    """Called at the beginning of model training."""

  def on_train_end(self):
    """Called at the end of model training."""

  def on_predict_begin(self):
    """Called at the beginning of prediction."""

  def on_predict_end(self):
    """Called at the end of prediction."""


class CallbackList(Callback):

  def __init__(self, callbacks=[]):
    self.callbacks = callbacks

  def set_model(self, model):
    for callback in self.callbacks:
      callback.set_model(model)

  def set_display_freq(self, display_freq):
    for callback in self.callbacks:
      callback.set_display_freq(display_freq)

  def on_epoch_begin(self):
    for callback in self.callbacks:
      callback.on_epoch_begin()

  def on_epoch_end(self):
    for callback in self.callbacks:
      callback.on_epoch_end()

  def on_batch_begin(self):
    for callback in self.callbacks:
      callback.on_batch_begin()

  def on_batch_end(self):
    for callback in self.callbacks:
      callback.on_batch_end()

  def on_train_begin(self):
    for callback in self.callbacks:
      callback.on_train_begin()

  def on_train_end(self):
    for callback in self.callbacks:
      callback.on_train_end()

  def on_predict_begin(self):
    for callback in self.callbacks:
      callback.on_predict_begin()

  def on_predict_end(self):
    for callback in self.callbacks:
      callback.on_predict_end()

  def append(self, callback):
    if (not isinstance(callback, Callback)):
      raise Exception(
        str(callback) + " is an invalid 'Callback' object."
      )
    self.callbacks.append(callback)
