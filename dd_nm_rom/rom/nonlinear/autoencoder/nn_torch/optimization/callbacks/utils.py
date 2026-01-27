from .callback import CallbackList
from .checkpoint import *
from .history import *
from .learning_rate import *
from .stopping import *


_CB_LIST = (
  "checkpoint",
  "early_stopping",
  "value_early_stopping",
  "history",
  "lr_tracker",
  "terminator"
)

def get_callbacks(
  callbacks=None
):
  cb_list = {}
  if (callbacks is not None):
    for (identifier, kwargs) in callbacks.items():
      cb_list[identifier] = _get_callback(identifier, kwargs)
  if ("history" not in cb_list):
    cb_list["history"] = _get_callback("history")
  if ("terminator" not in cb_list):
    cb_list["terminator"] = _get_callback("terminator")
  if ("early_stopping" not in cb_list):
    cb_list["early_stopping"] = _get_callback("early_stopping")
  return CallbackList(list(cb_list.values()))

def _get_callback(
  identifier,
  kwargs={}
):
  if callable(identifier):
    return identifier
  elif isinstance(identifier, str) and (identifier in _CB_LIST):
    if (identifier == "checkpoint"):
      return ModelCheckpoint(**kwargs)
    elif (identifier == "early_stopping"):
      return EarlyStopping(**kwargs)
    elif (identifier == "value_early_stopping"):
      return ValueEarlyStopping(**kwargs)
    elif (identifier == "history"):
      return History(**kwargs)
    elif (identifier == "lr_tracker"):
      return LearningRateTracker(**kwargs)
    elif (identifier == "terminator"):
      return Terminator(**kwargs)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(
      f"Could not interpret callback identifier: '{identifier}'. " \
        f"The available ones are {_CB_LIST}"
    )
