import abc
import copy
import numpy as np

import torch
from torch.nn import functional

from dd_nm_rom.ops import sp_diag
from dd_nm_rom import backend as bkd
import dd_nm_rom.config as cfg

#torch._logging.set_logs(graph_code=True)

_ACT_IDS = ("elu", "linear", "mixed", "relu", "sigmoid", "swish", "softplus")


def _maybe_compile(fun):
  if not hasattr(torch, "compile"):
    raise RuntimeError(
      "DDNMROM_ACT_COMPILE is enabled, but this PyTorch version does not "
      "provide torch.compile"
    )
  # Compilation is intentionally not wrapped in an eager fallback.  When
  # compilation is enabled, a compiler failure must be reported rather than
  # silently changing the execution path on one rank.
  return torch.compile(fun, backend="inductor", dynamic=True)


def _should_compile():
  """Return whether activation functions should use Torch Inductor.

  Inductor's CPU backend requires a working host C++ toolchain, while the
  activation kernels are small enough that eager CPU execution is preferable
  by default.  GPU/ROCm runs retain the compiled default.  An explicit
  ``DDNMROM_ACT_COMPILE`` value always takes precedence, so requesting
  compilation still surfaces a compiler failure rather than changing the
  execution mode on one distributed rank.
  """
  value = cfg.get_config_val("DDNMROM_ACT_COMPILE", get_default=False)
  if value is None:
    return torch.cuda.is_available()
  return value.lower() in ("1", "true", "yes", "on")


def warmup(act, size, device, dtype):
  """Trigger the first compiled activation call before distributed work."""
  if not _should_compile():
    return

  if isinstance(act, Mixed):
    # Mixed activations compile their component functions lazily.  Warm each
    # non-empty mask separately because the masked implementation does not
    # invoke components whose masks are empty.
    for (component, mask) in act.masks.values():
      if mask.size:
        component(
          torch.zeros(mask.size, device=device, dtype=dtype),
          with_jac=False
        )
  else:
    act(torch.zeros(size, device=device, dtype=dtype), with_jac=False)


def get(identifier='sigmoid', *args, **kwargs):
  if (isinstance(identifier, str) and (identifier.lower() in _ACT_IDS)):
    act = {
      "elu":     ELU,
      "linear":  Linear,
      "mixed":   Mixed,
      "relu":    ReLU,
      "sigmoid": Sigmoid,
      "swish":   Swish,
      "softplus": Softplus
    }[identifier.lower()](*args, **kwargs)
    if bkd.is_torch_backend():
        BaseAct.__call__ = BaseAct._call__torch
        BaseAct._fun = BaseAct._fun_torch
        BaseAct._jac = BaseAct._jac_torch

        # Keep the Torch path on compiled callables when the function is
        # simple enough to trace. Mixed activations retain the Python version
        # because their masked indexing is less stable under compilation.
        # The Torch Jacobian stays as a derivative vector. Callers apply it as
        # row/column scaling so we avoid sparse diagonal multiplies in CUDA.
        if _should_compile() and not isinstance(act, Mixed):
          compiled_fun = _maybe_compile(act._fun_torch)
          act._fun_torch = compiled_fun
          act._fun = compiled_fun
          act.fun = act._fun
          act.jac = act._jac_torch
        else:
          act._fun = act._fun_torch
          act._jac = act._jac_torch
          act.fun = act._fun
          act.jac = act._jac_torch
    return act
  else:
    raise ValueError(
      f"Could not interpret activation function identifier: '{identifier}'."
    )

# Base activation function
# -------------------------------------
class BaseAct(object):

  def __init__(self):
    self.fun = self._fun
    self.jac = lambda x: sp_diag(self._jac(x))

  def __call__(self, x, with_jac=True):
    return (self.fun(x), self.jac(x)) if with_jac else self.fun(x)

  def _call__torch(self, x, with_jac=True):
    if with_jac:
      return (self.fun(x), self.jac(x))
    else:
      return self.fun(x)

  @abc.abstractmethod
  def _fun(self, x):
    pass

  @abc.abstractmethod
  def _jac(self, x):
    pass

  @abc.abstractmethod
  def _fun_torch(self, x):
    pass

  def _jac_torch(self, x):
    raise NotImplementedError

  def _fun_jac(self, x):
    result = self._fun_torch(x)
    return result, result

# Linear
# -------------------------------------
class Linear(BaseAct):

  def _fun(self, x):
    return x

  def _jac(self, x):
    return np.ones_like(x)

  def _fun_torch(self, x):
    return x

  def _jac_torch(self, x):
    return torch.ones_like(x)


# Sigmoid
# -------------------------------------
class Sigmoid(BaseAct):

  def _fun(self, x):
    return 1.0 / (1.0+np.exp(-x))

  def _jac(self, x):
    ex = np.exp(-x)
    return ex / (1.0+ex)**2

  def _fun_torch(self, x):
    return torch.sigmoid(x)

  def _jac_torch(self, x):
    fx = torch.sigmoid(x)
    return fx * (1.0 - fx)

# Swish
# -------------------------------------
class Swish(BaseAct):

  def _fun(self, x):
    return x / (1.0 + np.exp(-x))

  def _jac(self, x):
    ex = np.exp(x)
    return ex * (1.0+x+ex) / (1.0+ex)**2

  def _fun_torch(self, x):
    return x * torch.sigmoid(x)

  def _jac_torch(self, x):
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)

# ReLU
# -------------------------------------
class ReLU(BaseAct):

  def __init__(self):
    self._fun = np.vectorize(self._fun)
    self._jac = np.vectorize(self._jac)
    super(ReLU, self).__init__()

  def _fun(self, x):
    return x if (x > 0.0) else 0.0

  def _jac(self, x):
    return 1.0 if (x > 0.0) else 0.0

  def _fun_torch(self, x):
    return torch.where(x > 0.0, x, torch.zeros_like(x))

  def _jac_torch(self, x):
    return torch.where(x > 0.0, torch.ones_like(x), torch.zeros_like(x))

# ELU
# -------------------------------------
class ELU(ReLU):

  def __init__(self, alpha=1.0):
    self.alpha = float(alpha)
    super(ELU, self).__init__()

  def _fun(self, x):
    return x if (x > 0.0) else self.alpha*(np.exp(x)-1.0)

  def _jac(self, x):
    return 1.0 if (x > 0.0) else self.alpha*np.exp(x)

  def _fun_torch(self, x):
    return torch.where(x > 0.0, x, self.alpha * (torch.exp(x) - 1.0))

  def _jac_torch(self, x):
    return torch.where(x > 0.0, torch.ones_like(x), self.alpha * torch.exp(x))

# Mixed
# -------------------------------------
class Mixed(BaseAct):

  def __init__(self, masks):
    super(Mixed, self).__init__()
    self.masks = {}
    for (act, mask) in masks.items():
      self.masks[act] = (get(act), mask.reshape(-1))

  def _fun(self, x):
    y = copy.deepcopy(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._fun(x[mask])
    return y

  def _jac(self, x):
    y = copy.deepcopy(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._jac(x[mask])
    return y

  def _fun_torch(self, x):
    y = torch.clone(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._fun_torch(x[mask])
    return y

  def _jac_torch(self, x):
    y = torch.zeros_like(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._jac_torch(x[mask])
    return y

# Softplus
# -------------------------------------
class Softplus(BaseAct):
  """
  Softplus: f(x) = log(1 + exp(x))
  Jacobian (elementwise): f'(x) = 1 / (1 + exp(-x))  == sigmoid(x)
  Uses stable formulas to avoid overflow/underflow.
  """

  def _fun(self, x):
    # Stable: log(1+exp(x)) = max(x,0) + log1p(exp(-|x|)) with minimal temporaries
    absx = np.abs(x)
    return np.where(x > 0, x + np.log1p(np.exp(-absx)), np.log1p(np.exp(-absx)))

  def _jac(self, x):
    # f'(x) = sigmoid(x); use tanh formulation for stability
    return 0.5 * (1.0 + np.tanh(0.5 * x))

  def _fun_torch(self, x):
    return functional.softplus(x)

  def _jac_torch(self, x):
    return torch.sigmoid(x)
