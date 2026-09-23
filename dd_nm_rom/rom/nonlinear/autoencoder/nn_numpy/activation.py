import abc
import copy
from collections import OrderedDict
import time
import numpy as np

import torch
from torch.nn import functional

from dd_nm_rom.ops import sp_diag
from dd_nm_rom import backend as bkd
import dd_nm_rom.config as cfg

#torch._logging.set_logs(graph_code=True)

_ACT_IDS = ("elu", "linear", "mixed", "relu", "sigmoid", "swish", "softplus")
_COMPILED_CACHE_MAXSIZE = 32
_COMPILED_CACHE = OrderedDict()
_COMPILE_STATS = {}


def reset_compile_stats():
  """Reset preparation-time activation compilation statistics."""
  global _COMPILE_STATS
  _COMPILE_STATS = {
    "warmup_calls": 0,
    "compile_disabled_calls": 0,
    "compile_enabled_calls": 0,
    "eligible_calls": 0,
    "skipped_small": 0,
    "skipped_linear": 0,
    "skipped_mixed": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "compiled_functions": 0,
    "compile_warmup_seconds": 0.0,
    "by_role_activation": {},
  }


def get_compile_stats():
  """Return JSON-serializable preparation-time compilation statistics."""
  return copy.deepcopy(_COMPILE_STATS)


def _role_stats(role, act):
  key = "{}:{}".format(role or "unknown", type(act).__name__)
  stats = _COMPILE_STATS["by_role_activation"].setdefault(
    key,
    {
      "warmup_calls": 0,
      "eligible_calls": 0,
      "cache_hits": 0,
      "cache_misses": 0,
      "size_min": None,
      "size_max": None,
      "size_total": 0,
    },
  )
  return stats


reset_compile_stats()


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


def warmup(act, size, device, dtype, role="unknown"):
  """Bind and warm an activation when compilation can amortize its overhead.

  Activation functions are called between sparse matrix operations, so
  compiling tiny or masked functions generally adds launch and wrapper
  overhead without enabling useful fusion.  The size threshold keeps the
  opt-in compiler path focused on sufficiently large, regular activations.
  """
  size = int(size)
  _COMPILE_STATS["warmup_calls"] += 1
  role_stats = _role_stats(role, act)
  role_stats["warmup_calls"] += 1
  role_stats["size_total"] += size
  role_stats["size_min"] = (
    size if role_stats["size_min"] is None
    else min(role_stats["size_min"], size)
  )
  role_stats["size_max"] = (
    size if role_stats["size_max"] is None
    else max(role_stats["size_max"], size)
  )

  if not _should_compile():
    _COMPILE_STATS["compile_disabled_calls"] += 1
    return False

  _COMPILE_STATS["compile_enabled_calls"] += 1
  min_size = int(cfg.get_config_val("DDNMROM_ACT_COMPILE_MIN_SIZE"))
  if size < min_size:
    _COMPILE_STATS["skipped_small"] += 1
    return False
  if isinstance(act, Linear):
    _COMPILE_STATS["skipped_linear"] += 1
    return False
  if isinstance(act, Mixed):
    _COMPILE_STATS["skipped_mixed"] += 1
    # Linear activations have no useful work to optimize. Mixed activations
    # contain Python-level masked indexing and are deliberately kept as one
    # eager operation rather than partially compiling their components.
    return False

  _COMPILE_STATS["eligible_calls"] += 1
  role_stats["eligible_calls"] += 1

  device_key = str(torch.device(device))
  dtype_key = str(dtype)
  signature = (type(act).__name__, getattr(act, "alpha", None))
  key = (signature, device_key, dtype_key)
  compiled = _COMPILED_CACHE.get(key)
  cache_miss = compiled is None
  if cache_miss:
    _COMPILE_STATS["cache_misses"] += 1
    role_stats["cache_misses"] += 1
    start = time.perf_counter()
    # Compile pure closures over activation parameters.  This allows the
    # result to be reused by equivalent activation objects and avoids making
    # the bound Python object part of the compiled graph.
    def fun(x, activation=act):
      return activation._fun_torch(x)

    # Compile the forward and Jacobian together for eligible large
    # activations. This avoids the compiled/eager split on the dominant
    # with_jac=True path. Small activations never reach this branch.
    def fun_jac(x, activation=act):
      return activation._fun_torch(x), activation._jac_torch(x)

    compiled = (_maybe_compile(fun), _maybe_compile(fun_jac))
    _COMPILED_CACHE[key] = compiled
    _COMPILED_CACHE.move_to_end(key)
    while len(_COMPILED_CACHE) > _COMPILED_CACHE_MAXSIZE:
      _COMPILED_CACHE.popitem(last=False)
  else:
    _COMPILE_STATS["cache_hits"] += 1
    role_stats["cache_hits"] += 1
    _COMPILED_CACHE.move_to_end(key)

  act.fun, act._compiled_fun_jac = compiled
  x = torch.zeros(size, device=device, dtype=dtype)
  act.fun(x)
  act._compiled_fun_jac(x)
  if cache_miss:
    # The first invocation is lazy for Torch Inductor, so include it in the
    # preparation statistic. Cache-hit warmups are intentionally not included.
    # Keep this branch free of CUDA synchronization; DD_NM_ROM synchronizes
    # once after all ranks finish preparation.
    _COMPILE_STATS["compile_warmup_seconds"] += time.perf_counter() - start
    _COMPILE_STATS["compiled_functions"] += 2
  return True


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

        # Compilation is deferred until warmup knows the actual shape.  The
        # eager path remains valid if compilation is disabled or unsupported.
        act._fun = act._fun_torch
        act._jac = act._jac_torch
        act.fun = act._fun
        act.jac = act._jac_torch
        act._compiled_fun_jac = None
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
    self._compiled_fun_jac = None

  def __call__(self, x, with_jac=True):
    return (self.fun(x), self.jac(x)) if with_jac else self.fun(x)

  def _call__torch(self, x, with_jac=True):
    if with_jac:
      if self._compiled_fun_jac is not None:
        return self._compiled_fun_jac(x)
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
