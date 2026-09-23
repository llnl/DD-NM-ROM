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
_MIXED_LAYOUT_COUNTS = {}
_COMPILE_STATS = {}


def reset_compile_stats():
  """Reset preparation-time activation compilation statistics."""
  global _COMPILE_STATS, _MIXED_LAYOUT_COUNTS
  _MIXED_LAYOUT_COUNTS = {}
  _COMPILE_STATS = {
    "warmup_calls": 0,
    "compile_disabled_calls": 0,
    "compile_enabled_calls": 0,
    "eligible_calls": 0,
    "skipped_small": 0,
    "skipped_linear": 0,
    "skipped_mixed": 0,
    "skipped_allowlist": 0,
    "skipped_mixed_frequency": 0,
    "skipped_role": 0,
    "skipped_no_path": 0,
    "skipped_cache_full": 0,
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
      "skipped_cache_full": 0,
      "skipped_allowlist": 0,
      "skipped_mixed_frequency": 0,
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
  return torch.compile(fun, backend="inductor", dynamic=False)


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


def _config_bool(name):
  """Read a boolean configuration value with support for typed defaults."""
  value = cfg.get_config_val(name)
  if isinstance(value, bool):
    return value
  return value.lower() in ("1", "true", "yes", "on")


def _role_should_compile(role):
  """Return whether compilation is enabled for an encoder or decoder."""
  if role == "encoder":
    return _config_bool("DDNMROM_ACT_COMPILE_ENCODER")
  if role == "decoder":
    return _config_bool("DDNMROM_ACT_COMPILE_DECODER")
  # Preserve direct warmup API behavior for callers that do not identify a
  # model role. Model warmup passes an explicit role.
  return True


def _activation_should_compile(act):
  """Return whether an activation type is selected by the allowlist."""
  value = cfg.get_config_val("DDNMROM_ACT_COMPILE_ACTIVATIONS")
  names = {
    name.strip().lower()
    for name in value.split(",")
    if name.strip()
  }
  return "all" in names or type(act).__name__.lower() in names


def _is_mixed_cache_key(key):
  return key[0] == "Mixed"


def _set_eager(act):
  """Restore an activation object to its eager Torch callables."""
  act.fun = act._fun_torch
  act.jac = act._jac_torch
  act._compiled_fun = None
  act._compiled_fun_jac = None


def _mixed_fun_from_parts(x, parts):
  activation_parts, full_coverage = parts
  y = torch.empty_like(x) if full_coverage else torch.clone(x)
  for act, indices in activation_parts:
    values = act._fun_torch(x.index_select(0, indices))
    y.index_copy_(0, indices, values)
  return y


def _mixed_fun_jac_from_parts(x, parts):
  activation_parts, full_coverage = parts
  y = torch.empty_like(x) if full_coverage else torch.clone(x)
  dy = torch.empty_like(x) if full_coverage else torch.zeros_like(x)
  for act, indices in activation_parts:
    values = x.index_select(0, indices)
    y.index_copy_(0, indices, act._fun_torch(values))
    dy.index_copy_(0, indices, act._jac_torch(values))
  return y, dy


def warmup(act, size, device, dtype, role="unknown", force=False):
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
    _set_eager(act)
    return False

  _COMPILE_STATS["compile_enabled_calls"] += 1
  if not _role_should_compile(role):
    _COMPILE_STATS["skipped_role"] += 1
    _set_eager(act)
    return False
  if not _activation_should_compile(act):
    _COMPILE_STATS["skipped_allowlist"] += 1
    role_stats["skipped_allowlist"] += 1
    _set_eager(act)
    return False
  compile_forward = _config_bool("DDNMROM_ACT_COMPILE_FORWARD")
  compile_jac = _config_bool("DDNMROM_ACT_COMPILE_JAC")
  if not compile_forward and not compile_jac:
    _COMPILE_STATS["skipped_no_path"] += 1
    _set_eager(act)
    return False
  min_size = int(cfg.get_config_val("DDNMROM_ACT_COMPILE_MIN_SIZE"))
  if size < min_size:
    _COMPILE_STATS["skipped_small"] += 1
    _set_eager(act)
    return False
  if isinstance(act, Linear):
    _COMPILE_STATS["skipped_linear"] += 1
    _set_eager(act)
    return False
  _COMPILE_STATS["eligible_calls"] += 1
  role_stats["eligible_calls"] += 1

  device_key = str(torch.device(device))
  dtype_key = str(dtype)
  input_shape = (size,)
  signature = (type(act).__name__, getattr(act, "alpha", None))
  if isinstance(act, Mixed):
    signature = (signature, act._compile_signature())
  key = (
    type(act).__name__,
    signature,
    device_key,
    dtype_key,
    input_shape,
    compile_forward,
    compile_jac,
  )
  compiled = _COMPILED_CACHE.get(key)
  cache_miss = compiled is None
  if isinstance(act, Mixed) and cache_miss:
    policy = cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_POLICY").lower()
    if policy not in ("all", "top_k"):
      raise ValueError(
        "DDNMROM_ACT_COMPILE_MIXED_POLICY must be 'all' or 'top_k', "
        f"got {policy!r}"
      )
    layout_count = _MIXED_LAYOUT_COUNTS.get(signature, 0) + 1
    _MIXED_LAYOUT_COUNTS[signature] = layout_count
    min_frequency = int(
      cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY")
    )
    if min_frequency < 1:
      raise ValueError(
        "DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY must be positive"
      )
    if policy == "top_k" and layout_count < min_frequency and not force:
      _COMPILE_STATS["skipped_mixed_frequency"] += 1
      role_stats["skipped_mixed_frequency"] += 1
      _set_eager(act)
      return False
  if cache_miss:
    # Static compilation creates one specialization per input shape. Retain
    # existing specializations, but avoid evicting and recompiling them when
    # a workload introduces too many unique shapes; those new shapes remain
    # on the eager path instead.
    mixed_cache_size = int(
      cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE")
    )
    if mixed_cache_size < 1:
      raise ValueError(
        "DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE must be positive"
      )
    cache_full = len(_COMPILED_CACHE) >= _COMPILED_CACHE_MAXSIZE
    if isinstance(act, Mixed):
      cache_full = cache_full or sum(
        _is_mixed_cache_key(existing_key)
        for existing_key in _COMPILED_CACHE
      ) >= mixed_cache_size
    if cache_full:
      _COMPILE_STATS["skipped_cache_full"] += 1
      role_stats["skipped_cache_full"] = role_stats.get("skipped_cache_full", 0) + 1
      _set_eager(act)
      return False
    _COMPILE_STATS["cache_misses"] += 1
    role_stats["cache_misses"] += 1
    start = time.perf_counter()
    # Compile pure closures over activation parameters. This allows the
    # result to be reused by equivalent activation objects and avoids making
    # the bound Python object part of the compiled graph. Compile only the
    # paths selected by configuration; the defaults target the decoder
    # Jacobian used by the Newton solve.
    compiled_fun = None
    compiled_fun_jac = None
    compile_parts = None
    if isinstance(act, Mixed):
      compile_parts = (
        act._torch_compile_parts_for_size(device, size),
        act._full_coverage(size),
      )
    if compile_forward:
      if compile_parts is None:
        def fun(x, activation=act):
          return activation._fun_torch(x)
      else:
        def fun(x, parts=compile_parts):
          return _mixed_fun_from_parts(x, parts)
      compiled_fun = _maybe_compile(fun)
    if compile_jac:
      if compile_parts is None:
        def fun_jac(x, activation=act):
          return activation._fun_torch(x), activation._jac_torch(x)
      else:
        def fun_jac(x, parts=compile_parts):
          return _mixed_fun_jac_from_parts(x, parts)
      compiled_fun_jac = _maybe_compile(fun_jac)

    compiled = (compiled_fun, compiled_fun_jac)
    _COMPILED_CACHE[key] = compiled
    _COMPILED_CACHE.move_to_end(key)
  else:
    _COMPILE_STATS["cache_hits"] += 1
    role_stats["cache_hits"] += 1
    _COMPILED_CACHE.move_to_end(key)

  compiled_fun, compiled_fun_jac = compiled
  act.fun = compiled_fun if compiled_fun is not None else act._fun_torch
  act.jac = act._jac_torch
  act._compiled_fun = compiled_fun
  act._compiled_fun_jac = compiled_fun_jac
  x = torch.zeros(size, device=device, dtype=dtype)
  if compiled_fun is not None:
    act.fun(x)
  if compiled_fun_jac is not None:
    act._compiled_fun_jac(x)
  if cache_miss:
    # The first invocation is lazy for Torch Inductor, so include it in the
    # preparation statistic. Cache-hit warmups are intentionally not included.
    # Keep this branch free of CUDA synchronization; DD_NM_ROM synchronizes
    # once after all ranks finish preparation.
    _COMPILE_STATS["compile_warmup_seconds"] += time.perf_counter() - start
    _COMPILE_STATS["compiled_functions"] += (
      int(compiled_fun is not None) + int(compiled_fun_jac is not None)
    )
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
        act._compiled_fun = None
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
    self._compiled_fun = None
    self._compiled_fun_jac = None

  def __call__(self, x, with_jac=True):
    return (self.fun(x), self.jac(x)) if with_jac else self.fun(x)

  def _call__torch(self, x, with_jac=True):
    # ``get`` installs this method on BaseAct globally for the Torch backend.
    # Direct NumPy activation objects can still be used after that happens,
    # so keep their original eager dispatch when the input is not a Tensor.
    if not torch.is_tensor(x):
      return (self.fun(x), self.jac(x)) if with_jac else self.fun(x)
    if with_jac:
      if self._compiled_fun_jac is not None:
        return self._compiled_fun_jac(x)
      if getattr(self, "_compiled_fun", None) is None:
        return self._fun_jac_torch(x)
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

  def _fun_jac_torch(self, x):
    return self._fun_torch(x), self._jac_torch(x)

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
    # Keep the public NumPy masks for model construction and HR handling, but
    # lazily materialize one device-resident index tensor per Torch device.
    # This avoids converting masks and creating indexing tensors on every
    # activation call.
    self._torch_masks = {}
    self._torch_parts_cache = {}
    coverage_indices = []
    coverage_known = True
    for (act, mask) in masks.items():
      if torch.is_tensor(mask):
        if mask.dtype != torch.bool and mask.dtype not in (
          torch.int8, torch.int16, torch.int32, torch.int64,
          torch.uint8,
        ):
          raise TypeError(
            "Mixed activation masks must have boolean or integer dtype"
          )
      else:
        mask_array = np.asarray(mask)
        if (
          mask_array.dtype != np.bool_
          and not np.issubdtype(mask_array.dtype, np.integer)
        ):
          raise TypeError(
            "Mixed activation masks must have boolean or integer dtype"
          )
      mask = (
        mask.reshape(-1) if hasattr(mask, "reshape")
        else np.asarray(mask).reshape(-1)
      )
      self.masks[act] = (get(act), mask)
      if torch.is_tensor(mask) and mask.device.type != "cpu":
        coverage_known = False
        continue
      mask_array = (
        mask.detach().cpu().numpy() if torch.is_tensor(mask)
        else np.asarray(mask)
      ).reshape(-1)
      if mask_array.dtype == np.bool_:
        mask_array = np.flatnonzero(mask_array)
      coverage_indices.extend(
        np.asarray(mask_array, dtype=np.int64).tolist()
      )
    if coverage_known:
      unique_indices = np.unique(np.asarray(coverage_indices, dtype=np.int64))
      self._coverage = (
        len(coverage_indices),
        len(unique_indices),
        int(unique_indices[0]) if len(unique_indices) else None,
        int(unique_indices[-1]) if len(unique_indices) else None,
      )
    else:
      self._coverage = None

  def _torch_mask(self, act_id, mask, device):
    device_key = str(torch.device(device))
    device_masks = self._torch_masks.setdefault(device_key, {})
    indices = device_masks.get(act_id)
    if indices is None:
      if torch.is_tensor(mask):
        mask = mask.reshape(-1)
        if mask.dtype == torch.bool:
          indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        else:
          indices = mask.to(dtype=torch.long)
        indices = indices.to(device=device)
      else:
        mask_array = np.asarray(mask).reshape(-1)
        if mask_array.dtype == np.bool_:
          mask_array = np.flatnonzero(mask_array)
        indices = torch.as_tensor(
          mask_array,
          dtype=torch.long,
          device=device,
        )
      device_masks[act_id] = indices
    return indices

  def _torch_compile_parts(self, device):
    return self._torch_compile_parts_for_size(device, None)

  def _torch_compile_parts_for_size(self, device, size):
    parts = []
    for act_id, (act, mask) in self.masks.items():
      is_boolean = (
        torch.is_tensor(mask) and mask.dtype == torch.bool
      ) or (
        not torch.is_tensor(mask)
        and np.asarray(mask).dtype == np.bool_
      )
      if (
        size is not None
        and is_boolean
        and int(mask.numel() if torch.is_tensor(mask) else np.size(mask))
        != int(size)
      ):
        raise IndexError(
          "Mixed activation boolean mask must match the input size"
        )
      parts.append(
        (
          act,
          self._normalize_indices(
            self._torch_mask(act_id, mask, device), size
          ),
        )
      )
    return tuple(parts)

  @staticmethod
  def _normalize_indices(indices, size):
    """Match eager advanced-indexing rules for integer masks."""
    if size is None or indices.numel() == 0:
      return indices
    invalid = (indices < -size) | (indices >= size)
    if bool(torch.any(invalid)):
      raise IndexError(
        "Mixed activation mask index is out of bounds for input size "
        f"{size}"
      )
    if bool(torch.any(indices < 0)):
      return indices.remainder(size)
    return indices

  def _torch_parts(self, device, size):
    device_key = str(torch.device(device))
    cache_key = (device_key, int(size))
    parts = self._torch_parts_cache.get(cache_key)
    if parts is None:
      parts = self._torch_compile_parts_for_size(device, int(size))
      self._torch_parts_cache[cache_key] = parts
    return parts

  def _full_coverage(self, size):
    """Return whether the masks form a disjoint partition of ``range(size)``."""
    if self._coverage is None:
      return False
    count, unique_count, first, last = self._coverage
    if size == 0:
      return count == 0
    return (
      count == size
      and unique_count == size
      and first == 0
      and last == size - 1
    )

  def _compile_signature(self):
    signature = []
    for act_id, (act, mask) in self.masks.items():
      if torch.is_tensor(mask):
        mask = mask.detach().to(device="cpu").reshape(-1)
        if mask.dtype == torch.bool:
          indices = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        else:
          indices = mask.to(dtype=torch.long).tolist()
      else:
        mask_array = np.asarray(mask).reshape(-1)
        if mask_array.dtype == np.bool_:
          mask_array = np.flatnonzero(mask_array)
        indices = np.asarray(mask_array, dtype=np.int64).tolist()
      signature.append(
        (
          act_id,
          type(act).__name__,
          getattr(act, "alpha", None),
          tuple(int(index) for index in indices),
        )
      )
    return tuple(signature)

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
    parts = self._torch_parts(x.device, x.numel())
    y = torch.empty_like(x) if self._full_coverage(x.numel()) else torch.clone(x)
    for act, indices in parts:
      values = act._fun_torch(x.index_select(0, indices))
      y.index_copy_(0, indices, values)
    return y

  def _jac_torch(self, x):
    parts = self._torch_parts(x.device, x.numel())
    y = torch.empty_like(x) if self._full_coverage(x.numel()) else torch.zeros_like(x)
    for act, indices in parts:
      values = act._jac_torch(x.index_select(0, indices))
      y.index_copy_(0, indices, values)
    return y

  def _fun_jac_torch(self, x):
    """Evaluate Mixed forward and derivative with one gather per mask."""
    parts = self._torch_parts(x.device, x.numel())
    full_coverage = self._full_coverage(x.numel())
    y = torch.empty_like(x) if full_coverage else torch.clone(x)
    dy = torch.empty_like(x) if full_coverage else torch.zeros_like(x)
    for act, indices in parts:
      values = x.index_select(0, indices)
      y.index_copy_(0, indices, act._fun_torch(values))
      dy.index_copy_(0, indices, act._jac_torch(values))
    return y, dy

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
