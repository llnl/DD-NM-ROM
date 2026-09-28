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


def clear_compiled_cache(mixed_only=False):
  """Remove cached Torch callables, optionally retaining pure activations.

  Mixed-layout selection samples eager activations and then reserves cache
  entries for the layouts it chooses.  Keeping stale Mixed entries from model
  construction would make that reservation depend on whichever layouts were
  encountered first.
  """
  if mixed_only:
    keys = [key for key in _COMPILED_CACHE if _is_mixed_cache_key(key)]
  else:
    keys = list(_COMPILED_CACHE)
  for key in keys:
    _COMPILED_CACHE.pop(key, None)


def reset_mixed_compile_state():
  """Reset workload-local Mixed layout frequency bookkeeping."""
  _MIXED_LAYOUT_COUNTS.clear()


def compiled_cache_slots_available():
  """Return the number of free entries in the shared compile cache."""
  return max(0, _COMPILED_CACHE_MAXSIZE - len(_COMPILED_CACHE))


def _set_eager(act):
  """Restore an activation object to its eager Torch callables."""
  act.fun = act._fun_torch
  act.jac = act._jac_torch
  act._compiled_fun = None
  act._compiled_fun_jac = None


def _mixed_fun_from_parts(x, parts):
  activation_parts, full_coverage, packed_plan = parts
  if packed_plan is not None:
    order, packed_groups, identity = packed_plan
    # A full DD-ROM Mixed layout can be evaluated in activation-group order,
    # reducing K gathers/scatters to one gather/scatter for K activation types.
    if identity:
      y = torch.empty_like(x)
      for act, start, end in packed_groups:
        y[start:end] = act._fun_torch(x[start:end])
      return y

    grouped = x.index_select(0, order)
    for act, start, end in packed_groups:
      grouped[start:end] = act._fun_torch(grouped[start:end])
    y = torch.empty_like(x)
    return y.index_copy_(0, order, grouped)

  y = torch.empty_like(x) if full_coverage else torch.clone(x)
  for act, indices in activation_parts:
    values = act._fun_torch(x.index_select(0, indices))
    y.index_copy_(0, indices, values)
  return y


def _mixed_fun_jac_from_parts(x, parts):
  activation_parts, full_coverage, packed_plan = parts
  if packed_plan is not None:
    order, packed_groups, identity = packed_plan
    if identity:
      y = torch.empty_like(x)
      dy = torch.empty_like(x)
      for act, start, end in packed_groups:
        values, derivatives = act._fun_jac_torch(x[start:end])
        y[start:end] = values
        dy[start:end] = derivatives
      return y, dy

    grouped = x.index_select(0, order)
    dy_grouped = torch.empty_like(grouped)
    for act, start, end in packed_groups:
      values, derivatives = act._fun_jac_torch(grouped[start:end])
      grouped[start:end] = values
      dy_grouped[start:end] = derivatives
    y = torch.empty_like(x)
    dy = torch.empty_like(x)
    y.index_copy_(0, order, grouped)
    dy.index_copy_(0, order, dy_grouped)
    return y, dy

  y = torch.empty_like(x) if full_coverage else torch.clone(x)
  dy = torch.empty_like(x) if full_coverage else torch.zeros_like(x)
  for act, indices in activation_parts:
    values = x.index_select(0, indices)
    result, derivative = act._fun_jac_torch(values)
    y.index_copy_(0, indices, result)
    dy.index_copy_(0, indices, derivative)
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
      activation_parts = act._torch_compile_parts_for_size(device, size)
      compile_parts = (
        activation_parts,
        act._full_coverage(size),
        act._torch_packed_plan(device, size, activation_parts),
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
          return activation._fun_jac_torch(x)
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
      # A forward-only compiled callable cannot share intermediates with the
      # Jacobian. Prefer the fused eager pair over running two paths.
      return self._fun_jac_torch(x)
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

  def _fun_jac_torch(self, x):
    return x, torch.ones_like(x)


# Sigmoid
# -------------------------------------
def _numpy_sigmoid(x):
  x = np.asarray(x)
  exp_neg_abs = np.exp(-np.abs(x))
  return np.where(
    x >= 0.0,
    1.0 / (1.0 + exp_neg_abs),
    exp_neg_abs / (1.0 + exp_neg_abs),
  )


class Sigmoid(BaseAct):

  def _fun(self, x):
    return _numpy_sigmoid(x)

  def _jac(self, x):
    fx = self._fun(x)
    return fx * (1.0 - fx)

  def _fun_torch(self, x):
    return torch.sigmoid(x)

  def _jac_torch(self, x):
    fx = torch.sigmoid(x)
    return fx * (1.0 - fx)

  def _fun_jac_torch(self, x):
    fx = torch.sigmoid(x)
    return fx, fx * (1.0 - fx)

# Swish
# -------------------------------------
class Swish(BaseAct):

  def _fun(self, x):
    fx = _numpy_sigmoid(x)
    return x * fx

  def _jac(self, x):
    fx = _numpy_sigmoid(x)
    return fx + x * fx * (1.0 - fx)

  def _fun_torch(self, x):
    return functional.silu(x)

  def _jac_torch(self, x):
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)

  def _fun_jac_torch(self, x):
    sig = torch.sigmoid(x)
    return x * sig, sig + x * sig * (1.0 - sig)

# ReLU
# -------------------------------------
class ReLU(BaseAct):

  def _fun(self, x):
    return np.maximum(x, 0.0)

  def _jac(self, x):
    return np.where(np.asarray(x) > 0.0, 1.0, 0.0)

  def _fun_torch(self, x):
    return torch.relu(x)

  def _jac_torch(self, x):
    return torch.gt(x, 0.0).to(dtype=x.dtype)

  def _fun_jac_torch(self, x):
    return torch.relu(x), torch.gt(x, 0.0).to(dtype=x.dtype)

# ELU
# -------------------------------------
class ELU(ReLU):

  def __init__(self, alpha=1.0):
    self.alpha = float(alpha)
    super(ELU, self).__init__()

  def _fun(self, x):
    x = np.asarray(x)
    return np.where(
      x > 0.0,
      x,
      self.alpha * np.expm1(np.minimum(x, 0.0)),
    )

  def _jac(self, x):
    x = np.asarray(x)
    return np.where(
      x > 0.0,
      1.0,
      self.alpha * np.exp(np.minimum(x, 0.0)),
    )

  def _fun_torch(self, x):
    return functional.elu(x, alpha=self.alpha)

  def _jac_torch(self, x):
    result = functional.elu(x, alpha=self.alpha)
    return torch.where(x > 0.0, torch.ones_like(x), result + self.alpha)

  def _fun_jac_torch(self, x):
    result = functional.elu(x, alpha=self.alpha)
    derivative = torch.where(
      x > 0.0,
      torch.ones_like(x),
      result + self.alpha,
    )
    return result, derivative

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
    self._torch_packed_cache = {}
    coverage_indices = []
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
      # Materialize coverage metadata once. Device-resident masks incur one
      # preparation-time host transfer, allowing subsequent calls to use the
      # packed execution plan without repeated coverage checks.
      mask_array = (
        mask.detach().cpu().numpy() if torch.is_tensor(mask)
        else np.asarray(mask)
      ).reshape(-1)
      if mask_array.dtype == np.bool_:
        mask_array = np.flatnonzero(mask_array)
      coverage_indices.extend(
        np.asarray(mask_array, dtype=np.int64).tolist()
      )
    unique_indices = np.unique(np.asarray(coverage_indices, dtype=np.int64))
    self._coverage = (
      len(coverage_indices),
      len(unique_indices),
      int(unique_indices[0]) if len(unique_indices) else None,
      int(unique_indices[-1]) if len(unique_indices) else None,
    )

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

  def _torch_packed_plan(self, device, size, parts=None):
    """Build one gather/scatter plan for a full, disjoint partition."""
    device_key = str(torch.device(device))
    cache_key = (device_key, int(size))
    if cache_key in self._torch_packed_cache:
      return self._torch_packed_cache[cache_key]

    if not self._full_coverage(size):
      self._torch_packed_cache[cache_key] = None
      return None
    if parts is None:
      parts = self._torch_parts(device, size)
    indices = tuple(indices for _, indices in parts)
    if indices:
      order = torch.cat(indices)
    else:
      order = torch.empty(0, dtype=torch.long, device=device)
    packed_groups = []
    start = 0
    for act, indices in parts:
      end = start + indices.numel()
      packed_groups.append((act, start, end))
      start = end
    identity = torch.equal(
      order,
      torch.arange(int(size), dtype=torch.long, device=device),
    )
    plan = (order, tuple(packed_groups), identity)
    self._torch_packed_cache[cache_key] = plan
    return plan

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
    y = x.copy()
    for (act, mask) in self.masks.values():
      y[mask] = act._fun(x[mask])
    return y

  def _jac(self, x):
    # Match the established Torch Mixed convention: only masked entries
    # contribute to the returned derivative vector; uncovered entries remain
    # zero even though the forward path preserves their values.
    y = np.zeros_like(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._jac(x[mask])
    return y

  def _fun_torch(self, x):
    parts = self._torch_parts(x.device, x.numel())
    packed_plan = self._torch_packed_plan(x.device, x.numel(), parts)
    if packed_plan is not None:
      return _mixed_fun_from_parts(
        x,
        (parts, True, packed_plan),
      )
    y = torch.empty_like(x) if self._full_coverage(x.numel()) else torch.clone(x)
    for act, indices in parts:
      values = act._fun_torch(x.index_select(0, indices))
      y.index_copy_(0, indices, values)
    return y

  def _jac_torch(self, x):
    parts = self._torch_parts(x.device, x.numel())
    packed_plan = self._torch_packed_plan(x.device, x.numel(), parts)
    if packed_plan is not None:
      order, packed_groups, identity = packed_plan
      if identity:
        dy = torch.empty_like(x)
        for act, start, end in packed_groups:
          dy[start:end] = act._jac_torch(x[start:end])
        return dy

      grouped = x.index_select(0, order)
      dy_grouped = torch.empty_like(grouped)
      for act, start, end in packed_groups:
        dy_grouped[start:end] = act._jac_torch(grouped[start:end])
      dy = torch.empty_like(x)
      return dy.index_copy_(0, order, dy_grouped)
    y = torch.empty_like(x) if self._full_coverage(x.numel()) else torch.zeros_like(x)
    for act, indices in parts:
      values = act._jac_torch(x.index_select(0, indices))
      y.index_copy_(0, indices, values)
    return y

  def _fun_jac_torch(self, x):
    """Evaluate Mixed forward and derivative with packed full partitions."""
    parts = self._torch_parts(x.device, x.numel())
    packed_plan = self._torch_packed_plan(x.device, x.numel(), parts)
    if packed_plan is not None:
      return _mixed_fun_jac_from_parts(
        x,
        (parts, True, packed_plan),
      )
    full_coverage = self._full_coverage(x.numel())
    y = torch.empty_like(x) if full_coverage else torch.clone(x)
    dy = torch.empty_like(x) if full_coverage else torch.zeros_like(x)
    for act, indices in parts:
      values = x.index_select(0, indices)
      result, derivative = act._fun_jac_torch(values)
      y.index_copy_(0, indices, result)
      dy.index_copy_(0, indices, derivative)
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

  def _fun_jac_torch(self, x):
    result = functional.softplus(x)
    return result, torch.sigmoid(x)
