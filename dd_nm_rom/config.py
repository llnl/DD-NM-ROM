import logging
import hashlib
import json
import os

logger = logging.getLogger(__name__)

# General options
_DDNMROM_VERBOSE = int(0)
_DDNMROM_LOG_LEVEL = int(0)

# Solver-specific options
# NOTE: not all options are used by all solvers
_DDNMROM_SOLVE_FORCE_SERIAL = False
_DDNMROM_SOLVE_FORCE_DENSE = False
_DDNMROM_SOLVE_FORCE_BACKEND = "pytorch" # (for torch-sla, e.g, pytorch, cudss, etc)
_DDNMROM_SOLVE_PRECONDITIONER = "none"
_DDNMROM_SOLVE_IGNORE_HIST = False
_DDNMROM_SOLVE_LINE_SEARCH = True

# Activation options
_DDNMROM_ACT_COMPILE = True

# Backend specific options
# NOTE: not all options are used by all backends or runtimes (numpy, pytorch, or cpu/gpu serial or parallel)
_DDNMROM_BACKEND_DEVICE_PER_NODE = 4
#_DDNMROM_BACKEND_MPI_FILLZERO = True
_DDNMROM_BACKEND_DTENSOR_CHECKS = False


_DDNMROM_BACKEND_MPI_FILLZERO = False
_DDNMROM_BACKEND_MPI_GPU_AWARE = False



_env_vars = {
  "DDNMROM_VERBOSE": _DDNMROM_VERBOSE, # whether to use verbose prints and enable logging (1: enable, 0: disable)
  "DDNMROM_LOG_LEVEL": _DDNMROM_LOG_LEVEL, # Logging level (40: error, 30: warning, 20: info, 10: debug, 0: disable)
  # Solver configuration options
  "DDNMROM_FORCE_SERIAL_SOLVE": _DDNMROM_SOLVE_FORCE_SERIAL,
  "DDNMROM_FORCE_DENSE_SOLVE": _DDNMROM_SOLVE_FORCE_DENSE,
  "DDNMROM_FORCE_SOLVE_BACKEND": _DDNMROM_SOLVE_FORCE_BACKEND,
  "DDNMROM_SOLVE_PRECONDITIONER": _DDNMROM_SOLVE_PRECONDITIONER, # distributed Krylov preconditioner; use "none" to disable
  # Solver behavior options
  "DDNMROM_SOLVE_IGNORE_HIST": _DDNMROM_SOLVE_IGNORE_HIST, # force solver to not track full-residual histories over each iteration
  "DDNMROM_SOLVE_LINE_SEARCH": _DDNMROM_SOLVE_LINE_SEARCH, # enable Armijo backtracking in the distributed Newton solver
  # Activation options
  "DDNMROM_ACT_COMPILE": _DDNMROM_ACT_COMPILE, # enable torch.compile for Torch activations
  # Backend specific options
  "DDNMROM_DEVICE_PER_NODE": _DDNMROM_BACKEND_DEVICE_PER_NODE, # physical devices per node, used to map gpus to physical layout, change depending on hardware
  "DDNMROM_MPI_BUFFER_ZEROFILL": _DDNMROM_BACKEND_MPI_FILLZERO, # whether to create communication buffers always filled with zero, or empty memory
  "DDNMROM_MPI_GPU_AWARE": _DDNMROM_BACKEND_MPI_GPU_AWARE, # enable device-resident mpi4py buffers (requires validated GPU-aware MPI)
  "DDNMROM_DTENSOR_CHECKS": _DDNMROM_BACKEND_DTENSOR_CHECKS, # enable torch.DTensor shape checks on creation (adds overhead)
  }


def _convert_env_value(value, default):
  """Convert an environment value using the registered default's type."""
  if isinstance(default, bool):
    return value.lower() in ("1", "true", "yes", "on")
  return type(default)(value)


def get_config_snapshot():
  """Return effective DDNM-ROM environment configuration and provenance.

  Registered options are reported even when unset, using their configured
  defaults.  Additional ``DDNMROM_*`` variables are retained under
  ``extra_environment`` so benchmark metadata also captures runner- or
  site-specific settings that are not library configuration options.
  """
  registered = {}
  effective = {}
  for var, default in _env_vars.items():
    raw_value = os.getenv(var)
    if raw_value is None:
      value = default
      source = "default"
    else:
      value = _convert_env_value(raw_value, default)
      source = "environment"
    registered[var] = {
      "value": value,
      "source": source,
      "default": default,
      "environment": raw_value,
    }
    effective[var] = value

  extras = {
    var: value for var, value in os.environ.items()
    if var.startswith("DDNMROM_") and var not in _env_vars
  }
  encoded = json.dumps(effective, sort_keys=True, separators=(",", ":"))
  fingerprint = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
  return {
    "registered": registered,
    "extra_environment": extras,
    "fingerprint": fingerprint,
  }


def print_config_env():
  lines = ["----------------------", "DDNMROM configuration:", "----------------------"]
  for (var, default_val) in _env_vars.items():
    env_value = os.getenv(var)
    if env_value is None:
      lines.append("   '{}': {} (default)".format(var, default_val))
    else:
      lines.append("   '{}': {} (env)".format(var, env_value))
  lines.extend(["----------------------", ""])
  logger.info("\n".join(lines))


def get_config_val(var, get_default=True):
  if var not in _env_vars:
    raise RuntimeError("Tried to find unexpected config var '{}'".format(var))
  
  if get_default:
    logger.debug("GETTING DEFAULT FOR '%s'", var)
    value = os.getenv(var, _env_vars[var])
    if value is None:
      raise RuntimeError(" Error getting config var '{}'".format(var))
    return value
  else:
    return os.getenv(var)


def update_from_env(var, current=None, verbose=True):
  """
  Returns a (possibly) new value for the given "current" variable, based on environment option var.
  If current variable is None, then lookup the default defined here
  If verbose is enabled, then this will print a message if the variable changed from the environment
  """
  if var not in _env_vars:
    raise RuntimeError("Tried to find unexpected config var '{}'".format(var))

  _current = current
  if _current is None:
    # set default
    _current = _env_vars[var]

  opt = get_config_val(var, get_default=False)
  if opt is not None:
    # if provided, we want to make sure the given env value matches type of the incoming
    # TODO: should do some better checking here
    if isinstance(_current, bool):
      opt = opt.lower() in ("1", "true", "yes", "on")
    else:
      opt = type(_current)(opt)

    # print a message if the existing value changed from the existing
    if verbose and _current != opt:
      logger.debug(
        "Overriding option from environment! '%s' (was %s, now %s)",
        var, _current, opt,
      )
    return opt
  else:
    # if env variable is not defined, then we do nothing and return current value
    return _current
