
"""Top-level package for DD-NM-ROM."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .log import _setup_logger

try:
  __version__ = version("dd_nm_rom")
except PackageNotFoundError:
  __version__ = "0.0.0"

_log = _setup_logger(__name__)

_LAZY_MODULES = {
  "backend",
  "config",
  "elements",
  "env",
  "field",
  "fom",
  "ops",
  "postproc",
  "rom",
  "solvers",
  "utils",
}

__all__ = sorted(_LAZY_MODULES | {"__version__"})


def __getattr__(name):
  if name in _LAZY_MODULES:
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
  return sorted(set(globals()) | _LAZY_MODULES | {"__version__"})
