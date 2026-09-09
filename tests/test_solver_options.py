"""Tests for the common solver configuration contract."""

import inspect

import pytest
import torch

from dd_nm_rom import backend as bkd
from dd_nm_rom.solvers import DistNewton, GaussNewton, Newton


pytestmark = pytest.mark.no_backend

_SOLVER_TYPES = (Newton, GaussNewton, DistNewton)
_COMMON_PARAMETERS = (
  "model",
  "tol",
  "maxit",
  "stepsize_min",
  "iostep",
  "verbose",
  "distributed",
  "use_line_search",
  "preconditioner",
)


class _Model:
  """Minimal model required for solver construction."""

  def __init__(self):
    self.runtime = {"total": 0.0, "lin_solve": 0.0}


@pytest.mark.parametrize("solver_type", _SOLVER_TYPES)
def test_solvers_share_common_constructor_options(monkeypatch, solver_type):
  """All solver implementations expose and retain the common options."""
  monkeypatch.delenv("DDNMROM_FORCE_DENSE_SOLVE", raising=False)
  monkeypatch.delenv("DDNMROM_FORCE_SERIAL_SOLVE", raising=False)
  monkeypatch.delenv("DDNMROM_FORCE_SOLVE_BACKEND", raising=False)
  monkeypatch.delenv("DDNMROM_SOLVE_LINE_SEARCH", raising=False)
  monkeypatch.delenv("DDNMROM_SOLVE_PRECONDITIONER", raising=False)
  monkeypatch.setattr(bkd, "device", lambda: torch.device("cpu"))

  assert tuple(inspect.signature(solver_type).parameters) == _COMMON_PARAMETERS

  solver = solver_type(
    model=_Model(),
    iostep=3,
    distributed=True,
    use_line_search=False,
    preconditioner="jacobi",
  )

  assert solver.iostep == 3
  assert solver.distributed_solve
  assert not solver.use_line_search
  assert solver.preconditioner == "jacobi"


@pytest.mark.parametrize("solver_type", _SOLVER_TYPES)
def test_solver_common_options_obey_environment_overrides(monkeypatch, solver_type):
  """Environment settings override the shared constructor options uniformly."""
  monkeypatch.delenv("DDNMROM_FORCE_DENSE_SOLVE", raising=False)
  monkeypatch.delenv("DDNMROM_FORCE_SERIAL_SOLVE", raising=False)
  monkeypatch.delenv("DDNMROM_FORCE_SOLVE_BACKEND", raising=False)
  monkeypatch.setenv("DDNMROM_SOLVE_LINE_SEARCH", "0")
  monkeypatch.setenv("DDNMROM_SOLVE_PRECONDITIONER", "none")
  monkeypatch.setattr(bkd, "device", lambda: torch.device("cpu"))

  solver = solver_type(
    model=_Model(),
    use_line_search=True,
    preconditioner="jacobi",
  )

  assert not solver.use_line_search
  assert solver.preconditioner is None
