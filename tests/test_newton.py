import numpy as np
import pytest
import torch
from scipy.optimize import newton_krylov

from dd_nm_rom import backend as bkd
from dd_nm_rom import field as field_mod
from dd_nm_rom import fom as fom_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.solvers.gauss_newton import GaussNewton
from dd_nm_rom.solvers.newton import Newton


def _build_burgers_fom():
  """Build the small monolithic Burgers2D case used by the FOM tests."""
  mesh = mesh_mod.MeshDD(
    nx_intr=5,
    ny_intr=5,
    lx_sub=0.5,
    ly_sub=0.5,
    n_sub_x=2,
    n_sub_y=2,
  )
  mesh.build()

  viscosity = 1.0e-1
  field = field_mod.Burgers2DExact(
    mesh=mesh,
    nu=viscosity,
    a_lim=[1.0, 1.0e4],
    k_lim=[5.0, 25.0],
  )
  field.set_params(np.array([1.0e3, 10.0]))

  fom = fom_mod.Burgers2D(mesh=mesh, nu=viscosity)
  fom.build(field)
  return fom


def _nearby_burgers_start(x_star, mesh):
  """Return a deterministic, nonzero perturbation of a Burgers root."""
  perturbation = field_mod.MultiPeak(
    mesh=mesh,
    mu_lim=[1.0e-2, 1.0e-2],
    bc_type="dirichlet",
    use_qmc=True,
  )
  delta = perturbation.get_init(
    np.full(2 * mesh.n_sub, 1.0e-2)
  )
  return x_star + delta


def _solve_result(model, x0):
  solver = Newton(
    model=model,
    tol=1.0e-8,
    maxit=50,
    stepsize_min=1.0e-20,
    verbose=False,
  )
  result = solver.solve(x0)
  return solver, result


def _install_serial_numpy_backend(monkeypatch):
  """Keep the SciPy reference independent of any active MPI state."""
  monkeypatch.setattr(bkd, "is_torch_backend", lambda: False)
  monkeypatch.setattr(bkd, "distributed", lambda: False)
  monkeypatch.setattr(bkd, "get_rank", lambda: 0)
  monkeypatch.setattr(bkd, "get_nranks", lambda: 1)
  monkeypatch.setattr(bkd, "root", lambda: True)
  monkeypatch.setattr(bkd, "barrier", lambda: None)


def _solve_scipy_reference(model, x0):
  """Solve with SciPy's independent Armijo inexact-Newton solver."""
  def residual(x):
    res, _ = model.res_jac(x)
    return res

  x = newton_krylov(
    residual,
    x0,
    f_tol=1.0e-8,
    maxiter=500,
    line_search="armijo",
  )
  return x, residual(x)


def _scalar_result_value(value):
  return np.asarray(value).reshape(-1)[-1].item()


class _OneStepNewtonModel:
  """Scalar residual that Newton solves in one step."""

  def __init__(self):
    self.runtime = {"total": 0.0, "lin_solve": 0.0}

  def res_jac(self, x):
    return np.asarray(x, dtype=float), np.eye(1)


class _OneStepGaussNewtonModel:
  """Overdetermined residual with a zero gradient after one Gauss step."""

  def __init__(self):
    self.runtime = {"total": 0.0, "lin_solve": 0.0}

  def res_jac(self, x):
    return np.array([x[0], 1.0]), np.array([[1.0], [0.0]])


@pytest.mark.no_backend
@pytest.mark.parametrize(
  ("solver_type", "model_type"),
  ((Newton, _OneStepNewtonModel), (GaussNewton, _OneStepGaussNewtonModel)),
)
@pytest.mark.parametrize("use_line_search", (True, False))
def test_final_allowed_iteration_can_converge(
  monkeypatch,
  solver_type,
  model_type,
  use_line_search,
):
  """A solve that converges at maxit reports success, not flag 3."""
  monkeypatch.setattr(bkd, "_BKD", "numpy", raising=False)
  monkeypatch.setattr(bkd, "device", lambda: torch.device("cpu"))
  solver = solver_type(
    model=model_type(),
    tol=1.0e-12,
    maxit=1,
    stepsize_min=1.0e-20,
    use_line_search=use_line_search,
  )

  result = solver.solve(np.array([1.0]))

  assert solver.use_line_search is use_line_search
  assert _scalar_result_value(result[-2]) == 1
  assert _scalar_result_value(result[-1]) == 0


@pytest.mark.no_backend
def test_newton_uses_scipy_by_default_on_cpu(monkeypatch):
  """CPU Torch solves prefer the available direct SciPy backend."""
  monkeypatch.delenv("DDNMROM_FORCE_SOLVE_BACKEND", raising=False)
  monkeypatch.setattr(bkd, "_BKD", "torch", raising=False)
  monkeypatch.setattr(bkd, "device", lambda: torch.device("cpu"))

  solver = Newton(model=_OneStepNewtonModel())

  assert solver.solve_backend == "scipy"


@pytest.mark.parametrize(
  ("force_dense_env", "force_dense"),
  [("0", False), ("1", True)],
  ids=("sparse", "dense"),
)
def test_newton_matches_serial_scipy(
  monkeypatch,
  force_dense_env,
  force_dense,
):
  """Compare the selected backend Newton solve with serial SciPy Newton."""
  # Make the Torch sparse path deterministic and serial. The dense setting is
  # ignored by the NumPy linear-solver selection, but is still checked below.
  monkeypatch.setenv("DDNMROM_FORCE_DENSE_SOLVE", force_dense_env)
  monkeypatch.setenv("DDNMROM_FORCE_SOLVE_BACKEND", "pytorch")

  active_backend = bkd.get_backend()

  # Build and solve the reference entirely with NumPy/SciPy on the CPU.
  with monkeypatch.context() as serial:
    _install_serial_numpy_backend(serial)
    bkd.set_backend("numpy")
    bkd.set_floatx("float64")
    reference = _build_burgers_fom()
    # The zero state is outside the robust convergence basin of the direct
    # Newton path for this nonlinear FOM.  Bootstrap its discrete solution,
    # then use a deterministic field-based nonzero perturbation as the common
    # starting point for the SciPy and selected-backend solves.
    root_seed, _ = _solve_scipy_reference(
      reference, np.zeros(reference.get_ndof())
    )
    x0 = _nearby_burgers_start(root_seed, reference.mesh)
    reference_x, reference_res = _solve_scipy_reference(reference, x0)
  assert np.linalg.norm(reference_res) < 1.0e-7

  # Restore the backend selected by tests/conftest.py and build the same model
  # after switching its sparse operators to the selected backend.
  bkd.set_backend(active_backend)
  bkd.set_floatx("float64")
  test_model = _build_burgers_fom()
  test_x0 = bkd.to_backend(x0)
  test_solver, test_result = _solve_result(test_model, test_x0)

  assert test_solver.use_dense_solve is force_dense
  assert test_solver.solve_backend == "pytorch"
  assert _scalar_result_value(test_result[-1]) == 0

  np.testing.assert_allclose(
    bkd.to_numpy(test_result[0]),
    reference_x,
    rtol=1.0e-6,
    atol=1.0e-8,
  )
  np.testing.assert_allclose(
    bkd.to_numpy(test_result[2][-1]),
    np.linalg.norm(reference_res),
    rtol=1.0e-6,
    atol=1.0e-8,
  )
