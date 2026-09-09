import numpy as np
import pytest
import torch
import torch_sla
from mpi4py import MPI
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import spsolve

from dd_nm_rom import backend as bkd
from dd_nm_rom import field as field_mod
from dd_nm_rom import fom as fom_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.fom.domain_dec.model import DDBurgers2D
from dd_nm_rom.rom.nonlinear.domain_dec.model import DD_NM_ROM
from dd_nm_rom.solvers.dist_newton import DistNewton


_BACKEND_INITIALIZED = False


class _ReplicatedVector:
  """Minimal replicated-vector stand-in for the line-search unit test."""

  def __init__(self, value):
    self.value = torch.as_tensor(value, dtype=torch.float64)

  def __add__(self, other):
    return _ReplicatedVector(self.value + other.value)

  def __rmul__(self, scalar):
    return _ReplicatedVector(scalar * self.value)

  def full_tensor(self):
    return self.value


class _ConstraintOnlyModel:
  """Residual model with zero state residual and nonzero constraints."""

  def __init__(self):
    self.runtime = {"total": 0.0}

  def residual(self, x, use_global=False):
    del x, use_global
    return torch.zeros(1, dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)


def test_line_search_norm_includes_constraint_residual():
  """A constraint-only residual must not appear converged."""
  solver = DistNewton(
    model=_ConstraintOnlyModel(),
    tol=1.0e-8,
    stepsize_min=1.0e-10,
  )
  x0 = _ReplicatedVector([0.0])
  dx = _ReplicatedVector([0.0])

  _, _, _, res_norm, _ = solver.line_search(
    x0,
    dx,
    eval_res_tol=lambda _: np.inf,
    use_global=False,
  )

  assert res_norm > solver.tol
  np.testing.assert_allclose(res_norm, 2.0)


def test_line_search_can_be_disabled_by_constructor_or_environment(monkeypatch):
  """The constructor option is enabled by default and env-overridable."""
  monkeypatch.delenv("DDNMROM_SOLVE_LINE_SEARCH", raising=False)
  solver = DistNewton(model=_ConstraintOnlyModel(), use_line_search=False)
  assert not solver.use_line_search

  monkeypatch.setenv("DDNMROM_SOLVE_LINE_SEARCH", "0")
  solver = DistNewton(model=_ConstraintOnlyModel(), use_line_search=True)
  assert not solver.use_line_search


def test_rom_trial_vector_uses_rank_state_and_global_multipliers(monkeypatch):
  model = DD_NM_ROM.__new__(DD_NM_ROM)
  model.local_sizes = np.array([2, 3])
  model.global_offsets = np.array([0, 4])
  model.global_sizes = np.array([4, 5])
  model.n_constraints = 2
  model.debug = False
  monkeypatch.setattr(bkd, "distributed", lambda: True)
  monkeypatch.setattr(bkd, "get_rank", lambda: 1)

  trial = torch.arange(11, dtype=torch.float64)
  local_trial = model.local_trial_vector(trial)

  torch.testing.assert_close(
    local_trial,
    torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
  )

  state, lambdas = model.split_lambdas(local_trial, use_global=False)
  torch.testing.assert_close(state, torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
  torch.testing.assert_close(lambdas, torch.tensor([9.0, 10.0]))


def test_fom_trial_vector_keeps_replicated_global_kkt_vector():
  """FOM local work uses global offsets, unlike the ROM local layout."""
  model = DDBurgers2D.__new__(DDBurgers2D)
  trial = torch.arange(11, dtype=torch.float64)

  assert model.local_trial_vector(trial) is trial


def test_shard_preconditioner_can_be_set_by_constructor_or_environment(monkeypatch):
  monkeypatch.delenv("DDNMROM_SOLVE_PRECONDITIONER", raising=False)

  assert DistNewton(model=_ConstraintOnlyModel()).preconditioner is None
  assert DistNewton(
    model=_ConstraintOnlyModel(), preconditioner="jacobi",
  ).preconditioner == "jacobi"
  assert DistNewton(
    model=_ConstraintOnlyModel(), preconditioner=None,
  ).preconditioner is None

  monkeypatch.setenv("DDNMROM_SOLVE_PRECONDITIONER", "block_jacobi")
  assert DistNewton(
    model=_ConstraintOnlyModel(), preconditioner="jacobi_l1",
  ).preconditioner == "block_jacobi"

  monkeypatch.setenv("DDNMROM_SOLVE_PRECONDITIONER", "none")
  assert DistNewton(
    model=_ConstraintOnlyModel(), preconditioner="jacobi_l1",
  ).preconditioner is None


def test_direct_solve_prefers_strumpack_and_requires_an_available_backend(monkeypatch):
  """Direct distributed solves use an available direct sparse backend."""
  monkeypatch.setenv("DDNMROM_FORCE_SERIAL_SOLVE", "1")
  monkeypatch.delenv("DDNMROM_FORCE_SOLVE_BACKEND", raising=False)
  monkeypatch.setattr(
    torch_sla.backends, "is_strumpack_available", lambda: True,
  )
  monkeypatch.setattr(
    torch_sla.backends, "is_cudss_available", lambda: True,
  )

  solver = DistNewton(model=_ConstraintOnlyModel())
  assert solver.use_direct_solve
  assert solver.direct_solve_backend == "strumpack"

  monkeypatch.setattr(
    torch_sla.backends, "is_strumpack_available", lambda: False,
  )
  monkeypatch.setattr(
    torch_sla.backends, "is_cudss_available", lambda: False,
  )
  with pytest.raises(RuntimeError, match="requires a sparse direct backend"):
    DistNewton(model=_ConstraintOnlyModel())


@pytest.fixture(scope="module", autouse=True)
def configured_backend(request):
  """Initialize the selected Torch backend when the test is MPI-launched."""
  global _BACKEND_INITIALIZED
  selected = request.config.getoption("--backend") or "torch_cpu"
  request.getfixturevalue("backend_" + selected)

  if not _BACKEND_INITIALIZED:
    bkd.set_floatx("float64")
    world_size = MPI.COMM_WORLD.Get_size()
    if bkd._NRANKS is None and world_size > 1:
      device = "cuda" if bkd.device().type == "cuda" else "cpu"
      if device == "cpu":
        # backend.init_distributed queries CUDA metadata even for Gloo.
        get_name = torch.cuda.get_device_name
        get_properties = torch.cuda.get_device_properties
        torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"
        torch.cuda.get_device_properties = lambda *args, **kwargs: "cpu"
        try:
          bkd.init_distributed(device)
        finally:
          torch.cuda.get_device_name = get_name
          torch.cuda.get_device_properties = get_properties
      else:
        bkd.init_distributed(device)
    bkd.set_seed(0)
    _BACKEND_INITIALIZED = True
  yield


def _build_case(subs_per_rank, world_size):
  mesh = mesh_mod.MeshDD(
    nx_intr=5,
    ny_intr=5,
    lx_sub=0.5,
    ly_sub=0.5,
    n_sub_x=2,
    n_sub_y=world_size,
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
  dd_fom = fom_mod.DDBurgers2D(
    monolithic=fom,
    constraint_type="strong",
    scaling=-1,
    subs_per_rank=subs_per_rank,
  )
  dd_fom.build()

  x_phys = np.concatenate([
    field.u(*mesh.grid).reshape(-1),
    field.v(*mesh.grid).reshape(-1),
  ])
  return dd_fom, x_phys


def _install_serial_numpy_backend(monkeypatch):
  """Use the NumPy reference path without disturbing MPI state."""
  monkeypatch.setattr(bkd, "is_torch_backend", lambda: False)
  monkeypatch.setattr(bkd, "distributed", lambda: False)
  monkeypatch.setattr(bkd, "get_rank", lambda: 0)
  monkeypatch.setattr(bkd, "get_nranks", lambda: 1)
  monkeypatch.setattr(bkd, "root", lambda: True)
  monkeypatch.setattr(bkd, "barrier", lambda: None)


def _solve_reference(monkeypatch, world_size):
  with monkeypatch.context() as serial:
    _install_serial_numpy_backend(serial)
    bkd.set_backend("numpy")
    bkd.set_floatx("float64")
    reference, _ = _build_case(
      subs_per_rank=2 * world_size,
      world_size=world_size,
    )
    physical_x = newton_krylov(
      lambda x: reference.monolithic.res_jac(x)[0],
      np.zeros(reference.monolithic.get_ndof()),
      f_tol=1.0e-8,
      maxiter=500,
      line_search="armijo",
    )
    x0 = reference.get_init_sol(physical_x)
    x0[0] += 1.0e-2
    x = np.array(x0, copy=True)
    res, jac = reference.res_jac(x)
    res_norm = np.linalg.norm(res)
    stepsize_min = 1.0e-20
    tol = 1.0e-8
    maxit = 50
    it = 0

    while res_norm >= tol and it < maxit:
      dx = spsolve(jac, -res)
      stepsize = 1.0
      while True:
        trial_x = x + stepsize * dx
        trial_res, trial_jac = reference.res_jac(trial_x)
        trial_norm = np.linalg.norm(trial_res)
        eval_res_tol = (1.0 - 2.0e-4 * stepsize) * res_norm
        if trial_norm < eval_res_tol or stepsize < stepsize_min:
          break
        stepsize *= 0.5

      x, res, jac, res_norm = (
        trial_x,
        trial_res,
        trial_jac,
        trial_norm,
      )
      it += 1
      if stepsize < stepsize_min:
        break

    return x, res, physical_x


def _last_scalar(value):
  return np.asarray(value).reshape(-1)[-1].item()


@pytest.mark.mpi(min_size=2)
def test_dist_newton_matches_serial_scipy(monkeypatch):
  """Compare distributed Torch DistNewton with a serial SciPy reference."""
  if not bkd.is_torch_backend() or not bkd.distributed():
    pytest.skip("requires a multi-rank Torch backend for DistNewton")

  monkeypatch.setenv("DDNMROM_FORCE_SERIAL_SOLVE", "0")
  world_size = bkd.get_nranks()
  reference_x, reference_res, x_phys = _solve_reference(monkeypatch, world_size)

  bkd.set_backend("torch")
  parallel, _ = _build_case(
    subs_per_rank=2,
    world_size=world_size,
  )
  x0 = parallel.get_init_sol(x_phys).clone()
  x0[0] += 1.0e-2
  solver = DistNewton(
    model=parallel,
    tol=1.0e-8,
    maxit=50,
    stepsize_min=1.0e-20,
    verbose=False,
    distributed=True,
  )

  # The distributed line-search norm must include the constraint block.  Use
  # a zero Newton step so the candidate is exactly x0, then compare its local
  # line-search norm with the norm of the globally assembled KKT residual.
  x0_replicated = bkd.to_replica_dtensor(x0)
  zero_step = bkd.to_replica_dtensor(torch.zeros_like(x0))
  candidate, _, _, line_search_norm, _ = solver.line_search(
    x0_replicated,
    zero_step,
    eval_res_tol=lambda _: np.inf,
    use_global=False,
  )
  assembled_res, _ = parallel.res_jac(candidate.full_tensor())
  np.testing.assert_allclose(
    _last_scalar(line_search_norm),
    np.linalg.norm(bkd.to_numpy(assembled_res)),
    rtol=1.0e-12,
    atol=1.0e-12,
  )

  result = solver.solve(x0)

  assert not solver.use_direct_solve
  assert np.linalg.norm(reference_res) < 1.0e-7
  assert _last_scalar(result[-1]) == 0
  np.testing.assert_allclose(
    bkd.to_numpy(result[0]),
    reference_x,
    rtol=1.0e-6,
    atol=1.0e-8,
  )
  np.testing.assert_allclose(
    bkd.to_numpy(result[2][-1]),
    np.linalg.norm(reference_res),
    rtol=1.0e-6,
    atol=1.0e-8,
  )
  np.testing.assert_allclose(
    np.asarray(result[2]).reshape(-1),
    np.linalg.norm(bkd.to_numpy(result[1]), axis=1),
    rtol=1.0e-6,
    atol=1.0e-8,
  )
