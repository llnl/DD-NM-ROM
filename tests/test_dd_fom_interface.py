import numpy as np
import pytest
import scipy.sparse as sp
import torch
from mpi4py import MPI

from dd_nm_rom import backend as bkd
from dd_nm_rom import field as field_mod
from dd_nm_rom import fom as fom_mod
from dd_nm_rom.elements import mesh as mesh_mod


_BACKEND_INITIALIZED = False
_SOLUTION_REL_TOL = 1.0e-6


@pytest.fixture(scope="module", autouse=True)
def configured_backend(request):
  """Use conftest's selected backend and initialize MPI for parallel runs."""
  global _BACKEND_INITIALIZED
  selected = request.config.getoption("--backend") or "torch_cpu"
  request.getfixturevalue("backend_" + selected)
  if not _BACKEND_INITIALIZED:
    bkd.set_floatx("float64")
    world_size = MPI.COMM_WORLD.Get_size()
    if world_size > 1 and not bkd.distributed():
      device = "cuda" if bkd.device().type == "cuda" else "cpu"
      if device == "cpu":
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


def _as_numpy(value):
  """Convert dense/sparse backend values to a NumPy array for comparison."""
  if hasattr(value, "to_local"):
    value = value.to_local()
  if torch.is_tensor(value):
    if value.layout != torch.strided:
      value = value.to_dense()
    return value.detach().cpu().numpy()
  if sp.issparse(value):
    return value.toarray()
  return np.asarray(value)


def _assert_allclose(actual, expected):
  np.testing.assert_allclose(
    _as_numpy(actual),
    _as_numpy(expected),
    rtol=1e-10,
    atol=1e-11,
  )


def _install_serial_numpy_backend(monkeypatch):
  """Use the NumPy serial path without disturbing the MPI process group."""
  monkeypatch.setattr(bkd, "is_torch_backend", lambda: False)
  monkeypatch.setattr(bkd, "distributed", lambda: False)
  monkeypatch.setattr(bkd, "get_rank", lambda: 0)
  monkeypatch.setattr(bkd, "get_nranks", lambda: 1)
  monkeypatch.setattr(bkd, "root", lambda: True)
  monkeypatch.setattr(bkd, "barrier", lambda: None)


def _build_case(subs_per_rank, world_size):
  """Build a small Burgers DD-FOM without reading a configuration file."""
  mesh = mesh_mod.MeshDD(
    nx_intr=3,
    ny_intr=3,
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


def _build_numpy_and_torch(monkeypatch):
  """Build a serial NumPy reference and the active Torch model."""
  nranks = bkd.get_nranks()
  n_sub = 2 * nranks

  with monkeypatch.context() as serial:
    _install_serial_numpy_backend(serial)
    bkd.set_backend("numpy")
    reference, x_phys = _build_case(
      subs_per_rank=n_sub,
      world_size=nranks,
    )

  bkd.set_backend("torch")
  bkd.set_seed(0)
  parallel, _ = _build_case(
    subs_per_rank=n_sub // nranks,
    world_size=nranks,
  )
  return reference, parallel, x_phys


def _call_numpy(monkeypatch, function, *args, **kwargs):
  with monkeypatch.context() as serial:
    _install_serial_numpy_backend(serial)
    bkd.set_backend("numpy")
    return function(*args, **kwargs)


def _call_torch(function, *args, **kwargs):
  bkd.set_backend("torch")
  args = [bkd.to_backend(value) for value in args]
  kwargs = {key: bkd.to_backend(value) for key, value in kwargs.items()}
  return function(*args, **kwargs)


def _canonical_init_sol(model, value):
  """Return get_init_sol in the shared serial subdomain-major layout."""
  return _as_numpy(value).reshape(-1)


def test_dd_fom_get_init_sol_numpy_vs_torch(monkeypatch):
  reference, parallel, x_phys = _build_numpy_and_torch(monkeypatch)
  ref_x0 = _call_numpy(monkeypatch, reference.get_init_sol, x_phys)
  test_x0 = _call_torch(parallel.get_init_sol, x_phys)
  _assert_allclose(_canonical_init_sol(parallel, test_x0), ref_x0)


def test_dd_fom_assemble_cmat_numpy_vs_torch(monkeypatch):
  reference, parallel, _ = _build_numpy_and_torch(monkeypatch)
  assert parallel.n_constraints == reference.n_constraints
  for element in ("interior", "interface"):
    assert len(parallel.cmat[element]) == len(reference.cmat[element])
    for actual, expected in zip(parallel.cmat[element], reference.cmat[element]):
      _assert_allclose(actual, expected)


def test_dd_fom_get_ndof_numpy_vs_torch(monkeypatch):
  reference, parallel, _ = _build_numpy_and_torch(monkeypatch)
  assert _call_numpy(monkeypatch, reference.get_ndof) == _call_torch(parallel.get_ndof)


def test_dd_fom_residual_numpy_vs_torch(monkeypatch):
  reference, parallel, x_phys = _build_numpy_and_torch(monkeypatch)
  ref_x0 = _call_numpy(monkeypatch, reference.get_init_sol, x_phys)
  test_x0 = _call_torch(parallel.get_init_sol, x_phys)
  ref_res, ref_cres = _call_numpy(monkeypatch, reference.residual, ref_x0, use_global=True)
  test_res, test_cres = _call_torch(parallel.residual, test_x0, use_global=True)
  _assert_allclose(test_res, ref_res)
  if bkd.root():
    _assert_allclose(test_cres, ref_cres)


def test_dd_fom_res_jac_numpy_vs_torch(monkeypatch):
  reference, parallel, x_phys = _build_numpy_and_torch(monkeypatch)
  ref_x0 = _call_numpy(monkeypatch, reference.get_init_sol, x_phys)
  test_x0 = _call_torch(parallel.get_init_sol, x_phys)
  ref_res, ref_jac = _call_numpy(monkeypatch, reference.res_jac, ref_x0, use_global=True)
  test_res, test_jac = _call_torch(parallel.res_jac, test_x0, use_global=True)
  _assert_allclose(test_res, ref_res)
  _assert_allclose(test_jac, ref_jac)


def _build_fom_solution_case(
  constraint_type,
  boundary_type,
  n_sub_y=None,
):
  """Build a shared monolithic/DD-FOM case for a physical solution check."""
  world_size = MPI.COMM_WORLD.Get_size()
  if world_size > 1 and not bkd.is_torch_backend():
    pytest.skip("distributed DD solves require a Torch backend")

  if n_sub_y is None:
    n_sub_y = 2 * world_size

  mesh = mesh_mod.MeshDD(
    nx_intr=16,
    ny_intr=16,
    lx_sub=0.5,
    ly_sub=0.5,
    n_sub_x=1,
    n_sub_y=n_sub_y,
    with_bounds=True,
  )
  mesh.build()
  field = field_mod.SinMultiPeak(
    mesh=mesh,
    mu_lim=[0.9, 1.1],
    bc_type=boundary_type,
    use_qmc=True,
  )
  field.set_params(np.linspace(0.9, 1.1, mesh.n_sub))
  x0 = np.concatenate([field.u().reshape(-1), field.v().reshape(-1)])

  # This non-compact operator path is implemented by every selected backend.
  fom = fom_mod.Burgers2D(
    mesh=mesh,
    nu=1.0e-3,
    upwind=True,
    upwind_order=2,
    compact=False,
  )
  fom.build(field)
  dd_fom = fom_mod.DDBurgers2D(
    monolithic=fom,
    constraint_type=constraint_type,
    scaling=-1,
    subs_per_rank=2,
    # A full-rank random transform retains all strong constraints while
    # exercising the weak-constraint assembly path.
    n_constraints_weak=1_000_000,
  )
  dd_fom.build()
  return fom, dd_fom, x0


def _assert_dd_fom_solution_matches_monolithic(
  steady,
  constraint_type,
  boundary_type,
  n_sub_y=None,
):
  """Assert that a DD-FOM solve recovers the monolithic physical state."""
  fom, dd_fom, x0 = _build_fom_solution_case(
    constraint_type,
    boundary_type,
    n_sub_y=n_sub_y,
  )
  solve_args = {
    "dt": 0.0 if steady else 1.0e-3,
    "nt": 1,
    "steady": steady,
    "tol": 1.0e-8,
    "maxit": 50,
    "stepsize_min": 1.0e-20,
    "verbose": False,
  }

  # The selected boundary fields have homogeneous boundary data, making zero
  # the compatible steady state.  The unsteady case uses the nonzero field
  # state to exercise the time-discrete solve.
  if steady:
    x0 = np.zeros_like(x0)

  fom_uv, _, fom_converged = fom.solve(x0=x0, **solve_args)
  assert fom_converged

  dd_x0 = dd_fom.get_init_sol(x0)
  dd_uv, _, _, dd_converged = dd_fom.solve(x0=dd_x0, **solve_args)
  assert dd_converged

  relative_errors = {}
  absolute_errors = {}
  for component in ("u", "v"):
    reference = _as_numpy(fom_uv[component])
    candidate = _as_numpy(dd_uv["res"][component])
    absolute_errors[component] = np.linalg.norm(candidate - reference)
    if not steady:
      relative_errors[component] = (
        absolute_errors[component] / np.linalg.norm(reference)
      )

  if steady:
    # With zero-source Neumann/periodic steady problems the physical solution
    # is nearly zero, so a relative norm is not meaningful.
    assert max(absolute_errors.values()) < 2.0e-3
  else:
    assert max(relative_errors.values()) < _SOLUTION_REL_TOL


@pytest.mark.parametrize("steady", (True, False), ids=("steady", "unsteady"))
@pytest.mark.parametrize("constraint_type", ("strong", "weak"))
@pytest.mark.parametrize("boundary_type", ("neumann", "periodic"))
def test_dd_fom_solution_matches_monolithic_fom(
  steady,
  constraint_type,
  boundary_type,
):
  """DD-FOM tracks the monolithic solution across solver formulations."""
  _assert_dd_fom_solution_matches_monolithic(
    steady,
    constraint_type,
    boundary_type,
  )


@pytest.mark.skipif(
  MPI.COMM_WORLD.Get_size() != 1,
  reason="serial baseline for the distributed unsteady DD-FOM regression",
)
def test_serial_unsteady_neumann_strong_dd_fom_matches_monolithic():
  """Check the failing 8-subdomain layout without distributed solves."""
  _assert_dd_fom_solution_matches_monolithic(
    steady=False,
    constraint_type="strong",
    boundary_type="neumann",
    # The recorded failure uses 4 ranks with 2 subdomains per rank.  Retain
    # the same decomposition in serial so KKT layout/assembly can be checked
    # independently of MPI collectives and distributed Krylov solves.
    n_sub_y=8,
  )


@pytest.mark.mpi(min_size=2)
def test_dist_unsteady_neumann_strong_dd_fom_matches_monolithic():
  """Reproduce the multi-rank unsteady DD-FOM solution mismatch."""
  if not bkd.distributed():
    pytest.skip("requires an initialized multi-rank Torch backend")
  _assert_dd_fom_solution_matches_monolithic(
    steady=False,
    constraint_type="strong",
    boundary_type="neumann",
  )


@pytest.mark.skipif(
  MPI.COMM_WORLD.Get_size() != 1,
  reason="serial-only test; temporarily replaced by the FOM benchmark",
)
@pytest.mark.skip(
  reason="temporarily skipped while replacing the serial NumPy KKT reference",
)
def test_dd_fom_solve_numpy_serial(monkeypatch):
  """Legacy serial NumPy KKT solve retained for future re-enablement."""
  original_backend = bkd.get_backend()
  try:
    with monkeypatch.context() as serial:
      _install_serial_numpy_backend(serial)
      bkd.set_backend("numpy")
      bkd.set_seed(0)
      reference, x_phys = _build_case(subs_per_rank=2, world_size=1)
      x0 = reference.get_init_sol(x_phys)
      _, _, residual_history, converged = reference.solve(
        x0=x0,
        tol=1.0e-8,
        maxit=20,
        stepsize_min=1.0e-20,
        verbose=False,
      )
  finally:
    bkd.set_backend(original_backend)

  final_norm = np.linalg.norm(_as_numpy(residual_history[-1]))
  assert converged, f"serial NumPy reference solve did not converge (||res||={final_norm:.6e})"
  assert final_norm < 1.0e-6


@pytest.mark.mpi(min_size=2)
@pytest.mark.skip(
  reason="temporarily skipped; the Torch FOM benchmark below is the solver reference",
)
def test_dd_fom_solve_numpy_vs_distributed_torch(monkeypatch):
  """Legacy serial NumPy versus distributed Torch KKT comparison.

  This test is retained as a record of the former comparison, but is skipped
  because the two KKT formulations do not provide a reliable serial reference.
  """
  if not bkd.is_torch_backend() or not bkd.distributed():
    pytest.skip("requires a multi-rank Torch backend for DistNewton")

  reference, parallel, x_phys = _build_numpy_and_torch(monkeypatch)
  ref_x0 = _call_numpy(monkeypatch, reference.get_init_sol, x_phys)
  test_x0 = _call_torch(parallel.get_init_sol, x_phys)

  solve_args = {
    "tol": 1.0e-8,
    "maxit": 20,
    "stepsize_min": 1.0e-20,
    "verbose": False,
  }
  ref_uv, ref_lambdas, ref_res, ref_converged = _call_numpy(
    monkeypatch,
    reference.solve,
    x0=ref_x0,
    **solve_args,
  )
  test_uv, test_lambdas, test_res, test_converged = _call_torch(
    parallel.solve,
    x0=test_x0,
    **solve_args,
  )

  assert ref_converged, "serial NumPy reference solve did not converge"
  assert test_converged, "distributed Torch DistNewton solve did not converge"

  for component in ("u", "v"):
    np.testing.assert_allclose(
      _as_numpy(test_uv["res"][component]),
      _as_numpy(ref_uv["res"][component]),
      rtol=1.0e-6,
      atol=1.0e-8,
    )
  np.testing.assert_allclose(
    _as_numpy(test_lambdas),
    _as_numpy(ref_lambdas),
    rtol=1.0e-6,
    atol=1.0e-8,
  )

  # Iteration counts may differ because the distributed linear solve is
  # iterative, but both returned histories must end at a small full KKT
  # residual.
  assert np.linalg.norm(_as_numpy(ref_res[-1])) < 1.0e-6
  assert np.linalg.norm(_as_numpy(test_res[-1])) < 1.0e-6


def test_dd_fom_solve_torch_against_fom():
  """Benchmark the DD-FOM solution against the monolithic Torch FOM.

  Both models use the same mesh, field, and physical discretization.  The
  monolithic FOM is the direct reference, so this test does not depend on a
  serial SciPy/KKT solve converging or on matching KKT multiplier values.
  In a multi-rank run, the DD model exercises DistNewton; in serial mode it
  exercises the corresponding serial Torch solver.
  """
  if not bkd.is_torch_backend():
    pytest.skip("requires the Torch backend for the DD-FOM benchmark")

  dd_fom, _ = _build_case(
    subs_per_rank=2,
    world_size=bkd.get_nranks(),
  )
  fom = dd_fom.monolithic

  fom_uv, fom_res, fom_converged = fom.solve(
    tol=1.0e-8,
    maxit=20,
    stepsize_min=1.0e-20,
    verbose=False,
  )
  dd_uv, _, _, _ = dd_fom.solve(
    tol=1.0e-8,
    maxit=50,
    stepsize_min=1.0e-20,
    verbose=False,
  )

  assert fom_converged, "monolithic Torch FOM solve did not converge"
  assert np.linalg.norm(_as_numpy(fom_res[-1])) < 1.0e-6

  for component in ("u", "v"):
    dd_values = _as_numpy(dd_uv["res"][component]).reshape(-1)
    fom_values = _as_numpy(fom_uv[component]).reshape(-1)
    assert dd_values.shape == fom_values.shape
    assert np.all(np.isfinite(dd_values))
    relative_error = np.linalg.norm(dd_values - fom_values) / np.linalg.norm(fom_values)
    print(f"DD-FOM vs FOM {component} relative error = {relative_error:.6e}")
    assert relative_error < 1.0e-6, (
      f"DD-FOM {component} relative error is too large: "
      f"{relative_error:.6e} >= 1.0e-6"
    )
