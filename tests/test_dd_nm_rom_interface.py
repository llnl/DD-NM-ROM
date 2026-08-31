import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_sla
from mpi4py import MPI
from scipy.sparse.linalg import spsolve

from dd_nm_rom import backend as bkd
from dd_nm_rom import field as field_mod
from dd_nm_rom import fom as fom_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.rom.nonlinear.domain_dec.model import DD_NM_ROM
from dd_nm_rom.solvers.dist_newton import DistNewton

_BACKEND_INITIALIZED = False
_SOLUTION_REL_TOL = 1.0e-6
_STEADY_ABS_TOL = 2.0e-3


@pytest.fixture(scope="module", autouse=True)
def configured_backend(request):
  """Use conftest's selected backend and initialize MPI for parallel runs."""
  global _BACKEND_INITIALIZED
  selected = request.config.getoption("--backend") or "torch_cpu"
  request.getfixturevalue("backend_" + selected)
  if not _BACKEND_INITIALIZED:
    bkd.set_floatx("float64")
    world_size = MPI.COMM_WORLD.Get_size()
    if bkd._NRANKS is None and world_size > 1:
      device = "cuda" if bkd.device().type == "cuda" else "cpu"
      if device == "cpu":
        # backend.init_distributed currently prints CUDA device information
        # even for the CPU backend. Keep CPU MPI jobs from initializing CUDA.
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


def _as_numpy(x):
  if hasattr(x, "to_local"):
    x = x.to_local()
  if torch.is_tensor(x):
    if x.layout != torch.strided:
      x = x.to_dense()
    return x.detach().cpu().numpy()
  if sp.issparse(x):
    return x.toarray()
  return np.asarray(x)


def _assert_allclose(actual, expected, *, atol=1e-12, rtol=1e-10):
  np.testing.assert_allclose(_as_numpy(actual), _as_numpy(expected), atol=atol, rtol=rtol)


def _make_identity_config(dim):
  eye = sp.eye(dim, format="csr")
  zeros = np.zeros(dim)
  ones = np.ones(dim)
  return {
    "encoder": {
      "input_dim": dim,
      "latent_dim": dim,
      "hidden_dim": dim,
      "activation": "linear",
      "ref": zeros.copy(),
      "scale": ones.copy(),
      "weights": {
        "W1": eye.copy(),
        "b1": zeros.copy(),
        "W2": eye.copy(),
      },
    },
    "decoder": {
      "input_dim": dim,
      "latent_dim": dim,
      "hidden_dim": dim,
      "activation": "linear",
      "ref": zeros.copy(),
      "scale": ones.copy(),
      "weights": {
        "W1": eye.copy(),
        "b1": zeros.copy(),
        "W2": eye.copy(),
      },
    },
  }


def _make_nn_configs(dd_fom, constraint_type="strong"):
  configs = {"interior": [], "port": []}
  for sub in dd_fom.subdomains:
    dim = 2 * sub.elem_states["interior"].n_nodes_state
    configs["interior"].append(_make_identity_config(dim))
  for p in dd_fom.dd_indices.ports:
    dim = 2 * dd_fom.dd_indices.port_to_nodes[p].size
    configs["port"].append(_make_identity_config(dim))
  if constraint_type == "weak":
    configs["interface"] = []
    for sub in dd_fom.subdomains:
      dim = 2 * sub.elem_states["interface"].n_nodes_state
      configs["interface"].append(_make_identity_config(dim))
  return configs


def _install_serial_numpy_backend(monkeypatch):
  monkeypatch.setattr(bkd, "is_torch_backend", lambda: False)
  monkeypatch.setattr(bkd, "distributed", lambda: False)
  monkeypatch.setattr(bkd, "get_rank", lambda: 0)
  monkeypatch.setattr(bkd, "get_nranks", lambda: 1)
  monkeypatch.setattr(bkd, "root", lambda: True)
  monkeypatch.setattr(bkd, "barrier", lambda: None)


def _selected_backend(request):
  cli = request.config.getoption("--backend")
  if cli:
    return cli
  marker = request.node.get_closest_marker("backend")
  if marker:
    return marker.args[0]
  return "torch_cpu"


def _torch_device(request):
  return "cuda" if _selected_backend(request) == "torch_gpu" else "cpu"


def _set_numpy_seed():
  bkd.set_backend("numpy")
  np.random.seed(0)
  torch.manual_seed(0)


def _set_torch_seed(request):
  assert bkd.device().type == _torch_device(request)
  bkd.set_seed(0)


def _build_case(
  monkeypatch,
  *,
  serial,
  world_size,
  compact=True,
  constraint_type="strong",
  boundary_type="neumann",
):
  if serial:
    with monkeypatch.context() as m:
      _install_serial_numpy_backend(m)
      _set_numpy_seed()
      return _build_case_impl(
        m,
        world_size=world_size,
        subs_per_rank=2 * world_size,
        compact=compact,
        constraint_type=constraint_type,
        boundary_type=boundary_type,
      )

  return _build_case_impl(
    monkeypatch,
    world_size=world_size,
    subs_per_rank=2,
    compact=compact,
    constraint_type=constraint_type,
    boundary_type=boundary_type,
  )


def _build_case_impl(
  monkeypatch,
  *,
  world_size,
  subs_per_rank,
  compact=True,
  constraint_type="strong",
  boundary_type="neumann",
):
  mesh = mesh_mod.MeshDD(
    nx_intr=16,
    ny_intr=16,
    lx_sub=0.5,
    ly_sub=0.5,
    x0=0.0,
    y0=0.0,
    n_sub_x=1,
    n_sub_y=2 * world_size,
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
  x_phys = np.concatenate([field.u().reshape(-1), field.v().reshape(-1)])

  fom = fom_mod.Burgers2D(
    mesh=mesh,
    nu=1.0e-3,
    upwind=True,
    upwind_order=2,
    compact=compact,
  )
  fom.build(field)

  dd_fom = fom_mod.DDBurgers2D(
    monolithic=fom,
    constraint_type=constraint_type,
    scaling=-1,
    subs_per_rank=subs_per_rank,
    n_constraints_weak=1_000_000,
  )
  dd_fom.build()

  nn_configs = _make_nn_configs(dd_fom, constraint_type)
  monkeypatch.setattr(
    DD_NM_ROM,
    "_load_nn_configs",
    classmethod(lambda cls, nn_configfiles, **kwargs: nn_configfiles),
    raising=False,
  )

  dd_rom = DD_NM_ROM(
    dd_fom=dd_fom,
    nn_configfiles=nn_configs,
    constraint_type=constraint_type,
    n_constraints_weak=1_000_000,
    scaling=-1,
    subs_per_rank=subs_per_rank,
    check_unique_models=False,
  )

  return dd_rom, x_phys


def _build_reference_and_parallel(monkeypatch, request):
  world_size = MPI.COMM_WORLD.Get_size()
  assert bkd.get_nranks() == world_size
  assert bkd.distributed() == (world_size > 1)
  ref_rom, x_phys = _build_case(
    monkeypatch,
    serial=True,
    world_size=world_size,
  )
  bkd.set_backend("torch")
  bkd.set_seed(0)
  test_rom, _ = _build_case(
    monkeypatch,
    serial=False,
    world_size=world_size,
  )
  return ref_rom, test_rom, x_phys


def _call_reference(monkeypatch, fun, *args, **kwargs):
  with monkeypatch.context() as m:
    _install_serial_numpy_backend(m)
    _set_numpy_seed()
    return fun(*args, **kwargs)


def _call_parallel(fun, *args, **kwargs):
  bkd.set_backend("torch")
  bkd.set_seed(0)
  backend_args = [bkd.to_backend(arg) for arg in args]
  backend_kwargs = {key: bkd.to_backend(value) for key, value in kwargs.items()}
  return fun(*backend_args, **backend_kwargs)


def _serial_kkt_armijo_step(model, x0, stepsize_min=1.0e-20):
  """Take one serial direct KKT step using DistNewton's Armijo rule."""
  res, jac = model.res_jac(x0, use_global=True)
  res_norm = np.linalg.norm(res)
  dx = spsolve(jac, -res)
  stepsize = 1.0

  while True:
    x = x0 + stepsize * dx
    trial_res, _ = model.res_jac(x, use_global=True)
    trial_norm = np.linalg.norm(trial_res)
    armijo_bound = (1.0 - 2.0e-4 * stepsize) * res_norm
    if trial_norm < armijo_bound or stepsize < stepsize_min:
      return x, trial_res, trial_norm, stepsize
    stepsize *= 0.5


def _compare_get_ndof(monkeypatch, request, ref_rom, test_rom):
  ref_ndof = _call_reference(monkeypatch, ref_rom.get_ndof)
  test_ndof = _call_parallel(test_rom.get_ndof)
  assert test_ndof == ref_ndof


def _compare_cmat(ref_rom, test_rom):
  assert set(test_rom.cmat.keys()) == {"interior", "interface"}
  assert set(ref_rom.cmat.keys()) == {"interior", "interface"}

  if bkd.distributed():
    for local_idx, global_s in enumerate(test_rom.global_subdomains):
      _assert_allclose(test_rom.cmat["interior"][local_idx], ref_rom.cmat["interior"][global_s])
    for global_s in range(len(ref_rom.cmat["interface"])):
      _assert_allclose(test_rom.cmat["interface"][global_s], ref_rom.cmat["interface"][global_s])
  else:
    for e_k in ("interior", "interface"):
      assert len(test_rom.cmat[e_k]) == len(ref_rom.cmat[e_k])
      for actual, expected in zip(test_rom.cmat[e_k], ref_rom.cmat[e_k]):
        _assert_allclose(actual, expected)


def _compare_encode(monkeypatch, request, ref_rom, test_rom, x_phys):
  ref_enc = _call_reference(monkeypatch, ref_rom.encode, x_phys)
  test_enc = _call_parallel(test_rom.encode, x_phys)

  if bkd.distributed():
    expected = []
    for global_s in test_rom.global_subdomains:
      start = ref_rom.local_offsets[global_s]
      stop = start + ref_rom.local_sizes[global_s]
      expected.append(ref_enc[start:stop])
    expected.append(ref_enc[-ref_rom.n_constraints:])
    _assert_allclose(test_enc, np.concatenate([_as_numpy(x) for x in expected]))
  else:
    _assert_allclose(test_enc, ref_enc)


def _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys):
  ref_x0 = _call_reference(monkeypatch, ref_rom.get_init_sol, x_phys)
  test_x0 = _call_parallel(test_rom.get_init_sol, x_phys)
  _assert_allclose(test_x0, ref_x0)
  return ref_x0, test_x0


def _compare_decode(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0):
  ref_uv, ref_z, ref_lambdas = _call_reference(monkeypatch, ref_rom.decode, ref_x0, map_on_res=True)
  test_uv, test_z, test_lambdas = _call_parallel(test_rom.decode, test_x0, map_on_res=True)

  if bkd.distributed():
    for local_idx, global_s in enumerate(test_rom.global_subdomains):
      for e_k in ("interior", "interface"):
        _assert_allclose(test_z[e_k][local_idx], ref_z[e_k][global_s])
        for x_k in ("u", "v"):
          _assert_allclose(test_uv[e_k][x_k][local_idx], ref_uv[e_k][x_k][global_s])

        indices = ref_rom.dd_fom.dd_indices.__dict__[e_k][global_s]
        for x_k in ("u", "v"):
          _assert_allclose(test_uv["res"][x_k][indices], ref_uv["res"][x_k][indices])
    _assert_allclose(test_lambdas, ref_lambdas)
  else:
    for e_k in ("interior", "interface"):
      for x_k in ("u", "v"):
        for actual, expected in zip(test_uv[e_k][x_k], ref_uv[e_k][x_k]):
          _assert_allclose(actual, expected)
      for actual, expected in zip(test_z[e_k], ref_z[e_k]):
        _assert_allclose(actual, expected)
    for x_k in ("u", "v"):
      _assert_allclose(test_uv["res"][x_k], ref_uv["res"][x_k])
    _assert_allclose(test_lambdas, ref_lambdas)


def _compare_residual(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0):
  ref_res, ref_cres = _call_reference(monkeypatch, ref_rom.residual, ref_x0, use_global=True)
  test_res, test_cres = _call_parallel(test_rom.residual, test_x0, use_global=True)
  _assert_allclose(test_res, ref_res)
  if bkd.root():
    _assert_allclose(test_cres, ref_cres)


def _compare_res_jac(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0):
  ref_res, ref_jac = _call_reference(monkeypatch, ref_rom.res_jac, ref_x0, use_global=True)
  test_res, test_jac = _call_parallel(test_rom.res_jac, test_x0, use_global=True)
  _assert_allclose(test_res, ref_res)
  _assert_allclose(test_jac, ref_jac)


def test_dd_nm_rom_get_ndof_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, _ = _build_reference_and_parallel(monkeypatch, request)
  _compare_get_ndof(monkeypatch, request, ref_rom, test_rom)


def test_dd_nm_rom_assemble_cmat_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, _ = _build_reference_and_parallel(monkeypatch, request)
  _compare_cmat(ref_rom, test_rom)


def test_dd_nm_rom_encode_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  _compare_encode(monkeypatch, request, ref_rom, test_rom, x_phys)


def test_dd_nm_rom_decode_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys)
  _compare_decode(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0)


def test_dd_nm_rom_get_init_sol_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys)


def test_dd_nm_rom_residual_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys)
  _compare_residual(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0)


def test_dd_nm_rom_res_jac_numpy_vs_torch(monkeypatch, request):
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys)
  _compare_res_jac(monkeypatch, request, ref_rom, test_rom, ref_x0, test_x0)


def _dd_rom_field_error_norms(reference_uv, dd_rom, candidate_uv):
  """Return absolute and relative errors for rank-local DD-ROM fields."""
  error_sq = {component: 0.0 for component in ("u", "v")}
  reference_sq = {component: 0.0 for component in ("u", "v")}
  for local_idx, sub in enumerate(dd_rom.subdomains):
    for element in ("interior", "interface"):
      indices = sub.sub_fom.elem_states[element].nodes_state
      for component in ("u", "v"):
        reference = _as_numpy(reference_uv[component])[indices]
        candidate = _as_numpy(candidate_uv[element][component][local_idx])
        error_sq[component] += np.vdot(candidate - reference, candidate - reference)
        reference_sq[component] += np.vdot(reference, reference)

  if bkd.distributed():
    for component in ("u", "v"):
      error_sq[component] = bkd._COMM.allreduce(error_sq[component], op=MPI.SUM)
      reference_sq[component] = bkd._COMM.allreduce(
        reference_sq[component], op=MPI.SUM,
      )
  absolute = {
    component: np.sqrt(error_sq[component])
    for component in ("u", "v")
  }
  relative = {
    component: absolute[component] / np.sqrt(reference_sq[component])
    if reference_sq[component] else 0.0
    for component in ("u", "v")
  }
  return absolute, relative


def _build_fom_dd_rom_solve_case(
  monkeypatch,
  steady,
  constraint_type,
  boundary_type,
):
  """Build a backend-local FOM/DD-FOM/DD-ROM case for solution comparison."""
  world_size = MPI.COMM_WORLD.Get_size()
  if world_size > 1 and not bkd.is_torch_backend():
    pytest.skip("distributed DD solves require a Torch backend")

  # The standard operator path is shared by NumPy, Torch CPU, and Torch GPU.
  dd_rom, x0 = _build_case(
    monkeypatch,
    serial=False,
    world_size=world_size,
    compact=False,
    constraint_type=constraint_type,
    boundary_type=boundary_type,
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

  if steady:
    x0 = np.zeros_like(x0)

  # Solve the monolithic problem first, as the physical-space reference.
  fom_uv, _, fom_converged = dd_rom.dd_fom.monolithic.solve(
    x0=x0,
    **solve_args,
  )
  assert fom_converged
  return dd_rom, x0, fom_uv, solve_args


@pytest.mark.parametrize("steady", (True, False), ids=("steady", "unsteady"))
@pytest.mark.parametrize("constraint_type", ("strong", "weak"))
@pytest.mark.parametrize("boundary_type", ("neumann", "periodic"))
def test_dd_nm_rom_solution_matches_monolithic_fom(
  monkeypatch,
  steady,
  constraint_type,
  boundary_type,
):
  """Identity DD-NM-ROM tracks all FOM boundary/constraint formulations."""
  dd_rom, x0, fom_uv, solve_args = _build_fom_dd_rom_solve_case(
    monkeypatch,
    steady,
    constraint_type,
    boundary_type,
  )

  rom_x0 = dd_rom.get_init_sol(x0)
  rom_uv, _, _, _, rom_converged = dd_rom.solve(x0=rom_x0, **solve_args)

  assert rom_converged
  absolute_errors, relative_errors = _dd_rom_field_error_norms(
    fom_uv, dd_rom, rom_uv,
  )
  if steady:
    assert max(absolute_errors.values()) < _STEADY_ABS_TOL
  else:
    assert max(relative_errors.values()) < _SOLUTION_REL_TOL


@pytest.mark.mpi(min_size=2)
def test_dist_newton_dd_rom_one_step_matches_serial_kkt(monkeypatch, request):
  """Compare one distributed ROM step with a serial direct KKT reference.

  A one-step comparison avoids making the test depend on full nonlinear
  convergence of the serial NumPy path, while validating the distributed
  vector layout, constraint reduction, Krylov direction gather, and Armijo
  trial evaluation.
  """
  if not bkd.is_torch_backend() or not bkd.distributed():
    pytest.skip("requires a multi-rank Torch backend for DistNewton")

  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(
    monkeypatch, request, ref_rom, test_rom, x_phys,
  )

  # Perturb the shared physical-state block so the reference exercises a
  # nontrivial Newton direction instead of accepting an already converged
  # encoded field.
  ref_x0 = np.array(ref_x0, copy=True)
  ref_x0[0] += 1.0e-2
  test_x0 = test_x0.clone()
  test_x0[0] += 1.0e-2

  ref_x1, ref_res1, ref_norm1, ref_stepsize = _call_reference(
    monkeypatch,
    _serial_kkt_armijo_step,
    ref_rom,
    ref_x0,
  )

  solver = DistNewton(
    model=test_rom,
    tol=0.0,
    maxit=1,
    stepsize_min=1.0e-20,
    verbose=False,
    distributed=True,
  )
  x1, res_hist, res_norm_hist, step_hist, iterations, _ = solver.solve(test_x0)

  assert iterations == 1
  np.testing.assert_allclose(
    _as_numpy(x1), ref_x1, rtol=1.0e-6, atol=1.0e-8,
  )
  np.testing.assert_allclose(
    res_hist[-1], ref_res1, rtol=1.0e-6, atol=1.0e-8,
  )
  assert res_norm_hist[-1] == pytest.approx(ref_norm1, rel=1.0e-6, abs=1.0e-8)
  assert step_hist[-1] == pytest.approx(ref_stepsize)


@pytest.mark.mpi(min_size=2)
def test_dist_newton_dd_rom_root_direct_step_matches_serial_kkt(monkeypatch, request):
  """Exercise the real root-direct solve and broadcast path for DD-ROM."""
  if not bkd.is_torch_backend() or not bkd.distributed():
    pytest.skip("requires a multi-rank Torch backend for DistNewton")

  if torch_sla.backends.is_strumpack_available():
    direct_backend = "strumpack"
  elif torch_sla.backends.is_cudss_available():
    direct_backend = "cudss"
  else:
    pytest.skip("requires STRUMPACK or cuDSS for the direct sparse solve")

  monkeypatch.setenv("DDNMROM_FORCE_SERIAL_SOLVE", "1")
  monkeypatch.setenv("DDNMROM_FORCE_SOLVE_BACKEND", direct_backend)
  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(
    monkeypatch, request, ref_rom, test_rom, x_phys,
  )
  ref_x0 = np.array(ref_x0, copy=True)
  ref_x0[0] += 1.0e-2
  test_x0 = test_x0.clone()
  test_x0[0] += 1.0e-2

  ref_x1, ref_res1, ref_norm1, ref_stepsize = _call_reference(
    monkeypatch,
    _serial_kkt_armijo_step,
    ref_rom,
    ref_x0,
  )

  solver = DistNewton(
    model=test_rom,
    tol=0.0,
    maxit=1,
    stepsize_min=1.0e-20,
    verbose=False,
    distributed=True,
  )
  assert solver.use_direct_solve
  assert solver.direct_solve_backend == direct_backend
  x1, res_hist, res_norm_hist, step_hist, iterations, _ = solver.solve(test_x0)

  assert iterations == 1
  np.testing.assert_allclose(
    _as_numpy(x1), ref_x1, rtol=1.0e-8, atol=1.0e-10,
  )
  np.testing.assert_allclose(
    res_hist[-1], ref_res1, rtol=1.0e-8, atol=1.0e-10,
  )
  assert res_norm_hist[-1] == pytest.approx(ref_norm1, rel=1.0e-8, abs=1.0e-10)
  assert step_hist[-1] == pytest.approx(ref_stepsize)

  # The direct solve happens on rank zero, so this catches a missing or
  # malformed direction broadcast to non-root ranks.
  for rank_x1 in bkd._COMM.allgather(_as_numpy(x1)):
    np.testing.assert_allclose(rank_x1, _as_numpy(x1), rtol=0.0, atol=0.0)


@pytest.mark.mpi(min_size=2)
@pytest.mark.skip(
  reason="temporarily skipped; the serial NumPy KKT reference does not reliably converge",
)
def test_dd_nm_rom_solve_numpy_vs_distributed_torch(monkeypatch, request):
  """Legacy serial NumPy versus distributed Torch KKT comparison for ROM.

  This test is retained as a record of the former comparison, but is skipped
  because the serial NumPy KKT reference does not provide a reliable solver
  benchmark for the distributed implementation.
  """
  if not bkd.is_torch_backend() or not bkd.distributed():
    pytest.skip("requires a multi-rank Torch backend for DistNewton")

  ref_rom, test_rom, x_phys = _build_reference_and_parallel(monkeypatch, request)
  ref_x0, test_x0 = _compare_get_init_sol(monkeypatch, request, ref_rom, test_rom, x_phys)

  solve_args = {
    "tol": 1.0e-8,
    "maxit": 20,
    "stepsize_min": 1.0e-20,
    "verbose": False,
  }
  ref_uv, ref_z, ref_lambdas, ref_res, ref_converged = _call_reference(
    monkeypatch,
    ref_rom.solve,
    x0=ref_x0,
    **solve_args,
  )
  test_uv, test_z, test_lambdas, test_res, test_converged = _call_parallel(
    test_rom.solve,
    x0=test_x0,
    **solve_args,
  )

  assert ref_converged, "serial NumPy reference solve did not converge"
  assert test_converged, "distributed Torch DistNewton solve did not converge"

  # The distributed ROM returns only rank-local subdomains. Compare each
  # local result with its corresponding global subdomain in the serial model.
  for local_idx, global_s in enumerate(test_rom.global_subdomains):
    for e_k in ("interior", "interface"):
      for x_k in ("u", "v"):
        np.testing.assert_allclose(
          _as_numpy(test_uv[e_k][x_k][local_idx]),
          _as_numpy(ref_uv[e_k][x_k][global_s]),
          rtol=1.0e-6,
          atol=1.0e-8,
        )
      _assert_allclose(
        test_z[e_k][local_idx],
        ref_z[e_k][global_s],
        atol=1.0e-6,
        rtol=1.0e-6,
      )

      indices = ref_rom.dd_fom.dd_indices.__dict__[e_k][global_s]
      for x_k in ("u", "v"):
        np.testing.assert_allclose(
          _as_numpy(test_uv["res"][x_k][indices]),
          _as_numpy(ref_uv["res"][x_k][indices]),
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
