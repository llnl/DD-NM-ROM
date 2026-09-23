import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pytest
import torch

pytestmark = pytest.mark.no_backend


def _load_activation_module():
  saved = {
    name: sys.modules.get(name)
    for name in ("dd_nm_rom", "dd_nm_rom.ops", "dd_nm_rom.backend", "dd_nm_rom.config")
  }
  try:
    pkg = types.ModuleType("dd_nm_rom")
    pkg.__path__ = []
    sys.modules["dd_nm_rom"] = pkg

    ops = types.ModuleType("dd_nm_rom.ops")
    ops.sp_diag = lambda x: torch.diag(x) if torch.is_tensor(x) else sp.spdiags(x, 0, x.size, x.size)
    sys.modules["dd_nm_rom.ops"] = ops

    bkd = types.ModuleType("dd_nm_rom.backend")
    bkd.is_torch_backend = lambda: True
    sys.modules["dd_nm_rom.backend"] = bkd

    cfg_path = Path(__file__).resolve().parents[1] / "dd_nm_rom/config.py"
    cfg_spec = importlib.util.spec_from_file_location("dd_nm_rom.config", cfg_path)
    cfg_module = importlib.util.module_from_spec(cfg_spec)
    sys.modules[cfg_spec.name] = cfg_module
    cfg_spec.loader.exec_module(cfg_module)

    path = Path(__file__).resolve().parents[1] / "dd_nm_rom/rom/nonlinear/autoencoder/nn_numpy/activation.py"
    spec = importlib.util.spec_from_file_location("_activation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
  finally:
    for name, value in saved.items():
      if value is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = value


activation = _load_activation_module()


def _torch_activation_input(values):
  """Create inputs on the device selected by the activation compiler.

  On accelerator hosts, activations use Torch Inductor by default.  Supplying
  a CPU tensor there would instead make Inductor compile a CPU kernel, which
  is not the path exercised by the GPU test configuration.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return torch.as_tensor(deepcopy(values), device=device)


def test_softplus():
  x = np.random.randn(16)

  act_np = activation.Softplus()
  fun_np = act_np(deepcopy(x), with_jac=False)
  fx, fdx = act_np(deepcopy(x), with_jac=True)

  act_torch = activation.get("softplus")
  x_torch = _torch_activation_input(x)
  fun_torch = act_torch(x_torch, with_jac=False)
  fx_t, fdx_t = act_torch(x_torch, with_jac=True)

  np.testing.assert_allclose(fun_torch.cpu().numpy(), fun_np)
  np.testing.assert_allclose(fx_t.cpu().numpy(), fx)
  np.testing.assert_allclose(fdx_t.cpu().numpy(), np.diag(fdx.todense()))


@pytest.mark.parametrize(
  "name, values, kwargs",
  [
    ("relu", np.array([-2.0, -0.5, 0.0, 0.25, 3.0]), {}),
    ("elu", np.array([-2.0, -0.5, 0.0, 0.25, 3.0]), {"alpha": 1.5}),
  ],
)
def test_compiled_torch_activations(name, values, kwargs):
  x_torch = _torch_activation_input(values)
  act_cls = {
    "relu": activation.ReLU,
    "elu": activation.ELU,
  }[name]
  act_np = act_cls(**kwargs)

  fun_np = act_np(deepcopy(values), with_jac=False)
  _, jac_np = act_np(deepcopy(values), with_jac=True)

  act_torch = activation.get(name, **kwargs)
  fun_torch = act_torch(x_torch, with_jac=False)
  _, jac_torch = act_torch(x_torch, with_jac=True)

  np.testing.assert_allclose(fun_torch.cpu().numpy(), fun_np)
  np.testing.assert_allclose(jac_torch.cpu().numpy(), np.diag(jac_np.todense()))


def test_numpy_activation_dispatch_survives_torch_activation_setup():
  # ``activation.get`` installs the Torch dispatcher on BaseAct globally.
  activation.get("softplus")
  act = activation.ReLU()
  values = np.array([-1.0, 0.0, 2.0])

  actual, actual_jac = act(values, with_jac=True)

  np.testing.assert_allclose(actual, np.maximum(values, 0.0))
  np.testing.assert_allclose(
    actual_jac.todense(),
    np.diag([0.0, 0.0, 1.0]),
  )


def test_mixed_torch_activation_uses_device_indices():
  values = np.array([-2.0, -0.5, 0.0, 0.25, 3.0, 1.5])
  masks = {
    "softplus": np.array([0, 2, 5]),
    "relu": np.array([1, 3, 4]),
  }
  x_torch = _torch_activation_input(values)

  expected = deepcopy(values)
  expected[masks["softplus"]] = activation.Softplus()._fun(
    values[masks["softplus"]]
  )
  expected[masks["relu"]] = activation.ReLU()._fun(values[masks["relu"]])
  expected_jac = np.zeros_like(values)
  expected_jac[masks["softplus"]] = activation.Softplus()._jac(
    values[masks["softplus"]]
  )
  expected_jac[masks["relu"]] = activation.ReLU()._jac(values[masks["relu"]])
  mixed_torch = activation.get("mixed", masks=masks)
  actual, actual_jac = mixed_torch(x_torch, with_jac=True)

  np.testing.assert_allclose(actual.cpu().numpy(), expected)
  np.testing.assert_allclose(
    actual_jac.cpu().numpy(),
    expected_jac,
  )
  assert len(mixed_torch._torch_masks) == 1
  assert set(mixed_torch._torch_masks[next(iter(mixed_torch._torch_masks))]) == \
    {"softplus", "relu"}
  assert mixed_torch._full_coverage(values.size) is True


def test_mixed_torch_activation_preserves_partial_masks():
  values = np.array([-2.0, -0.5, 0.0, 0.25])
  masks = {
    "softplus": np.array([0, 2]),
    "relu": np.array([1]),
  }
  x_torch = _torch_activation_input(values)
  mixed_torch = activation.get("mixed", masks=masks)

  actual, actual_jac = mixed_torch(x_torch, with_jac=True)
  expected = values.copy()
  expected[[0, 2]] = activation.Softplus()._fun(values[[0, 2]])
  expected[1] = activation.ReLU()._fun(values[1])
  expected_jac = np.zeros_like(values)
  expected_jac[[0, 2]] = activation.Softplus()._jac(values[[0, 2]])
  expected_jac[1] = activation.ReLU()._jac(values[1])

  np.testing.assert_allclose(actual.cpu().numpy(), expected)
  np.testing.assert_allclose(actual_jac.cpu().numpy(), expected_jac)
  assert mixed_torch._full_coverage(values.size) is False


def test_mixed_torch_activation_supports_negative_indices():
  values = np.array([-2.0, 0.25, 3.0, -0.5])
  masks = {
    "softplus": np.array([-1, 0]),
    "relu": np.array([1]),
  }
  mixed_torch = activation.get("mixed", masks=masks)
  actual, actual_jac = mixed_torch(
    _torch_activation_input(values), with_jac=True
  )

  expected = values.copy()
  expected[[-1, 0]] = activation.Softplus()._fun(values[[-1, 0]])
  expected[1] = activation.ReLU()._fun(values[1])
  expected_jac = np.zeros_like(values)
  expected_jac[[-1, 0]] = activation.Softplus()._jac(values[[-1, 0]])
  expected_jac[1] = activation.ReLU()._jac(values[1])

  np.testing.assert_allclose(actual.cpu().numpy(), expected)
  np.testing.assert_allclose(actual_jac.cpu().numpy(), expected_jac)


def test_mixed_masks_reject_noninteger_and_out_of_bounds_indices():
  with pytest.raises(TypeError, match="boolean or integer"):
    activation.get("mixed", masks={"relu": np.array([0.5])})

  mixed_torch = activation.get(
    "mixed", masks={"relu": np.array([3])}
  )
  with pytest.raises(IndexError, match="out of bounds"):
    mixed_torch(torch.zeros(3), with_jac=False)

  boolean_mask = activation.get(
    "mixed", masks={"relu": np.array([True, False])}
  )
  with pytest.raises(IndexError, match="boolean mask"):
    boolean_mask(torch.zeros(3), with_jac=False)


def test_mixed_jacobian_can_be_compiled(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  masks = {
    "softplus": np.array([0, 2, 5]),
    "relu": np.array([1, 3, 4]),
  }
  act = activation.get("mixed", masks=masks)
  assert activation.warmup(
    act, 8192, "cpu", torch.float32, role="decoder"
  ) is True

  x = torch.randn(8192)
  actual, actual_jac = act(x, with_jac=True)
  expected = act._fun_torch(x)
  expected_jac = act._jac_torch(x)
  assert len(calls) == 1
  assert torch.allclose(actual, expected)
  assert torch.allclose(actual_jac, expected_jac)
  assert activation.get_compile_stats()["skipped_mixed"] == 0


def test_mixed_compile_cache_includes_mask_layout(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  first = activation.get(
    "mixed",
    masks={"softplus": np.array([0, 1]), "relu": np.array([2, 3])},
  )
  second = activation.get(
    "mixed",
    masks={"softplus": np.array([0, 2]), "relu": np.array([1, 3])},
  )
  assert activation.warmup(first, 8192, "cpu", torch.float32, role="decoder")
  assert activation.warmup(second, 8192, "cpu", torch.float32, role="decoder")
  assert len(calls) == 2


def test_activation_allowlist_can_select_mixed(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_ACTIVATIONS", "mixed")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  softplus = activation.get("softplus")
  mixed = activation.get("mixed", masks={})
  assert activation.warmup(
    softplus, 8192, "cpu", torch.float32, role="decoder"
  ) is False
  assert activation.warmup(
    mixed, 8192, "cpu", torch.float32, role="decoder"
  ) is True

  assert len(calls) == 1
  stats = activation.get_compile_stats()
  assert stats["skipped_allowlist"] == 1
  assert stats["compiled_functions"] == 1


def test_mixed_top_k_policy_uses_frequency_and_mixed_cache_limit(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_ACTIVATIONS", "mixed")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_MIXED_POLICY", "top_k")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY", "2")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  first = activation.get("mixed", masks={"softplus": np.array([0, 1])})
  second = activation.get("mixed", masks={"softplus": np.array([0, 1])})
  other = activation.get("mixed", masks={"softplus": np.array([1, 2])})
  other_repeat = activation.get("mixed", masks={"softplus": np.array([1, 2])})
  assert activation.warmup(first, 8192, "cpu", torch.float32, role="decoder") is False
  assert activation.warmup(second, 8192, "cpu", torch.float32, role="decoder") is True
  assert activation.warmup(other, 8192, "cpu", torch.float32, role="decoder") is False
  assert activation.warmup(
    other_repeat, 8192, "cpu", torch.float32, role="decoder"
  ) is False

  assert len(calls) == 1
  stats = activation.get_compile_stats()
  assert stats["skipped_mixed_frequency"] == 2
  assert stats["skipped_cache_full"] == 1


def test_activation_compile_flag_controls_torch_compile(monkeypatch):
  calls = []

  def fake_compile(fun, **kwargs):
    calls.append((fun, kwargs))
    return fun

  monkeypatch.setattr(torch, "compile", fake_compile, raising=True)
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "0")
  activation.get("softplus")
  assert calls == []

  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  act = activation.get("softplus")
  activation.warmup(act, 4096, "cpu", torch.float32, role="decoder")
  assert len(calls) == 1
  assert all(
    call[1] == {"backend": "inductor", "dynamic": False}
    for call in calls
  )


def test_compiled_activations_are_cached_by_shape(monkeypatch):
  calls = []

  def fake_compile(fun, **kwargs):
    calls.append((fun, kwargs))
    return fun

  monkeypatch.setattr(torch, "compile", fake_compile, raising=True)
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()
  activation.warmup(activation.get("softplus"), 4096, "cpu", torch.float32, role="decoder")
  activation.warmup(activation.get("softplus"), 4096, "cpu", torch.float32, role="decoder")
  activation.warmup(activation.get("softplus"), 8192, "cpu", torch.float32, role="decoder")
  assert len(calls) == 2


def test_static_compilation_falls_back_when_cache_is_full(monkeypatch):
  calls = []

  def fake_compile(fun, **kwargs):
    calls.append((fun, kwargs))
    return fun

  monkeypatch.setattr(torch, "compile", fake_compile, raising=True)
  monkeypatch.setattr(activation, "_COMPILED_CACHE_MAXSIZE", 1)
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  act = activation.get("softplus")
  first = activation.warmup(act, 4096, "cpu", torch.float32, role="decoder")
  second = activation.warmup(
    act, 8192, "cpu", torch.float32, role="decoder"
  )

  assert first is True
  assert second is False
  assert len(calls) == 1
  assert act._compiled_fun_jac is None
  stats = activation.get_compile_stats()
  assert stats["skipped_cache_full"] == 1


def test_compile_stats_report_eligibility_and_cache_use(monkeypatch):
  def fake_compile(fun, **kwargs):
    return fun

  monkeypatch.setattr(torch, "compile", fake_compile, raising=True)
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_ENCODER", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  activation.warmup(
    activation.get("softplus"),
    4096,
    "cpu",
    torch.float32,
    role="decoder",
  )
  activation.warmup(
    activation.get("softplus"),
    4096,
    "cpu",
    torch.float32,
    role="decoder",
  )
  activation.warmup(
    activation.get("softplus"),
    1024,
    "cpu",
    torch.float32,
    role="encoder",
  )

  stats = activation.get_compile_stats()
  assert stats["warmup_calls"] == 3
  assert stats["eligible_calls"] == 2
  assert stats["cache_misses"] == 1
  assert stats["cache_hits"] == 1
  assert stats["skipped_small"] == 1
  assert stats["compiled_functions"] == 1
  assert stats["by_role_activation"]["decoder:Softplus"]["size_max"] == 4096
  assert stats["by_role_activation"]["encoder:Softplus"]["size_max"] == 1024


def test_compile_options_select_decoder_jacobian(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  encoder = activation.get("softplus")
  decoder = activation.get("softplus")
  assert activation.warmup(encoder, 8464, "cpu", torch.float32, role="encoder") is False
  assert activation.warmup(decoder, 21179, "cpu", torch.float32, role="decoder") is True

  assert len(calls) == 1
  assert encoder._compiled_fun_jac is None
  assert decoder._compiled_fun_jac is not None
  stats = activation.get_compile_stats()
  assert stats["skipped_role"] == 1
  assert stats["compiled_functions"] == 1


def test_compile_forward_and_jacobian_options(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_FORWARD", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE_JAC", "0")
  activation._COMPILED_CACHE.clear()
  activation.reset_compile_stats()

  act = activation.get("softplus")
  assert activation.warmup(act, 4096, "cpu", torch.float32, role="decoder") is True
  assert len(calls) == 1
  assert act._compiled_fun_jac is None
  assert act.fun is not act._fun_torch


def test_small_and_trivial_activations_stay_eager(monkeypatch):
  calls = []

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation._COMPILED_CACHE.clear()

  assert activation.warmup(activation.get("softplus"), 4095, "cpu", torch.float32) is False
  assert activation.warmup(activation.get("linear"), 8192, "cpu", torch.float32) is False
  assert activation.warmup(activation.get("mixed", masks={}), 8192, "cpu", torch.float32) is True
  assert len(calls) == 1


def test_cpu_uses_eager_activations_by_default(monkeypatch):
  calls = []

  monkeypatch.delenv("DDNMROM_ACT_COMPILE", raising=False)
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: calls.append((fun, kwargs)) or fun,
    raising=True,
  )

  activation.get("softplus")
  assert calls == []


def test_compile_failure_is_not_silently_replaced_with_eager(monkeypatch):
  def fake_compile(fun):
    def compiled(*args, **kwargs):
      raise RuntimeError("compile failed")
    return compiled

  monkeypatch.setattr(
    torch,
    "compile",
    lambda fun, **kwargs: fake_compile(fun),
    raising=True
  )
  compiled = activation._maybe_compile(lambda x: x + 1)
  x = torch.tensor([1.0, 2.0])
  with pytest.raises(RuntimeError, match="compile failed"):
    compiled(x)
