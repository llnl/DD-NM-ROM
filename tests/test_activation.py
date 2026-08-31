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


def test_activation_compile_flag_controls_torch_compile(monkeypatch):
  calls = []

  def fake_compile(fun, **kwargs):
    calls.append((fun, kwargs))
    return fun

  monkeypatch.setattr(torch, "compile", fake_compile, raising=True)
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "0")
  activation.get("softplus")
  assert calls == []

  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "1")
  activation.get("softplus")
  assert len(calls) == 1
  assert calls[0][1] == {"backend": "inductor", "dynamic": True}


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
