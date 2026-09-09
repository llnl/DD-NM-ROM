import pytest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

"""
Defines testing fixtures for setting different testing backends:
- (numpy, torch-cpu, torch-gpu)

Default fixture used for all tests is torch-cpu
"""

_BACKENDS = ["numpy", "torch_cpu", "torch_gpu"]
_DEFAULT_BACKEND = "torch_cpu"


def pytest_addoption(parser):
  parser.addoption("--backend",
                   choices=_BACKENDS,
                   help="DD-NM-ROM backend setup")


@pytest.fixture(scope="session")
def backend_numpy():
  # Set numpy as backend
  from dd_nm_rom import backend as bkd
  bkd.set_backend("numpy")
  bkd.set_device("cpu")
  bkd.set_floatx("float64")

  yield

  bkd.finalize_distributed()


@pytest.fixture(scope="session")
def backend_torch_cpu():
  # Set numpy as backend
  from dd_nm_rom import backend as bkd
  bkd.set_backend("torch")
  bkd.set_device("cpu")
  bkd.set_floatx("float64")

  yield

  bkd.finalize_distributed()


@pytest.fixture(scope="session")
def backend_torch_gpu():
  # Set numpy as backend
  from dd_nm_rom import backend as bkd
  bkd.set_backend("torch")
  bkd.set_device("cuda")
  bkd.set_floatx("float64")

  yield

  bkd.finalize_distributed()


# global default environment for all tests
@pytest.fixture(autouse=True)
def backend_default(request):
  if request.node.get_closest_marker("no_backend"):
    return

  cli = request.config.getoption("--backend")
  marker = request.node.get_closest_marker("backend")

  if cli:
    fixture_to_use = "backend_" + cli
  elif marker:
    fixture_to_use = marker.args[0]
  else:
    fixture_to_use = "backend_" + _DEFAULT_BACKEND

  print("*** USING FIXTURE '{}' ({})".format(fixture_to_use, marker))
  request.getfixturevalue(fixture_to_use)
