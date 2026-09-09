import pytest

import numpy as np
import os
import json
import dill as pickle
from pathlib import Path
import torch
import torch.distributed as dist
import torch.testing

from pyinstrument import Profiler

if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))

from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print

from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import rom as rom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.rom.utils import pod as pod_mod
from dd_nm_rom.elements import mesh as mesh_mod


@pytest.fixture(scope="module", autouse=True)
def configured_backend(request):
    """Use the selected conftest backend rather than forcing Torch CPU."""
    selected = request.config.getoption("--backend") or "torch_cpu"
    request.getfixturevalue("backend_" + selected)
    bkd.set_seed(0)
    yield

def test_dd_fom_indices():
    n_sub_x = 2
    n_sub_y = 2
    mesh = mesh_mod.MeshDD(nx_intr=16,
                           ny_intr=16,
                           lx_sub=0.5,
                           ly_sub=0.5,
                           x0=0.0,
                           y0=0.0,
                           n_sub_x=n_sub_x,
                           n_sub_y=n_sub_y,
                           with_bounds=True)
    mesh.build()

    field = field_mod.SinPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="periodic")
    field.set_params(mu=field.sample_design_space())

    fom = fom_mod.Burgers2D(nu=1e-3, mesh=mesh)
    fom.build(field)

    fom_subs_per_rank = (n_sub_x * n_sub_y) // bkd.get_nranks()
    dd_fom = fom_mod.DDBurgers2D(fom, constraint_type='strong', subs_per_rank=fom_subs_per_rank)
    dd_fom.build()

    # each rank has a global map for all other ranks
    assert np.concatenate(dd_fom.global_submap).size == mesh.n_sub

    indices = dd_fom.dd_indices



if __name__ == "__main__":
    pytest.main()
