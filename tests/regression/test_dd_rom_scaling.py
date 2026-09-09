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
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))

from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print

from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import postproc
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import rom as rom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.rom.utils import pod as pod_mod
from dd_nm_rom.elements import mesh as mesh_mod


def _profile_max_solver_iterations():
    """Return the Newton budget used by the distributed scaling benchmark."""
    value = os.environ.get("DDNMROM_PROFILE_MAXIT", "10")
    try:
        maxit = int(value)
    except ValueError as exc:
        raise ValueError(
            "DDNMROM_PROFILE_MAXIT must be a positive integer, "
            f"got {value!r}"
        ) from exc
    if maxit < 1:
        raise ValueError(
            "DDNMROM_PROFILE_MAXIT must be a positive integer, "
            f"got {maxit}"
        )
    return maxit


def setup_module(module):
    #from dd_nm_rom import env
    #env.set(**inputs["env"])

    #bkd.set(backend="numpy", device="cpu", seed=0)
    bkd.set(backend="torch", device="cuda", seed=0)


def teardown_module(module):
    if bkd.is_torch_backend():
        bkd.finalize_distributed()


def test_ddnmrom_simple():
    rank = bkd.get_rank()
    max_solver_iterations = _profile_max_solver_iterations()
    parallel_print(
        "Distributed ROM profile Newton iteration budget: "
        f"{max_solver_iterations}"
    )

    inputs_file = "./tests/inputs/test_dd_nmrom_scale.json"
    with open(inputs_file) as file:
        inputs = json.load(file)

    profiler = Profiler()
    profile_out = "profiles/test_dd_rom_scale_new_profile_gpu"
    if bkd.distributed():
        profile_out += "_{}_{}".format(bkd.get_nranks(), bkd.get_rank())

    print("\nInitialization ...")
    # Mesh
    mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
    mesh.build()
    X, Y = mesh.grid

    fom_subs_per_rank = (mesh.n_sub) // bkd.get_nranks()
    print(" SUBDOMAINS PER RANK: {}".format(fom_subs_per_rank))

    # Field
    field = utils.get_class(modules=[field_mod],
                            name=inputs["field"]["name"]
    )(mesh=mesh, **inputs["field"]["kwargs"])
    field.set_params(mu=field.sample_design_space())
    # FOM
    fom = utils.get_class(modules=[fom_mod],
                          name="Burgers2D"
    )(mesh=mesh, **inputs["fom"]["kwargs"])
    fom.build(field)
    # DD-FOM
    dd_fom = utils.get_class(modules=[fom_mod],
                             name="DDBurgers2D"
    )(monolithic=fom, subs_per_rank=fom_subs_per_rank, **inputs["dd_fom"]["kwargs"])
    dd_fom.build()

    test_num_timesteps = 1
    #return

    # Data loading
    # =====================================
    print("\nLoading test cases ...")
    #test_cases = utils.load_case_parallel(**inputs["data_load"])
    #parallel_print("rank {} test_cases1 = {}".format(rank, test_cases))
    #test_cases = [case for case in test_cases if case is not None]

    # NN models loading
    # =====================================
    print("\nLoading NN configuration files ...")
    path_to_nets = {}
    for (element, tag) in inputs["paths"]["nets_tag"].items():
        suffix = f"/{tag}/merged/{element}/"
        path_to_nets[element] = inputs["paths"]["nets_dir"] + suffix
    # for (element, tag) in inputs["paths"]["nets_tag"].items():
    #     suffix = f"/{tag}/multi/{element}/"
    #     path_to_nets[element] = inputs["paths"]["nets_dir"] + suffix
    print("\n  PATH TO NETS = {}".format(path_to_nets))
    nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles_new(
        mesh=mesh, dd_fom=dd_fom, path_to_nets=path_to_nets
    )

    # Test case set up
    # ---------------

    #print(teval,ieval)
    # DD-FOM
    # ---------------
    # > Building
    U0 = field.u()
    V0 = field.v()

    x0 = np.concatenate([U0.reshape(-1), V0.reshape(-1)])
    #field.set_params(icase["mu"])
    #fom.build(field)
    #dd_fom.build()

    # > Solution

    # DD-NM-ROM
    # ---------------
    # > Configuration
    # > Building
    dd_rom = rom_mod.DD_NM_ROM(dd_fom=dd_fom,
                               nn_configfiles=nn_configfiles,
                               constraint_type="strong",
                               #constraint_type="weak",
                               n_constraints_weak=-1,
                               scaling=-1,
                               subs_per_rank=fom_subs_per_rank,
                               check_unique_models=True
    )

    profiler.start()
    # >> Solving
    uv_rom, z, lambdas, res, iconverged = dd_rom.solve(
        x0=dd_rom.get_init_sol(x=x0),
        runtime=0.0,
        use_guess=False,
        tol=1.0e-8,
        nt=test_num_timesteps,
        maxit=max_solver_iterations,
        verbose=True
    )

    profiler.stop()
    with open(profile_out + ".out", 'w') as file:
        profiler.print(file)

    #print(" UV ROM = {}".format(uv_rom))
    #print(" z = {}".format(z))
    #print(" lambdas = {}".format(lambdas))
    #print(" res = {}".format(res))
    #print(" converged = {}".format(iconverged))

    # >> Statistics - Single case
    iruntime = dd_rom.runtime
    # ierror = dd_rom.compute_error_new(
    #     uv_fom, uv_rom, scaling=True, relative=True, axis=0
    # )
    # ispeedup = {k: tk/iruntime[k] for (k, tk) in runtime_fom.items()}

    print(" IRUNTIME = {}".format(iruntime))
    #print(" IERROR = {}".format(ierror))
    #print(" ISPEEDUP = {}".format(ispeedup))

if __name__ == "__main__":
    setup_module(None)
    test_ddnmrom_simple()
    teardown_module(None)
