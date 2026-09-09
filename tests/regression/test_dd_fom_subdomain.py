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
from dd_nm_rom import postproc
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import rom as rom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.rom.utils import pod as pod_mod
from dd_nm_rom.elements import mesh as mesh_mod

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

    #inputs_file = Path(Path.cwd(), "./inputs/test_dd_nmrom_simple.json").resolve()
    inputs_file = "./tests/inputs/test_dd_nmrom_simple.json"
    with open(inputs_file) as file:
        inputs = json.load(file)

    profiler = Profiler()
    profile_out = "./profiles/test_dd_fom_rom_subdomain_profile_gpu"
    if bkd.distributed():
        profile_out += "_{}_{}".format(bkd.get_nranks(), bkd.get_rank())

    print("\nInitialization ...")
    # Mesh
    mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
    mesh.build()
    X, Y = mesh.grid
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
    )(monolithic=fom, **inputs["dd_fom"]["kwargs"])
    dd_fom.build()


    test_num_timesteps = 1
    #return

    # Data loading
    # =====================================
    print("\nLoading test cases ...")
    test_cases = utils.load_case_parallel(**inputs["data_load"])
    #parallel_print("rank {} test_cases1 = {}".format(rank, test_cases))
    test_cases = [case for case in test_cases if case is not None]
    #parallel_print("rank {} test_cases2 = {}".format(rank, test_cases))

    parallel_print(" RANK {}: TEST CASES = len = {} {}".format(bkd.get_rank(), len(test_cases), test_cases))

    print("\nLoading POD data ...")
    filename = inputs["paths"]["pod_dir"] + "/merged/svd.p"
    svd = pickle.load(open(filename, "rb"))["res"][0]

    # NN models loading
    # =====================================
    print("\nLoading NN configuration files ...")
    path_to_nets = {}
    for (element, tag) in inputs["paths"]["nets_tag"].items():
        suffix = f"/{tag}/merged/{element}/"
        path_to_nets[element] = inputs["paths"]["nets_dir"] + suffix
    nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles(
        mesh=mesh, dd_fom=dd_fom, path_to_nets=path_to_nets
    )

    for icase in test_cases:
        # Test case set up
        # ---------------
        print(icase)
        x0 = icase["snapshots"][0]
        parallel_print("RANK {}: SNAPSHOTS = ({}) {}".format(bkd.get_rank(), x0.shape, x0))
        solver = icase["solver"]
        print(solver)
        uv_fom = icase["solution"]
        runtime_fom = icase["runtime"]
        # > Time instants plotted
        # teval = icase["time"].squeeze()[::25]
        # ieval = np.arange(len(icase["time"]))[::25]
        teval = icase["time"].squeeze()[::25*solver['iostep']]
        ieval = (np.arange(len(icase["time"]))[::25*solver['iostep']])//solver['iostep']
        print(teval,ieval)
        # DD-FOM
        # ---------------
        # > Building
        field.set_params(icase["mu"])
        fom.build(field)
        dd_fom.build()
        # > Solution

        # DD-NM-ROM
        # ---------------
        # > Configuration
        hr_active = False
        e_min = 1e-6
        res_bases = pod_mod.get_pod_bases(svd=svd, energy_min=e_min)
        hr_n_samples = int(2*res_bases.shape[1])
        # > Building
        dd_rom = rom_mod.DD_NM_ROM(dd_fom=dd_fom,
                                   nn_configfiles=nn_configfiles,
                                   res_bases=res_bases,
                                   hr_active=hr_active,
                                   hr_n_samples=hr_n_samples,
                                   hr_n_edge_samples_ratio=2.0/3.0,
                                   hr_sample_small_ports=True,
                                   hr_small_ports_dim=5,
                                   constraint_type="strong",
                                   n_constraints_weak=-1,
                                   scaling=-1
        )

        profiler.start()
        # >> Solving
        solver["verbose"] = True
        solver["nt"] = test_num_timesteps
        parallel_print(" BEFORE SOLVE: RANK {}: X0 shape = {} = {}".format(rank, x0.shape, x0))
        #uv_rom, *_, iconverged = dd_rom.solve(
        uv_rom, z, lambdas, res, iconverged = dd_rom.solve(
            x0=dd_rom.get_init_sol(x=x0),
            runtime=0.0,
            use_guess=False,
            **solver
        )

        profiler.stop()
        with open(profile_out + ".out", 'w') as file:
            profiler.print(file)

        # >> Statistics - Single case
        iruntime = dd_rom.runtime
        ierror = dd_rom.compute_error_new(
            uv_fom, uv_rom, scaling=True, relative=True, axis=0
        )
        ispeedup = {k: tk/iruntime[k] for (k, tk) in runtime_fom.items()}

        print(" IRUNTIME = {}".format(iruntime))
        print(" IERROR = {}".format(ierror))
        print(" ISPEEDUP = {}".format(ispeedup))

        # TODO: fix this to run cpu and gpu case and compare, instead of hardcoded values!
        '''
        CPU results:
            Solver terminated after 1 iterations with residual norm of 1.6263e-08.
            Execution time: 4.63775e-01 s
            IRUNTIME = {'total': 43.04410171508789, 'lin_solve': 4.289948225021362, 'res_jac': 38.73490619659424}
            IERROR = {'L2_timemax': 0.001960687233002323, 'L2_timeavg': 0.001481475956931282, 'Linf_timeavg': 0.5042307588169733, 'Linf_timemax': 0.6460970156614413}
            ISPEEDUP = {'total': 23.68131718066232, 'lin_solve': 234.93744721312828, 'res_jac': 0.29337850820847583}
        '''
        '''
        # for nt = 200:
        np.testing.assert_almost_equal(ierror['L2_timemax'], 0.001960687233002323)
        np.testing.assert_almost_equal(ierror['L2_timeavg'], 0.001481475956931282)
        np.testing.assert_almost_equal(ierror['Linf_timeavg'], 0.5042307588169733)
        np.testing.assert_almost_equal(ierror['Linf_timemax'], 0.6460970156614413)
        '''

        # for nt = 5:
        '''
        CPU results:
            Solver terminated after 2 iterations with residual norm of 9.8904e-08.
            Execution time: 1.03765e+00 s
            DDROM SOLVE: RANK 0: POST_SOLVE X size = torch.Size([1280, 3]), res size = 5
            IRUNTIME = {'total': 4.215246915817261, 'lin_solve': 0.17498230934143066, 'res_jac': 4.005152702331543}
            IERROR = {'L2_timemax': 0.00024161770924130115, 'L2_timeavg': 0.00020730815822363234, 'Linf_timeavg': 0.028933408273142558, 'Linf_timemax': 0.03264850879615309}
            ISPEEDUP = {'total': 241.822376204516, 'lin_solve': 5759.836456932482, 'res_jac': 2.8373422538762822}
        '''
        np.testing.assert_almost_equal(ierror['L2_timemax'], 0.00024161770924130115)
        np.testing.assert_almost_equal(ierror['Linf_timemax'], 0.03264850879615309)
        if test_num_timesteps > 1:
          np.testing.assert_almost_equal(ierror['L2_timeavg'], 0.00020730815822363234)
          np.testing.assert_almost_equal(ierror['Linf_timeavg'], 0.028933408273142558)


if __name__ == "__main__":
    setup_module(None)
    test_ddnmrom_simple()
    teardown_module(None)
