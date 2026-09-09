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
    #bkd.set(backend="numpy", device="cpu", seed=0)
    bkd.set(backend="torch", device="cuda", seed=0)


def teardown_module(module):
    if bkd.is_torch_backend():
        bkd.finalize_distributed()


# Unsteady time-dependent Burgers 2D
#  Single sin peak, periodic BC
#  no-HR
def test_ddfom_distributed_unsteady():
    rank = bkd.get_rank()

    profiler = Profiler()
    profile_out = "test_dd_fom_unsteady_profile_gpu"
    if bkd.distributed():
        profile_out += "_{}".format(bkd.get_rank())

    print("\nInitialization ...")
    
    # DD Mesh
    nx_intr = 48
    ny_intr = 48
    lx_sub = 0.5
    ly_sub = 0.5
    x0 = 0.0
    y0 = 0.0
    n_sub_x = 2
    n_sub_y = 2
    # Time grid
    dt = 0.03
    nt = 1
    t_lim = [0, dt*nt]
    # PDE
    viscosity = 1e-3

    mesh = mesh_mod.MeshDD(
        nx_intr=nx_intr,
        ny_intr=ny_intr,
        lx_sub=lx_sub,
        ly_sub=ly_sub,
        x0=x0,
        y0=y0,
        n_sub_x=n_sub_x,
        n_sub_y=n_sub_y,
        with_bounds=True
    )
    mesh.build()

    X, Y = mesh.grid
    print(" MESH X: {} {}".format(X.shape, X))
    print(" MESH Y: {} {}".format(Y.shape, Y))

    field = field_mod.SinPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="periodic")
    field.set_params(mu=field.sample_design_space())
    U0 = field.u()
    V0 = field.v()

    # Monolithic Burgers 2D
    fom = fom_mod.Burgers2D(nu=viscosity, 
                            mesh=mesh
    )
    fom.build(field)

    # get initial condition
    x0 = np.concatenate([U0.reshape(-1), V0.reshape(-1)])
    print("U0 = {}".format(U0))
    print("V0 = {}".format(V0))

    print("x0 = {}".format(x0))

    # Solve FOM
    profiler.start()
    uv, rhs, converged = fom.solve(x0, dt=dt, nt=nt, steady=False, tol=1e-8, maxit=20, stepsize_min=1e-10, verbose=True)
    profiler.stop()
    with open(profile_out + "_fom.out", 'w') as file:
        profiler.print(file)
    print("FOM RUNTIME:", fom.runtime)
    print("FOM RHS: {}".format(rhs))

    #print(uv)
    for (k, v) in uv.items():
        print(k, v.shape)

    for (iter, r) in enumerate(rhs):
        print(" RES NORM ITER: {} = {}".format(iter, np.linalg.norm(r)))

    # compute exact u and v on grid
    uv = ops.map_nested_dict(uv, bkd.to_numpy)

    # plot FD u and v
    #U = uv["u"].reshape(ny, nx) # note ny nx here is for mono mesh
    #V = uv["v"].reshape(ny, nx)

    #U = uv["u"].reshape(len(Y), len(X))
    #V = uv["v"].reshape(len(Y), len(X))

    # compute DD model
    # Burgers 2D DD-FOM
    dd_fom = fom_mod.DDBurgers2D(fom, constraint_type='strong')
    dd_fom.build()

    x0_dd = dd_fom.get_init_sol(x=x0)
    print("x0_dd = {} {}".format(x0_dd.shape, x0_dd))
    profiler.reset()
    profiler.start()
    uv_dd, lambdas, rhs_dd, converged = dd_fom.solve(x0=x0_dd,
                                                     dt=dt,
                                                     nt=nt,
                                                     steady=False,
                                                     tol=1e-8,
                                                     maxit=20,
                                                     stepsize_min=1e-10,
                                                     verbose=True)
    profiler.stop()
    with open(profile_out + "_ddfom.out", 'w') as file:
        profiler.print(file)
    print("DD-FOM RUNTIME:", dd_fom.runtime)
    print("DD-FOM lambdas: {}".format(lambdas))
    print("DD-FOM RHS: {}".format(rhs_dd))
    #print(uv_dd_s)
    uv_dd = ops.map_nested_dict(uv_dd, bkd.to_numpy)

  
    dd_u_rel_err = np.abs(uv_dd["res"]["u"]-uv["u"]).T.reshape(-1, mesh.n["y"], mesh.n["x"])
    dd_v_rel_err = np.abs(uv_dd["res"]["v"]-uv["v"]).T.reshape(-1, mesh.n["y"], mesh.n["x"])
    print("DD u relative error =", dd_u_rel_err)
    print("DD v relative error =", dd_v_rel_err)

    # Compare DD-FOM solver against monolithic FOM solver
    #  - Note: Norms below are from serial numpy version of code
    # TODO: fix - below is only valid for same grid, FOM parameters, etc
    #np.testing.assert_almost_equal(dd_u_rel_err, 1.8152893989448393)
    #np.testing.assert_almost_equal(dd_v_rel_err, 10.93394163183537)

