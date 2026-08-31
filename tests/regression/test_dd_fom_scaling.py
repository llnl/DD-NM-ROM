import pytest

import numpy as np
import os
import json
import dill as pickle
from pathlib import Path
import torch
import torch.distributed as dist
import torch.testing
from mpi4py import MPI
import copy

if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))

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
    #from dd_nm_rom import env
    #env.set(**inputs["env"])

    #bkd.set(backend="numpy", device="cpu", seed=0)
    bkd.set(backend="torch", device="cuda", seed=0)


def teardown_module(module):
    if bkd.is_torch_backend():
        bkd.finalize_distributed()


# Steady Burgers-2D
#  constraint type = strong
#  no-HR
def test_ddfom_distributed_steady():
    rank = bkd.get_rank()

    profiler = Profiler()
    profile_out = "./profiles/test_dd_fom_steady_scaling_profile_gpu"
    if bkd.distributed():
        profile_out += "_{}_{}".format(bkd.get_nranks(), bkd.get_rank())

    print("\nInitialization ...")
    
    nx, ny = 480, 24
    x_lim = [-1.0, 1.0]
    y_lim = [0.0, 0.05]
    a_lim = [1.0, 10000.0]
    k_lim = [5.0, 25.0]
    a1 = 1e4
    lam = 5.0
    mu = np.array([a1, lam])
    viscosity = 1e-1

    nx_intr = 3
    ny_intr = 3
    lx_sub = 0.5
    ly_sub = 0.5
    x0 = 0.0
    y0 = 0.0
    #n_sub_x = 2
    n_sub_x = 4
    n_sub_y = 4

    fom_subs_per_rank = (n_sub_x * n_sub_y) // bkd.get_nranks()
    print(" SUBDOMAINS PER RANK: {}".format(fom_subs_per_rank))

    mesh = mesh_mod.MeshDD(
        nx_intr=nx_intr,
        ny_intr=ny_intr,
        lx_sub=lx_sub,
        ly_sub=ly_sub,
        x0=x0,
        y0=y0,
        n_sub_x=n_sub_x,
        n_sub_y=n_sub_y,
        #with_bounds=True
    )
    mesh.build()
    mesh_dd = mesh
    field = field_mod.Burgers2DExact(
        mesh=mesh,
        nu=viscosity,
        a_lim=a_lim,
        k_lim=k_lim
    )
    field.set_params(mu)
    fom = fom_mod.Burgers2D(
        nu=viscosity,
        mesh=mesh
    )
    fom.build(field)
    dd_fom = fom_mod.DDBurgers2D(fom, subs_per_rank=fom_subs_per_rank)
    dd_fom.build()

    # checks that constraint matrices are computed correctly
    c = np.zeros(dd_fom.n_constraints)
    vec = np.random.rand(2*mesh.nxy)
    for s in dd_fom.subdomains:
        interface = s.elem_states["interface"].nodes_state
        #sub_cmat_np = bkd.to_numpy(s.cmat["interface"])
        sub_cmat_np = bkd.torch_csr_to_scipy(s.cmat["interface"].to_sparse_csr())
        c += sub_cmat_np@np.concatenate([vec[interface],vec[mesh.nxy+interface]])
        #print(" RANK {} SUM c = {}".format(bkd.get_rank(), c))
    c = bkd._COMM.allreduce(c, op=MPI.SUM)
    print("||sum(A[i] x[i])||=", np.linalg.norm(c))
    #np.testing.assert_almost_equal(np.linalg.norm(c), 0.0)

    X, Y = mesh.grid
    #print(" MESH X: {} {}".format(X.shape, X))
    #print(" MESH Y: {} {}".format(Y.shape, Y))


    # generate Burgers FOM on coarse grid for visualization
    # uv, res, converged = fom.solve(tol=1e-8, maxit=20, stepsize_min=1e-20, verbose=True)
    # print("FOM RUNTIME:", fom.runtime)
    # print("FOM RES: {}".format(res))
    # #print(uv)
    # # for (k, v) in uv.items():
    # #     print(k, v.shape)

    # # for (iter, r) in enumerate(res):
    # #     print(" RES NORM ITER: {} = {}".format(iter, np.linalg.norm(r)))

    # # compute exact u and v on grid
    # Uex, Vex = field.u(X, Y), field.v(X, Y)
    # #Uex = bkd.to_backend(Uex)
    # #Vex = bkd.to_backend(Vex)
    # #print("uv before: ", uv)
    # uv = ops.map_nested_dict(uv, bkd.to_numpy)
    # #print("uv after: ", uv)

    # # plot FD u and v
    # #U = uv["u"].reshape(ny, nx) # note ny nx here is for mono mesh
    # #V = uv["v"].reshape(ny, nx)

    # U = uv["u"].reshape(Y.shape[0], X.shape[1])
    # V = uv["v"].reshape(Y.shape[0], X.shape[1])
    # uerr = np.abs(U-Uex) / np.linalg.norm(Uex)
    # verr = np.abs(V-Vex) / np.linalg.norm(Vex)
    # #print("U-Uex", uerr)
    # #print("V-Vex", verr)

    # uerr = np.linalg.norm(uerr)
    # verr = np.linalg.norm(verr)
    # print("U-Uex norm = ", uerr)
    # print("V-Vex norm = ", verr)

    # Compare monolithic FOM solver against exact solution
    #  - Note: Norms below are from serial numpy version of code
    # TODO: fix - below is only valid for same grid, FOM parameters, etc
    #np.testing.assert_almost_equal(uerr, 0.0002792348355655072)
    #np.testing.assert_almost_equal(verr, 0.014938729901817153)

    # for 2x4:
    #np.testing.assert_almost_equal(uerr, 0.00025529812284760)
    #np.testing.assert_almost_equal(verr, 0.01706449507086961)
    # for 4x4:
    #np.testing.assert_almost_equal(uerr, 0.000228173117397447)
    #np.testing.assert_almost_equal(verr, 0.018192105339752386)


    # compute DD model
    dd_fom_s = fom_mod.DDBurgers2D(fom, constraint_type="strong", scaling=-1, subs_per_rank=fom_subs_per_rank)
    dd_fom_s.build()

    profiler.reset()
    profiler.start()
    uv_dd_s, lambdas, res_dd, converged = dd_fom_s.solve(tol=1e-8, maxit=50, stepsize_min=1e-20, verbose=True)
    profiler.stop()
    with open(profile_out + "_ddfom{}x{}.out".format(n_sub_x, n_sub_y), 'w') as file:
        profiler.print(file)
    print("DD-FOM RUNTIME:", dd_fom_s.runtime)
    print("DD-FOM RES: {}".format(res_dd))
    #print(uv_dd_s)
    uv_dd_s = ops.map_nested_dict(uv_dd_s, bkd.to_numpy)

    bkd.barrier()

    dd_u_rel_err = np.linalg.norm(uv_dd_s["res"]["u"]-uv["u"].reshape(-1))/np.linalg.norm(uv["u"])
    dd_v_rel_err = np.linalg.norm(uv_dd_s["res"]["v"]-uv["v"].reshape(-1))/np.linalg.norm(uv["v"])
    print("DD u relative error =", dd_u_rel_err)
    print("DD v relative error =", dd_v_rel_err)

    #parallel_print(" RANK {} LAMBDAS = {}".format(rank, lambdas))

    # Compare DD-FOM solver against monolithic FOM solver
    #  - Note: Norms below are from serial numpy version of code
    # TODO: fix - below is only valid for same grid, FOM parameters, etc
    #np.testing.assert_almost_equal(dd_u_rel_err, 1.8152893989448393)
    #np.testing.assert_almost_equal(dd_v_rel_err, 10.93394163183537)

    # for 2x4
    #np.testing.assert_almost_equal(dd_u_rel_err, 2.6591856954101725)
    #np.testing.assert_almost_equal(dd_v_rel_err, 16.64303125604579)

    # for 4x4
    np.testing.assert_almost_equal(dd_u_rel_err, 6.972791319923508)
    np.testing.assert_almost_equal(dd_v_rel_err, 25.046511081234925)


if __name__ == "__main__":
    setup_module(None)
    test_ddfom_distributed_steady()
    teardown_module(None)
