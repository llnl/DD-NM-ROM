"""Benchmark-native DD-FOM and DD-ROM workloads.

These adapters mirror the setup in ``tests/regression/*scaling.py`` while
keeping model construction outside the measured region.  The trained ROM
network files are intentionally independent of the global ``n_sub_x`` and
``n_sub_y`` values; the same trained local model may therefore be reused for
each global domain size supported by the experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _subdomains(config: dict[str, Any]) -> tuple[int, int, int]:
    n_sub_x = int(config["n_sub_x"])
    n_sub_y = int(config["n_sub_y"])
    total = n_sub_x * n_sub_y
    return n_sub_x, n_sub_y, total


def _metrics(converged: Any, residuals: Any, runtime: Any) -> dict[str, Any]:
    try:
        iterations = len(residuals)
    except TypeError:
        iterations = None
    residual_norm = None
    if residuals is not None and iterations:
        try:
            from dd_nm_rom import backend as bkd

            residual_norm = float(np.linalg.norm(bkd.to_numpy(residuals[-1])))
        except (IndexError, TypeError, ValueError):
            residual_norm = None
    return {
        "converged": bool(converged),
        "newton_iterations": iterations,
        "residual_norm": residual_norm,
        # The runtime dictionary belongs to the mutable model.  Snapshot it
        # so a later reset cannot rewrite an earlier repetition's metrics.
        "internal_timing": dict(runtime) if isinstance(runtime, dict) else runtime,
    }


def _restore_solve_state(model: Any) -> None:
    """Restore fields mutated by the Newton/time-integration driver.

    Rebuilding a distributed DD model for every repetition repeats model
    loading, activation compilation, and MPI collectives.  Those operations
    are setup, not solver state, and repeated construction can make a
    multi-rank benchmark appear to hang.  The DD models keep the mutable
    solve state in a small set of fields, so restore those fields in place.
    """
    runtime = getattr(model, "runtime", None)
    if isinstance(runtime, dict):
        for key in runtime:
            runtime[key] = 0.0
    for name, value in (
        ("steady", True),
        ("x_old", None),
        ("dt", 0.0),
        ("local_res_shapes", None),
        ("resjac_it", 0),
    ):
        if hasattr(model, name):
            setattr(model, name, value)


def dd_fom_steady_prepare(config: dict[str, Any]) -> dict[str, Any]:
    from dd_nm_rom import field as field_mod
    from dd_nm_rom import fom as fom_mod
    from dd_nm_rom import backend as bkd
    from dd_nm_rom.elements import mesh as mesh_mod

    n_sub_x, n_sub_y, total = _subdomains(config)
    mesh = mesh_mod.MeshDD(
        nx_intr=int(config["nx_intr"]),
        ny_intr=int(config["ny_intr"]),
        lx_sub=float(config["lx_sub"]),
        ly_sub=float(config["ly_sub"]),
        x0=float(config.get("x0", 0.0)),
        y0=float(config.get("y0", 0.0)),
        n_sub_x=n_sub_x,
        n_sub_y=n_sub_y,
    )
    mesh.build()
    field = field_mod.Burgers2DExact(
        mesh=mesh,
        nu=float(config["viscosity"]),
        a_lim=config["a_lim"],
        k_lim=config["k_lim"],
    )
    field.set_params(np.asarray(config["mu"]))
    fom = fom_mod.Burgers2D(nu=float(config["viscosity"]), mesh=mesh)
    fom.build(field)
    dd_fom = fom_mod.DDBurgers2D(
        fom,
        constraint_type=config.get("constraint_type", "strong"),
        scaling=float(config.get("scaling", -1)),
        subs_per_rank=total // bkd.get_nranks(),
    )
    dd_fom.build()
    return {"dd_fom": dd_fom, "config": config}


def dd_fom_steady_run(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"] if "config" in state else {}
    dd_fom = state["dd_fom"]
    uv, lambdas, residuals, converged = dd_fom.solve(
        tol=float(config.get("tol", 1e-8)),
        maxit=int(config.get("maxit", 50)),
        stepsize_min=float(config.get("stepsize_min", 1e-20)),
        verbose=bool(config.get("verbose", False)),
    )
    del uv, lambdas
    return _metrics(converged, residuals, dd_fom.runtime)


def dd_fom_steady_reset(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Restore an independent steady DD-FOM solve state.

    A DD solve mutates only its solver-related state; restoring it avoids
    repeating distributed model construction between repetitions.
    """
    del config
    _restore_solve_state(state["dd_fom"])
    return state


def dd_fom_steady(config: dict[str, Any]) -> dict[str, Any]:
    return dd_fom_steady_run(dd_fom_steady_prepare(config))


def dd_fom_unsteady_prepare(
    config: dict[str, Any], mu: np.ndarray | None = None
) -> dict[str, Any]:
    from dd_nm_rom import backend as bkd
    from dd_nm_rom import field as field_mod
    from dd_nm_rom import fom as fom_mod
    from dd_nm_rom.elements import mesh as mesh_mod

    n_sub_x, n_sub_y, total = _subdomains(config)
    mesh = mesh_mod.MeshDD(
        nx_intr=int(config["nx_intr"]),
        ny_intr=int(config["ny_intr"]),
        lx_sub=float(config["lx_sub"]),
        ly_sub=float(config["ly_sub"]),
        x0=float(config.get("x0", 0.0)),
        y0=float(config.get("y0", 0.0)),
        n_sub_x=n_sub_x,
        n_sub_y=n_sub_y,
        with_bounds=bool(config.get("with_bounds", True)),
    )
    mesh.build()
    field = field_mod.SinPeak(
        mesh=mesh,
        mu_lim=config.get("mu_lim", [0.9, 1.1]),
        bc_type=config.get("bc_type", "periodic"),
    )
    if mu is None:
        mu = field.sample_design_space()
    field.set_params(mu=mu)
    fom = fom_mod.Burgers2D(nu=float(config["viscosity"]), mesh=mesh)
    fom.build(field)
    x0 = np.concatenate([field.u().reshape(-1), field.v().reshape(-1)])
    dd_fom = fom_mod.DDBurgers2D(
        fom,
        constraint_type=config.get("constraint_type", "strong"),
        scaling=float(config.get("scaling", -1)),
        subs_per_rank=total // bkd.get_nranks(),
    )
    dd_fom.build()
    return {
        "dd_fom": dd_fom,
        "x0": dd_fom.get_init_sol(x=x0),
        "config": config,
        "mu": np.array(mu, copy=True),
    }


def dd_fom_unsteady_run(state: dict[str, Any]) -> dict[str, Any]:
    config = state.get("config", {})
    dd_fom = state["dd_fom"]
    uv, lambdas, residuals, converged = dd_fom.solve(
        x0=state["x0"],
        dt=float(config.get("dt", 0.03)),
        nt=int(config.get("nt", 1)),
        steady=bool(config.get("steady", False)),
        tol=float(config.get("tol", 1e-8)),
        maxit=int(config.get("maxit", 20)),
        stepsize_min=float(config.get("stepsize_min", 1e-10)),
        verbose=bool(config.get("verbose", False)),
    )
    del uv, lambdas
    return _metrics(converged, residuals, dd_fom.runtime)


def dd_fom_unsteady_reset(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Restore an independent unsteady DD-FOM solve state."""
    del config
    _restore_solve_state(state["dd_fom"])
    return state


def dd_fom_unsteady(config: dict[str, Any]) -> dict[str, Any]:
    return dd_fom_unsteady_run(dd_fom_unsteady_prepare(config))


def _network_paths(config: dict[str, Any]) -> dict[str, str]:
    root = Path(config["nets_dir"])
    layout = config.get("nets_layout", "merged")
    paths = {}
    for element, tag in config.get("nets_tag", {"interior": "", "port": ""}).items():
        element_root = root / str(tag) / layout / element if tag else root / layout / element
        paths[element] = str(element_root)
    return paths


def dd_rom_prepare(
    config: dict[str, Any], mu: np.ndarray | None = None
) -> dict[str, Any]:
    from dd_nm_rom import backend as bkd
    from dd_nm_rom import field as field_mod
    from dd_nm_rom import fom as fom_mod
    from dd_nm_rom import rom as rom_mod
    from dd_nm_rom.elements import mesh as mesh_mod

    n_sub_x, n_sub_y, total = _subdomains(config)
    mesh = mesh_mod.MeshDD(
        nx_intr=int(config["nx_intr"]),
        ny_intr=int(config["ny_intr"]),
        lx_sub=float(config["lx_sub"]),
        ly_sub=float(config["ly_sub"]),
        x0=float(config.get("x0", 0.0)),
        y0=float(config.get("y0", 0.0)),
        n_sub_x=n_sub_x,
        n_sub_y=n_sub_y,
        with_bounds=bool(config.get("with_bounds", True)),
    )
    mesh.build()
    field = field_mod.SinMultiPeak(
        mesh=mesh,
        mu_lim=config.get("mu_lim", [0.5, 1.5]),
        bc_type=config.get("bc_type", "periodic"),
    )
    if mu is None:
        mu = field.sample_design_space()
    field.set_params(mu=mu)
    fom = fom_mod.Burgers2D(
        mesh=mesh,
        nu=float(config["viscosity"]),
        upwind=bool(config.get("upwind", True)),
        upwind_order=int(config.get("upwind_order", 2)),
        compact=bool(config.get("compact", True)),
    )
    fom.build(field)
    dd_fom = fom_mod.DDBurgers2D(
        monolithic=fom,
        subs_per_rank=total // bkd.get_nranks(),
        constraint_type=config.get("constraint_type", "strong"),
        scaling=float(config.get("scaling", -1)),
    )
    dd_fom.build()
    path_to_nets = _network_paths(config)
    nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles_new(
        mesh=mesh,
        dd_fom=dd_fom,
        path_to_nets=path_to_nets,
    )
    dd_rom = rom_mod.DD_NM_ROM(
        dd_fom=dd_fom,
        nn_configfiles=nn_configfiles,
        constraint_type=config.get("constraint_type", "strong"),
        n_constraints_weak=-1,
        scaling=float(config.get("scaling", -1)),
        subs_per_rank=total // bkd.get_nranks(),
        check_unique_models=bool(config.get("check_unique_models", True)),
    )
    x0 = np.concatenate([field.u().reshape(-1), field.v().reshape(-1)])
    return {
        "dd_rom": dd_rom,
        "x0": x0,
        "config": config,
        "mu": np.array(mu, copy=True),
    }


def dd_rom_run(state: dict[str, Any]) -> dict[str, Any]:
    config = state.get("config", {})
    dd_rom = state["dd_rom"]
    maxit = int(config.get("maxit", 10))
    env_name = config.get("profile_maxit_env")
    if env_name and os.environ.get(env_name):
        maxit = int(os.environ[env_name])
    x0 = dd_rom.get_init_sol(x=state["x0"])
    uv, z, lambdas, residuals, converged = dd_rom.solve(
        x0=x0,
        runtime=0.0,
        use_guess=False,
        tol=float(config.get("tol", 1e-8)),
        nt=int(config.get("nt", 1)),
        maxit=maxit,
        verbose=bool(config.get("verbose", False)),
    )
    del uv, z, lambdas
    return _metrics(converged, residuals, dd_rom.runtime)


def dd_rom_reset(
    state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Restore an independent DD-ROM solve state.

    ``DD_NM_ROM.solve`` mutates the ROM solver fields and its DD-FOM backing
    model.  Restore both without rebuilding the distributed model, which
    avoids repeating MPI setup and activation compilation.
    """
    del config
    dd_rom = state["dd_rom"]
    _restore_solve_state(dd_rom)
    _restore_solve_state(dd_rom.dd_fom)
    return state


def dd_rom(config: dict[str, Any]) -> dict[str, Any]:
    return dd_rom_run(dd_rom_prepare(config))
