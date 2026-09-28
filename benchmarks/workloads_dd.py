"""Benchmark-native DD-FOM and DD-ROM workloads.

These adapters mirror the setup in ``tests/regression/*scaling.py`` while
keeping model construction outside the measured region.  The trained ROM
network files are intentionally independent of the global ``n_sub_x`` and
``n_sub_y`` values; the same trained local model may therefore be reused for
each global domain size supported by the experiment.
"""

from __future__ import annotations

import os
import hashlib
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np


def _subdomains(config: dict[str, Any]) -> tuple[int, int, int]:
    n_sub_x = int(config["n_sub_x"])
    n_sub_y = int(config["n_sub_y"])
    total = n_sub_x * n_sub_y
    return n_sub_x, n_sub_y, total


def _metrics(
    converged: Any,
    residuals: Any,
    runtime: Any,
    activation_compile: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    metrics = {
        "converged": bool(converged),
        "newton_iterations": iterations,
        "residual_norm": residual_norm,
        "internal_timing": runtime,
    }
    if activation_compile is not None:
        metrics["activation_compile"] = activation_compile
    return metrics


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


def dd_fom_steady(config: dict[str, Any]) -> dict[str, Any]:
    return dd_fom_steady_run(dd_fom_steady_prepare(config))


def dd_fom_unsteady_prepare(config: dict[str, Any]) -> dict[str, Any]:
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
    field.set_params(mu=field.sample_design_space())
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
    return {"dd_fom": dd_fom, "x0": dd_fom.get_init_sol(x=x0), "config": config}


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


def dd_rom_prepare(config: dict[str, Any]) -> dict[str, Any]:
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
    field.set_params(mu=field.sample_design_space())
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
    return {"dd_rom": dd_rom, "x0": x0, "config": config}


def dd_rom_run(state: dict[str, Any]) -> dict[str, Any]:
    config = state.get("config", {})
    dd_rom = state["dd_rom"]
    maxit = int(config.get("maxit", 10))
    env_name = config.get("profile_maxit_env")
    if env_name and os.environ.get(env_name):
        maxit = int(os.environ[env_name])
    uv, z, lambdas, residuals, converged = dd_rom.solve(
        x0=dd_rom.get_init_sol(x=state["x0"]),
        runtime=0.0,
        use_guess=False,
        tol=float(config.get("tol", 1e-8)),
        nt=int(config.get("nt", 1)),
        maxit=maxit,
        verbose=bool(config.get("verbose", False)),
    )
    del uv, z, lambdas
    return _metrics(
        converged,
        residuals,
        dd_rom.runtime,
        activation_compile=dd_rom.activation_compile_stats,
    )


def dd_rom(config: dict[str, Any]) -> dict[str, Any]:
    return dd_rom_run(dd_rom_prepare(config))


def _decoder_activation_inputs(
    dd_rom: Any,
    x0: Any,
) -> list[tuple[str, Any, Any]]:
    """Build actual decoder preactivations from the DD-ROM initial state."""
    z0 = dd_rom.get_init_sol(x=x0)
    z = dd_rom.extract_z_sub_from_vec(z0, use_global=True)
    inputs = []
    for s, sub in enumerate(dd_rom.subdomains):
        for element in ("interior", "interface"):
            state = sub.elem_states[element]
            state.set_decoder_hr(active=False)
            decoder = state.nn_model.decoder
            hidden = decoder.w["W1"] @ z[s][element] + decoder.w["b1"]
            inputs.append(("decoder", decoder.activation, hidden))
    return inputs


def _activation_path(role: str, activation: Any) -> str:
    """Return the stable profile key for one activation execution path."""
    return f"{role}:{type(activation).__name__}"


def _mixed_layout_id(activation: Any, width: int | None = None) -> str:
    signature = repr((activation._compile_signature(), width)).encode()
    return hashlib.sha256(signature).hexdigest()[:16]


def _mixed_candidate_is_eligible(
    candidate: dict[str, Any],
    min_frequency: int,
    min_size: int,
    max_spread: float,
) -> bool:
    """Apply selection limits using production workload frequency."""
    return (
        candidate["workload_calls_per_pass"] >= min_frequency
        and candidate["width"] >= min_size
        and candidate["relative_spread"] <= max_spread
    )


def _inference_context(bkd):
    """Return a no-autograd context for Torch activation-only work."""
    if bkd.is_torch_backend():
        import torch

        return torch.inference_mode()
    return nullcontext()


def _aggregate_mixed_candidates(candidates, bkd):
    """Make Mixed-layout measurements and ranking identical across ranks."""
    if not bkd.distributed():
        for candidate in candidates:
            candidate["rank_count"] = 1
        return candidates

    fields = (
        "layout_id", "role", "width", "mask_count",
        "workload_calls_per_pass", "warmup_calls", "calls", "seconds",
        "sample_seconds_per_call",
    )
    local = [{key: candidate[key] for key in fields} for candidate in candidates]
    gathered = bkd._COMM.allgather(local)
    merged = {}
    for rank_records in gathered:
        for record in rank_records:
            layout_id = record["layout_id"]
            candidate = merged.get(layout_id)
            if candidate is None:
                candidate = {
                    key: record[key] for key in fields
                    if key != "sample_seconds_per_call"
                }
                candidate["sample_seconds_per_call"] = list(
                    record["sample_seconds_per_call"]
                )
                merged[layout_id] = candidate
                continue
            if candidate["role"] != record["role"]:
                # The layout ID does not include the model role. Keep the
                # deterministic lexical choice for reporting and compilation.
                candidate["role"] = min(candidate["role"], record["role"])
            candidate["workload_calls_per_pass"] += record["workload_calls_per_pass"]
            candidate["calls"] += record["calls"]
            candidate["seconds"] += record["seconds"]
            candidate["sample_seconds_per_call"].extend(
                record["sample_seconds_per_call"]
            )

    # Rank count is tracked separately so layouts absent on a rank can be
    # excluded from a synchronized compile set.
    for layout_id, candidate in merged.items():
        records_present = sum(
            layout_id in {record["layout_id"] for record in rank_records}
            for rank_records in gathered
        )
        candidate["rank_count"] = records_present
        samples = candidate["sample_seconds_per_call"]
        candidate["seconds_per_call"] = statistics.median(samples) if samples else 0.0
        candidate["median_seconds_per_call"] = candidate["seconds_per_call"]
        candidate["mean_seconds_per_call"] = statistics.mean(samples) if samples else 0.0
        candidate["min_seconds_per_call"] = min(samples) if samples else 0.0
        candidate["max_seconds_per_call"] = max(samples) if samples else 0.0
        candidate["relative_spread"] = (
            candidate["max_seconds_per_call"] / candidate["seconds_per_call"]
            if candidate["seconds_per_call"] > 0.0 else float("inf")
        )
    return list(merged.values())


def _sample_mixed_layouts(
    inputs: list[tuple[str, Any, Any]],
    with_jac: bool,
    warmups: int,
    repetitions: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Measure eager Mixed layouts before selecting compiled candidates.

    Sampling deliberately warms every layout before timing it.  CUDA events
    cover a small batch of calls so event creation and launch overhead do not
    dominate the per-layout estimate.  A fixed local RNG changes the order on
    every pass without making benchmark reports non-reproducible.
    """
    from dd_nm_rom import backend as bkd

    candidates = {}
    for role, activation, hidden in inputs:
        if type(activation).__name__ != "Mixed":
            continue
        signature = (activation._compile_signature(), int(hidden.numel()))
        candidate = candidates.setdefault(
            signature,
            {
                "layout_id": _mixed_layout_id(activation, hidden.numel()),
                "activation": activation,
                "role": role,
                "hidden": hidden,
                "inputs": [],
                "workload_calls_per_pass": 0,
                "warmup_calls": 0,
                "calls": 0,
                "seconds": 0.0,
                "sample_seconds_per_call": [],
            },
        )
        candidate["inputs"].append((activation, hidden))
        candidate["workload_calls_per_pass"] += 1

    if not candidates or repetitions <= 0:
        return []

    is_cuda = (
        bkd.is_torch_backend()
        and getattr(bkd.device(), "type", str(bkd.device())) == "cuda"
    )
    ordered_candidates = list(candidates.values())
    rng = random.Random(0)

    def run_candidate(candidate, count):
        for _ in range(count):
            for activation, hidden in candidate["inputs"]:
                result = activation(hidden, with_jac=with_jac)
                del result

    with _inference_context(bkd):
        # Warmups are intentionally excluded from both the call count and
        # timing used for selection. This removes first-use allocation and
        # kernel setup costs from the ranking signal.
        for _ in range(warmups):
            rng.shuffle(ordered_candidates)
            for candidate in ordered_candidates:
                run_candidate(candidate, 1)
                candidate["warmup_calls"] += len(candidate["inputs"])

        if is_cuda:
            import torch

            torch.cuda.synchronize()

        if is_cuda:
            import torch

            events = []
            for _ in range(repetitions):
                rng.shuffle(ordered_candidates)
                for candidate in ordered_candidates:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    run_candidate(candidate, batch_size)
                    end.record()
                    batch_calls = len(candidate["inputs"]) * batch_size
                    events.append((candidate, start, end, batch_calls))
                    candidate["calls"] += batch_calls
            torch.cuda.synchronize()
            for candidate, start, end, batch_calls in events:
                seconds = start.elapsed_time(end) / 1000.0
                candidate["seconds"] += seconds
                candidate["sample_seconds_per_call"].append(seconds / batch_calls)
        else:
            for _ in range(repetitions):
                rng.shuffle(ordered_candidates)
                for candidate in ordered_candidates:
                    start = time.perf_counter()
                    run_candidate(candidate, batch_size)
                    seconds = time.perf_counter() - start
                    batch_calls = len(candidate["inputs"]) * batch_size
                    candidate["seconds"] += seconds
                    candidate["sample_seconds_per_call"].append(seconds / batch_calls)
                    candidate["calls"] += batch_calls

    result = []
    for candidate in candidates.values():
        candidate = dict(candidate)
        candidate.pop("inputs")
        candidate["width"] = int(candidate["hidden"].numel())
        candidate["mask_count"] = len(candidate["activation"].masks)
        samples = candidate["sample_seconds_per_call"]
        candidate["seconds_per_call"] = (
            statistics.median(samples) if samples else 0.0
        )
        candidate["median_seconds_per_call"] = candidate["seconds_per_call"]
        candidate["mean_seconds_per_call"] = (
            statistics.mean(samples) if samples else 0.0
        )
        candidate["min_seconds_per_call"] = min(samples) if samples else 0.0
        candidate["max_seconds_per_call"] = max(samples) if samples else 0.0
        candidate["relative_spread"] = (
            candidate["max_seconds_per_call"] / candidate["seconds_per_call"]
            if candidate["seconds_per_call"] > 0.0 else float("inf")
        )
        candidate.pop("activation")
        candidate.pop("hidden")
        result.append(candidate)
    return result


def _prepare_mixed_compile_selection(
    state: dict[str, Any],
    inputs: list[tuple[str, Any, Any]],
    with_jac: bool,
) -> dict[str, Any] | None:
    """Sample Mixed layouts and compile the highest-value candidates."""
    from dd_nm_rom import backend as bkd
    from dd_nm_rom import config as cfg
    from dd_nm_rom.rom.nonlinear.autoencoder.nn_numpy import activation as activation_mod

    if not bkd.is_torch_backend():
        return None
    if cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_POLICY").lower() != "top_k":
        return None
    if not activation_mod._should_compile():
        return None
    if not (
        activation_mod._config_bool("DDNMROM_ACT_COMPILE_FORWARD")
        or activation_mod._config_bool("DDNMROM_ACT_COMPILE_JAC")
    ):
        return None
    warmups = int(
        cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_SAMPLE_WARMUPS")
    )
    repetitions = int(
        cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_SAMPLE_REPETITIONS")
    )
    batch_size = int(
        cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_SAMPLE_BATCH_SIZE")
    )
    max_spread = float(
        cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_SAMPLE_MAX_SPREAD")
    )
    if warmups < 0:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_WARMUPS must be nonnegative"
        )
    if repetitions < 1:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_REPETITIONS must be positive"
        )
    if batch_size < 1:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_BATCH_SIZE must be positive"
        )
    if max_spread <= 0.0:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_MAX_SPREAD must be positive"
        )

    # Model construction may have compiled a different set of Mixed layouts.
    # Sampling must see eager callables, and selected layouts must start with
    # predictable cache capacity and frequency accounting.
    for _, activation, _ in inputs:
        if isinstance(activation, activation_mod.Mixed):
            activation_mod._set_eager(activation)
    activation_mod.clear_compiled_cache(mixed_only=True)
    activation_mod.reset_mixed_compile_state()

    local_candidates = _sample_mixed_layouts(
        inputs, with_jac, warmups, repetitions, batch_size
    )
    candidates = _aggregate_mixed_candidates(local_candidates, bkd)
    min_frequency = int(
      cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY")
    )
    maxsize = int(
      cfg.get_config_val("DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE")
    )
    min_size = int(cfg.get_config_val("DDNMROM_ACT_COMPILE_MIN_SIZE"))
    if min_frequency < 1:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY must be positive"
        )
    if maxsize < 1:
        raise ValueError(
            "DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE must be positive"
        )
    eligible = [
        candidate for candidate in candidates
        if _mixed_candidate_is_eligible(
            candidate, min_frequency, min_size, max_spread
        )
        and candidate.get("rank_count", 1) >= bkd.get_nranks()
    ]
    for candidate in candidates:
        candidate["stable"] = candidate["relative_spread"] <= max_spread
        candidate["selection_score"] = (
            candidate["workload_calls_per_pass"]
            * candidate["median_seconds_per_call"]
        )
        candidate["selected"] = False
        candidate["selection_rank"] = None

    # Rank by expected workload frequency times robust steady-state eager
    # cost.  Sampling calls are intentionally not used as frequency because
    # every layout receives the same number of sample repetitions.
    eligible.sort(
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["width"],
        ),
        reverse=True,
    )
    available_slots = activation_mod.compiled_cache_slots_available()
    if bkd.distributed():
        available_slots = min(
            bkd._COMM.allgather(available_slots)
        )
    selection_limit = min(maxsize, available_slots)
    selected = eligible[:selection_limit]
    selected_ids = {candidate["layout_id"] for candidate in selected}
    for rank, candidate in enumerate(selected, start=1):
        candidate["selected"] = True
        candidate["selection_rank"] = rank

    # Compile every selected layout on every rank. Compile errors are
    # exchanged before any rank proceeds to a later distributed collective;
    # otherwise one rank can fail while another enters the solver and hangs.
    compile_exception = None
    try:
        with _inference_context(bkd):
            for role, activation, hidden in inputs:
                if type(activation).__name__ != "Mixed":
                    continue
                if _mixed_layout_id(activation, hidden.numel()) not in selected_ids:
                    continue
                activation_mod.warmup(
                    activation,
                    int(hidden.numel()),
                    device=hidden.device,
                    dtype=hidden.dtype,
                    role=role,
                    force=True,
                )

            if (
                bkd.is_torch_backend()
                and getattr(bkd.device(), "type", str(bkd.device())) == "cuda"
            ):
                import torch

                torch.cuda.synchronize()
    except Exception as exc:
        compile_exception = exc

    if bkd.distributed():
        compile_errors = bkd._COMM.allgather(
            None if compile_exception is None else repr(compile_exception)
        )
        if any(error is not None for error in compile_errors):
            details = "; ".join(
                "rank {}: {}".format(rank, error)
                for rank, error in enumerate(compile_errors)
                if error is not None
            )
            raise RuntimeError(
                "DDNMROM Mixed activation compilation failed before "
                "synchronization: " + details
            ) from compile_exception
    elif compile_exception is not None:
        raise compile_exception

    if selected_ids:
        bkd.barrier()

    report = {
        "sample_warmups": warmups,
        "sample_repetitions": repetitions,
        "sample_batch_size": batch_size,
        "sample_max_spread": max_spread,
        "candidate_layouts": len(candidates),
        "eligible_layouts": len(eligible),
        "selected_layouts": len(selected),
        "available_cache_slots": available_slots,
        "by_layout": {
            candidate["layout_id"]: {
                key: value for key, value in candidate.items()
                if key not in ("layout_id",)
            }
            for candidate in candidates
        },
    }
    state["dd_rom"].activation_compile_stats = activation_mod.get_compile_stats()
    return report


def dd_rom_activation_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare the same DD-ROM workload for activation-only profiling."""
    state = dd_rom_prepare(config)
    state["activation_inputs"] = _decoder_activation_inputs(
        state["dd_rom"], state["x0"]
    )
    state["activation_compile_mixed_sampling"] = _prepare_mixed_compile_selection(
        state,
        state["activation_inputs"],
        bool(config.get("activation_with_jac", True)),
    )
    return state


def dd_rom_activation_run(state: dict[str, Any]) -> dict[str, Any]:
    """Time decoder activation/Jacobian calls using workload-shaped inputs.

    Model construction, activation compilation, and latent-state preparation
    remain outside the measured region. CUDA events measure device execution
    without synchronizing once per activation call.
    """
    from dd_nm_rom import backend as bkd

    inputs = state["activation_inputs"]
    with_jac = bool(state["config"].get("activation_with_jac", True))
    activation_elements = sum(
        int(hidden.numel()) if hasattr(hidden, "numel") else int(np.size(hidden))
        for _, _, hidden in inputs
    )
    device = bkd.device()
    is_cuda = bkd.is_torch_backend() and getattr(device, "type", str(device)) == "cuda"

    by_path = {}
    grouped_inputs = {}
    for role, activation, hidden in inputs:
        path = _activation_path(role, activation)
        path_metrics = by_path.setdefault(
            path,
            {"calls": 0, "elements": 0, "seconds": 0.0},
        )
        path_metrics["calls"] += 1
        path_metrics["elements"] += (
            int(hidden.numel()) if hasattr(hidden, "numel") else int(np.size(hidden))
        )
        grouped_inputs.setdefault(path, []).append((activation, hidden))

    with _inference_context(bkd):
        if is_cuda:
            import torch

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            path_events = []
            for path, path_inputs in grouped_inputs.items():
                path_start = torch.cuda.Event(enable_timing=True)
                path_end = torch.cuda.Event(enable_timing=True)
                path_start.record()
                for activation, hidden in path_inputs:
                    result = activation(hidden, with_jac=with_jac)
                    del result
                path_end.record()
                path_events.append((path, path_start, path_end))
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
            for path, path_start, path_end in path_events:
                by_path[path]["seconds"] = path_start.elapsed_time(path_end) / 1000.0
        else:
            start_time = time.perf_counter()
            for path, path_inputs in grouped_inputs.items():
                path_start = time.perf_counter()
                for activation, hidden in path_inputs:
                    result = activation(hidden, with_jac=with_jac)
                    del result
                by_path[path]["seconds"] = time.perf_counter() - path_start
            elapsed = time.perf_counter() - start_time

    return {
        "activation_timing_seconds": elapsed,
        "activation_calls": len(inputs),
        "activation_elements": activation_elements,
        "activation_with_jac": with_jac,
        "activation_by_path": by_path,
        "activation_compile_mixed_sampling": state.get(
            "activation_compile_mixed_sampling"
        ),
        "activation_compile": state["dd_rom"].activation_compile_stats,
    }


def dd_rom_activation(config: dict[str, Any]) -> dict[str, Any]:
    return dd_rom_activation_run(dd_rom_activation_prepare(config))
