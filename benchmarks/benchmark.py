#!/usr/bin/env python3
"""Run serial or MPI benchmark workloads locally or through a batch system.

The target in a JSON spec is either a module-level callable accepting a config
dictionary, or a module containing ``prepare(config)`` and ``run(state)``.
Project workloads should keep model construction in ``prepare`` and one
measured operation in ``run``.

Examples:
  .venv/bin/python benchmarks/benchmark.py --spec benchmarks/specs/backend_dense_linear.json
  .venv/bin/python benchmarks/benchmark.py --spec benchmarks/specs/backend_device.json \
      --launcher slurm --ranks 1,2,4 --output-dir benchmark_jobs
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from itertools import product
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


BACKENDS = ("numpy", "torch_cpu", "torch_gpu")
LAUNCHERS = ("local", "slurm", "flux")
ENV_PROFILE_DIR = Path(__file__).resolve().parent / "env"


def _parse_ranks(value: str) -> list[int]:
    ranks = []
    for item in value.split(","):
        try:
            rank = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid rank count: {item!r}") from exc
        if rank < 1:
            raise argparse.ArgumentTypeError("rank counts must be positive")
        ranks.append(rank)
    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank count is required")
    return ranks


def _value_slug(value: Any) -> str:
    text = str(value).replace("/", "-").replace(" ", "")
    return text.replace(".", "p")


def _set_override(spec: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"invalid sweep override path: {path!r}")
    target: Any = spec
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise ValueError(f"sweep override path does not exist: {path!r}")
        target = target[part]
    if not isinstance(target, dict):
        raise ValueError(f"sweep override parent is not an object: {path!r}")
    target[parts[-1]] = value


def _apply_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(spec)
    for path, value in overrides.items():
        _set_override(effective, path, value)
    effective.pop("sweep", None)
    effective.pop("sweep_cases", None)
    return effective


def _independent_sweep(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sweep = spec.get("sweep", {})
    if sweep is None:
        return [("", {})]
    if not isinstance(sweep, dict):
        raise ValueError("'sweep' must be an object mapping paths to lists")
    if not sweep:
        return [("", {})]
    keys = list(sweep)
    values = []
    for key in keys:
        choices = sweep[key]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"sweep values for {key!r} must be a nonempty list")
        values.append(choices)
    cases = []
    for combination in product(*values):
        overrides = dict(zip(keys, combination))
        label = "_".join(
            f"{key.split('.')[-1]}-{_value_slug(value)}"
            for key, value in overrides.items()
        )
        cases.append((label, overrides))
    return cases


def _expand_cases(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any], list[int] | None, dict[str, Any]]]:
    independent = _independent_sweep(spec)
    explicit = spec.get("sweep_cases")
    if explicit is None:
        explicit = [{"name": "base", "overrides": {}, "ranks": None}]
    if not isinstance(explicit, list) or not explicit:
        raise ValueError("'sweep_cases' must be a nonempty list")

    expanded = []
    for case in explicit:
        if not isinstance(case, dict):
            raise ValueError("each sweep case must be an object")
        case_name = str(case.get("name", "case"))
        case_overrides = case.get("overrides", {})
        if not isinstance(case_overrides, dict):
            raise ValueError(f"overrides for sweep case {case_name!r} must be an object")
        case_ranks = case.get("ranks")
        if case_ranks is not None:
            if not isinstance(case_ranks, list) or not all(isinstance(rank, int) for rank in case_ranks):
                raise ValueError(f"ranks for sweep case {case_name!r} must be a list of integers")
            case_ranks = list(case_ranks)
        for independent_name, independent_overrides in independent:
            overrides = dict(independent_overrides)
            overrides.update(case_overrides)
            suffix = f"_{independent_name}" if independent_name else ""
            name = f"{case_name}{suffix}"
            effective = _apply_overrides(spec, overrides)
            effective["case_name"] = name
            expanded.append((name, effective, case_ranks, overrides))
    return expanded


def _load_spec(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict):
        raise ValueError("benchmark spec must contain a JSON object")
    if "target" not in spec:
        raise ValueError("benchmark spec requires a 'target' field")
    return spec


def _load_target(target: str) -> tuple[Callable[..., Any], Callable[..., Any] | None]:
    if ":" not in target:
        raise ValueError("target must have the form 'module:function'")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    target_fn = getattr(module, function_name)
    prepare = getattr(module, f"{function_name}_prepare", None)
    run = getattr(module, f"{function_name}_run", None)
    if prepare is None or run is None:
        prepare = getattr(module, "prepare", None)
        run = getattr(module, "run", None)
    return target_fn, (prepare, run) if prepare and run else None


def _validate_rank_counts(spec: dict[str, Any], ranks: list[int]) -> None:
    config = spec.get("config", {})
    if "n_sub_x" not in config or "n_sub_y" not in config:
        return
    n_sub_x = int(config["n_sub_x"])
    n_sub_y = int(config["n_sub_y"])
    if n_sub_x < 1 or n_sub_y < 1:
        raise ValueError("n_sub_x and n_sub_y must be positive")
    n_sub = n_sub_x * n_sub_y
    invalid = [rank for rank in ranks if rank > n_sub or n_sub % rank != 0]
    if invalid:
        raise ValueError(
            f"rank counts {invalid} are invalid for {n_sub_x}x{n_sub_y} "
            f"({n_sub} subdomains); ranks must divide the subdomain count"
        )


def _configure_backend(name: str, threads: int, seed: int | None) -> Any:
    from dd_nm_rom import backend as bkd

    if name == "numpy":
        bkd.set_backend("numpy")
        bkd.set_device("cpu", nb_threads=threads)
        bkd.set_floatx("float64")
        bkd.set_seed(seed)
    else:
        device = "cuda" if name == "torch_gpu" else "cpu"
        if name == "torch_gpu":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch_gpu benchmark requested, but CUDA is unavailable")
        bkd.set(
            backend="torch",
            device=device,
            nb_threads=threads,
            floatx="float64",
            seed=seed,
        )
    return bkd


def _synchronize(bkd: Any, device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    bkd.barrier()


def _run_worker(spec: dict[str, Any], backend: str, rank_count: int) -> dict[str, Any] | None:
    _validate_rank_counts(spec, [rank_count])
    if backend == "numpy" and rank_count > 1:
        raise ValueError("parallel benchmarks require torch_cpu or torch_gpu; NumPy is serial-only")
    config = dict(spec.get("config", {}))
    config.update({"backend": backend, "ranks": rank_count})
    threads = int(spec.get("threads", config.get("threads", 1)))
    seed = spec.get("seed", config.get("seed"))
    device = "cuda" if backend == "torch_gpu" else "cpu"
    bkd = _configure_backend(backend, threads, seed)

    try:
        rank = bkd.get_rank()
        target, protocol = _load_target(spec["target"])
        if protocol:
            prepare, run = protocol
            state = prepare(config)
            invoke = lambda: run(state)
        else:
            state = None
            invoke = lambda: target(config)

        warmups = int(spec.get("warmups", 1))
        repetitions = int(spec.get("repetitions", 3))
        if warmups < 0 or repetitions < 1:
            raise ValueError("warmups must be nonnegative and repetitions must be positive")

        for _ in range(warmups):
            invoke()
        timings = []
        last_metrics: Any = None
        for _ in range(repetitions):
            _synchronize(bkd, device)
            start = time.perf_counter()
            last_metrics = invoke()
            _synchronize(bkd, device)
            timings.append(time.perf_counter() - start)

        record = {
            "rank": rank,
            "ranks": bkd.get_nranks(),
            "timings_seconds": timings,
            "metrics": last_metrics if isinstance(last_metrics, dict) else {},
        }
        if bkd.get_nranks() > 1:
            records = bkd._COMM.gather(record, root=0)
            if rank != 0:
                return None
        else:
            records = [record]

        all_times = [value for item in records for value in item["timings_seconds"]]
        critical_path = [max(item["timings_seconds"][i] for item in records)
                         for i in range(repetitions)]
        return {
            "backend": backend,
            "device": device,
            "ranks": bkd.get_nranks(),
            "target": spec["target"],
            "subdomains": {
                "x": config.get("n_sub_x"),
                "y": config.get("n_sub_y"),
                "total": config.get("n_sub_x", 1) * config.get("n_sub_y", 1),
            },
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "timings_seconds": critical_path,
            "summary": {
                "min": min(all_times),
                "max": max(all_times),
                "mean": sum(all_times) / len(all_times),
                "critical_path_mean": sum(critical_path) / len(critical_path),
            },
            "solver_metrics": records[0].get("metrics", {}),
            "rank_records": records,
        }
    finally:
        if backend != "numpy":
            bkd.finalize_distributed()


def _base_command(
    args: argparse.Namespace,
    spec_path: Path,
    rank: int,
    output: Path | None = None,
    case_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        args.python,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--spec",
        str(spec_path.resolve()),
        "--backend",
        args.backend,
        "--rank-count",
        str(rank),
    ]
    if output is not None:
        command += ["--output", str(output)]
    if args.threads is not None:
        command += ["--threads", str(args.threads)]
    if case_name is not None:
        command += ["--case-name", case_name]
    if overrides:
        command += ["--case-overrides", json.dumps(overrides, separators=(",", ":"))]
    return command


def _resolve_env_profile(configured: Path | None) -> Path | None:
    """Resolve the optional site module/runtime profile for batch jobs."""
    if configured is not None:
        profile = configured.expanduser().resolve()
        if not profile.is_file():
            raise ValueError(f"benchmark environment profile does not exist: {profile}")
        return profile

    configured_by_env = os.environ.get("DDNMROM_BENCHMARK_ENV_PROFILE")
    if configured_by_env:
        profile = Path(configured_by_env).expanduser().resolve()
        if not profile.is_file():
            raise ValueError(f"benchmark environment profile does not exist: {profile}")
        return profile

    system_name = {
        "toss_4_x86_64_ib": "toss",
        "toss_4_x86_64_ib_cray": "tuo",
        "blueos_3_ppc64le_ib_p9": "coral",
    }.get(os.environ.get("SYS_TYPE"))
    if system_name is None:
        return None
    profile = ENV_PROFILE_DIR / f"{system_name}.bash"
    return profile if profile.is_file() else None


def _scheduler_script(
    args: argparse.Namespace,
    spec_path: Path,
    rank: int,
    case_name: str,
    overrides: dict[str, Any],
    output: Path,
) -> str:
    from dd_nm_rom import jobs

    tag = f"{spec_path.stem}_{_value_slug(case_name)}_{args.backend}_{rank}ranks"
    if args.launcher == "slurm":
        header = jobs.generate_batch_stub_slurm(
            tag, queue=args.queue, nodes=args.nodes,
            walltime=args.walltime, account=args.account,
            ntasks=rank,
            cpus_per_task=args.cpus_per_task,
            gpus_per_task=args.gpus_per_task or None,
        )
        launch = ["srun", "--ntasks", str(rank)]
        if args.cpus_per_task:
            launch += ["--cpus-per-task", str(args.cpus_per_task)]
        if args.gpus_per_task:
            launch += ["--gpus-per-task", str(args.gpus_per_task)]
    else:
        flux_cpus = 1
        flux_gpus = 1
        header = jobs.generate_batch_stub_flux(
            tag, queue=args.queue, nodes=args.nodes,
            walltime=args.walltime, account=args.account,
            nslots=rank,
            cores_per_slot=flux_cpus,
            gpus_per_slot=flux_gpus,
            exclusive=True,
        )
        launch = [
            "flux", "run", "-N", str(args.nodes), "-x",
            "-n", str(rank), "-g", str(flux_gpus), "-c", str(flux_cpus),
            "-vvv", "--setopt=mpibind=verbose:1",
        ]
    command = launch + _base_command(
        args, spec_path, rank, output, case_name=case_name, overrides=overrides
    )
    profile = _resolve_env_profile(args.env_profile)
    setup = f"source {shlex.quote(str(profile))}\n" if profile else ""
    return (
        "#!/bin/bash\n" + header + "\nset -euo pipefail\n" + setup
        + " ".join(shlex.quote(x) for x in command) + "\n"
    )


def _write_result(path: Path, spec: dict[str, Any], result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec": spec,
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": _git_commit(),
        },
        "result": result,
    }
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, default=str)


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _result_path(
    args: argparse.Namespace,
    spec_path: Path,
    spec: dict[str, Any],
    case_name: str,
    rank: int,
    total_jobs: int,
) -> Path:
    configured = args.output or spec.get("output")
    if total_jobs == 1 and configured:
        return Path(configured)
    stem = Path(configured).stem if configured else spec_path.stem
    case_slug = _value_slug(case_name)
    return args.output_dir / f"{stem}_{case_slug}_{args.backend}_{rank}ranks.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="torch_cpu")
    parser.add_argument("--ranks", type=_parse_ranks, default=[1])
    parser.add_argument("--launcher", choices=LAUNCHERS, default="local")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_jobs"))
    parser.add_argument("--submit", action="store_true", help="Submit generated scheduler scripts")
    parser.add_argument(
        "--python",
        default=".venv/bin/python",
        help="Python executable for benchmark workers (default: .venv/bin/python)",
    )
    parser.add_argument(
        "--env-profile",
        type=Path,
        help=(
            "Shell profile to source in scheduler jobs; otherwise use "
            "DDNMROM_BENCHMARK_ENV_PROFILE or a SYS_TYPE-matched profile"
        ),
    )
    parser.add_argument(
        "--mpi-launcher",
        default="mpiexec -n {ranks}",
        help="Local MPI launcher command template; {ranks} is replaced with the rank count",
    )
    parser.add_argument("--rank-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-name", help=argparse.SUPPRESS)
    parser.add_argument("--case-overrides", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--queue", default="pbatch")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--walltime", default="01:00:00")
    parser.add_argument("--account", default="")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--gpus-per-task", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_spec = _load_spec(args.spec)
    if args.worker and (args.rank_count is None or args.rank_count < 1):
        raise SystemExit("worker mode requires a positive --rank-count")

    if args.worker:
        overrides = {}
        if args.case_overrides:
            try:
                overrides = json.loads(args.case_overrides)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid --case-overrides JSON: {exc}") from exc
            if not isinstance(overrides, dict):
                raise SystemExit("--case-overrides must contain a JSON object")
        spec = _apply_overrides(base_spec, overrides)
        spec["case_name"] = args.case_name or spec.get("name", "base")
        if args.threads is not None:
            spec = dict(spec)
            spec["threads"] = args.threads
        result = _run_worker(spec, args.backend, args.rank_count)
        if result is not None:
            output = args.output or spec.get("output")
            if output:
                _write_result(Path(output), spec, result)
            print(json.dumps(result, indent=2, default=str))
        return 0

    spec = copy.deepcopy(base_spec)
    if args.threads is not None:
        spec["threads"] = args.threads
    cases = _expand_cases(spec)
    prepared_cases = []
    for case_name, effective_spec, case_ranks, overrides in cases:
        ranks = case_ranks if case_ranks is not None else args.ranks
        if not ranks:
            raise ValueError(f"no ranks configured for sweep case {case_name!r}")
        _validate_rank_counts(effective_spec, ranks)
        if args.backend == "numpy" and any(rank > 1 for rank in ranks):
            raise ValueError(
                f"sweep case {case_name!r} requests parallel NumPy ranks; "
                "use torch_cpu or torch_gpu for parallel benchmarks"
            )
        prepared_cases.append((case_name, effective_spec, ranks, overrides))
    total_jobs = sum(len(case[2]) for case in prepared_cases)

    if args.launcher == "local":
        for case_name, effective_spec, ranks, overrides in prepared_cases:
            for rank in ranks:
                output = _result_path(
                    args, args.spec, effective_spec, case_name, rank, total_jobs
                )
                if rank == 1:
                    result = _run_worker(effective_spec, args.backend, 1)
                    if result is not None:
                        if output:
                            _write_result(output, effective_spec, result)
                        print(json.dumps(result, indent=2, default=str))
                    continue

                command = args.mpi_launcher.format(ranks=rank)
                command = shlex.split(command) + _base_command(
                    args, args.spec, rank, output,
                    case_name=case_name, overrides=overrides,
                )
                subprocess.run(command, check=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = []
    for case_name, effective_spec, ranks, overrides in prepared_cases:
        for rank in ranks:
            output = _result_path(
                args, args.spec, effective_spec, case_name, rank, total_jobs
            )
            script_stem = effective_spec.get("name", args.spec.stem)
            script_path = args.output_dir / (
                f"{script_stem}_{_value_slug(case_name)}_{args.backend}_{rank}ranks.sh"
            )
            script_path.write_text(_scheduler_script(
                args, args.spec, rank, case_name, overrides, output
            ))
            script_path.chmod(0o750)
            scripts.append(script_path)
            if args.submit:
                submit = ["sbatch", str(script_path)] if args.launcher == "slurm" else ["flux", "batch", "--flags", "waitable", str(script_path)]
                subprocess.run(submit, check=True)
    for script in scripts:
        print(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
