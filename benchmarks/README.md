# Benchmark helper

`benchmark.py` provides a small launcher for serial workloads and for
SLURM/FLUX batch-script generation. Use the repository virtual environment:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_dense_linear.json --backend numpy
```

A workload is either a callable accepting a configuration dictionary, or a
module exposing `prepare(config)` and `run(state)`. Keep construction and
loading in `prepare`; `run` should perform exactly one operation to measure.

For scheduler scripts, generation is side-effect free unless `--submit` is
given:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_device.json --backend torch_gpu \
  --launcher slurm --ranks 1,2,4 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --gpus-per-task 1 \
  --output-dir benchmark_jobs
```

The scheduler resource options are available for both SLURM and FLUX:

```text
# SLURM
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json \
  --backend torch_cpu --launcher slurm \
  --ranks 1,2,4,8,16 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --output-dir benchmark_jobs/slurm

# FLUX
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json \
  --backend torch_gpu --launcher flux \
  --ranks 1,2,4,8,16 \
  --account asccasc --queue pbatch --walltime 01:00:00 \
  --output-dir benchmark_jobs/flux
```

These options populate the account, queue, and walltime directives in the
generated batch scripts. For SLURM, `--cpus-per-task` and
`--gpus-per-task` control the allocation and `srun` task. FLUX follows this
convention by default: `-N <nodes> -x -n <tasks> -g 1 -c 1
-vvv --setopt=mpibind=verbose:1`. Add `--submit` to submit the generated
scripts.

Scheduler jobs can source a site-specific module and runtime profile:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json \
  --backend torch_gpu --launcher flux --ranks 1,2,4 \
  --env-profile benchmarks/env/tuo.bash \
  --python .venv/bin/python
```

If `--env-profile` is omitted, the runner checks
`DDNMROM_BENCHMARK_ENV_PROFILE`, then selects a profile from `SYS_TYPE` when a
matching file exists. Profiles only load modules and export runtime settings;
the benchmark is still launched once per Flux task. The default worker
interpreter is `.venv/bin/python` and can be replaced with `--python`.

When already inside an allocation, use the local scheduler mode with the
allocation's task launcher. `auto` detects Flux or Slurm and otherwise retains
the traditional `mpiexec` behavior:

```text
# Inside Flux
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json --backend torch_gpu \
  --launcher local --local-launcher flux --ranks 1,2,4

# Inside Slurm
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady.json --backend torch_gpu \
  --launcher local --local-launcher srun --ranks 1,2,4
```

Use `--local-launcher mpiexec` to force the existing MPI launcher, or leave
the option at `auto` to select Flux, Slurm, or `mpiexec` from the allocation
environment. Any selected environment profile is sourced before the local
task launcher runs.

Benchmark result records include a `configuration` section containing every
registered `DDNMROM_*` option, its effective value, default, and whether it
came from the environment. Unregistered `DDNMROM_*` variables are included
under `extra_environment`; a stable configuration fingerprint is also
recorded. Parallel results report whether the configuration was consistent
across ranks.

Project DD-FOM/DD-ROM workloads can be added as separate modules using the
same protocol. This keeps benchmark timing independent of pytest setup,
assertions, and profiler output. The rank count must divide the workload's
subdomain count, and GPU parallel jobs should allocate one visible GPU per
rank.

The included DD workloads are in `workloads_dd.py`. They provide
`prepare/run` hooks for the three regression-derived specs and return
`converged`, `newton_iterations`, `residual_norm`, and `internal_timing`
metrics. These appear both in each rank record and in the top-level
`solver_metrics` result field.

Use one spec per decomposition size. Change `n_sub_x` and `n_sub_y` in a
copied spec, then provide only rank counts that divide the total number of
subdomains:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/dd_fom_steady_8x8.json \
  --backend torch_cpu \
  --launcher slurm \
  --ranks 1,2,4,8,16,32,64
```

Full DD-FOM/DD-ROM solves default to one measured repetition because solver
state may be mutated by a solve.

Backend microbenchmarks
-----------------------

`workloads_backend.py` covers operations that are otherwise hidden inside
solver timings:

* `backend_dense_linear.json` measures dense matrix-matrix and matrix-vector
  products through `backend.to_backend`.
* `backend_sparse_linear.json` measures sparse matrix-vector multiplication
  through `backend.to_sp_backend`.
* `backend_sparse_assembly.json` measures `speye` and `torch_bmat_new` on the
  Torch CPU backend.
* `backend_device.json` measures a device matmul and activation. Run it with
  `--backend torch_gpu` to measure GPU kernels; the runner synchronizes CUDA
  before and after every measured repetition.
* `backend_collective.json` expands into timing cases for the backend barrier,
  broadcast, gather, scatter, and local-size helpers. Use Torch with a batch
  launcher and rank counts greater than one for meaningful MPI timings.

For example, serial CPU coverage is:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_dense_linear.json --backend numpy
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_sparse_linear.json --backend torch_cpu
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_sparse_assembly.json --backend torch_cpu
```

GPU and MPI measurements are opt-in and should be launched only on matching
hardware. GPU-aware MPI paths in `gatherv_tensor` are selected by setting
`DDNMROM_MPI_GPU_AWARE=1` in the scheduler environment and using
`--backend torch_gpu`. The benchmark records the operation and result shape,
while the runner records rank-level timings and the active backend/device.

Adding a benchmark
------------------

Add a workload module with a `prepare(config)`/`run(state)` pair, keeping
allocation and setup in `prepare` so they are excluded from the measured
interval. Then add a JSON spec such as this dense backend-operation example:

```json
{
  "name": "backend_dense_linear",
  "target": "benchmarks.workloads_backend:backend_dense_linear",
  "warmups": 2,
  "repetitions": 5,
  "threads": 1,
  "config": {
    "rows": 1024,
    "inner": 1024,
    "cols": 1024,
    "seed": 0
  }
}
```

Run it against the desired backend:

```text
.venv/bin/python benchmarks/benchmark.py \
  --spec benchmarks/specs/backend_dense_linear.json --backend numpy
```

The same spec can be compared with `torch_cpu`, or with `torch_gpu` on GPU
hardware. The target may also be a single callable accepting the config
dictionary when no separate setup phase is needed.

Generic sweeps are supported without creating temporary spec files. Independent
axes use a Cartesian product:

```json
"sweep": {
  "config.size": [256, 512, 1024],
  "threads": [1, 8]
}
```

This creates six cases. Use `sweep_cases` when values must stay paired, such as
square DD decompositions:

```json
"sweep_cases": [
  {
    "name": "4x4",
    "overrides": {
      "config.n_sub_x": 4,
      "config.n_sub_y": 4
    },
    "ranks": [1, 2, 4, 8, 16]
  },
  {
    "name": "8x8",
    "overrides": {
      "config.n_sub_x": 8,
      "config.n_sub_y": 8
    },
    "ranks": [1, 2, 4, 8, 16, 32, 64]
  }
]
```

Each case/rank combination becomes one local execution or scheduler script.
Rank counts are validated separately for every effective configuration, and
case overrides are passed to workers in memory rather than materialized as
additional JSON files.
