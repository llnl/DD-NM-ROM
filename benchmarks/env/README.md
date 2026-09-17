# Benchmark environment profiles

Environment profiles are sourced by generated SLURM or Flux benchmark jobs.
They contain site-specific module loads and runtime environment variables, but
do not launch Python, Flux, MPI, or `torchrun`.

Select a profile explicitly with `--env-profile benchmarks/env/tuo.bash`, or
set `DDNMROM_BENCHMARK_ENV_PROFILE`. When neither is supplied, the benchmark
runner looks for a profile matching `SYS_TYPE` (`tuo`, `toss`, or `coral`).
Missing automatic profiles are allowed, keeping the runner usable on systems
without repository-local module profiles.

The benchmark worker Python executable is independent of the environment
profile and defaults to `.venv/bin/python`. Override it with:

```text
--python /path/to/other/venv/bin/python
```

Profiles may consume exported variables such as `ROCM_VERSION` and
`MPICH_VERSION` to select site-specific module versions.
