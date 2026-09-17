"""Tests for benchmark launcher command construction."""

from argparse import Namespace

import pytest

from benchmarks import benchmark


def _args(**overrides):
  values = {
    "cpus_per_task": 8,
    "gpus_per_task": 1,
    "mpi_launcher": "mpiexec -n {ranks}",
  }
  values.update(overrides)
  return Namespace(**values)


def test_local_launcher_auto_detects_flux(monkeypatch):
  monkeypatch.setenv("FLUX_JOB_SIZE", "4")
  monkeypatch.delenv("SLURM_JOB_ID", raising=False)

  assert benchmark._select_local_launcher("auto") == "flux"


def test_local_launcher_auto_detects_slurm(monkeypatch):
  monkeypatch.delenv("FLUX_JOB_SIZE", raising=False)
  monkeypatch.setenv("SLURM_JOB_ID", "123")

  assert benchmark._select_local_launcher("auto") == "srun"


def test_local_launcher_falls_back_to_mpiexec(monkeypatch):
  monkeypatch.delenv("FLUX_JOB_SIZE", raising=False)
  monkeypatch.delenv("SLURM_JOB_ID", raising=False)

  assert benchmark._select_local_launcher("auto") == "mpiexec"


def test_local_task_commands():
  args = _args()
  assert benchmark._local_task_command(args, "flux", 4) == [
    "flux", "run", "-n", "4", "-g", "1", "-c", "1",
    "-vvv", "--setopt=mpibind=verbose:1",
  ]
  assert benchmark._local_task_command(args, "srun", 4) == [
    "srun", "-n", "4", "--cpus-per-task", "8",
    "--gpus-per-task", "1", "--gpu-bind=closest",
  ]
  assert benchmark._local_task_command(args, "mpiexec", 4) == [
    "mpiexec", "-n", "4",
  ]


def test_explicit_scheduler_launcher_requires_allocation(monkeypatch):
  monkeypatch.delenv("FLUX_JOB_SIZE", raising=False)
  monkeypatch.delenv("FLUX_JOB_ID", raising=False)
  monkeypatch.delenv("FLUX_URI", raising=False)

  with pytest.raises(ValueError, match="active Flux allocation"):
    benchmark._select_local_launcher("flux")
