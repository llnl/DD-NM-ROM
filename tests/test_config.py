"""Tests for DDNM-ROM environment configuration snapshots."""

from dd_nm_rom import config


def test_config_snapshot_reports_defaults_and_extra_environment(monkeypatch):
  monkeypatch.delenv("DDNMROM_VERBOSE", raising=False)
  for var in (
    "DDNMROM_ACT_COMPILE",
    "DDNMROM_ACT_COMPILE_ENCODER",
    "DDNMROM_ACT_COMPILE_DECODER",
    "DDNMROM_ACT_COMPILE_FORWARD",
    "DDNMROM_ACT_COMPILE_JAC",
    "DDNMROM_ACT_COMPILE_ACTIVATIONS",
    "DDNMROM_ACT_COMPILE_MIXED_POLICY",
    "DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE",
    "DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY",
    "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_WARMUPS",
    "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_REPETITIONS",
    "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_BATCH_SIZE",
    "DDNMROM_ACT_COMPILE_MIXED_SAMPLE_MAX_SPREAD",
  ):
    monkeypatch.delenv(var, raising=False)
  monkeypatch.setenv("DDNMROM_BENCHMARK_ENV_PROFILE", "benchmarks/env/tuo.bash")

  snapshot = config.get_config_snapshot()

  verbose = snapshot["registered"]["DDNMROM_VERBOSE"]
  assert verbose["value"] == 0
  assert verbose["default"] == 0
  assert verbose["source"] == "default"
  assert verbose["environment"] is None
  registered = snapshot["registered"]
  assert registered["DDNMROM_ACT_COMPILE_ENCODER"]["value"] is False
  assert registered["DDNMROM_ACT_COMPILE_DECODER"]["value"] is True
  assert registered["DDNMROM_ACT_COMPILE_FORWARD"]["value"] is False
  assert registered["DDNMROM_ACT_COMPILE_JAC"]["value"] is True
  assert registered["DDNMROM_ACT_COMPILE_ACTIVATIONS"]["value"] == "all"
  assert registered["DDNMROM_ACT_COMPILE_MIXED_POLICY"]["value"] == "all"
  assert registered["DDNMROM_ACT_COMPILE_MIXED_CACHE_MAXSIZE"]["value"] == 32
  assert registered["DDNMROM_ACT_COMPILE_MIXED_MIN_FREQUENCY"]["value"] == 1
  assert registered["DDNMROM_ACT_COMPILE_MIXED_SAMPLE_WARMUPS"]["value"] == 2
  assert registered["DDNMROM_ACT_COMPILE_MIXED_SAMPLE_REPETITIONS"]["value"] == 8
  assert registered["DDNMROM_ACT_COMPILE_MIXED_SAMPLE_BATCH_SIZE"]["value"] == 4
  assert registered["DDNMROM_ACT_COMPILE_MIXED_SAMPLE_MAX_SPREAD"]["value"] == 4.0
  assert snapshot["extra_environment"]["DDNMROM_BENCHMARK_ENV_PROFILE"] == \
    "benchmarks/env/tuo.bash"
  assert snapshot["fingerprint"].startswith("sha256:")


def test_config_snapshot_converts_environment_values(monkeypatch):
  monkeypatch.setenv("DDNMROM_VERBOSE", "1")
  monkeypatch.setenv("DDNMROM_ACT_COMPILE", "false")
  monkeypatch.setenv("DDNMROM_DEVICE_PER_NODE", "8")

  snapshot = config.get_config_snapshot()["registered"]

  assert snapshot["DDNMROM_VERBOSE"]["value"] == 1
  assert snapshot["DDNMROM_VERBOSE"]["source"] == "environment"
  assert snapshot["DDNMROM_ACT_COMPILE"]["value"] is False
  assert snapshot["DDNMROM_DEVICE_PER_NODE"]["value"] == 8
