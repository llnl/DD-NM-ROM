"""Tests for DDNM-ROM environment configuration snapshots."""

from dd_nm_rom import config


def test_config_snapshot_reports_defaults_and_extra_environment(monkeypatch):
  monkeypatch.delenv("DDNMROM_VERBOSE", raising=False)
  monkeypatch.setenv("DDNMROM_BENCHMARK_ENV_PROFILE", "benchmarks/env/tuo.bash")

  snapshot = config.get_config_snapshot()

  verbose = snapshot["registered"]["DDNMROM_VERBOSE"]
  assert verbose["value"] == 0
  assert verbose["default"] == 0
  assert verbose["source"] == "default"
  assert verbose["environment"] is None
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
