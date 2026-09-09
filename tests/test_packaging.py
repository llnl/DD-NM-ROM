"""Checks for the distributable Python package."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


pytestmark = [pytest.mark.no_backend, pytest.mark.packaging]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_build_contains_package_and_metadata(tmp_path):
  """Build the project wheel without resolving dependencies and inspect it."""
  project_dir = tmp_path / "project"
  project_dir.mkdir()

  for filename in ("pyproject.toml", "README.md", "LICENSE"):
    shutil.copy2(PROJECT_ROOT / filename, project_dir / filename)
  shutil.copytree(PROJECT_ROOT / "dd_nm_rom", project_dir / "dd_nm_rom")

  wheel_dir = tmp_path / "wheel"
  result = subprocess.run(
    [
      sys.executable,
      "-m",
      "pip",
      "wheel",
      "--no-deps",
      "--no-build-isolation",
      "--wheel-dir",
      str(wheel_dir),
      str(project_dir),
    ],
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stdout + result.stderr

  wheels = list(wheel_dir.glob("dd_nm_rom-*.whl"))
  assert len(wheels) == 1

  with zipfile.ZipFile(wheels[0]) as wheel:
    contents = set(wheel.namelist())
    metadata_path = next(path for path in contents if path.endswith(".dist-info/METADATA"))
    metadata = wheel.read(metadata_path).decode()

  package_root = project_dir / "dd_nm_rom"
  ignored_source_dir = package_root / "rom" / "tests"
  source_files = {
    path.relative_to(project_dir).as_posix()
    for path in package_root.rglob("*.py")
    if ignored_source_dir not in path.parents
  }
  missing_sources = source_files - contents

  assert "Name: dd-nm-rom" in metadata
  assert "Version: 0.1.0" in metadata
  assert not missing_sources, f"wheel is missing source files: {sorted(missing_sources)}"
  assert any(path.endswith(".dist-info/RECORD") for path in contents)
