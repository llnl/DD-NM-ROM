"""Tests for benchmark result analysis."""

import json

from benchmarks.analyze import add_scaling_columns, load_rows


def _write_result(path, ranks, timing):
  path.write_text(json.dumps({
    "spec": {
      "name": "dd_fom",
      "target": "benchmarks.workloads_dd:dd_fom_steady",
      "analysis": {"study": "strong"},
      "config": {"n_sub_x": 4, "n_sub_y": 4, "nx_intr": 3, "ny_intr": 3},
    },
    "result": {
      "backend": "torch",
      "device": "cpu",
      "ranks": ranks,
      "subdomains": {"x": 4, "y": 4, "total": 16},
      "summary": {"critical_path_mean": timing, "min": timing, "max": timing},
      "solver_metrics": {"newton_iterations": 5},
    },
  }))


def test_load_rows_and_scaling_columns(tmp_path):
  _write_result(tmp_path / "rank1.json", 1, 10.0)
  _write_result(tmp_path / "rank2.json", 2, 6.0)

  rows = load_rows([], tmp_path)
  add_scaling_columns(rows)

  by_rank = {row["ranks"]: row for row in rows}
  assert by_rank[1]["timing_mean"] == 10
  assert by_rank[2]["speedup"] == 10 / 6
  assert by_rank[2]["efficiency"] == (10 / 6) / 2
  assert by_rank[1]["solver_metrics.newton_iterations"] == 5
