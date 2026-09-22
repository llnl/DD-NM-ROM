#!/usr/bin/env python3
"""Summarize and plot benchmark result JSON files.

Examples:
  .venv/bin/python benchmarks/analyze.py summarize \
      --input-dir benchmark_jobs/flux \
      --output benchmark_results/dd_fom.csv --format csv --study strong
  .venv/bin/python benchmarks/analyze.py plot \
      --table benchmark_results/dd_fom.csv \
      --output-dir benchmark_results/plots --study strong --format png,pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _as_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _slug(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "benchmark"


def _flatten_metrics(value: Any, prefix: str = "solver_metrics") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not isinstance(value, dict):
        return flattened
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, dict):
            flattened.update(_flatten_metrics(item, name))
        else:
            flattened[name] = item
    return flattened


def _configuration_fingerprint(spec: dict[str, Any], result: dict[str, Any]) -> str:
    configuration = result.get("configuration", {})
    fingerprint = configuration.get("fingerprint") if isinstance(configuration, dict) else None
    if fingerprint:
        return str(fingerprint)
    encoded = json.dumps(
        {
            "target": spec.get("target"),
            "config": spec.get("config", {}),
            "backend": result.get("backend"),
            "device": result.get("device"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _infer_study(spec: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    analysis = spec.get("analysis", {})
    if isinstance(analysis, dict) and analysis.get("study") in ("strong", "weak"):
        return str(analysis["study"])
    return "custom"


def _result_row(path: Path, payload: dict[str, Any], requested_study: str) -> dict[str, Any]:
    spec = payload.get("spec", {})
    result = payload.get("result", {})
    if not isinstance(spec, dict) or not isinstance(result, dict):
        raise ValueError(f"result file must contain object-valued spec/result: {path}")

    subdomains = result.get("subdomains", {})
    if not isinstance(subdomains, dict):
        subdomains = {}
    config = spec.get("config", {})
    if not isinstance(config, dict):
        config = {}
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    timing = summary.get("critical_path_mean", summary.get("mean"))
    row = {
        "source_file": str(path),
        "study": _infer_study(spec, requested_study),
        "case_name": spec.get("case_name", spec.get("name", path.stem)),
        "target": spec.get("target", result.get("target", "unknown")),
        "backend": result.get("backend", "unknown"),
        "device": result.get("device", "unknown"),
        "ranks": _as_number(result.get("ranks")),
        "subdomains_x": _as_number(subdomains.get("x", config.get("n_sub_x"))),
        "subdomains_y": _as_number(subdomains.get("y", config.get("n_sub_y"))),
        "subdomains_total": _as_number(subdomains.get("total")),
        "timing_mean": _as_number(timing),
        "timing_min": _as_number(summary.get("min")),
        "timing_max": _as_number(summary.get("max")),
        "configuration_fingerprint": _configuration_fingerprint(spec, result),
        "config_json": json.dumps(config, sort_keys=True, separators=(",", ":"), default=str),
        "spec_name": spec.get("name", path.stem),
    }
    row.update({key: value for key, value in _flatten_metrics(result.get("solver_metrics", {})).items()})
    row.update({
        "config.nx_intr": config.get("nx_intr"),
        "config.ny_intr": config.get("ny_intr"),
        "config.threads": spec.get("threads", config.get("threads")),
    })
    return row


def _input_paths(input_files: list[Path], input_dir: Path | None) -> list[Path]:
    paths = list(input_files)
    if input_dir is not None:
        paths.extend(sorted(input_dir.rglob("*.json")))
    unique = {path.resolve() for path in paths}
    return sorted(unique)


def load_rows(input_files: list[Path], input_dir: Path | None, study: str = "auto") -> list[dict[str, Any]]:
    rows = []
    paths = _input_paths(input_files, input_dir)
    if not paths:
        raise ValueError("no JSON result files were found")
    for path in paths:
        with path.open() as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or "result" not in payload:
            continue
        rows.append(_result_row(path, payload, study))
    if not rows:
        raise ValueError("no benchmark result records were found")
    return rows


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("study", "")), str(row.get("target", "")),
        str(row.get("backend", "")), str(row.get("device", "")),
        row.get("subdomains_total") or 0, row.get("ranks") or 0,
    )


def _scaling_group(row: dict[str, Any], study: str) -> tuple[Any, ...]:
    common = (
        row.get("study"), row.get("case_name"), row.get("target"),
        row.get("backend"), row.get("device"), row.get("configuration_fingerprint"),
    )
    if study == "weak":
        return common + (row.get("config.nx_intr"), row.get("config.ny_intr"))
    return common + (row.get("subdomains_x"), row.get("subdomains_y"), row.get("config_json"))


def add_scaling_columns(rows: list[dict[str, Any]], baseline_ranks: int | None = None) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("study") in ("strong", "weak"):
            groups[_scaling_group(row, row["study"])].append(row)
    for group_rows in groups.values():
        eligible = [row for row in group_rows if row.get("timing_mean") is not None]
        if not eligible:
            continue
        if baseline_ranks is None:
            baseline = min(eligible, key=lambda row: row.get("ranks") or math.inf)
        else:
            matching = [row for row in eligible if row.get("ranks") == baseline_ranks]
            baseline = matching[0] if matching else min(
                eligible, key=lambda row: abs((row.get("ranks") or 0) - baseline_ranks)
            )
        base_time = float(baseline["timing_mean"])
        base_ranks = float(baseline.get("ranks") or 1)
        for row in group_rows:
            timing = row.get("timing_mean")
            ranks = row.get("ranks")
            if timing is None or not timing:
                continue
            row["normalized_time"] = float(timing) / base_time
            row["speedup"] = base_time / float(timing)
            row["efficiency"] = row["speedup"] / (float(ranks or 1) / base_ranks)


def _write_table(rows: list[dict[str, Any]], output: Path, format_name: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "study", "case_name", "target", "backend", "device", "ranks",
        "subdomains_x", "subdomains_y", "subdomains_total", "timing_mean",
        "timing_min", "timing_max", "normalized_time", "speedup", "efficiency",
        "configuration_fingerprint", "source_file",
    ]
    columns = [key for key in preferred if key in columns] + [
        key for key in columns if key not in preferred
    ]
    if format_name == "json":
        output.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    elif format_name == "csv":
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    elif format_name == "markdown":
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = [str(row.get(key, "")).replace("|", "\\|") for key in columns]
            lines.append("| " + " | ".join(values) + " |")
        output.write_text("\n".join(lines) + "\n")
    else:
        raise ValueError(f"unsupported table format: {format_name}")


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON table must contain a list")
        rows = data
    else:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    numeric_fields = {
        "ranks", "subdomains_x", "subdomains_y", "subdomains_total",
        "timing_mean", "timing_min", "timing_max", "normalized_time",
        "speedup", "efficiency", "config.nx_intr", "config.ny_intr",
        "config.threads",
    }
    for row in rows:
        for field in numeric_fields:
            if field in row:
                row[field] = _as_number(row[field])
    return rows


def _plot_value(row: dict[str, Any], field: str) -> float | None:
    if field.startswith("metric:"):
        field = "solver_metrics." + field.removeprefix("metric:")
    return _as_number(row.get(field))


def plot_table(rows: list[dict[str, Any]], output_dir: Path, study: str, x_field: str,
               y_fields: list[str], formats: list[str]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    if study != "auto":
        rows = [row for row in rows if row.get("study") == study]
    if not rows:
        raise ValueError("no rows match the requested study")
    if x_field == "auto":
        x_field = "ranks" if study == "strong" else "subdomains_total"

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = (row.get("target"), row.get("backend"), row.get("device"), row.get("case_name"))
        groups[group].append(row)

    written: list[Path] = []
    for y_field in y_fields:
        figure, axis = plt.subplots()
        for group, group_rows in groups.items():
            points = []
            for row in group_rows:
                x = _plot_value(row, x_field)
                y = _plot_value(row, y_field)
                if x is not None and y is not None:
                    points.append((x, y))
            if not points:
                continue
            points.sort()
            label = "/".join(str(value) for value in group)
            axis.plot([point[0] for point in points], [point[1] for point in points], "o-", label=label)
        axis.set_xlabel(x_field)
        axis.set_ylabel(y_field)
        axis.grid(True, which="both", alpha=0.3)
        if len(groups) > 1:
            axis.legend()
        figure.tight_layout()
        stem = _slug(f"{study}_{y_field}")
        for format_name in formats:
            path = output_dir / f"{stem}.{format_name}"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            written.append(path)
        plt.close(figure)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", type=Path, action="append", default=[])
    summarize.add_argument("--input-dir", type=Path)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--format", choices=("csv", "markdown", "json"), default="csv")
    summarize.add_argument("--study", choices=("auto", "strong", "weak", "custom"), default="auto")
    summarize.add_argument("--baseline-ranks", type=int)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--table", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--study", choices=("auto", "strong", "weak", "custom"), default="auto")
    plot.add_argument("--x", dest="x_field", default="auto")
    plot.add_argument("--y", dest="y_fields", action="append", default=[])
    plot.add_argument("--format", dest="formats", default="png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        rows = load_rows(args.input, args.input_dir, args.study)
        add_scaling_columns(rows, args.baseline_ranks)
        _write_table(rows, args.output, args.format)
        print(args.output)
        return 0

    rows = _read_table(args.table)
    if args.study != "auto":
        rows = [row for row in rows if row.get("study") == args.study]
    # CSV values are strings, so re-apply the scaling-derived fields when a
    # table was written without them or when a caller supplies a custom table.
    add_scaling_columns(rows)
    studies = {row.get("study") for row in rows}
    plot_study = args.study
    if plot_study == "auto" and len(studies) == 1:
        plot_study = next(iter(studies))
    y_fields = args.y_fields or ["timing_mean"]
    written = plot_table(rows, args.output_dir, plot_study, args.x_field, y_fields,
                         [value.strip() for value in args.formats.split(",") if value.strip()])
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
