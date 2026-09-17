"""Small reference workloads for :mod:`benchmarks.benchmark`.

Project-specific workloads can follow the same protocol without importing the
launcher: ``prepare(config)`` builds a state object and ``run(state)`` performs
one measured operation.
"""

from __future__ import annotations

import numpy as np


def matmul_prepare(config: dict) -> dict:
    """Build a deterministic NumPy matrix-multiplication benchmark state."""
    size = int(config.get("size", 512))
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    return {
        "a": rng.standard_normal((size, size)),
        "b": rng.standard_normal((size, size)),
    }


def matmul_run(state: dict) -> dict:
    """Run one matrix multiplication and return lightweight metadata."""
    result = state["a"] @ state["b"]
    return {"output_shape": list(result.shape), "output_norm": float(np.linalg.norm(result))}


def matmul(config: dict) -> dict:
    """Convenience workload using the ``prepare/run`` protocol."""
    return matmul_run(matmul_prepare(config))

