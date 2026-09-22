"""Microbenchmarks for operations implemented by :mod:`dd_nm_rom.backend`.

The workloads deliberately keep allocation and conversion in ``prepare``.
Only the backend operation being measured is performed in ``run``.  The
collective workload is safe in serial mode, but is intended to be launched
with the benchmark runner's MPI/scheduler options when measuring communication.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as scipy_sparse


def _backend():
    from dd_nm_rom import backend as bkd

    return bkd


def _norm(value: Any) -> float:
    bkd = _backend()
    return float(np.linalg.norm(bkd.to_numpy(value)))


def backend_dense_linear_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare dense matrix/vector operands for backend linear algebra."""
    bkd = _backend()
    rows = int(config.get("rows", 1024))
    inner = int(config.get("inner", 1024))
    cols = int(config.get("cols", 1024))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    matrix = bkd.to_backend(rng.standard_normal((rows, inner)))
    right = bkd.to_backend(rng.standard_normal((inner, cols)))
    vector = bkd.to_backend(rng.standard_normal(inner))
    return {"matrix": matrix, "right": right, "vector": vector}


def backend_dense_linear_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure dense matrix-matrix and matrix-vector products."""
    matrix = state["matrix"]
    product = matrix @ state["right"]
    vector_product = matrix @ state["vector"]
    return {
        "matrix_product_shape": list(product.shape),
        "matrix_product_norm": _norm(product),
        "vector_product_norm": _norm(vector_product),
    }


def backend_dense_linear(config: dict[str, Any]) -> dict[str, Any]:
    return backend_dense_linear_run(backend_dense_linear_prepare(config))


def backend_sparse_linear_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare a sparse matrix and vector for backend sparse matvec."""
    bkd = _backend()
    size = int(config.get("size", 2048))
    diagonals = int(config.get("diagonals", 5))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    offsets = np.arange(-(diagonals // 2), diagonals // 2 + 1)
    values = rng.standard_normal((len(offsets), size))
    matrix = scipy_sparse.diags(values, offsets, shape=(size, size), format="csr")
    return {
        "matrix": bkd.to_sp_backend(matrix),
        "vector": bkd.to_backend(rng.standard_normal(size)),
    }


def backend_sparse_linear_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure sparse matrix-vector multiplication."""
    result = state["matrix"] @ state["vector"]
    return {"output_shape": list(result.shape), "output_norm": _norm(result)}


def backend_sparse_linear(config: dict[str, Any]) -> dict[str, Any]:
    return backend_sparse_linear_run(backend_sparse_linear_prepare(config))


def backend_sparse_assembly_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare sparse blocks for ``torch_bmat_new`` and ``speye``."""
    bkd = _backend()
    size = int(config.get("block_size", 256))
    block = scipy_sparse.diags(
        np.ones(size), offsets=0, shape=(size, size), format="csr"
    )
    block = bkd.to_sp_backend(block)
    return {"blocks": [[block, None], [None, block]], "size": size}


def backend_sparse_assembly_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure backend sparse identity and block-matrix assembly helpers."""
    bkd = _backend()
    identity = bkd.speye(state["size"], format="csr")
    assembled = bkd.torch_bmat_new(state["blocks"], format="csr")
    return {
        "identity_shape": list(identity.shape),
        "assembled_shape": list(assembled.shape),
        "assembled_nnz": int(assembled._nnz()),
    }


def backend_sparse_assembly(config: dict[str, Any]) -> dict[str, Any]:
    return backend_sparse_assembly_run(backend_sparse_assembly_prepare(config))


def backend_device_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare device-resident operands for a Torch CPU/GPU kernel."""
    bkd = _backend()
    size = int(config.get("size", 2048))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    return {
        "left": bkd.to_backend(rng.standard_normal((size, size))),
        "right": bkd.to_backend(rng.standard_normal((size, size))),
    }


def backend_device_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure a device matmul followed by an elementwise activation."""
    result = state["left"] @ state["right"]
    # Keep the operation backend-native without importing torch in NumPy runs.
    result = result.clip(min=0)
    return {"output_shape": list(result.shape), "output_norm": _norm(result)}


def backend_device(config: dict[str, Any]) -> dict[str, Any]:
    return backend_device_run(backend_device_prepare(config))


def backend_collective_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare rank-local data for a backend collective operation."""
    bkd = _backend()
    rank = bkd.get_rank()
    nranks = bkd.get_nranks()
    rows = int(config.get("rows", 256))
    cols = int(config.get("cols", 256))
    rng = np.random.default_rng(int(config.get("seed", 0)) + rank)
    operation = config.get("operation", "barrier")
    local_rows = rows + rank if operation == "gatherv_tensor" else rows
    local = bkd.to_backend(rng.standard_normal((local_rows, cols)))
    root_data = None
    if rank == 0:
        root_data = bkd.to_backend(rng.standard_normal((rows * nranks, cols)))
    return {"local": local, "root_data": root_data, "operation": operation}


def backend_collective_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure one backend collective, including GPU-aware MPI paths."""
    bkd = _backend()
    operation = state["operation"]
    local = state["local"]
    if operation == "barrier":
        bkd.barrier()
        result = local
    elif operation == "bcast":
        result = bkd.bcast(1.0 if bkd.root() else None)
    elif operation == "get_local_sizes_all":
        result = bkd.get_local_sizes_all(local)
    elif operation == "gather_tensor":
        result = bkd.gather_tensor(local)
    elif operation == "gatherv_tensor":
        result = bkd.gatherv_tensor(local)
    elif operation == "scatter_tensor":
        result = bkd.scatter_tensor(state["root_data"])
    elif operation == "broadcast_tensor":
        result = bkd.broadcast_tensor(state["root_data"] if bkd.root() else None)
    else:
        raise ValueError(f"unknown backend collective: {operation!r}")
    return {
        "operation": operation,
        "result_type": type(result).__name__,
        "result_shape": list(result.shape) if hasattr(result, "shape") else None,
    }


def backend_collective(config: dict[str, Any]) -> dict[str, Any]:
    return backend_collective_run(backend_collective_prepare(config))
