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
    uneven_operation = operation in ("gatherv_tensor", "gatherv_batch")
    local_rows = rows + rank if uneven_operation else rows
    local = None if operation == "gatherv_dd_assembly" else bkd.to_backend(
        rng.standard_normal((local_rows, cols))
    )
    batch_local = []
    if operation == "gatherv_batch":
        batch_local = [
            local,
            bkd.to_backend(rng.standard_normal((local_rows, cols))),
            bkd.to_backend(rng.standard_normal((local_rows, cols))),
        ]

    assembly_fields = {}
    assembly_total_subdomains = 0
    assembly_active_subdomains = 0
    assembly_slot_count = 0
    if operation == "gatherv_dd_assembly":
        n_subdomains = int(config.get("n_subdomains", 100))
        rows_per_subdomain = int(config.get("rows_per_subdomain", 256))
        assembly_total_subdomains = n_subdomains
        if n_subdomains < 1 or rows_per_subdomain < 1:
            raise ValueError("n_subdomains and rows_per_subdomain must be positive")

        # Every rank must post the same number of collective operations.  For
        # non-divisible rank counts, pad the local schedule with zero-row
        # tensors while keeping the active payload equal to n_subdomains.
        quotient, remainder = divmod(n_subdomains, nranks)
        assembly_active_subdomains = quotient + int(rank < remainder)
        assembly_slot_count = (n_subdomains + nranks - 1) // nranks
        assembly_fields = {
            "residual": [],
            "cjac": [],
            "hessian": [],
        }
        for slot in range(assembly_slot_count):
            active = slot < assembly_active_subdomains
            subdomain_rows = rows_per_subdomain if active else 0
            residual = rng.standard_normal((subdomain_rows, cols))
            cjac = rng.standard_normal((cols, subdomain_rows))
            hessian = rng.standard_normal((subdomain_rows, cols))
            assembly_fields["residual"].append(bkd.to_backend(residual))
            assembly_fields["cjac"].append(bkd.to_backend(cjac))
            assembly_fields["hessian"].append(bkd.to_backend(hessian))

    # Gatherv sizes are expressed along the gather dimension, while MPI
    # receive offsets are expressed in flattened scalar elements.  This
    # workload gathers along dim=0, so each row contributes ``cols`` values.
    gatherv_sizes = [rows + other_rank for other_rank in range(nranks)]
    gatherv_offsets = [0]
    for size in gatherv_sizes[:-1]:
        gatherv_offsets.append(gatherv_offsets[-1] + size * cols)

    root_data = None
    if rank == 0 and operation in ("scatter_tensor", "broadcast_tensor"):
        root_data = bkd.to_backend(rng.standard_normal((rows * nranks, cols)))
    return {
        "local": local,
        "batch_local": batch_local,
        "root_data": root_data,
        "operation": operation,
        "gatherv_mode": config.get("gatherv_mode", "baseline"),
        "gatherv_sizes": gatherv_sizes,
        "gatherv_offsets": gatherv_offsets,
        "gatherv_cache_key": "backend-collective-gatherv",
        "assembly_fields": assembly_fields,
        "assembly_total_subdomains": assembly_total_subdomains,
        "assembly_active_subdomains": assembly_active_subdomains,
        "assembly_slot_count": assembly_slot_count,
    }


def _run_gatherv(local: Any, state: dict[str, Any]) -> Any:
    """Run one gatherv variant selected by the benchmark case."""
    bkd = _backend()
    mode = state["gatherv_mode"]
    sizes = state["gatherv_sizes"]
    offsets = state["gatherv_offsets"]
    cache_key = state["gatherv_cache_key"]

    if mode == "baseline":
        return bkd.gatherv_tensor(local)
    if mode == "cache":
        return bkd.gatherv_tensor(local, cache_key=cache_key)
    if mode == "explicit":
        return bkd.gatherv_tensor(local, sizes=sizes, offsets=offsets)
    if mode == "cache_async":
        request = bkd.gatherv_tensor(
            local, cache_key=cache_key, async_op=True
        )
        return request.wait()
    if mode == "explicit_async":
        request = bkd.gatherv_tensor(
            local, sizes=sizes, offsets=offsets, async_op=True
        )
        return request.wait()
    if mode == "final":
        # Exercise the complete public call shape.  With the current backend,
        # explicit sizes/offsets take precedence over the internal cache, so
        # the cache_key is retained for API coverage but does not add runtime
        # work to this particular call.
        request = bkd.gatherv_tensor(
            local,
            sizes=sizes,
            offsets=offsets,
            cache_key=cache_key,
            async_op=True,
        )
        return request.wait()
    raise ValueError(f"unknown gatherv mode: {mode!r}")


def _run_gatherv_batch(batch_local: list[Any], state: dict[str, Any]) -> list[Any]:
    """Post or execute a three-gather batch like DD-FOM/DD-ROM assembly."""
    bkd = _backend()
    mode = state["gatherv_mode"]
    cache_key = state["gatherv_cache_key"]

    if mode == "batch_cache":
        return [
            bkd.gatherv_tensor(
                local,
                as_list=True,
                cache_key=(cache_key, "batch", index),
            )
            for index, local in enumerate(batch_local)
        ]
    if mode == "batch_cache_async":
        requests = [
            bkd.gatherv_tensor(
                local,
                as_list=True,
                cache_key=(cache_key, "batch", index),
                async_op=True,
            )
            for index, local in enumerate(batch_local)
        ]
        # Deliberately wait only after every request has been posted.  The
        # dedicated DD assembly operation below mirrors the full production
        # residual, cJAC, and Hessian schedule.
        return [request.wait() for request in requests]
    raise ValueError(f"unknown gatherv batch mode: {mode!r}")


def _run_gatherv_dd_assembly(
    fields: dict[str, list[Any]], state: dict[str, Any]
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Measure the DD-FOM/DD-ROM per-subdomain gather schedule.

    Each local slot represents one subdomain for each of the residual, cJAC,
    and Hessian collectives.  The asynchronous mode deliberately posts every
    field and slot before waiting, matching the production assembly paths.
    """
    bkd = _backend()
    mode = state["gatherv_mode"]
    cache_key = state["gatherv_cache_key"]
    field_dims = {"residual": 0, "cjac": 1, "hessian": 0}

    def post(field: str, slot: int, tensor: Any, async_op: bool) -> Any:
        return bkd.gatherv_tensor(
            tensor,
            dim=field_dims[field],
            as_list=True,
            cache_key=(cache_key, "dd-assembly", field, slot),
            async_op=async_op,
        )

    if mode == "dd_cache":
        results = {
            field: [
                post(field, slot, tensor, async_op=False)
                for slot, tensor in enumerate(tensors)
            ]
            for field, tensors in fields.items()
        }
    elif mode == "dd_cache_async_fieldwise":
        results = {}
        for field, tensors in fields.items():
            requests = [
                post(field, slot, tensor, async_op=True)
                for slot, tensor in enumerate(tensors)
            ]
            results[field] = [request.wait() for request in requests]
    elif mode == "dd_cache_async":
        requests = {
            field: [
                post(field, slot, tensor, async_op=True)
                for slot, tensor in enumerate(tensors)
            ]
            for field, tensors in fields.items()
        }
        results = {
            field: [request.wait() for request in field_requests]
            for field, field_requests in requests.items()
        }
    else:
        raise ValueError(f"unknown DD assembly mode: {mode!r}")

    slot_count = state["assembly_slot_count"]
    active_subdomains = state["assembly_active_subdomains"]
    return results, {
        "assembly_total_subdomains": state["assembly_total_subdomains"],
        "assembly_local_active_subdomains": active_subdomains,
        "assembly_local_padded_slots": slot_count - active_subdomains,
        "assembly_slots_per_field": slot_count,
        "assembly_requests_per_field": slot_count,
        "assembly_total_requests": 3 * slot_count,
        "assembly_local_active_requests": 3 * active_subdomains,
    }


def _result_shape(result: Any) -> Any:
    if isinstance(result, (list, tuple)):
        return [list(value.shape) if hasattr(value, "shape") else None
                for value in result]
    return list(result.shape) if hasattr(result, "shape") else None


def backend_collective_run(state: dict[str, Any]) -> dict[str, Any]:
    """Measure one backend collective, including GPU-aware MPI paths."""
    bkd = _backend()
    operation = state["operation"]
    local = state["local"]
    extra_metrics = {}
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
        result = _run_gatherv(local, state)
    elif operation == "gatherv_batch":
        result = _run_gatherv_batch(state["batch_local"], state)
    elif operation == "gatherv_dd_assembly":
        result, extra_metrics = _run_gatherv_dd_assembly(
            state["assembly_fields"], state
        )
    elif operation == "scatter_tensor":
        result = bkd.scatter_tensor(state["root_data"])
    elif operation == "broadcast_tensor":
        result = bkd.broadcast_tensor(state["root_data"] if bkd.root() else None)
    else:
        raise ValueError(f"unknown backend collective: {operation!r}")
    return {
        "operation": operation,
        "result_type": type(result).__name__,
        "result_shape": _result_shape(result),
        **extra_metrics,
    }


def backend_collective(config: dict[str, Any]) -> dict[str, Any]:
    return backend_collective_run(backend_collective_prepare(config))
