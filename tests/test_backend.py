import pytest

import numpy as np
import torch
import torch.distributed as dist
import torch.testing
from mpi4py import MPI

from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print


_BACKEND_INITIALIZED = False


@pytest.fixture(scope="module", autouse=True)
def configured_backend(request):
    """Use conftest's selected backend and initialize MPI only when needed."""
    global _BACKEND_INITIALIZED
    selected = request.config.getoption("--backend") or "torch_cpu"
    request.getfixturevalue("backend_" + selected)

    if not _BACKEND_INITIALIZED:
        bkd.set_floatx("float64")
        if MPI.COMM_WORLD.Get_size() > 1 and not bkd.distributed():
            device = "cuda" if bkd.device().type == "cuda" else "cpu"
            if device == "cpu":
                get_name = torch.cuda.get_device_name
                get_properties = torch.cuda.get_device_properties
                torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"
                torch.cuda.get_device_properties = lambda *args, **kwargs: "cpu"
                try:
                    bkd.init_distributed(device)
                finally:
                    torch.cuda.get_device_name = get_name
                    torch.cuda.get_device_properties = get_properties
            else:
                bkd.init_distributed(device)
        bkd.set_seed(0)
        _BACKEND_INITIALIZED = True
    yield

def _generate_matrix(size):
    return torch.randn(size, device=bkd.device())

@pytest.mark.mpi_skip
def test_distributed_serial():
    assert bkd.distributed() == False

@pytest.mark.mpi(min_size=2)
def test_distributed_parallel():
    assert bkd.distributed() == True
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == bkd.get_nranks()
    assert comm.Get_rank() == bkd.get_rank()
    if comm.Get_rank() == 0:
        assert bkd.root() == True
    else:
        assert bkd.root() == False

def _check_sizes(sizes, expected):
    if not bkd.distributed():
        assert sizes is not None
        assert len(sizes) == 1
        assert sizes[0] == expected
    else:
        assert sizes is not None
        assert len(sizes) == bkd.get_nranks()
        for rank in range(bkd.get_nranks()):
            assert sizes[rank] == expected


def test_get_local_sizes_np():
    x = np.random.randn(2, 4)

    sizes = bkd.get_local_sizes(x)
    if bkd.root():
        _check_sizes(sizes, 2)
    else:
        assert sizes is None
    
    sizes_dim1 = bkd.get_local_sizes(x, dim=1)
    if bkd.root():
        _check_sizes(sizes_dim1, 4)
    else:
        assert sizes_dim1 is None


def test_get_local_sizes():
    x = torch.randn((2, 4), device=bkd.device())

    sizes = bkd.get_local_sizes(x)
    if bkd.root():
        _check_sizes(sizes, 2)
    else:
        assert sizes is None
    
    sizes_dim1 = bkd.get_local_sizes(x, dim=1)
    if bkd.root():
        _check_sizes(sizes_dim1, 4)
    else:
        assert sizes_dim1 is None


def test_get_local_sizes_all():
    x = torch.randn((2, 4), device=bkd.device())

    sizes = bkd.get_local_sizes_all(x)
    _check_sizes(sizes, 2)
    
    sizes_dim1 = bkd.get_local_sizes_all(x, dim=1)
    _check_sizes(sizes_dim1, 4)


def test_gather_tensor_calc_sizes():
    local_size = 2
    global_size = local_size * bkd.get_nranks()
    x_local = torch.randn((local_size, 4), device=bkd.device())

    x_global = bkd.gather_tensor(x_local)
    #parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    #parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if bkd.root():
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 4
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_tensor_calc_sizes():
    local_size = 2 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 4), device=bkd.device())

    x_global = bkd.gatherv_tensor(x_local)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if bkd.root():
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 4
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_tensor_calc_sizes_dim():
    local_size = 2 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((4, local_size), device=bkd.device())

    x_global = bkd.gatherv_tensor(x_local, dim=1)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if bkd.root():
        assert x_global.shape[0] == 4
        assert x_global.shape[1] == global_size
    else:
        torch.testing.assert_close(x_local, x_global)

def test_gatherv_tensor_calc_sizes_list():
    local_size = 2 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 4), device=bkd.device())

    x_global = bkd.gatherv_tensor(x_local, as_list=True)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        torch.testing.assert_close(x_global, x_local)
        return
    if bkd.root():
        assert len(x_global) == bkd.get_nranks()
        for (rank, x) in enumerate(x_global):
          lsize = 2 + rank
          assert x.shape[0] == lsize
          assert x.shape[1] == 4
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_tensor_calc_sizes_3d():
    nlocal = 3

    local_size = 2 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size * nlocal, op=MPI.SUM)

    x_local = []
    for i in range(nlocal):
        x_local.append(torch.randn((local_size, 4), device=bkd.device()))
    
    x_global = bkd.gatherv_tensor(x_local)
    parallel_print(" RANK {}: x_local = {} (shape {})".format(bkd.get_rank(), x_local, len(x_local)))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        assert len(x_global) == len(x_local)
        for actual, expected in zip(x_global, x_local):
            torch.testing.assert_close(actual, expected)
        return
    if bkd.root():
        assert x_global.ndim == 2
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 4
    else:
        #for (sub, x) in enumerate(x_global):
        torch.testing.assert_close(torch.cat(x_local), x_global)


def test_gatherv_tensor_calc_sizes_3d_list():
    nlocal = 3

    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size * nlocal, op=MPI.SUM)

    x_local = []
    for i in range(nlocal):
        x_local.append(torch.randn((local_size, 4), device=bkd.device()))
    
    x_global = bkd.gatherv_tensor(x_local, as_list=True)
    parallel_print(" RANK {}: x_local = {} (shape {})".format(bkd.get_rank(), x_local, len(x_local)))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        assert len(x_global) == len(x_local)
        for actual, expected in zip(x_global, x_local):
            torch.testing.assert_close(actual, expected)
        return
    if bkd.root():
        assert len(x_global) == bkd.get_nranks()
        for (rank, x) in enumerate(x_global):
            lsize = 16 + rank
            assert len(x) == nlocal
            for sub in x:
                assert sub.shape[0] == lsize
                assert sub.shape[1] == 4
    else:
        torch.testing.assert_close(torch.cat(x_local), x_global)


def test_gatherv_spcoo_calc_sizes():
    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 16), device=bkd.device())
    x_local = x_local.to_sparse(layout=torch.sparse_coo)

    x_global = bkd.gatherv_tensor(x_local)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    assert x_global.layout == torch.sparse_coo
    if bkd.root():
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 16
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_spcoo_calc_sizes_list():
    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 16), device=bkd.device())
    x_local = x_local.to_sparse(layout=torch.sparse_coo)

    x_global = bkd.gatherv_tensor(x_local, as_list=True)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        torch.testing.assert_close(x_global, x_local)
        return
    if bkd.root():
        assert len(x_global) == bkd.get_nranks()
        for (rank, x) in enumerate(x_global):
          assert x.layout == torch.sparse_coo
          lsize = 16 + rank
          assert x.shape[0] == lsize
          assert x.shape[1] == 16
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_spcsr_calc_sizes():
    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 16), device=bkd.device())
    x_local = x_local.to_sparse_csr()

    x_global = bkd.gatherv_tensor(x_local)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    assert x_global.layout == torch.sparse_csr
    if bkd.root():
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 16
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_spcsr_calc_sizes_list():
    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.randn((local_size, 16), device=bkd.device())
    x_local = x_local.to_sparse_csr()

    x_global = bkd.gatherv_tensor(x_local, as_list=True)
    parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        torch.testing.assert_close(x_global, x_local)
        return
    if bkd.root():
        assert len(x_global) == bkd.get_nranks()
        for (rank, x) in enumerate(x_global):
          assert x.layout == torch.sparse_csr
          lsize = 16 + rank
          assert x.shape[0] == lsize
          assert x.shape[1] == 16
    else:
        torch.testing.assert_close(x_local, x_global)


def test_gatherv_spcsr_calc_sizes_3d():
    nlocal = 3

    local_size = 16 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size * nlocal, op=MPI.SUM)

    x_local = []
    for i in range(nlocal):
        x_local.append(torch.randn((local_size, 16), device=bkd.device()).to_sparse_csr())
    
    x_global = bkd.gatherv_tensor(x_local)
    parallel_print(" RANK {}: x_local = {} (shape {})".format(bkd.get_rank(), x_local, len(x_local)))
    parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if not bkd.distributed():
        assert len(x_global) == len(x_local)
        for actual, expected in zip(x_global, x_local):
            torch.testing.assert_close(
                actual.to_sparse_coo(), expected.to_sparse_coo(),
            )
        return
    assert x_global.layout == torch.sparse_csr
    if bkd.root():
        assert x_global.ndim == 2
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 16
    else:
        #for (sub, x) in enumerate(x_global):
        torch.testing.assert_close(torch.cat([x.to_sparse_coo() for x in x_local]), x_global.to_sparse_coo())


def test_scatter_tensor_allocate():
    local_size = 2
    global_size = local_size * bkd.get_nranks()
    x = None
    if bkd.root():
        x = torch.randn((global_size, 4), device=bkd.device())
        #print(" ROOT x to scatter = {}".format(x))

    x_rank = bkd.scatter_tensor(x, x_out=None)
    #parallel_print(" RANK {}: x_rank = {}".format(bkd.get_rank(), x_rank))
    assert x_rank is not None
    assert x_rank.shape[0] == local_size
    assert x_rank.shape[1] == 4

    if not bkd.root():
        x = torch.empty((global_size, 4), device=bkd.device())
    x = bkd.broadcast_tensor(x, root=0)
    x = torch.tensor_split(x, bkd.get_nranks())
    torch.testing.assert_close(x[bkd.get_rank()], x_rank)


# def test_scatter_tensor_list_allocate():
#     local_size = 2
#     global_size = local_size * bkd.get_nranks()
#     x = None
#     x_split = None
#     if bkd.root():
#         x = torch.randn((global_size, 4), device=bkd.device())
#         x_split = list(torch.tensor_split(torch.clone(x), bkd.get_nranks()))
#         print(" ROOT x to scatter = {}, SPLIT = {}".format(x, x_split))
#         print(" ROOT x to scatter = {}, SPLIT = {}".format(x.shape, len(x_split)))

#     x_rank = bkd.scatter_tensor(x_split, x_out=None, as_list=True)
#     parallel_print(" RANK {}: x_rank = {}".format(bkd.get_rank(), x_rank))
#     assert x_rank is not None
#     assert len(x_rank) == bkd.get_nranks()
#     for rank in range(bkd.get_ranks()):
#         assert x_rank[rank].shape[0] == local_size
#         assert x_rank[rank].shape[1] == 4

#     parallel_print(" RANK {}: x_rank = {}".format(bkd.get_rank(), x_rank[bkd.get_rank()]))

#     if not bkd.root():
#         x = torch.empty((global_size, 4), device=bkd.device())
#     dist.broadcast(x, src=0)
#     x = torch.tensor_split(x, bkd.get_nranks())
#     torch.testing.assert_close(x[bkd.get_rank()], x_rank[bkd.get_rank()])


@pytest.mark.mpi
def test_create_shard_dtensor():
    local_size = 2
    global_size = local_size * bkd.get_nranks()

    x_local = torch.full((local_size, 4), bkd.get_rank(), device=bkd.device(), dtype=bkd.floatx())
    
    x_global = torch.zeros((global_size, 4), device=bkd.device())
    for rank in range(bkd.get_nranks()):
        x_global[rank * local_size : (rank+1) * local_size,] = rank

    dist_x = bkd.to_sharded_dtensor(x_local)

    torch.testing.assert_close(dist_x.to_local(), x_local)
    torch.testing.assert_close(dist_x.full_tensor(), x_global)


@pytest.mark.mpi(min_size=2)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "PyTorch DTensor Shard(0) reconstructs uneven local inputs "
        "incorrectly; distributed Newton avoids this path."
    ),
)
def test_create_uneven_shard_dtensor_round_trip():
    """DTensor Shard(0) must preserve rank-local lengths that differ."""
    if not bkd.distributed():
        pytest.skip("requires at least two MPI ranks to create uneven shards")

    local_size = 2 + bkd.get_rank()
    global_size = MPI.COMM_WORLD.allreduce(local_size, op=MPI.SUM)
    x_local = torch.arange(
        local_size,
        device=bkd.device(),
        dtype=bkd.floatx(),
    ) + 100 * bkd.get_rank()

    # Use the repository's variable-size gather as the reference for
    # DTensor.full_tensor(). Broadcasting makes the expected global value
    # available on every rank for the assertion below.
    expected = bkd.gatherv_tensor(x_local)
    expected = bkd.broadcast_tensor(expected, root=0)

    dist_x = bkd.to_sharded_dtensor(
        x_local,
        shape=(global_size,),
        stride=(1,),
    )

    assert dist_x.shape == (global_size,)
    torch.testing.assert_close(dist_x.to_local(), x_local)
    torch.testing.assert_close(dist_x.full_tensor(), expected)


@pytest.mark.mpi
def test_create_replica_dtensor():
    local_size = 2
    global_size = local_size * bkd.get_nranks()

    x_local = torch.full((local_size, 4), bkd.get_rank(), device=bkd.device(), dtype=bkd.floatx())
    
    x_global = torch.zeros((global_size, 4), device=bkd.device())
    for rank in range(bkd.get_nranks()):
        x_global[rank * local_size : (rank+1) * local_size,] = rank

    dist_x = bkd.to_sharded_dtensor(x_local)

    torch.testing.assert_close(dist_x.to_local(), x_local)
    torch.testing.assert_close(dist_x.full_tensor(), x_global)

    g = torch.Generator(device=bkd.device())
    g.manual_seed(0)
    i = torch.randperm(global_size, generator=g)
    di = bkd.to_replica_dtensor(i)

    xdi = dist_x[di]
    torch.testing.assert_close(xdi.full_tensor(), x_global[i])
