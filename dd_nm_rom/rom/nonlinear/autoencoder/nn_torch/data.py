import torch
import numpy as np
import torch.distributed as dist
import torch.distributed.tensor as dtensor

from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print

class Data(object):

  def __init__(
    self,
    snapshots,
    validation_split=0.2,
    batch_size=32,
    eps=1e-5,
    tile=None,
    redistribute=False
  ):
    self.rank = 0
    if bkd.distributed():
      self.rank = bkd.get_rank()

    self.snapshots = bkd.to_backend(snapshots)

    self.epoch = 0
    self.redistribute = redistribute

    if tile is not None:
      # debugging option to extend snapshots
      self.snapshots = torch.tile(self.snapshots, (tile, 1))

    self._rank_sizes = self._get_snapshot_sizes(self.snapshots)

    self.local_size = self.snapshots.shape[0]
    self.global_size = self.local_size
    if bkd.distributed():
      if bkd.get_rank() == 0:
        self.global_size = np.sum(self._rank_sizes)
      self.global_size = bkd._COMM.bcast(self.global_size, root=0)

      parallel_print("RANK {}:  Local snapshot matrix size = {}, global size = {}".format(bkd.get_rank(), self.local_size, self.global_size))

    # Normalization
    #self.normalize(self.snapshots, eps=eps)

    # For distributed training, gather samples from all ranks, shuffle, then scatter back
    if bkd.distributed() and redistribute:
      bkd.barrier()

      # gather full snapshot matrix
      self.snapshots = self._gather_samples(self.snapshots, self._rank_sizes)


      # renormalize snapshots using entire matrix
      self.ref = torch.empty(self.snapshots.shape[-1], device=bkd.device())
      self.scale = torch.empty(self.snapshots.shape[-1], device=bkd.device())

      if bkd.get_rank() == 0:
        self.normalize(self.snapshots, eps=eps)

      dist.broadcast(self.ref, src = 0)
      dist.broadcast(self.scale, src = 0)

      # split snapshots across all ranks:
      #self.snapshots = self._scatter_samples(snapshots_all)

      # all ranks have full snapshot matrix:
      if bkd.get_rank() > 0:
        self.snapshots = torch.empty((self.global_size, self.snapshots.shape[-1]), device=bkd.device())

      bkd.barrier()
      dist.broadcast(self.snapshots, src=0)
      bkd.barrier()

    # Data
    self.train = None
    self.valid = None
    self.batches = None
    self.batches_valid = None
    # Split train and validation
    self.validation_split = validation_split
    #self.split_train_valid(self.snapshots)
    self.batch_size = batch_size

    '''
    if bkd.distributed():
      self.batch_size = batch_size / bkd.get_nranks()

      #self.snapshots = self._scatter_samples(snapshots_all)

      if bkd.get_rank() == 0:
        self.split_train_valid(snapshots_all)
      
      bkd.barrier()
      self.train = self._scatter_samples(self.train)
      self.valid = self._scatter_samples(self.valid)

      self.normalize(torch.cat((self.train, self.valid), dim=0), eps=eps)
    '''
    if bkd.distributed() and redistribute:
      self.batch_size = batch_size
      #self.batch_size = batch_size / bkd.get_nranks()
    
      # splits full snapshot matrix into train/valid split on root rank
      # full train and valid snapshots are broadcast out after initial shuffle
      train_size = 0
      valid_size = 0
      if bkd.get_rank() == 0:
        self.split_train_valid(self.snapshots)

        train_size = self.train.shape[0]
        valid_size = self.valid.shape[0]
  
      train_size = bkd._COMM.bcast(train_size, 0)
      valid_size = bkd._COMM.bcast(valid_size, 0)
      
      bkd.barrier()

      if bkd.get_rank() > 0:
        self.train = torch.empty((train_size, self.snapshots.shape[-1]), device=bkd.device())
        self.valid = torch.empty((valid_size, self.snapshots.shape[-1]), device=bkd.device())
      dist.broadcast(self.train, src=0)
      dist.broadcast(self.valid, src=0)

      #self.train = self._scatter_samples(self.train)
      #self.valid = self._scatter_samples(self.valid)
    else:
      if bkd.distributed() and not redistribute:
        self.normalize_dist(self.snapshots, eps=eps)
      else:
        self.normalize(self.snapshots, eps=eps)
      self.split_train_valid(self.snapshots)
      self.batch_size = batch_size

    parallel_print("RANK {}: data: snapshot size = {} batch size = {} train size = {} valid size = {}".format(self.rank, self.snapshots.shape, self.batch_size, self.train.shape, self.valid.shape))
    self._validate_sizes()



  def normalize(self, data, eps=1e-5):
    amin = torch.amin(data, dim=0)
    amax = torch.amax(data, dim=0)
    ref, scale = 0.5*(amax+amin), 0.5*(amax-amin)
    indices = torch.isclose(scale, torch.tensor(0.0), rtol=0.0, atol=eps)
    scale[indices] = 1.0
    self.ref = bkd.to_backend(ref)
    self.scale = bkd.to_backend(scale)


  def normalize_dist(self, data, eps=1e-5):
    amin = torch.amin(data, dim=0)
    amax = torch.amax(data, dim=0)
    dist.all_reduce(amin, op=dist.ReduceOp.MIN)
    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
    bkd.barrier()
    ref, scale = 0.5*(amax+amin), 0.5*(amax-amin)
    indices = torch.isclose(scale, torch.tensor(0.0), rtol=0.0, atol=eps)
    scale[indices] = 1.0
    self.ref = bkd.to_backend(ref)
    self.scale = bkd.to_backend(scale)

  def split_train_valid(self, data):
    data = self.shuffle(data, seed=bkd.seed())
    if (self.validation_split > 0.0):
      nb_samples = data.shape[0]
      valid_size = int(self.validation_split*nb_samples)
      self.valid = data[:valid_size]
      self.train = data[valid_size:]
    else:
      self.train = data

  def batch(self, data):
    data[:,] = self.shuffle(data, seed=bkd.seed())

    if bkd.distributed() and self.redistribute:
      data = torch.tensor_split(data, bkd.get_nranks(), dim=0)
      nb_samples = data[bkd.get_rank()].shape[0]
      nb_batches = int(np.ceil(nb_samples/self.batch_size))
      return torch.tensor_split(data[bkd.get_rank()], nb_batches, dim=0)
    else:
      nb_samples = data.shape[0]
      nb_batches = int(np.ceil(nb_samples/self.batch_size))
      return torch.tensor_split(data, nb_batches, dim=0)


  def batch_dist_scatter(self, data):
    # distributed version of batch function, assuming scattered snapshots
    # performs the permutation on the root rank, then distributes samples after permuting
    if not bkd.distributed():
      return self.batch(data)

    # NOTE: refactor to avoid communication
    data_all = self._gather_samples(data)

    if bkd.get_rank() == 0:
      # shuffle on root rank using all data
      data_all = self.shuffle(data_all)

    bkd.barrier()

    data = self._scatter_samples(data_all, data)

    nb_samples = data.shape[0]
    nb_batches = int(np.ceil(nb_samples/self.batch_size))
    return torch.tensor_split(data, nb_batches, dim=0)


  def batch_dist(self, data):
    # distributed version of batch function, assuming full shared snapshot
    # each rank permutes on full data size, then permutation is split across ranks
    if not bkd.distributed():
      return self.batch(data)

    g = torch.Generator(device=bkd.device())
    g.manual_seed(bkd.seed())

    i = torch.randperm(data.shape[0], generator=g)
    data[:,] = data[i]
    i = torch.tensor_split(i, bkd.get_nranks(), dim=0)
    i = i[bkd.get_rank()]

    rank_data = torch.tensor_split(data, bkd.get_nranks(), dim=0)
    rank_data = rank_data[bkd.get_rank()]
    
    nb_samples = i.shape[0]
    nb_batches = int(np.ceil(nb_samples/self.batch_size))
    return torch.tensor_split(rank_data, nb_batches, dim=0)


  def shuffle(self, data, seed=None):
    np.random.seed(seed)
    #i = np.random.permutation(data.shape[0])
    g = torch.Generator(device=bkd.device())
    g.manual_seed(seed)
    i = torch.randperm(data.shape[0], generator=g)
    return data[i]

  def on_epoch_begin(self):
    if bkd.distributed() and self.redistribute:
      bkd.barrier()
      self.batches = self.batch_dist(self.train)
      if (self.valid is not None):
        self.batches_valid = self.batch_dist(self.valid)
    else:
      self.batches = self.batch(self.train)
      if (self.valid is not None):
        self.batches_valid = self.batch(self.valid)
    self.epoch += 1


  def _get_snapshot_sizes(self, data):
    # returns the data size (tensor dim 0) for each rank in a list
    # NOTE: should be called from root rank only
    if not bkd.distributed():
      return [data.shape[0]]
    lsize = data.shape[0]
    rank_sizes = bkd._COMM.gather(lsize, root=0)

    bkd.barrier()

    return rank_sizes


  def _validate_sizes(self):
    # performs check to ensure snapshot sizes across all ranks are equal
    if not bkd.distributed():
      return

    train_sizes = bkd._COMM.gather(self.train.shape[0], root=0)
    valid_sizes = bkd._COMM.gather(self.valid.shape[0], root=0)
    if bkd.get_rank() == 0:
      '''
      print(" RANK 0 TRAIN SIZES:")
      for i in range(bkd.get_nranks()):
        print("   -- RANK {}: local size = {}".format(i, train_sizes[i]))
      print(" RANK 0 VALID SIZES:")
      for i in range(bkd.get_nranks()):
        print("   -- RANK {}: local size = {}".format(i, valid_sizes[i]))
      '''

      train_sizes = np.asarray(train_sizes)
      valid_sizes = np.asarray(valid_sizes)
      train_per_rank = np.sum(train_sizes)/train_sizes.size
      valid_per_rank = np.sum(valid_sizes)/valid_sizes.size
      print(" Avg train size per rank = {} Avg valid size per rank = {}".format(train_per_rank, valid_per_rank))
      # ensure the amount of train / valid per rank is the same, otherwise there are uneven inputs which may hang the training
      stop = False
      if train_per_rank != train_sizes[0]:
        stop = True
        raise RuntimeWarning("Detected uneven training samples across ranks!")
      if valid_per_rank != valid_sizes[0]:
        stop = True
        raise RuntimeWarning("Detected uneven valid samples across ranks!")
      
      if stop:
        bkd.finalize_distributed()
        bkd._COMM.Abort()

    bkd.barrier()


  def _gather_samples(self, data, data_sizes=None):
    # gathers data from all processes to this rank
    # if data_sizes = None, then the size of data on each rank will be determined first
    # otherwise, pre-populated list of data size on each rank will be used

    if not bkd.distributed():
      return data

    snapshots = None

    rank_sizes = data_sizes
    calc_sizes = False
    if rank_sizes is None and bkd.get_rank() == 0:
      # if rank sizes is undefined on root rank, then we need to determine the sizes
      calc_sizes = True
    calc_sizes = bkd._COMM.bcast(calc_sizes, root=0)
    bkd.barrier()

    if calc_sizes:
      rank_sizes = self._get_snapshot_sizes(data)

    if bkd.get_rank() == 0:
      snapshots = []
      for rank in range(bkd.get_nranks()):
        # NOTE: assumes data is 2D tensor, where the second dim is always constant across ranks (e.g domain size)
        snapshots.append(torch.zeros_like(data, device=bkd.device()))

    dist.gather(data, snapshots, 0)

    bkd.barrier()

    if bkd.get_rank() == 0:
      snapshots = torch.vstack(snapshots)

    bkd.barrier()

    if bkd.get_rank() == 0:
      return snapshots
    else:
      return data


  def _scatter_samples(self, data, data_out=None):
    # scatters distributed data tensor across all ranks and returns result
    # each rank gets 1/nranks portion of snapshot matrix
    # NOTE: assumes data is distributed across all ranks, and each rank must have same size
    # 
    # If data_out=None, then result tensor will be allocated with size according to data // nranks
    # data_out will be overwritten by this operation
    if not bkd.distributed():
      return data

    # overwrites each self.snapshots in-place
    full_size = 0
    samples = None
    if bkd.get_rank() == 0:
      samples = list(torch.tensor_split(data, bkd.get_nranks(), dim=0))
      full_size = data.shape[0]
      assert len(samples) == bkd.get_nranks()

    if data_out is None:
      full_size = bkd._COMM.bcast(full_size, root=0)
      out_size = full_size // bkd.get_nranks()
      data_out = torch.empty((out_size, self.snapshots.shape[-1]), device=bkd.device())

    bkd.barrier()

    dist.scatter(data_out, samples, 0)

    bkd.barrier()

    return data_out

