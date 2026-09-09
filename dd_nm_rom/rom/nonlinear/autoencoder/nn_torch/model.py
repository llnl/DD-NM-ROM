import os
import numpy as np
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import optimization as optim
from dd_nm_rom import backend as bkd


class Model(object):

  def __init__(
    self,
    net=None,
    data=None,
    path="./",
    chk_meta=None,
  ):
    self.net = net
    self.data = data
    # Attributes for optimization
    self.loss = None
    self.optimizer = None
    self.callbacks = None
    # Training state
    logs_ids = ["loss"]
    if (self.data.valid is not None):
      logs_ids.append("val_loss")
    self.train_state = optim.TrainState()
    self.train_state.init_logs(logs_ids)
    # Define paths
    self.path = path
    self.dirs = {
      "save": self.path + "/saving/",
      "train": self.path + "/training/",
      "ckpt": self.path + "/training/ckpt/"
    }
    for d in self.dirs.values():
      os.makedirs(d, exist_ok=True)
    # Control vars
    self.is_compiled = False
    self.stop_training = False

    self.meta = dict()
    if chk_meta is not None:
      self.meta = dict(chk_meta)

    self.rank = 0
    if bkd.distributed():
        self.rank = bkd.get_rank()
        if torch.accelerator.device_count() > 1:
          print("MODEL:: creating DDP with device_ids = {}".format(bkd.get_rank()))
          self.ddp_net = DDP(self.net, device_ids=[bkd.get_rank()])
        else:
          print("MODEL:: creating DDP with device_ids = {}".format(bkd.device()))
          self.ddp_net = DDP(self.net, device_ids=None)
        self.net = self.ddp_net.module
    else:
        self.ddp_net = self.net

  # Compiling
  # ---------------------------------
  def compile(
    self,
    optimizer="adam",
    lr=1e-3,
    lr_decay=None,
    weight_decay=0.0,
    loss="mse",
    monitor="loss",
    reduction="sum",
    callbacks=None
  ):
    print("Compiling the model ...")
    # Write nn summary
    summary_fname = self.dirs["save"] + "/summary.txt"
    if bkd.distributed():
      summary_fname = summary_fname + ".{:d}".format(bkd.get_rank())
    self.net.summary(filename=summary_fname)
    # Initializing loss function
    self.loss = optim.losses.get(loss, reduction=reduction)
    # Monitor metrics
    self.monitor = monitor
    options = list(self.train_state.logs.keys())
    if (self.monitor not in options):
      raise ValueError(f"Monitor metrics not valid. Please choose: {options}")
    # Initializing the optimizer
    self.optimizer, self.lr_scheduler = optim.optimizers.get(
      self.ddp_net.parameters(),
      optimizer,
      lr=lr,
      lr_decay=lr_decay,
      weight_decay=weight_decay
    )
    # Callbacks
    self.callbacks = optim.callbacks.get_callbacks(callbacks)
    self.callbacks.set_model(self)
    self.is_compiled = True

  # Training
  # ---------------------------------
  def train(
    self,
    epochs=100,
    display_freq=1,
    saving=True
  ):
    if self.net.trainable:
      print("Training the model ...")
      # Training state
      self.train_state.epochs = epochs
      self.train_state.display_freq = display_freq
      # Training
      self.callbacks.set_display_freq(display_freq)
      self.callbacks.on_train_begin()
      if bkd.distributed():
        with self.ddp_net.join(throw_on_early_termination=True):
          self.train_sgd()
        if bkd.device().type == "cuda":
          torch.cuda.synchronize()
        bkd._COMM.Barrier()
      else:
        self.train_sgd()
      self.callbacks.on_train_end()
    else:
      print("Warning! Training skipped since the model is not trainable!")
    # Saving
    if saving:
      print("Saving the model ...")
      self.save()

  def train_sgd(self):
    for _ in range(self.train_state.epoch, self.train_state.epochs):
      # On epoch begin calls
      self.train_state.on_epoch_begin()
      self.callbacks.on_epoch_begin()
      self.data.on_epoch_begin()
      # Train step
      self.ddp_net.train(mode=True)
      for batch in self.data.batches:
        # On batch begin calls
        self.callbacks.on_batch_begin()
        # Training step
        self.train_step(batch.to(bkd.device()))
        # On batch end calls
        self.callbacks.on_batch_end()
      # Test step
      with torch.set_grad_enabled(False):
        self.evaluate()
      # On epoch end calls
      self.train_state.on_epoch_end()
      # Update lr
      self.update_lr()
      self.callbacks.on_epoch_end()
      self.train_state.epoch += 1
      if self.stop_training:
        break

  def train_step(self, data):
    # Run forward pass
    loss = self.evaluate_step(data)
    # Run backwards pass
    self.optimizer.zero_grad()
    loss.backward()
    # Update parameters
    self.optimizer.step()
    # Update logs
    self.train_state.on_batch_end({"loss": loss})

  def update_lr(self):
    if (self.lr_scheduler is not None):
      if self.lr_scheduler.metrics_needed:
        metrics = self.train_state.logs[self.monitor]
        self.lr_scheduler.step(metrics)
      else:
        self.lr_scheduler.step()

  # Testing
  # ---------------------------------
  def evaluate(self):
    if (self.data.valid is not None):
      self.ddp_net.train(mode=False)
      for batch in self.data.batches_valid:
        loss = self.evaluate_step(batch.to(bkd.device()))
        # Update logs
        self.train_state.on_batch_end({"val_loss": loss})

  def evaluate_step(self, data):
    return self.loss(self.ddp_net(data), data)

  # Saving
  # ---------------------------------
  def save(self, filename=None):
    if (filename is None):
      filename = self.dirs["save"] + "/model_last"
    if bkd.distributed():
      dist.barrier()
      bkd._COMM.Barrier()
    if not bkd.distributed() or bkd.get_rank() == 0:
        start_time = time.time()
        torch.save(self.ddp_net.state_dict(), filename+"_torch.p")
        torch.save(self.net.state_dict_np(), filename+"_numpy.p")

        # Save complete checkpoint with optimizer and scheduler state
        random_state = (
            torch.cuda.get_rng_state(device=bkd.device())
            if bkd.device().type == "cuda"
            else torch.get_rng_state()
        )
        checkpoint = {
            'model_state_dict': self.ddp_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.train_state.epoch,
            'random_state_np': np.random.get_state(),
            'random_state': random_state,
            'metadata': self.meta
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, filename + "_checkpoint.p")

        end_time = time.time() - start_time
        print("  -- Checkpoint time: {:.5e} s".format(end_time))
    if bkd.distributed():
      dist.barrier()
      bkd._COMM.Barrier()

  def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path,  weights_only=False)

    self.train_state.epoch = checkpoint['epoch']

    # Load scheduler if available
    if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load optimizer state - must be done after LR Scheduler!
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore random state
    if 'random_state' in checkpoint:
      np.random.set_state(checkpoint['random_state_np'])
      if bkd.device().type == "cuda":
        torch.cuda.set_rng_state(checkpoint['random_state'], device=bkd.device())
      else:
        torch.set_rng_state(checkpoint['random_state'])

    if bkd.distributed():
      dist.barrier()
      bkd._COMM.Barrier()

    return checkpoint
