import numpy as np
import scipy.sparse as sp
import torch.sparse
import torch_sla
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from torch_sla.nonlinear_solve import _newton_solve

from time import time
from typing import Tuple, Union
from typing_extensions import Unpack

from . import dtypes
from .basic import Solver
from .. import backend as bkd

from dd_nm_rom.utils import parallel_print
import dd_nm_rom.config as cfg

import logging
logger = logging.getLogger(__name__)

class DistNewton(Solver):
  r"""
  A solver for optimization problems using the Newton's method.

  This class implements Newton's method to solve the equation:

  .. math::
    \\mathbf{r}(\\mathbf{x}) = \\mathbf{0}

  where :math:`\mathbf{r}(\mathbf{x})` is the residual vector. The method
  iteratively updates the solution until convergence criteria are met or
  the maximum number of iterations is reached.

  :param model: The physical model to be used by the solver. It should be a
                callable that provides residual and Jacobian calculations.
  :type model: callable
  :param tol: Tolerance for convergence. The solver stops when the residual norm
              is below this threshold. Defaults to 1e-3.
  :type tol: float
  :param maxit: Maximum number of iterations for the solver. Defaults to 20.
  :type maxit: int
  :param stepsize_min: Minimum step size for the line search. Defaults to 1e-10.
  :type stepsize_min: float
  :param iostep: Store every ``iostep``-th solution during time integration.
                 Defaults to 1.
  :type iostep: int
  :param verbose: Whether to print iteration details. Defaults to False.
  :type verbose: bool
  :param distributed: Whether the solver is used in a distributed execution.
                      Defaults to False.
  :type distributed: bool
  :param use_line_search: Whether to use Armijo backtracking. If false, the
                          full Newton step is used while residual evaluation,
                          convergence checks, and history tracking remain active.
                          Defaults to True.
  :type use_line_search: bool
  :param preconditioner: Preconditioner passed to the distributed Krylov
                         solver.  ``None`` (or ``"none"``) disables
                         preconditioning. Defaults to ``None``.
  :type preconditioner: str or None
  """

  def __init__(
    self,
    model: callable,
    tol: float = 1e-3,
    maxit: int = 20,
    stepsize_min: float = 1e-10,
    iostep: int = 1,
    verbose: bool = False,
    distributed: bool = False,
    use_line_search: bool = True,
    preconditioner: str | None = None,
  ) -> None:
    # The global KKT matrix is replicated, so a direct solve is performed on
    # rank zero and its direction is broadcast to the remaining ranks.
    # Keep the existing environment variable for compatibility, even though
    # the operation is more accurately described as a root-only direct solve.
    self.use_direct_solve = cfg.update_from_env(
      "DDNMROM_FORCE_SERIAL_SOLVE", False,
    )

    # Match the serial Newton backend preference: STRUMPACK supports CPU,
    # CUDA, and ROCm, while cuDSS is the NVIDIA CUDA direct-solver option.
    self.direct_solve_backend = "pytorch"
    if torch_sla.backends.is_strumpack_available():
      self.direct_solve_backend = "strumpack"
    elif torch_sla.backends.is_cudss_available():
      self.direct_solve_backend = "cudss"

    self.direct_solve_backend = cfg.update_from_env(
      "DDNMROM_FORCE_SOLVE_BACKEND", self.direct_solve_backend,
    )
    direct_backend_available = {
      "strumpack": torch_sla.backends.is_strumpack_available,
      "cudss": torch_sla.backends.is_cudss_available,
    }
    if self.use_direct_solve and self.direct_solve_backend not in direct_backend_available:
      raise RuntimeError(
        "DDNMROM_FORCE_SERIAL_SOLVE requires a sparse direct backend "
        "(STRUMPACK or cuDSS), but '{}' is selected. Set "
        "DDNMROM_FORCE_SOLVE_BACKEND to an available direct backend or "
        "disable DDNMROM_FORCE_SERIAL_SOLVE.".format(
          self.direct_solve_backend,
        )
      )
    if (self.use_direct_solve
        and not direct_backend_available[self.direct_solve_backend]()):
      raise RuntimeError(
        "DDNMROM_FORCE_SERIAL_SOLVE selected direct backend '{}', but it "
        "is not available in this environment.".format(
          self.direct_solve_backend,
        )
      )
    self.debug = logger.isEnabledFor(logging.DEBUG)

    super(DistNewton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      iostep=iostep,
      verbose=verbose,
      distributed=distributed,
      use_line_search=use_line_search,
      preconditioner=preconditioner,
    )

  @staticmethod
  def _full_tensor(x):
    """Return the underlying tensor for replicated and serial inputs."""
    return x.full_tensor() if isinstance(x, DTensor) else x

  def _global_trial_direction(self, dx):
    """Gather a simple-partition Newton direction as a regular tensor.

    ``DSparseTensor`` is constructed with ``partition_method="simple"`` in
    :meth:`solve`, so its owned rows are rank-contiguous global slices.  The
    variable-size MPI gather therefore reconstructs the global direction in
    the same order as the KKT vector.  Keeping this result as a regular
    tensor avoids DTensor arithmetic on uneven shards.
    """
    if not isinstance(dx, DTensor):
      return dx
    direction = bkd.gatherv_tensor(dx.to_local())
    return bkd.broadcast_tensor(direction, root=0)

  def _global_kkt_norm(self, res, cres):
    """Return the global KKT norm and globally accumulated constraints."""
    if bkd.is_torch_backend():
      res_local = res.to_local() if isinstance(res, DTensor) else res
      cres_global = cres.to_local().clone() if isinstance(cres, DTensor) else cres.clone()
      state_sq = torch.sum(res_local * res_local)
      if bkd.distributed():
        dist.all_reduce(state_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(cres_global, op=dist.ReduceOp.SUM)
      total_sq = state_sq + torch.sum(cres_global * cres_global)
      return torch.sqrt(total_sq).item(), cres_global

    state = np.asarray(res)
    constraint = np.asarray(cres)
    return float(np.sqrt(np.dot(state, state) + np.dot(constraint, constraint))), constraint

  def _evaluate_trial(self, x0, dx, stepsize):
    """Evaluate a trial point using the distributed local residual path."""
    start = time()
    # Do not add a replicated DTensor to a sharded KKT direction.  Torch
    # DTensor assumes even Shard(0) layouts for this redistribution and can
    # silently reconstruct uneven local slices incorrectly.
    x0 = self._full_tensor(x0)
    dx = self._global_trial_direction(dx)
    x = x0 + stepsize * dx
    self.model.runtime["total"] += time()-start

    # Domain-decomposition models define the residual-vector layout.  The
    # ROM consumes local state plus global multipliers; the FOM uses global
    # state offsets and therefore keeps the complete replicated KKT vector.
    # The fallback preserves compatibility with external solver models.
    local_trial_vector = getattr(self.model, "local_trial_vector", lambda y: y)
    res, cres = self.model.residual(local_trial_vector(x), use_global=False)
    res_norm, cres = self._global_kkt_norm(res, cres)
    return x, res, cres, res_norm

  def full_step(self, x0, dx):
    """Take and evaluate one undamped Newton step."""
    x, res, cres, res_norm = self._evaluate_trial(x0, dx, 1.0)
    return x, res, cres, res_norm, 1.0

  def line_search(
    self,
    x0: np.ndarray,
    dx: np.ndarray,
    eval_res_tol: callable,
    use_global: bool = True
  ) -> Tuple[np.ndarray, Unpack[dtypes.EVAL_TYPE], float]:
    # Initialize
    # -------------
    # A backtracking search can evaluate several trial steps. Reconstruct the
    # sparse-solve direction once, then keep every trial vector regular and
    # globally replicated.
    x0 = self._full_tensor(x0)
    dx = self._global_trial_direction(dx)
    stepsize = 1.0
    x, res, cres, res_norm = self._evaluate_trial(x0, dx, stepsize)
    if self.debug: logger.debug(" LINE SEARCH: local res = {} cres = {}".format(res.shape, cres.shape))

    # Condition
    # -------------
    start = time()
    cond_fun = lambda res_norm, stepsize: (
      (res_norm >= eval_res_tol(stepsize)) and (stepsize >= self.stepsize_min)
    )
    cond = cond_fun(res_norm, stepsize)

    self.model.runtime["total"] += time()-start
    while cond:
      # Update solution
      # -------------
      stepsize *= 0.5
      x, res, cres, res_norm = self._evaluate_trial(x0, dx, stepsize)

      # Condition
      # -------------
      start = time()
      cond = cond_fun(res_norm, stepsize)
      self.model.runtime["total"] += time()-start

      if self.debug: logger.debug(" LOCAL RES = {}".format(res.shape))
    return x, res, cres, res_norm, stepsize

  def evaluate(
    self,
    x: np.ndarray,
    use_global: bool = True
  ) -> dtypes.EVAL_TYPE:
    """
    Evaluate the residual, Jacobian, and residual norm.

    :param x: Current solution.
    :type x: np.ndarray

    :return: A tuple containing:
      - res (np.ndarray): Right-hand side vector.
      - jac (np.ndarray): Jacobian matrix.
      - res_norm (float): Residual norm.
    :rtype: EVAL_TYPE
    """
    res, jac = self.model.res_jac(x, use_global)
    
    start = time()

    res_norm = torch.dot(res, res) if bkd.is_torch_backend() else np.dot(res,res)
    if (not self.squared_res):
      res_norm = res_norm.sqrt_() if bkd.is_torch_backend() else np.sqrt(res_norm)
    self.model.runtime["total"] += time()-start
    return res, jac, float(res_norm)

  def solve(
    self,
    x0: np.ndarray
  ) -> dtypes.SOL_TYPE:
    """
    Solve a minimization problem using the Newton's method.

    :param x0: Initial guess.
    :type x0: np.ndarray

    :return: A tuple containing:
      - x (np.ndarray): Solution of the equation.
      - res_hist (np.ndarray): Residual vector history.
      - res_norm_hist (np.ndarray): Residual norm history.
      - step_hist (np.ndarray): Step size history.
      - it (np.ndarray): Number of iterations.
      - flag (np.ndarray): Convergence flag.
    :rtype: SOL_TYPE
    """
    # Initialize
    # ---------------
    # > Set first step
    it, x = 0, x0

    if self.debug: logger.debug("BEGIN SOLVE:")
  
    global_size = x.shape[0]


    #   res, cres = self.model.residual(u_full, use_global=False)


    # #x, info = torch_sla.nonlinear_solve(calc_res, x0, jacobian_fn=calc_jac, method="newton", max_iter=self.maxit, verbose=True)
    # # x, info = _newton_solve(x0, tuple(), calc_res, calc_jac, 1e-6, 1e-10, self.maxit, line_search=True, verbose=True, linear_solver="pytorch", linear_method="cg")


    # res, jac, res_norm = self.evaluate(x.full_tensor())


    # #x, info = torch_sla.nonlinear_solve(calc_res, x0, jacobian_fn=calc_jac, method="newton", max_iter=self.maxit, verbose=True)
    # x, info = _newton_solve(x0, tuple(), calc_res, calc_jac, 1e-6, 1e-10, self.maxit, line_search=True, verbose=True, linear_solver="pytorch", linear_method="cg")

    # ``x`` is already globally replicated by the ROM initialization path.
    # Keep it as a regular tensor; only the sparse linear solve uses a
    # sharded DTensor internally.
    res, jac, res_norm = self.evaluate(x)
    jac = jac.to_sparse_coo()


    # > Set histories
    start = time()
    res_hist = [bkd.to_numpy(res)]
    res_norm_hist = [res_norm]
    step_hist = [0.0]
    self.model.runtime["total"] += time()-start
    # > Choose a sparse or dense linear solver depending on the problem
    if isinstance(x0, np.ndarray):
        solve = sp.linalg.spsolve if sp.issparse(jac) else np.linalg.solve
    else:
        if self.distributed_solve:
            solve = torch_sla.solve # TODO: this is the distributed iterative solve method, use direct for non-distributed cases?
        else:
            solve = torch_sla.spsolve_csr
    # > Print first step
    self.print_step(it, step_hist[-1], res_norm_hist[-1], header=True)

    if self.debug: logger.debug(" GLOBAL X SIZE = {}".format(global_size))
    if self.debug: logger.debug(" GLOBAL RES SIZE = {} JAC SHAPE = {}".format(res.shape, jac.shape))

    res_full = None
    local_res = None

    # Loop until convergence
    # ---------------
    flag = 0
    while ((res_norm_hist[-1] >= self.tol) and (it < self.maxit)):
      # > Initialize line search
      start = time()

      if self.use_direct_solve:
        # Assemble the current global KKT system after the first accepted
        # trial.  The initial residual/Jacobian were assembled above.
        if res_full is not None:
          res, jac, res_norm = self.evaluate(self._full_tensor(x))
          jac = jac.to_sparse_coo()

        # Both direct backends accept COO input.  Coalescing guarantees that
        # duplicate KKT entries are assembled before factorization.
        jac = jac.coalesce()
        if bkd.root():
          x_dt = torch_sla.spsolve_coo(
            jac,
            -res,
            backend=self.direct_solve_backend,
            # The saddle-point KKT matrix is generally indefinite.
            method="lu",
          )
        else:
          x_dt = torch.empty_like(res)

        # The Newton iterate is replicated, so every rank needs the complete
        # direction even though only rank zero factorizes the KKT matrix.
        x_dt = bkd.broadcast_tensor(x_dt, root=0)
      else:
        
        # TODO: fix
        # extract local portion of residual (b)

        if res_full is not None:
          logger.info(" CONVERTING RES TO LOCAL SHARD")

          #res, jac, res_norm = self.evaluate(self._full_tensor(x), False)
          res, jac, res_norm = self.evaluate(self._full_tensor(x))
          jac = jac.to_sparse_coo()

          inds = jac._indices()
          vals = jac._values()

          # TODO FIX:
          D = torch_sla.DSparseTensor.from_global_distributed(vals,
                                                              inds[0],
                                                              inds[1],
                                                              shape=jac.shape,
                                                              rank=bkd.get_rank(),
                                                              world_size=bkd.get_nranks(),
                                                              partition_method="simple",
                                                              device=bkd.device(),
                                                              verbose=True)

          local_res = D.scatter(-res)

        else:
          inds = jac._indices()
          vals = jac._values()

          # TODO FIX:
          D = torch_sla.DSparseTensor.from_global_distributed(vals,
                                                              inds[0],
                                                              inds[1],
                                                              shape=jac.shape,
                                                              rank=bkd.get_rank(),
                                                              world_size=bkd.get_nranks(),
                                                              partition_method="simple",
                                                              device=bkd.device(),
                                                              verbose=True)
          if self.debug: logger.debug(" EXTRACTING RES FROM D PARTITION")
          local_res = D.scatter(-res)
          if self.debug: logger.debug(" LOCAL RES = {}".format(local_res.shape))
        # The outer Newton solve targets 1e-8; a 1e-6 inner absolute
        # tolerance leaves the Newton direction too inaccurate near
        # convergence and causes line-search step exhaustion.
        with torch_sla.SolverConfig(
            backend="auto",
            method="fgmres",
            preconditioner=self.preconditioner,
            atol=1e-9,
            rtol=1e-9,
            maxiter=100,
            verbose=bkd.root(),
        ):

          x_dt = D.solve_distributed_shard(
              local_res,
              restart=30,
          )
      
      dx = x_dt


      delta = time()-start
      self.model.runtime["total"] += delta
      self.model.runtime["lin_solve"] += delta
      # > Armijo line search or an undamped Newton step
      if self.use_line_search:
        eval_res_tol = lambda stepsize: (1.0 - 2e-4*stepsize)*res_norm_hist[-1]
        x, res, cres, res_norm, stepsize = self.line_search(
          x,
          dx,
          eval_res_tol,
          use_global=False,
        )
      else:
        x, res, cres, res_norm, stepsize = self.full_step(x, dx)
      
      if self.debug: logger.debug(" DONE LINE SEARCH")
      if self.debug: logger.debug(" CRES SHAPE {} RES SHAPE = {}".format(cres.shape, res.shape))
      res_local = res.to_local() if isinstance(res, DTensor) else res
      res_full = bkd.gatherv_tensor(res_local)
      res_full = bkd.broadcast_tensor(res_full)
      res_full = torch.cat((res_full, cres))

      if self.debug: logger.debug("END NEWTON ITER:")

      if self.debug: logger.debug(" RES FULL SHAPE = {}".format(res_full.shape))
      # > Update
      start = time()
      it += 1
      res_hist.append(bkd.to_numpy(res_full))
      res_norm_hist.append(float(res_norm))
      step_hist.append(stepsize)
      self.model.runtime["total"] += time()-start
      # > Print step
      self.print_step(it, step_hist[-1], res_norm_hist[-1])
      # > Check convergence
      if (stepsize < self.stepsize_min):
        flag = 1
        break
      if not np.isfinite(res_norm):
        flag = 2
        break
      # Check if residual has plateaued (no decrease in past 5 iterations)
      if len(res_norm_hist) >= 6:
        # Check if there's no improvement in all of the last 5 consecutive iterations
        # Relative tolerance: 0.001% improvement required
        plateau_rel_tol = 1e-5
        no_improvement = True
        for i in range(5):
          improvement = res_norm_hist[-6+i] - res_norm_hist[-5+i]
          relative_improvement = improvement / res_norm_hist[-6+i] if res_norm_hist[-6+i] != 0 else 0
          if relative_improvement >= plateau_rel_tol:
            no_improvement = False
            break
        if no_improvement:
          # Residual plateaued - continue with current solution
          flag = 4
          # Exit the loop but keep the current solution
          break
    # Preserve a specific termination reason (or success) from the final
    # iteration.  Reaching maxit is a failure only if the residual remains
    # above tolerance and no earlier condition stopped the solve.
    if (flag == 0 and res_norm_hist[-1] >= self.tol and it >= self.maxit):
      flag = 3
    
    # Return result
    # ---------------
    start = time()
    if bkd.is_torch_backend():
        out = (
            self._full_tensor(x),
            np.vstack(res_hist),
            np.array(res_norm_hist),
            np.array(step_hist),
            np.array(it, dtype=int).tolist(),
            np.array(flag, dtype=int).tolist()
        )
    else:
        out = (
            x,
            np.vstack(res_hist),
            np.array(res_norm_hist),
            np.array(step_hist),
            np.array(it).reshape(1),
            np.array(flag).reshape(1)
        )
    self.model.runtime["total"] += time()-start
    return out
