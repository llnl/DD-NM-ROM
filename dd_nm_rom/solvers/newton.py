import numpy as np
import scipy.sparse as sp

from time import time

from . import dtypes
from .basic import Solver


class Newton(Solver):
  """
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
  :param verbose: Whether to print iteration details. Defaults to False.
  :type verbose: bool
  """

  def __init__(
    self,
    model: callable,
    tol: float = 1e-3,
    maxit: int = 20,
    stepsize_min: float = 1e-10,
    iostep: int = 1,
    verbose: bool = False
  ) -> None:
    super(Newton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      iostep=iostep,
      verbose=verbose
    )

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
    res, jac, res_norm = self.evaluate(x)
    # > Set histories
    start = time()
    res_hist = [res]
    res_norm_hist = [res_norm]
    step_hist = [0.0]
    self.model.runtime["total"] += time()-start
    # > Choose a sparse or dense linear solver depending on the problem
    solve = sp.linalg.spsolve if sp.issparse(jac) else np.linalg.solve
    # > Print first step
    self.print_step(it, step_hist[-1], res_norm_hist[-1], header=True)
    # Loop until convergence
    # ---------------
    flag = 0
    while ((res_norm_hist[-1] >= self.tol) and (it < self.maxit)):
      # > Initialize line search
      start = time()
      dx = solve(jac,-res)
      delta = time()-start
      self.model.runtime["total"] += delta
      self.model.runtime["lin_solve"] += delta
      # > Armijo line search
      eval_res_tol = lambda stepsize: (1.0 - 2e-4*stepsize)*res_norm_hist[-1]
      x, res, jac, res_norm, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      start = time()
      it += 1
      res_hist.append(res)
      res_norm_hist.append(res_norm)
      step_hist.append(stepsize)
      self.model.runtime["total"] += time()-start
      # > Print step
      self.print_step(it, step_hist[-1], res_norm_hist[-1])
      # > Check convergence
      if (stepsize < self.stepsize_min):
        flag = 1
        break
      if np.isnan(res_norm):
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
    if (it == self.maxit):
      flag = 3
    # Return result
    # ---------------
    start = time()
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
