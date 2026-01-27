import numpy as np
import scipy as sp

from time import time

from . import dtypes
from .basic import Solver


class GaussNewton(Solver):
  """
  A solver for optimization problems using the Gauss-Newton method.

  This class implements the Gauss-Newton method, which is used to solve 
  nonlinear least squares problems. The Gauss-Newton method approximates 
  the Hessian matrix using the Jacobian of the residuals, making it 
  suitable for overdetermined nonlinear problems.

  The solver aims to minimize the sum of squared residuals, formulated as 
  minimizing the following function:

  .. math::
    \\min \\frac{1}{2} \\| \\mathbf{r}(\\mathbf{x}) \\|^2

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
    verbose: bool = False
  ) -> None:
    super(GaussNewton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    self.squared_res = True

  def solve(
    self,
    x0: np.ndarray
  ) -> dtypes.SOL_TYPE:
    """
    Solve a minimization problem using the Gauss-Newton method.

    :param x0: Initial guess.
    :type x0: np.ndarray

    :return: A tuple containing:
      - x (np.ndarray): Solution of the minimization problem.
      - res_hist (np.ndarray): Residual vector history.
      - conv_hist (np.ndarray): Convergence history as 
        :math:\\|\\mathbf{J}^\intercal \mathbf{r}(\mathbf{x})\\|_2.
      - step_hist (np.ndarray): History of step sizes.
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
    conv_hist = [np.linalg.norm(jac.T@res)]
    step_hist = [0.0]
    self.model.runtime["total"] += time()-start
    # > Print first step
    self.print_step(it, step_hist[-1], conv_hist[-1], header=True)
    # Loop until convergence
    # ---------------
    flag = 0
    while ((conv_hist[-1] >= self.tol) & (it < self.maxit)):
      # > Initialize line search
      start = time()
      dx, minval = sp.linalg.lstsq(jac,-res)[:2]
      delta = time()-start
      self.model.runtime["total"] += delta
      self.model.runtime["lin_solve"] += delta
      # > Armijo line search
      eval_res_tol = lambda stepsize: res_norm+2e-4*stepsize*(minval-res_norm)
      x, res, jac, res_norm, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      start = time()
      it += 1
      res_hist.append(res)
      conv_hist.append(np.linalg.norm(jac.T@res))
      step_hist.append(stepsize)
      self.model.runtime["total"] += time()-start
      # > Print step
      self.print_step(it, step_hist[-1], conv_hist[-1])
      # > Check convergence
      if (stepsize < self.stepsize_min):
        flag = 1
        break
      if np.isnan(res_norm):
        flag = 2
        break
    if (it == self.maxit):
      flag = 3
    # Return result
    # ---------------
    start = time()
    out = (
      x,
      np.vstack(res_hist),
      np.array(conv_hist),
      np.array(step_hist),
      np.array(it).reshape(1),
      np.array(flag).reshape(1)
    )
    self.model.runtime["total"] += time()-start
    return out
