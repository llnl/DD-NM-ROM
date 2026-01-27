import abc
import numpy as np

from time import time
from typing import Tuple, Union
from typing_extensions import Unpack

from . import dtypes


class Solver(object):
  """
  Base class for solvers.

  This class provides the basic framework for solvers that integrate a
  physical model and use iterative methods for solving an optimization 
  problem at each time step.

  :param model: The physical model to be used by the solver. It should be a
                callable that provides residual and Jacobian calculations.
  :type model: callable
  :param tol: Tolerance for convergence. The solver stops when the residual 
              norm is below this threshold. Defaults to 1e-3.
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
    self.model = model
    self.tol = tol
    self.maxit = maxit
    self.stepsize_min = stepsize_min
    self.iostep = iostep
    self.verbose = verbose
    self.squared_res = False
    self.set_header()

  def set_header(self) -> None:
    """Initialize the header for printing iteration details."""
    self.header = "|   Iteration |    Stepsize |    Residual |\n| "
    n_chars = len(self.header.split("|")[1])-2
    for _ in range(3):
      self.header += "-"*n_chars + " | "
    self.header = self.header[:-1]

  # Call function
  # ===================================
  def __call__(
    self,
    x0: np.ndarray,
    dt: float = 0.0,
    nt: float = 1,
    guess: Union[np.ndarray, None] = None,
    use_guess: bool = False
  ) -> Tuple[np.ndarray, Unpack[dtypes.SOL_TYPE]]:
    """
    Integrate the model over time steps using the solver.

    :param x0: Initial guess for the solution.
    :type x0: np.ndarray
    :param dt: Time step size. Defaults to 0.0 (no time-stepping).
    :type dt: float
    :param nt: Number of time steps. Defaults to 1.
    :type nt: float
    :param guess: Optional array of guesses for each time step.
    :type guess: Union[np.ndarray, None]
    :param use_guess: Whether to use the provided guesses for the initial guess.
    :type use_guess: bool

    :return: A tuple containing:
      - x (np.ndarray): Array of solutions at each time step.
      - Additional elements depending on the `solve` method output.
    :rtype: Tuple[np.ndarray, Unpack[SOL_TYPE]]
    """
    return self.integrate(x0, dt, nt, guess, use_guess)

  def integrate(
    self,
    x0: np.ndarray,
    dt: float = 0.0,
    nt: float = 1,
    guess: Union[np.ndarray, None] = None,
    use_guess: bool = False
  ) -> Tuple[np.ndarray, Unpack[dtypes.SOL_TYPE]]:
    """
    Integrate the model over time steps using the solver.

    :param x0: Initial guess for the solution.
    :type x0: np.ndarray
    :param dt: Time step size. Defaults to 0.0 (no time-stepping).
    :type dt: float
    :param nt: Number of time steps. Defaults to 1.
    :type nt: float
    :param guess: Optional array of guesses for each time step.
    :type guess: Union[np.ndarray, None]
    :param use_guess: Whether to use the provided guesses for the initial guess.
    :type use_guess: bool

    :return: A tuple containing:
      - x (np.ndarray): Array of solutions at each time step.
      - Additional elements depending on the `solve` method output.
    :rtype: Tuple[np.ndarray, Unpack[SOL_TYPE]]
    """
    # Initialize
    start = time()
    x = [x0]
    self.model.t = 0.0
    self.model.dt = dt
    self.model.x_old = x0
    self.model.runtime["total"] += time()-start
    # Loop over time steps
    for i in range(nt):
      if self.verbose:
        print("Time step {0:4d}/{1:d}".format(i+1,nt))
        texec = time()
      # Solve
      self.model.t += dt
      xi, *step = self.solve(self.model.x_old)
      # Check convergence
      res_norm, it, flag = step[1][-1], int(step[-2]), int(step[-1])
      self.print_conv(res_norm, it, flag)
      start = time()
      # Update
      self.model.x_old = xi
      if use_guess:
        self.model.x_old = guess[i]
      # Store
      if ((i+1) % self.iostep == 0):
        x.append(xi)
      if (i == 0):
        steps = [[] for _ in step]
      for (j, obj) in enumerate(step):
        steps[j].append(obj)
      if (flag != 0) and (flag != 1) and (flag != 4):
        break
      self.model.runtime["total"] += time()-start
      if self.verbose:
        print("Execution time: {:.5e} s".format(time()-texec))
    # Return
    start = time()
    x = np.vstack(x).T
    if ((dt == 0.0) and (nt == 1)):
      x = x[:,-1]
      if x.ndim == 1:
        x = x[np.newaxis, :].T
      steps = [obj[-1] for obj in steps]
    self.model.runtime["total"] += time()-start
    return x, *steps

  @abc.abstractmethod
  def solve(
    self,
    x0: np.ndarray
  ) -> dtypes.SOL_TYPE:
    """
    Solve an optimization problem.

    :param x0: Initial guess.
    :type x0: np.ndarray

    :return: The solution and additional information based on the solver.
    :rtype: SOL_TYPE
    """
    pass

  # Util functions
  # ===================================
  # Solving
  # -----------------------------------
  def line_search(
    self,
    x0: np.ndarray,
    dx: np.ndarray,
    eval_res_tol: callable
  ) -> Tuple[np.ndarray, Unpack[dtypes.EVAL_TYPE], float]:
    """
    Perform Armijo line search to find a suitable step size.

    :param x0: Current solution.
    :type x0: np.ndarray
    :param dx: Direction vector for the line search.
    :type dx: np.ndarray
    :param eval_res_tol: Function to evaluate the residual tolerance.
    :type eval_res_tol: callable

    :return: A tuple containing:
      - x (np.ndarray): Updated solution after line search.
      - res (np.ndarray): Residual vector.
      - jac (np.ndarray): Jacobian matrix.
      - res_norm (float): Residual norm.
      - stepsize (float): Final step size used.
    :rtype: Tuple[np.ndarray, Unpack[EVAL_TYPE], float]
    """
    # Initialize
    # -------------
    start = time()
    stepsize = 1.0
    x = x0 + stepsize*dx
    self.model.runtime["total"] += time()-start
    res, jac, res_norm = self.evaluate(x)
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
      start = time()
      stepsize *= 0.5
      x = x0 + stepsize*dx
      self.model.runtime["total"] += time()-start
      res, jac, res_norm = self.evaluate(x)
      # Condition
      # -------------
      start = time()
      cond = cond_fun(res_norm, stepsize)
      self.model.runtime["total"] += time()-start
    return x, res, jac, res_norm, stepsize

  def evaluate(
    self,
    x: np.ndarray
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
    res, jac = self.model.res_jac(x)
    start = time()
    res_norm = np.dot(res,res)
    if (not self.squared_res):
      res_norm = np.sqrt(res_norm)
    self.model.runtime["total"] += time()-start
    return res, jac, float(res_norm)

  # Printing
  # -----------------------------------
  def print_step(
    self,
    it: int,
    stepsize: float,
    res_norm: float,
    header: bool = False
  ) -> None:
    """
    Print the current step details, including the iteration number, step size, 
    and residual.

    :param it: Current iteration number.
    :type it: int
    :param stepsize: Size of the step taken.
    :type stepsize: float
    :param res_norm: Residual value.
    :type res_norm: float
    :param header: Whether to print the header. Defaults to False.
    :type header: bool

    :return: None
    :rtype: None
    """
    if self.verbose:
      if header:
        print(self.header)
      print(dtypes.PRINT_FMT.format(it, stepsize, res_norm))

  def print_conv(
    self,
    res_norm: float,
    it: int,
    flag: int
  ) -> None:
    """
    Print the convergence status based on the residual value, iteration count,
    and flag.

    The convergence flag provides information about why the solver terminated.
    The possible values are:
      - `0`: The solver terminated successfully. The method converged to a
             solution with a residual norm below the tolerance.
      - `1`: The stepsize became too small to continue the search. This usually
             means that the solver is having trouble making progress, possibly
             due to a highly non-linear problem or an ill-conditioned Jacobian.
      - `2`: The residual value became 'nan' (not a number). This indicates
             that the computation encountered numerical issues, such as
             overflow or invalid operations.
      - `3`: The solver failed to converge within the maximum number of
             iterations specified. This might occur if the problem is too
             difficult or if the solver parameters (e.g., tolerance, maximum
             iterations) are not suitable for the given problem.

    :param res_norm: The residual value at the end of the computation.
    :type res_norm: float
    :param it: The number of iterations completed before termination.
    :type it: int
    :param flag: The convergence flag indicating the reason for termination.
    :type flag: int

    :return: None
    :rtype: None
    """
    if (flag == 1):
      print(
        f"Too small stepsize found at iteration {it}. " \
        "This may indicate that the problem is highly " \
        "non-linear or the Jacobian is ill-conditioned."
      )
    elif (flag == 2):
      print(
        "The residual value is 'nan'. This suggests " \
        "a numerical issue occurred during the computation, " \
        "such as overflow or invalid operations."
      )
    elif (flag == 3):
      print(
        f"Solver failed to converge in {self.maxit} iterations. " \
        "The solver did not reach the desired tolerance " \
        "within the maximum number of iterations."
      )
    elif (flag == 4):
      print(
        f"Residual plateaued at iteration {it}. " \
        "Relative improvement < 0.001% over the past 5 consecutive iterations. " \
        "Continuing with current solution."
      )
    else:
      if self.verbose:
        print(
          f"Solver terminated after {it} iterations " \
          f"with residual norm of {res_norm:1.4e}."
        )
