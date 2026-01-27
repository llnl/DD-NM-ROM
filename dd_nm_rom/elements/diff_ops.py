import numpy as np
import scipy.sparse as sp

from typing import List

from . import mesh as mesh_mod
from . import bound_cond as bc_mod


class DiffOperators(object):
  """
  A class to build and manage differential operators for a given mesh 
  and boundary conditions.

  :param nu: Viscosity or diffusion coefficient.
  :type nu: float
  :param bc: Boundary conditions.
  :type bc: BC_TYPES
  :param mesh: Mesh object containing grid information.
  :type mesh: MESH_TYPES
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    nu: float,
    bc: bc_mod.BC_TYPES,
    mesh: mesh_mod.MESH_TYPES,
    upwind: bool,
    upwind_order: int = 2,
  ) -> None:
    self.nu = nu
    self.bc = bc
    self.mesh = mesh
    self.built = False
    self.upwind = upwind
    self.upwind_order = upwind_order  # 1 for first-order, 2 for second-order

  # Building
  # ===================================
  def is_built(self) -> None:
    """
    Check if the differential operators have been built.

    :raises ValueError: If the differential operators are not built.
    """
    if (not self.built):
      raise ValueError(
        "Differential operators not built. Please, call 'build' method first."
      )

  def build(self) -> None:
    """
    Build the differential operators for the mesh and boundary conditions.
    """

    if self.upwind:
      if self.upwind_order == 1:
          print('using first order')
          return self.build_upwind_1st()
      else:
          return self.build_upwind_2nd_gen()
#         return self.build_upwind_2nd()

    self.ops = {"D": 0.0}
    for axis in ("x", "y"):
      h = self.mesh.h[axis]
      Ai = self._build_op(axis, stencil=[-1,  0, 1], diags=[-1, 0, 1])
      Di = self._build_op(axis, stencil=[ 1, -2, 1], diags=[-1, 0, 1])
      self.ops[f"A{axis}"] = (-0.5/h) * Ai
      self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di
    self.built = True

  def _build_op(
    self,
    axis: str,
    stencil: List[int],
    diags: List[int]
  ) -> sp.spmatrix:
    """
    Build a differential operator for a given axis.

    :param axis: The axis for which to build the operator ('x' or 'y').
    :type axis: str
    :param stencil: Coefficients for the finite difference stencil.
    :type stencil: List[int]
    :param diags: Diagonals for the sparse matrix representation.
    :type diags: List[int]

    :return: The constructed sparse matrix operator.
    :rtype: sp.spmatrix
    """
    n = self.mesh.n
    e = np.ones(n[axis])
    # Operator
    op = sp.spdiags([c*e for c in stencil], diags, n[axis], n[axis])
    # Update with BC
    if (self.bc.op is not None):
      for (method, bc_op) in self.bc.op[axis].items():
        index = 0 if (method == "fwd") else -1
        op += stencil[index] * bc_op
    # Map over 2D grid
    if (axis == "x"):
      op = sp.kron(sp.eye(n["y"]), op)
    else:
      op = sp.kron(op, sp.eye(n["x"]))
    return op.tocsr()

  def build_upwind_1st(self) -> None:
    """
    Build differential operators using first-order upwind scheme,
    assuming positive flow direction for both x and y.
    """
    self.ops = {"D": 0.0}
    for axis in ("x", "y"):
        h = self.mesh.h[axis]

        # First-order backward difference for first derivative (for positive flow)
        # Formula: (f_i - f_{i-1})/h
        upwind_stencil = [-1, 1, 0]  # Coefficients for points i-1, i, i+1
        upwind_diags = [-1, 0, 1]   # Diagonal positions

        # We can use the standard _build_op since first-order has same stencil width
        Ai = self._build_op(axis, stencil=upwind_stencil, diags=upwind_diags)
        self.ops[f"A{axis}"] = (-1.0/h) * Ai  # Negative sign as in 2nd order

        # Standard second-order central for diffusion (unchanged)
        Di = self._build_op(axis, stencil=[1, -2, 1], diags=[-1, 0, 1])
        self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di

    self.built = True


  def build_upwind_2nd(self) -> None:
      """
      Build differential operators using second-order upwind scheme,
      assuming positive flow direction for both x and y.
      """
      self.ops = {"D": 0.0}
      for axis in ("x", "y"):
          h = self.mesh.h[axis]
          # Second-order backward difference for first derivative (for positive flow)
          # Formula: (3f_i - 4f_{i-1} + f_{i-2})/(2h)
          upwind_stencil = [1, -4, 3, 0]  # Coefficients for points i-2, i-1, i, i+1
          upwind_diags = [-2, -1, 0, 1]   # Diagonal positions
          Ai = self._build_extended_op(axis, upwind_stencil, upwind_diags)
          self.ops[f"A{axis}"] = (-1.0/(2*h)) * Ai
          # Standard second-order central for diffusion
          Di = self._build_op(axis, stencil=[1, -2, 1], diags=[-1, 0, 1])
          self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di
      self.built = True

  def _build_extended_op(
      self,
      axis: str,
      stencil: List[float],
      diags: List[int]
  ) -> sp.spmatrix:
      """
      Build a differential operator for wider stencils (second-order upwind).

      :param axis: The axis for which to build the operator ('x' or 'y')
      :param stencil: Coefficients for the finite difference stencil
      :param diags: Diagonals for the sparse matrix representation
      :return: The constructed sparse matrix operator
      """
      n = self.mesh.n
      e = np.ones(n[axis])
      # Create basic operator with the stencil
      op = sp.spdiags([c*e for c in stencil], diags, n[axis], n[axis])

      # Handle periodic boundary conditions for wider stencil
      if isinstance(self.bc, bc_mod.PeriodicBC):

          op = op.tolil()
          # For positive flow with backward bias [1, -4, 3, 0]
          # We need to connect:
          # - Point 0 needs data from points n-2 and n-1
          # - Point 1 needs data from point n-1
          op[0, n[axis]-2] = stencil[0]  # Connect point 0 to point n-2
          op[0, n[axis]-1] = stencil[1]  # Connect point 0 to point n-1
          op[1, n[axis]-1] = stencil[0]  # Connect point 1 to point n-1
      elif self.bc.op is not None:
          # For non-periodic boundaries, apply standard BC handling
          for (method, bc_op) in self.bc.op[axis].items():
              index = 0 if (method == "fwd") else -1
              op += stencil[index] * bc_op

      # Map to 2D grid as before
      if axis == "x":
          op = sp.kron(sp.eye(n["y"]), op)
      else:
          op = sp.kron(op, sp.eye(n["x"]))

      return op.tocsr()

  def build_upwind_2nd_gen(self) -> None:
      """
      Build differential operators using second-order upwind scheme,
      which handles general flow direction for both x and y.
      """
      self.ops = {"D": 0.0}

      for axis in ("x", "y"):
          h = self.mesh.h[axis]

          # Second-order backward difference for first derivative (for positive flow)
          # Formula: (3f_i - 4f_{i-1} + f_{i-2})/(2h)
#         pos_stencil = [1, -4, 3, 0]  # Coefficients for points i-2, i-1, i, i+1
#         pos_diags = [-2, -1, 0, 1]   # Diagonal positions
          pos_stencil = [1, -4, 3]  # Coefficients for points i-2, i-1, i, i+1
          pos_diags = [-2, -1, 0]   # Diagonal positions
#         Ai = self._build_extended_op(axis, upwind_stencil, upwind_diags)
          pos_op = self._build_extended_op_gen(axis, pos_stencil, pos_diags, flow_dir="pos")

          # Build negative flow operator (forward differencing)
          neg_stencil = [-1, 4, -3]  # f_{i+2}, f_{i+1}, f_i
          neg_diags = [2, 1, 0]
#         neg_stencil = [0, 3, -4, 1]  # [i-1, i, i+1, i+2]
#         neg_diags = [-1, 0, 1, 2]
          neg_op = self._build_extended_op_gen(axis, neg_stencil, neg_diags, flow_dir="neg")

          self.ops[f"A{axis}_pos"] = (-1.0/(2*h)) * pos_op
          self.ops[f"A{axis}_neg"] = (-1.0/(2*h)) * neg_op

          # For backward compatibility, default to positive flow
#         self.ops[f"A{axis}"] = self.ops[f"A{axis}_pos"]
          self.ops[f"A{axis}"] = self.ops[f"A{axis}_neg"]

          # Standard second-order central for diffusion
          Di = self._build_op(axis, stencil=[1, -2, 1], diags=[-1, 0, 1])
          self.ops["D"] = self.ops["D"] + (self.nu/h**2) * Di
      self.built = True

  def _build_extended_op_gen(
      self,
      axis: str,
      stencil: List[float],
      diags: List[int],
      flow_dir: str = "pos"  # Add flow direction parameter
  ) -> sp.spmatrix:
      """
      Build a differential operator for wider stencils (second-order upwind).

      :param axis: The axis for which to build the operator ('x' or 'y')
      :param stencil: Coefficients for the finite difference stencil
      :param diags: Diagonals for the sparse matrix representation
      :param flow_dir: Flow direction, either "pos" or "neg"
      :return: The constructed sparse matrix operator
      """
      n = self.mesh.n
      e = np.ones(n[axis])
      # Create basic operator with the stencil
      op = sp.spdiags([c*e for c in stencil], diags, n[axis], n[axis])

      # Handle periodic boundary conditions for wider stencil
      if isinstance(self.bc, bc_mod.PeriodicBC):
          op = op.tolil()

          if flow_dir == "pos":
            # For positive flow with backward bias [1, -4, 3, 0]
            # We need to connect:
            # - Point 0 needs data from points n-2 and n-1
            # - Point 1 needs data from point n-1
            op[0, n[axis]-2] = stencil[0]  # Connect point 0 to point n-2
            op[0, n[axis]-1] = stencil[1]  # Connect point 0 to point n-1
            op[1, n[axis]-1] = stencil[0]  # Connect point 1 to point n-1

          else:  # flow_dir == "neg"
            # For negative flow with forward bias [0, 3, -4, 1]
            # We need to connect:
            # - Point n-1 needs data from points 0 and 1
            # - Point n-2 needs data from point 0
#           op[n[axis]-1, 0] = stencil[3]  # Connect point n-1 to point 0
#           op[n[axis]-1, 1] = stencil[2]  # Connect point n-1 to point 1
#           op[n[axis]-2, 0] = stencil[3]  # Connect point n-2 to point 0
            op[n[axis]-2, 0] = stencil[0]  # f_{i+2} = f_0 for i = n-2
            op[n[axis]-1, 0] = stencil[1]  # f_{i+1} = f_0 for i = n-1
            op[n[axis]-1, 1] = stencil[0]  # f_{i+2} = f_1 for i = n-1

      elif self.bc.op is not None:
          # For non-periodic boundaries, apply standard BC handling
          # Note that the code below has not been extended to handle non-periodic bc
          # Need to update bc_op
          for (method, bc_op) in self.bc.op[axis].items():
              index = 0 if (method == "fwd") else -1
              op += stencil[index] * bc_op

      # Map to 2D grid as before
      if axis == "x":
          op = sp.kron(sp.eye(n["y"]), op)
      else:
          op = sp.kron(op, sp.eye(n["x"]))

      return op.tocsr()