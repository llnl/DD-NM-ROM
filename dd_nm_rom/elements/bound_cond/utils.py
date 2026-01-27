from typing import Tuple


# Constants for sides and types of boundary conditions
SIDES = {
  "x": ("left", "right"),
  "y": ("bottom", "top")
}
SIDES["all"] = SIDES["x"] + SIDES["y"]
TYPES = ("dirichlet", "neumann", "periodic")

def check_bc_type(bc_type: str) -> None:
  """
  Check if the given boundary condition type is valid.

  :param bc_type: The type of boundary condition.
  :type bc_type: str

  :raises ValueError: If the boundary condition type is not valid.
  """
  if (bc_type not in TYPES):
    raise ValueError(
      f"Could not interpret b.c. type: '{bc_type}'. " \
        f"Valid options are: {TYPES}"
    )

def check_side(side: str) -> None:
  """
  Check if the given boundary side is valid.

  :param side: The side of the boundary.
  :type side: str

  :raises ValueError: If the boundary side is not valid.
  """
  if (side not in SIDES["all"]):
    raise ValueError(
      f"Could not interpret b.c. side: '{side}'. " \
        f"Valid options are: {SIDES['all']}"
    )

def get_axis_method(side: str) -> Tuple[str, str]:
  """
  Get the axis and method (forward or backward) for the given boundary side.

  :param side: The side of the boundary.
  :type side: str
  :return: A tuple containing the axis ('x' or 'y')
           and the method ('fwd' or 'bwd').
  :rtype: Tuple[str, str]

  :raises ValueError: If the boundary side is not valid.
  """
  check_side(side)
  axis = "x" if (side in SIDES["x"]) else "y"
  method = "fwd" if (side in ("left", "bottom")) else "bwd"
  return axis, method
