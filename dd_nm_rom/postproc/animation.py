import os
import numpy as np

from typing import List, Union
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from dd_nm_rom.fom import domain_dec as dd_mod
from dd_nm_rom.elements import mesh as mesh_mod

from .utils import set_style
plt = set_style(plt)


def _create_animation(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  lim: List[float],
  label: str,
  frames: int,
  show_labels: bool
) -> FuncAnimation:
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Set up axes
  if show_labels:
    ax.set_xlabel("$x$", labelpad=4)
    ax.set_ylabel("$y$", labelpad=12)
  # Initialize lines
  style = dict(
    cmap="viridis",
    shading="auto",
    vmin=lim[0],
    vmax=lim[1]
  )
  data = [plt.pcolormesh(x, y, z[0], **style)]
  # Add color bar
  label = label if show_labels else None
  cbar = plt.colorbar(orientation="vertical", label=label)
  # cbar.formatter.set_powerlimits((0, 0))
  # Tight layout
  plt.tight_layout()
  # Animate function
  def _animate(frame):
    # Set data
    data[0].set_array(z[frame+1])
    # Rescale axis limits
    ax.relim()
    ax.autoscale_view(tight=True)
    return data
  # Get animation
  anim = FuncAnimation(
    fig,
    _animate,
    frames=frames-1,
    blit=True
  )
  plt.close("all")
  return anim

def animate(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  lim: List[float],
  label: str,
  show_labels: bool = True,
  frames: Union[int, None] = None,
  fps: int = 10,
  filename: str = "./state.gif",
  dpi: int = 600,
  save: bool = True
) -> FuncAnimation:
  if (frames is None):
    frames = len(z)
  # Create animation
  anim = _create_animation(x, y, z, lim, label, frames, show_labels)
  # Save animation
  if save:
    anim.save(filename, writer="imagemagick", fps=fps, dpi=dpi)
  return anim

def animate_fom_rom(
  path: str,
  mesh: mesh_mod.MESH_TYPES,
  uv_fom: dd_mod.dtypes.UV_TYPE,
  uv_rom: Union[dd_mod.dtypes.UV_TYPE, None] = None,
  show_labels: bool = True
) -> None:
  for x_k in ("u", "v"):
    path_k = path + f"/{x_k}/"
    os.makedirs(path_k, exist_ok=True)
    zt = uv_fom["res"][x_k]
    lim = [zt.min(), zt.max()]
    if (uv_rom is not None):
      z = uv_rom["res"][x_k]
      label = "$\hat{%s}$" % x_k
      err = np.abs(z - zt)
      err_lim = [err.min(), err.max()]
    else:
      z = zt
      label = "$%s$" % x_k
      err, err_lim = None, None
    # Solution
    z = z.T.reshape(-1, mesh.n["y"], mesh.n["x"])
    animate(
      *mesh.grid,
      z=z,
      lim=lim,
      label=label,
      show_labels=show_labels,
      filename=path_k + "/state.gif",
      dpi=300,
      save=True
    )
    # Error
    if (err is not None):
      z = err.T.reshape(-1, mesh.n["y"], mesh.n["x"])
      label = "$|\hat{%s}-%s|$" % (x_k, x_k)
      animate(
        *mesh.grid,
        z=z,
        lim=err_lim,
        label=label,
        show_labels=show_labels,
        filename=path_k + "/state_err.gif",
        dpi=300,
        save=True
      )
