import os
import numpy as np

from matplotlib import pyplot as plt
from typing import List, Union, Tuple
from dd_nm_rom.fom import domain_dec as dd_mod
from dd_nm_rom.elements import mesh as mesh_mod

from .utils import set_style
plt = set_style(plt)


def plot_field(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  label: str,
  show_labels: bool = True,
  lim: Union[List[float], None] = None,
  figsize: Union[Tuple[int], None] = None,
  cmap: str = "viridis",
  filename: str = "./state.png",
  save: bool = True,
  show: bool = False
) -> None:
  if (figsize is not None):
    plt.figure(figsize=figsize)
  else:
    plt.figure()
  style = dict(
    cmap=cmap,
    shading="auto"
  )
  if (lim is not None):
    style["vmin"] = lim[0]
    style["vmax"] = lim[1]
  plt.pcolormesh(x, y, z, **style)
  if show_labels:
    plt.xlabel("$x$", labelpad=4)
    plt.ylabel("$y$", labelpad=12)
  label = label if show_labels else None
  cbar = plt.colorbar(orientation="vertical", label=label)
  # cbar.formatter.set_powerlimits((0, 0))
  if save:
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
  if show:
    plt.show()
  plt.close()

def plot_field_fom_rom(
  path: str,
  mesh: mesh_mod.MESH_TYPES,
  uv_fom: dd_mod.dtypes.UV_TYPE,
  uv_rom: Union[dd_mod.dtypes.UV_TYPE, None] = None,
  index: Union[int, None] = None,
  figsize: Union[Tuple[int], None] = None,
  show_labels: bool = True,
  use_fom_snapshot_limits: bool = False
) -> None:
  for x_k in ("u", "v"):
    path_k = path + f"/{x_k}/"
    os.makedirs(path_k, exist_ok=True)

    zt = uv_fom["res"][x_k]

    # --- State limits ---
    if use_fom_snapshot_limits and index is not None:
      # scale to FOM at this index
      zt_snap = zt.T[index]
      lim = [zt_snap.min(), zt_snap.max()]
    else:
      # original: global min/max over all snapshots
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
    filename = path_k + "state"
    if (index is not None):
      z = z.T[index]
      filename += f"_i{str(index).zfill(5)}"
    filename += ".png"
    plot_field(
      *mesh.grid,
      z=z.reshape(mesh.n["y"], mesh.n["x"]),
      lim=lim,
      figsize=figsize,
      label=label,
      show_labels=show_labels,
      filename=filename,
      save=True,
      show=False
    )
    # Error
    if (err is not None):
      filename = path_k + "state_err"
      if (index is not None):
        err = err.T[index]
        filename += f"_i{str(index).zfill(5)}"
      filename += ".png"
      label = "$|\hat{%s}-%s|$" % (x_k, x_k)
      plot_field(
        *mesh.grid,
        z=err.reshape(mesh.n["y"], mesh.n["x"]),
        lim=err_lim,
        figsize=figsize,
        label=label,
        show_labels=show_labels,
        filename=filename,
        save=True,
        show=False
      )
