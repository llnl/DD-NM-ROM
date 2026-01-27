import os
import numpy as np

from matplotlib import pyplot as plt
from typing import List, Union, Tuple
from dd_nm_rom.elements import mesh as mesh_mod

from .utils import set_style
plt = set_style(plt)


def plot_hr_nodes(
  mesh: mesh_mod.MESH_TYPES,
  dd_rom: callable,
  figsize: Union[Tuple[int], None] = None,
  path: str = "./",
  save: bool = True,
  show: bool = False
) -> None:
  path = path + "/hr_nodes/"
  os.makedirs(path, exist_ok=True)
  x, y = mesh.grid
  x, y = x.flatten(), y.flatten()
  for k in ("u", "v"):
    fig = plt.figure() if (figsize is None) else plt.figure(figsize=figsize)
    plt.xlim(mesh.phylim["x"])
    plt.ylim(mesh.phylim["y"])
    plt.xlabel("$x$", labelpad=4)
    plt.ylabel("$y$", labelpad=12)
    for sub in dd_rom.subdomains:
      i = sub.hr_nodes_res[k]
      j = sub.sub_fom.elem_states["res"].nodes_state[i]
      plt.scatter(x[j], y[j], s=10)
    # plt.title(f"Sampled HR nodes for ${k}$")
    if save:
      plt.savefig(path+f"/{k}.png", bbox_inches="tight", pad_inches=0.1)
    if show:
      plt.show()
    plt.close()
